#include "robmove/workflow.hpp"
#include <utility>

namespace robmove {
Workflow::Workflow(Config config, RobotPort& robot, PerceptionPort& perception, Clock& clock,
                   EventSink* events, CancelCheck cancel)
    : config_(std::move(config)), task_(config_, robot, perception, clock, std::move(cancel)), clock_(clock), events_(events) {}

void Workflow::state(State value, std::string message) {
    if (events_) events_->emit({value, cycle_, placed_, clock_.now() - start_time_, std::move(message)});
}
WorkflowResult Workflow::hold(const std::string& reason, bool cancelled) {
    task_.hold(reason);
    state(State::Hold, reason);
    return {false, placed_, true, cancelled, task_.custody(), reason};
}
WorkflowResult Workflow::done(const std::string& reason) {
    task_.check_active();
    state(State::Done, reason);
    return {true, placed_, false, false, task_.custody(), reason};
}

WorkflowResult Workflow::run(Mode mode) {
    if (started_) throw std::logic_error("workflow can run only once");
    started_ = true;
    struct Cleanup { TaskExecutor& task; ~Cleanup() { task.shutdown(); } } cleanup{task_};
    start_time_ = clock_.now();
    try {
        config_.validate(mode);
        state(State::Init);
        task_.initialize(mode == Mode::Cyclic);
        return mode == Mode::Single ? single() : cyclic();
    } catch (const Cancelled& error) { return hold(error.what(), true); }
    catch (const std::exception& error) { return hold(error.what()); }
    catch (...) { return hold("unknown adapter exception"); }
}

PerceptionResult Workflow::perceive_with_retries(bool retry_empty) {
    PerceptionResult result;
    for (int attempt = 0; attempt < config_.task.perception_retries; ++attempt) {
        state(State::CameraPerception, "attempt=" + std::to_string(attempt + 1));
        result = task_.perceive();
        state(State::CameraPerception, result.reason);
        if (result.status == PerceptionStatus::Found || (!retry_empty && result.status == PerceptionStatus::Empty)) return result;
        if (attempt + 1 < config_.task.perception_retries) task_.wait(config_.task.perception_retry_interval_s);
    }
    return result;
}

WorkflowResult Workflow::single() {
    state(State::ArmHomeForPerception);
    if (!task_.home()) return hold("home before perception failed");
    state(State::LoadPerceptionModel);
    task_.load_perception();
    const auto result = perceive_with_retries(true);
    if (result.status == PerceptionStatus::Empty) return done("no eligible target after retries");
    if (result.status != PerceptionStatus::Found || !result.target) return hold("perception unavailable: " + result.reason);
    const auto& target = *result.target;
    state(State::Grasp, target.class_name);
    if (!task_.pick(target)) return hold("grasp or grip confirmation failed");
    state(State::Lift);
    if (!task_.lift(target)) return hold("lift failed");
    state(State::LocalPlace);
    if (!task_.place(target, false)) return hold("local place or release failed");
    ++placed_;
    state(State::RetreatFromPlace);
    if (!task_.retreat(target, false)) return hold("retreat after place failed");
    state(State::ArmHomeAfterRelease);
    if (!task_.home()) return hold("home after release failed");
    return done("single pick and place complete");
}

WorkflowResult Workflow::cyclic() {
    state(State::NavToPickup);
    if (!task_.navigate(*config_.navigation.pickup_pose, false)) return hold("navigation to pickup failed");
    state(State::ArmHomeForPerception);
    if (!task_.home()) return hold("home before perception failed");
    state(State::LoadPerceptionModel);
    task_.load_perception();
    int empty_count = 0;
    while (true) {
        const auto result = perceive_with_retries(false);
        if (result.status == PerceptionStatus::Unavailable) return hold("perception retries exhausted: " + result.reason);
        if (result.status == PerceptionStatus::Empty) {
            if (++empty_count >= config_.task.empty_confirmations) {
                state(State::NavToOrigin);
                if (!task_.navigate(*config_.navigation.origin_pose, false)) return hold("return to origin failed");
                return done("pickup area confirmed empty");
            }
            task_.wait(config_.task.perception_retry_interval_s);
            continue;
        }
        empty_count = 0;
        if (!result.target) return hold("FOUND observation has no target");
        const auto& target = *result.target;
        state(State::Grasp, target.class_name);
        if (!task_.pick(target)) return hold("grasp or grip confirmation failed");
        state(State::Lift);
        if (!task_.lift(target)) return hold("lift failed");
        state(State::LiftSafe);
        if (!task_.lift_safe(target)) return hold("horizontal retreat before navigation failed");
        state(State::ArmHomeWithObject);
        if (!task_.home()) return hold("home with object failed");
        state(State::NavToPlace);
        if (!task_.navigate(*config_.navigation.place_pose, true)) return hold("navigation to place failed");
        state(State::Release);
        if (!task_.place(target, true)) return hold("place or release failed");
        ++placed_;
        state(State::RetreatToLiftSafe);
        if (!task_.retreat(target, true)) return hold("retreat after release failed");
        state(State::ArmHomeAfterRelease);
        if (!task_.home()) return hold("home after release failed");
        if (placed_ >= config_.task.max_cycles) return hold("max_cycles reached; inspection required");
        cycle_ = placed_ + 1;
        state(State::NavToPickup);
        if (!task_.navigate(*config_.navigation.pickup_pose, true)) return hold("return to pickup failed");
        state(State::ArmHomeForPerception);
        if (!task_.home()) return hold("home at pickup failed");
    }
}
}  // namespace robmove
