#include "robmove/task.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace robmove {
GripMonitor::GripMonitor(const RobotConfig& config)
    : threshold_(config.gripper_grasp_effort_threshold),
      tolerance_(config.gripper_grasp_width_stable_tolerance_m),
      maximum_width_(config.gripper_open_width + tolerance_),
      required_(config.gripper_grasp_effort_stable_samples) {
    if (!std::isfinite(threshold_) || threshold_ <= 0 || !std::isfinite(tolerance_) || tolerance_ < 0 || required_ <= 0)
        throw std::invalid_argument("invalid grip monitor settings");
}

bool GripMonitor::update(const std::optional<GripperState>& sample) {
    if (!sample || !std::isfinite(sample->width) || sample->width < 0 || sample->width > maximum_width_ ||
        !std::isfinite(sample->effort)) {
        last_width_.reset(); stable_ = 0; return false;
    }
    const bool stable_width = last_width_ && std::abs(sample->width - *last_width_) <= tolerance_;
    last_width_ = sample->width;
    if (std::abs(sample->effort) >= threshold_ && stable_width) ++stable_;
    else stable_ = 0;
    // Some grippers report is_moving even while holding an object; force and
    // consecutive width samples are the authoritative confirmation.
    return stable_ >= required_;
}

TaskExecutor::TaskExecutor(const Config& config, RobotPort& robot, PerceptionPort& perception,
                           Clock& clock, CancelCheck cancel)
    : config_(config), robot_(robot), perception_(perception), clock_(clock), cancel_(std::move(cancel)) {}
TaskExecutor::~TaskExecutor() { shutdown(); }

void TaskExecutor::check_active() const {
    if (cancel_ && cancel_()) throw Cancelled{};
    if (stopped_) throw std::logic_error("task is held and cannot resume");
}

void TaskExecutor::initialize(bool navigation) {
    check_active();
    if (robot_started_) throw std::logic_error("task already initialized");
    navigation_ = navigation;
    robot_started_ = true;  // Cleanup is required even if initialization partially fails.
    if (!robot_.initialize(config_.robot, navigation)) throw std::runtime_error("robot initialization failed");
    const auto current = robot_.read_joints();
    if (current.empty() || !std::all_of(current.begin(), current.end(), [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("cannot read finite home joint positions");
    home_joints_ = config_.robot.home_joint_positions.value_or(current);
    if (home_joints_.size() != current.size()) throw std::runtime_error("home joint count does not match the arm");
}

void TaskExecutor::load_perception() {
    check_active();
    if (model_loaded_) return;
    perception_started_ = true;
    perception_.load();
    model_loaded_ = true;
}

bool TaskExecutor::home() {
    check_active();
    const auto current = robot_.read_joints();
    if (current.size() != home_joints_.size() || current.empty()) return false;
    double error = 0;
    for (std::size_t index = 0; index < current.size(); ++index) {
        if (!std::isfinite(current[index])) return false;
        error = std::max(error, std::abs(current[index] - home_joints_[index]));
    }
    if (error <= config_.robot.home_tolerance_rad) return true;
    return robot_.move_joints(home_joints_, config_.robot.joint_speed, config_.robot.home_timeout);
}

PerceptionResult TaskExecutor::perceive() {
    check_active();
    try {
        load_perception();
        const auto observation = perception_.observe();
        check_active();
        if (observation.status == PerceptionStatus::Unavailable) return {observation.status, {}, observation.reason};
        if (observation.status == PerceptionStatus::Empty) {
            if (!observation.candidates.empty()) throw std::runtime_error("EMPTY observation contains candidates");
            return {PerceptionStatus::Empty, {}, observation.reason};
        }
        if (observation.status != PerceptionStatus::Found || observation.candidates.empty())
            throw std::runtime_error("FOUND observation requires candidates");
        const auto candidate = select_candidate(observation.candidates, config_);
        if (!candidate) return {PerceptionStatus::Empty, {}, "no eligible material above confidence threshold"};
        return {PerceptionStatus::Found, plan_grasp(*candidate, config_), observation.reason};
    } catch (const Cancelled&) { throw; }
    catch (const std::exception& error) { return {PerceptionStatus::Unavailable, {}, error.what()}; }
}

bool TaskExecutor::move(const Pose& pose) {
    check_active();
    validate_pose(pose);
    return robot_.move_pose(pose, config_.robot.move_timeout, config_.robot.collision_check);
}

bool TaskExecutor::open_gripper() {
    check_active();
    const auto& robot = config_.robot;
    return robot_.command_gripper(robot.gripper_open_width, robot.gripper_speed, robot.gripper_effort, true, robot.move_timeout);
}

void TaskExecutor::wait(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) throw std::invalid_argument("invalid wait duration");
    const double deadline = clock_.now() + seconds;
    while (clock_.now() < deadline) {
        check_active();
        clock_.sleep_for(std::min(0.05, deadline - clock_.now()));
    }
    check_active();
}

bool TaskExecutor::close_and_confirm(double width) {
    check_active();
    const auto& robot = config_.robot;
    custody_ = Custody::PossiblyHolding;
    if (!robot_.command_gripper(width, robot.gripper_speed, robot.gripper_effort, false, robot.move_timeout)) return false;
    GripMonitor monitor(robot);
    const double deadline = clock_.now() + robot.gripper_grasp_timeout_s;
    while (clock_.now() < deadline) {
        check_active();
        const auto sample = robot_.read_gripper();
        check_active();
        if (clock_.now() >= deadline) return false;
        if (monitor.update(sample)) { custody_ = Custody::Holding; return true; }
        wait(std::min(robot.gripper_state_poll_interval_s, std::max(0.0, deadline - clock_.now())));
    }
    return false;
}

bool TaskExecutor::pick(const PickTarget& target) {
    check_active();
    if (custody_ != Custody::Empty) return false;
    double width = config_.robot.gripper_close_width;
    if (config_.robot.auto_gripper_close_width && target.object_width)
        width = gripper_close_width(*target.object_width, config_.robot.gripper_width_margin);
    if (!std::isfinite(width) || width < 0 || width >= config_.robot.gripper_open_width) return false;
    return open_gripper() && move(target.plan.pre_grasp) && move(target.plan.grasp) && close_and_confirm(width);
}

bool TaskExecutor::lift(const PickTarget& target) {
    return custody_ == Custody::Holding && move(lift_pose(target.plan.grasp, config_.robot.lift_height));
}
bool TaskExecutor::lift_safe(const PickTarget& target) {
    return custody_ == Custody::Holding && move(lift_safe_pose(target, config_));
}
bool TaskExecutor::place(const PickTarget& target, bool cyclic) {
    if (custody_ != Custody::Holding) return false;
    if (cyclic && !move(lift_safe_pose(target, config_))) return false;
    if (!move(cyclic ? lift_pose(target.plan.grasp, config_.robot.lift_height) : target.plan.grasp)) return false;
    if (!open_gripper()) return false;
    custody_ = Custody::Empty;
    return true;
}
bool TaskExecutor::retreat(const PickTarget& target, bool cyclic) {
    return custody_ == Custody::Empty && move(cyclic ? lift_safe_pose(target, config_) : target.plan.pre_grasp);
}
bool TaskExecutor::navigate(const Pose& pose, bool lock_heading) {
    check_active();
    validate_pose(pose);
    if (!navigation_) return false;
    const auto& nav = config_.navigation;
    const NavigationOptions options{nav.timeout_s, nav.collision_check, nav.omni_plan, lock_heading && nav.lock_heading};
    for (int attempt = 0; attempt < nav.retries; ++attempt) {
        check_active();
        if (robot_.navigate(pose, options)) return true;
    }
    return false;
}

void TaskExecutor::hold(const std::string& reason) {
    (void)reason;
    if (stopped_) return;
    stopped_ = true;
    if (robot_started_) robot_.stop();
}
void TaskExecutor::shutdown() noexcept {
    if (perception_started_) { perception_.shutdown(); perception_started_ = false; model_loaded_ = false; }
    if (robot_started_) { robot_.shutdown(); robot_started_ = false; }
}
}  // namespace robmove
