#include "test_support.hpp"
#include "robmove/workflow.hpp"
#include <algorithm>
#include <deque>
#include <limits>

using namespace robmove;

namespace {
struct FakeClock final : Clock {
    double time = 0;
    double now() const override { return time; }
    void sleep_for(double seconds) override { time += seconds; }
};

struct RecordingRobot final : RobotPort {
    std::vector<std::string> calls;
    std::vector<Pose> moves;
    std::vector<bool> collision_checks;
    std::vector<Pose> navigation_goals;
    std::vector<bool> heading_locks;
    std::vector<double> widths;
    std::vector<double> joints = std::vector<double>(7, 0);
    std::deque<std::optional<GripperState>> samples;
    int fail_action = 0, actions = 0, stops = 0, shutdowns = 0;
    bool init_ok = true, throw_on_move = false, grip_ok = true;
    std::function<void()> on_gripper_read;
    bool action(const std::string& name) {
        calls.push_back(name);
        return ++actions != fail_action;
    }
    bool initialize(const RobotConfig&, bool) override { calls.push_back("init"); return init_ok; }
    void shutdown() noexcept override { ++shutdowns; }
    std::vector<double> read_joints() override { return joints; }
    bool move_joints(std::span<const double> target, double, double) override {
        if (!action("home")) return false;
        joints.assign(target.begin(), target.end());
        return true;
    }
    bool move_pose(const Pose& pose, double, bool collision) override {
        if (throw_on_move) throw std::runtime_error("motion transport failed");
        moves.push_back(pose); collision_checks.push_back(collision);
        return action("move");
    }
    bool command_gripper(double width, double, double, bool blocking, double) override {
        widths.push_back(width);
        return action(blocking ? "open" : "close");
    }
    std::optional<GripperState> read_gripper() override {
        if (on_gripper_read) on_gripper_read();
        if (!samples.empty()) { const auto value = samples.front(); samples.pop_front(); return value; }
        return GripperState{0.05, grip_ok ? 10.0 : 0.0, true};
    }
    bool navigate(const Pose& pose, const NavigationOptions& options) override {
        navigation_goals.push_back(pose); heading_locks.push_back(options.lock_heading);
        return action("navigate");
    }
    void stop() noexcept override { ++stops; }
};

struct ScriptedPerception final : PerceptionPort {
    std::deque<Observation> frames;
    int reads = 0, loads = 0, shutdowns = 0;
    bool load_throws = false;
    void load() override { ++loads; if (load_throws) throw std::runtime_error("model unavailable"); }
    Observation observe() override {
        ++reads;
        if (frames.empty()) return {PerceptionStatus::Unavailable, {}, "end of stream"};
        auto result = std::move(frames.front()); frames.pop_front(); return result;
    }
    void shutdown() noexcept override { ++shutdowns; }
};

struct Events final : EventSink {
    std::vector<State> states;
    std::vector<Event> records;
    void emit(const Event& event) noexcept override { states.push_back(event.state); records.push_back(event); }
};

Observation found() {
    return {PerceptionStatus::Found, {{"wulongcha", 0.9, pose_from_array({0.6, -0.1, 0.4, 0, 0, 0, 1}), 0.06, std::nullopt}}, ""};
}
Observation empty() { return {PerceptionStatus::Empty, {}, "valid empty frame"}; }
Observation unavailable() { return {PerceptionStatus::Unavailable, {}, "no depth frame"}; }
Config cyclic_config() {
    Config config;
    config.navigation.origin_pose = Pose{};
    config.navigation.pickup_pose = pose_from_array({1, 0, 0, 0, 0, 0, 1});
    config.navigation.place_pose = pose_from_array({2, 0, 0, 0, 0, 0, 1});
    return config;
}
}

TEST(grip_monitor_requires_consecutive_valid_force_and_width_samples) {
    GripMonitor monitor(RobotConfig{});
    CHECK(!monitor.update(GripperState{0.06, 9, true}));
    CHECK(!monitor.update(GripperState{0.06, 9, true}));
    CHECK(!monitor.update(std::nullopt));
    CHECK(!monitor.update(GripperState{0.06, 9, true}));
    CHECK(!monitor.update(GripperState{0.06, 9, true}));
    CHECK(!monitor.update(GripperState{0.06, 9, true}));
    CHECK(monitor.update(GripperState{0.06, -9, true}));
}

TEST(grip_monitor_resets_on_bad_feedback_and_width_changes) {
    GripMonitor monitor(RobotConfig{});
    monitor.update(GripperState{0.06, 10, false});
    monitor.update(GripperState{0.06, 10, false});
    CHECK(!monitor.update(GripperState{0.02, 10, false}));
    CHECK(!monitor.update(GripperState{0.02, std::numeric_limits<double>::quiet_NaN(), false}));
    CHECK(!monitor.update(GripperState{std::numeric_limits<double>::infinity(), 10, false}));
}

TEST(single_workflow_preserves_pick_lift_place_and_retreat_sequence) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock; Events events;
    perception.frames = {found()};
    Workflow workflow(Config{}, robot, perception, clock, &events);
    const auto result = workflow.run(Mode::Single);
    CHECK(result.success && !result.held && result.placed_count == 1);
    CHECK(result.custody == Custody::Empty);
    CHECK(robot.calls == std::vector<std::string>{"init", "open", "move", "move", "close", "move", "move", "open", "move"});
    CHECK(robot.moves.size() == 5);
    NEAR(robot.moves[2].position.z(), robot.moves[1].position.z() + 0.12, 1e-12);
    CHECK((robot.moves[3].position - robot.moves[1].position).norm() < 1e-12);
    CHECK((robot.moves[4].position - robot.moves[0].position).norm() < 1e-12);
    CHECK(std::all_of(robot.collision_checks.begin(), robot.collision_checks.end(), [](bool value) { return value; }));
    CHECK(robot.shutdowns == 1 && perception.shutdowns == 1 && robot.stops == 0);
    CHECK(events.states.back() == State::Done);
    THROWS(workflow.run(Mode::Single));
}

TEST(cyclic_workflow_releases_at_lift_height_then_confirms_empty_and_returns) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
    perception.frames = {found(), empty(), empty(), empty()};
    Workflow workflow(cyclic_config(), robot, perception, clock);
    const auto result = workflow.run(Mode::Cyclic);
    CHECK(result.success && result.placed_count == 1);
    CHECK(robot.navigation_goals.size() == 4);
    NEAR(robot.navigation_goals[0].position.x(), 1, 1e-12);
    NEAR(robot.navigation_goals[1].position.x(), 2, 1e-12);
    NEAR(robot.navigation_goals[2].position.x(), 1, 1e-12);
    NEAR(robot.navigation_goals[3].position.x(), 0, 1e-12);
    CHECK(robot.heading_locks == std::vector<bool>{false, true, true, false});
    CHECK(robot.moves.size() == 7);
    NEAR(robot.moves[5].position.z(), robot.moves[1].position.z() + 0.12, 1e-12);
    CHECK((robot.moves[6].position - robot.moves[4].position).norm() < 1e-12);
    CHECK(perception.loads == 1);
}

TEST(single_empty_is_success_but_unavailable_holds_after_bounded_retries) {
    for (const bool valid_empty : {true, false}) {
        RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
        perception.frames = valid_empty ? std::deque{empty(), empty(), empty()} : std::deque{unavailable(), unavailable(), unavailable()};
        Workflow workflow(Config{}, robot, perception, clock);
        const auto result = workflow.run(Mode::Single);
        CHECK(result.success == valid_empty);
        CHECK(result.held != valid_empty);
        CHECK(result.placed_count == 0 && robot.actions == 0 && perception.reads == 3);
        CHECK(robot.stops == (valid_empty ? 0 : 1));
    }
}

TEST(cyclic_retries_transport_failures_and_resets_empty_count_after_a_pick) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
    perception.frames = {empty(), empty(), unavailable(), found(), empty(), empty(), empty()};
    Workflow workflow(cyclic_config(), robot, perception, clock);
    const auto result = workflow.run(Mode::Cyclic);
    CHECK(result.success && result.placed_count == 1 && perception.reads == 7);
}

TEST(cycle_budget_holds_only_after_release_retreat_and_home) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
    auto config = cyclic_config(); config.task.max_cycles = 1;
    perception.frames = {found()};
    Workflow workflow(config, robot, perception, clock);
    const auto result = workflow.run(Mode::Cyclic);
    CHECK(result.held && !result.success && result.placed_count == 1 && result.custody == Custody::Empty);
    CHECK(robot.navigation_goals.size() == 2 && robot.stops == 1);
    CHECK(robot.moves.size() == 7);
}

TEST(failed_close_confirmation_never_lifts_or_opens_on_exit) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
    robot.grip_ok = false; perception.frames = {found()};
    Workflow workflow(Config{}, robot, perception, clock);
    const auto result = workflow.run(Mode::Single);
    CHECK(result.held && result.custody == Custody::PossiblyHolding);
    CHECK(robot.moves.size() == 2 && robot.widths.size() == 2);
    NEAR(clock.time, 3, 1e-8);
    CHECK(robot.stops == 1 && robot.shutdowns == 1);
}

TEST(each_failed_single_action_stops_sequence_and_preserves_release_count) {
    for (int failure = 1; failure <= 8; ++failure) {
        RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
        robot.fail_action = failure; perception.frames = {found()};
        Workflow workflow(Config{}, robot, perception, clock);
        const auto result = workflow.run(Mode::Single);
        CHECK(result.held && !result.success && robot.actions == failure);
        CHECK(result.placed_count == (failure == 8 ? 1 : 0));
        CHECK(robot.stops == 1 && robot.shutdowns == 1);
    }
}

TEST(initialization_model_and_motion_exceptions_always_release_resources) {
    for (int scenario = 0; scenario < 3; ++scenario) {
        RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
        robot.init_ok = scenario != 0;
        perception.load_throws = scenario == 1;
        robot.throw_on_move = scenario == 2;
        perception.frames = {found()};
        Workflow workflow(Config{}, robot, perception, clock);
        const auto result = workflow.run(Mode::Single);
        CHECK(result.held && robot.shutdowns == 1 && robot.stops == 1);
        CHECK(perception.shutdowns == (scenario == 0 ? 0 : 1));
    }
}

TEST(cancellation_during_grasp_keeps_possible_object_and_cleans_up) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
    perception.frames = {found()};
    Workflow workflow(Config{}, robot, perception, clock, nullptr, [&] { return clock.time >= 0.1; });
    const auto result = workflow.run(Mode::Single);
    CHECK(result.held && result.cancelled && result.custody == Custody::PossiblyHolding);
    CHECK(robot.moves.size() == 2 && robot.stops == 1);
}

TEST(bad_home_feedback_and_invalid_config_do_not_send_motion) {
    for (int scenario = 0; scenario < 3; ++scenario) {
        RecordingRobot robot; ScriptedPerception perception; FakeClock clock; Config config;
        if (scenario == 0) robot.joints.clear();
        if (scenario == 1) robot.joints[0] = std::numeric_limits<double>::quiet_NaN();
        if (scenario == 2) config.robot.home_joint_positions = std::vector<double>{0, 0};
        Workflow workflow(config, robot, perception, clock);
        CHECK(workflow.run(Mode::Single).held);
        CHECK(robot.actions == 0 && robot.shutdowns == 1);
    }
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock; Config config;
    config.task.perception_retries = 0;
    Workflow workflow(config, robot, perception, clock);
    CHECK(workflow.run(Mode::Single).held);
    CHECK(robot.calls.empty());
}

TEST(configured_home_is_executed_and_auto_close_respects_opening) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock; Config config;
    config.robot.home_joint_positions = std::vector<double>(7, 0.2);
    config.robot.auto_gripper_close_width = true;
    perception.frames = {found()};
    Workflow workflow(config, robot, perception, clock);
    CHECK(workflow.run(Mode::Single).success);
    CHECK(robot.calls[1] == "home");
    NEAR(robot.widths[1], 0.04, 1e-12);
}

TEST(malformed_found_frame_cannot_be_counted_as_an_empty_scene) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock;
    perception.frames = {{PerceptionStatus::Found, {}, "malformed"}, unavailable(), unavailable()};
    Workflow workflow(Config{}, robot, perception, clock);
    CHECK(workflow.run(Mode::Single).held);
    CHECK(robot.actions == 0);
}

TEST(state_trace_keeps_release_and_retreat_in_the_same_cycle) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock; Events events;
    perception.frames = {found()};
    Workflow workflow(Config{}, robot, perception, clock, &events);
    CHECK(workflow.run(Mode::Single).success);
    for (const auto& event : events.records) CHECK(event.cycle == 1);
}

TEST(grip_feedback_arriving_after_deadline_cannot_authorize_lift) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock; Config config;
    config.robot.gripper_grasp_effort_stable_samples = 1;
    int reads = 0;
    robot.on_gripper_read = [&] { if (++reads == 2) clock.time = 4; };
    perception.frames = {found()};
    Workflow workflow(config, robot, perception, clock);
    const auto result = workflow.run(Mode::Single);
    CHECK(result.held && result.custody == Custody::PossiblyHolding);
    CHECK(robot.moves.size() == 2);
}

TEST(oversized_auto_grip_target_fails_before_any_gripper_command) {
    RecordingRobot robot; ScriptedPerception perception; FakeClock clock; Config config;
    config.robot.auto_gripper_close_width = true;
    auto observation = found(); observation.candidates[0].object_width = 0.3;
    perception.frames = {observation};
    Workflow workflow(config, robot, perception, clock);
    CHECK(workflow.run(Mode::Single).held);
    CHECK(robot.widths.empty() && robot.moves.empty());
}
