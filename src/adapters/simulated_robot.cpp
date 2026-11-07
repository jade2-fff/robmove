#include "robmove/adapters/simulated_robot.hpp"
#include "robmove/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace robmove {
namespace {
bool positive(double value) { return std::isfinite(value) && value > 0; }
SimulatedCommand command(std::string operation) {
    SimulatedCommand result;
    result.operation = std::move(operation);
    return result;
}
}

bool SimulatedRobot::initialize(const RobotConfig& config, bool navigation) {
    if (initialized_) return false;
    config_ = config;
    navigation_ = navigation;
    stopped_ = false;
    initialized_ = true;
    gripper_ = {config.gripper_open_width, 0, false};
    commands_.push_back(command("initialize"));
    return true;
}
void SimulatedRobot::shutdown() noexcept {
    if (!initialized_) return;
    initialized_ = false;
    try { commands_.push_back(command("shutdown")); } catch (...) {}
}
std::vector<double> SimulatedRobot::read_joints() { return ready() ? joints_ : std::vector<double>{}; }
bool SimulatedRobot::move_joints(std::span<const double> joints, double speed, double timeout_s) {
    if (!ready() || joints.size() != joints_.size() || !positive(speed) || !positive(timeout_s) ||
        !std::all_of(joints.begin(), joints.end(), [](double value) { return std::isfinite(value); })) return false;
    joints_.assign(joints.begin(), joints.end());
    auto item = command("move_joints"); item.joints = joints_; commands_.push_back(std::move(item));
    return true;
}
bool SimulatedRobot::move_pose(const Pose& pose, double timeout_s, bool collision_check) {
    if (!ready() || !positive(timeout_s)) return false;
    validate_pose(pose);
    auto item = command("move_pose"); item.pose = pose; item.collision_check = collision_check;
    commands_.push_back(std::move(item));
    // Mark the simulated arm away from home without pretending to solve IK.
    for (double& joint : joints_) joint += 0.01;
    return true;
}
bool SimulatedRobot::command_gripper(double width, double speed, double effort, bool blocking, double timeout_s) {
    if (!ready() || !std::isfinite(width) || width < 0 || width > config_.gripper_open_width ||
        !positive(speed) || !positive(effort) || !positive(timeout_s)) return false;
    auto item = command("gripper"); item.width = width; item.blocking = blocking;
    commands_.push_back(std::move(item));
    gripper_ = blocking ? GripperState{width, 0, false}
                        : GripperState{std::max(width, config_.gripper_open_width / 2), effort, true};
    return true;
}
std::optional<GripperState> SimulatedRobot::read_gripper() {
    return ready() ? std::optional{gripper_} : std::nullopt;
}
bool SimulatedRobot::navigate(const Pose& goal, const NavigationOptions& options) {
    if (!ready() || !navigation_ || !positive(options.timeout_s)) return false;
    validate_pose(goal);
    auto item = command("navigate"); item.pose = goal;
    item.collision_check = options.collision_check; item.lock_heading = options.lock_heading;
    commands_.push_back(std::move(item));
    return true;
}
void SimulatedRobot::stop() noexcept {
    if (!initialized_ || stopped_) return;
    stopped_ = true;
    try { commands_.push_back(command("stop")); } catch (...) {}
}
void VirtualClock::sleep_for(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0 || !std::isfinite(now_ + seconds))
        throw std::invalid_argument("invalid virtual clock duration");
    now_ += seconds;
}
}  // namespace robmove
