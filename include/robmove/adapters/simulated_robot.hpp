#pragma once

#include "robmove/ports.hpp"

namespace robmove {
struct SimulatedCommand {
    std::string operation;
    std::optional<Pose> pose;
    std::vector<double> joints;
    std::optional<double> width;
    bool blocking = true;
    bool collision_check = true;
    bool lock_heading = false;
};

// Deterministic command simulator. No device I/O, IK, dynamics or collision model.
class SimulatedRobot final : public RobotPort {
public:
    bool initialize(const RobotConfig& config, bool navigation) override;
    void shutdown() noexcept override;
    std::vector<double> read_joints() override;
    bool move_joints(std::span<const double> joints, double speed, double timeout_s) override;
    bool move_pose(const Pose& pose, double timeout_s, bool collision_check) override;
    bool command_gripper(double width, double speed, double effort, bool blocking, double timeout_s) override;
    std::optional<GripperState> read_gripper() override;
    bool navigate(const Pose& goal, const NavigationOptions& options) override;
    void stop() noexcept override;
    bool initialized() const noexcept { return initialized_; }
    const std::vector<SimulatedCommand>& commands() const noexcept { return commands_; }
private:
    bool ready() const noexcept { return initialized_ && !stopped_; }
    RobotConfig config_;
    std::vector<double> joints_ = std::vector<double>(7, 0);
    std::vector<SimulatedCommand> commands_;
    GripperState gripper_{0.1, 0, false};
    bool initialized_ = false, stopped_ = false, navigation_ = false;
};

class VirtualClock final : public Clock {
public:
    double now() const override { return now_; }
    void sleep_for(double seconds) override;
private:
    double now_ = 0;
};
}  // namespace robmove
