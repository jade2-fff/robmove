#pragma once

#include "robmove/config.hpp"
#include <chrono>
#include <functional>
#include <span>
#include <string_view>

namespace robmove {
struct GripperState { double width; double effort; bool is_moving; };
struct NavigationOptions {
    double timeout_s = 30;
    bool collision_check = true;
    bool omni_plan = true;
    bool lock_heading = false;
};

// Adapters own SDK initialization, localization checks, transport error details
// and resource teardown. Blocking calls must honor their timeout. A false motion
// return must mean execution has stopped, never that it is still in progress.
class RobotPort {
public:
    virtual ~RobotPort() = default;
    virtual bool initialize(const RobotConfig& config, bool navigation) = 0;
    virtual void shutdown() noexcept = 0;
    virtual std::vector<double> read_joints() = 0;
    virtual bool move_joints(std::span<const double> joints, double speed, double timeout_s) = 0;
    virtual bool move_pose(const Pose& pose, double timeout_s, bool collision_check) = 0;
    virtual bool command_gripper(double width, double speed, double effort, bool blocking, double timeout_s) = 0;
    virtual std::optional<GripperState> read_gripper() = 0;
    virtual bool navigate(const Pose& goal, const NavigationOptions& options) = 0;
    // Stop base and arm trajectories; never open the gripper or command a home pose.
    virtual void stop() noexcept = 0;
};

class PerceptionPort {
public:
    virtual ~PerceptionPort() = default;
    virtual void load() = 0;
    virtual Observation observe() = 0;
    virtual void shutdown() noexcept = 0;
};

class Clock {
public:
    virtual ~Clock() = default;
    virtual double now() const = 0;  // Monotonic seconds, not wall clock.
    virtual void sleep_for(double seconds) = 0;
};

class SteadyClock final : public Clock {
public:
    double now() const override;
    void sleep_for(double seconds) override;
};

using CancelCheck = std::function<bool()>;
enum class Custody { Empty, PossiblyHolding, Holding };
enum class State {
    Init, NavToOrigin, NavToPickup, ArmHomeForPerception, LoadPerceptionModel,
    CameraPerception, Grasp, Lift, LiftSafe, ArmHomeWithObject, NavToPlace,
    LocalPlace, RetreatFromPlace, Release, RetreatToLiftSafe, ArmHomeAfterRelease, Hold, Done
};
std::string_view to_string(State state) noexcept;
std::string_view to_string(Custody custody) noexcept;

struct Event {
    State state;
    int cycle;
    int placed_count;
    double elapsed_s;
    std::string message;
};
class EventSink {
public:
    virtual ~EventSink() = default;
    virtual void emit(const Event& event) noexcept = 0;
};
}  // namespace robmove
