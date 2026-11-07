#pragma once

#include "robmove/planning.hpp"
#include "robmove/ports.hpp"
#include <stdexcept>

namespace robmove {
class Cancelled final : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("task cancelled") {}
};

class GripMonitor {
public:
    explicit GripMonitor(const RobotConfig& config);
    bool update(const std::optional<GripperState>& sample);
private:
    double threshold_, tolerance_, maximum_width_;
    int required_, stable_ = 0;
    std::optional<double> last_width_;
};

// Resource lifetime and atomic actions. Referenced ports and config must outlive
// this object. No motion is performed by construction or destruction.
class TaskExecutor {
public:
    TaskExecutor(const Config& config, RobotPort& robot, PerceptionPort& perception, Clock& clock, CancelCheck cancel = {});
    ~TaskExecutor();
    TaskExecutor(const TaskExecutor&) = delete;
    TaskExecutor& operator=(const TaskExecutor&) = delete;

    void initialize(bool navigation);
    void load_perception();
    bool home();
    PerceptionResult perceive();
    bool pick(const PickTarget& target);
    bool lift(const PickTarget& target);
    bool lift_safe(const PickTarget& target);
    bool place(const PickTarget& target, bool cyclic);
    bool retreat(const PickTarget& target, bool cyclic);
    bool navigate(const Pose& pose, bool lock_heading);
    void wait(double seconds);
    void check_active() const;
    void hold(const std::string& reason);
    void shutdown() noexcept;
    Custody custody() const noexcept { return custody_; }
private:
    bool move(const Pose& pose);
    bool open_gripper();
    bool close_and_confirm(double width);

    const Config& config_;
    RobotPort& robot_;
    PerceptionPort& perception_;
    Clock& clock_;
    CancelCheck cancel_;
    std::vector<double> home_joints_;
    bool robot_started_ = false, perception_started_ = false, model_loaded_ = false;
    bool stopped_ = false, navigation_ = false;
    Custody custody_ = Custody::Empty;
};
}  // namespace robmove
