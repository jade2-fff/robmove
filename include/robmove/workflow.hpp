#pragma once

#include "robmove/task.hpp"

namespace robmove {
struct WorkflowResult {
    bool success = false;
    int placed_count = 0;
    bool held = false;
    bool cancelled = false;
    Custody custody = Custody::Empty;
    std::string reason;
};

// One workflow instance is one mission. Construct another for another run.
class Workflow {
public:
    Workflow(Config config, RobotPort& robot, PerceptionPort& perception, Clock& clock,
             EventSink* events = nullptr, CancelCheck cancel = {});
    WorkflowResult run(Mode mode);
    Workflow(const Workflow&) = delete;
    Workflow& operator=(const Workflow&) = delete;
private:
    WorkflowResult single();
    WorkflowResult cyclic();
    WorkflowResult hold(const std::string& reason, bool cancelled = false);
    WorkflowResult done(const std::string& reason);
    PerceptionResult perceive_with_retries(bool retry_empty);
    void state(State state, std::string message = {});
    Config config_;
    TaskExecutor task_;
    Clock& clock_;
    EventSink* events_;
    double start_time_ = 0;
    int placed_ = 0;
    int cycle_ = 1;
    bool started_ = false;
};
}  // namespace robmove
