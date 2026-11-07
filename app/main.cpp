#include "robmove/adapters/replay.hpp"
#include "robmove/adapters/simulated_robot.hpp"
#include "robmove/workflow.hpp"
#include <nlohmann/json.hpp>

#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

namespace {
using namespace robmove;
using Json = nlohmann::json;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

struct Options {
    std::filesystem::path config, replay, trace;
    Mode mode = Mode::Single;
    bool demo = false, json = false, realtime = false, validate = false, help = false, version = false;
};

Options parse(int argc, char** argv) {
    Options options;
    options.help = argc == 1;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!seen.insert(arg).second) throw std::invalid_argument("duplicate option: " + arg);
        const auto value = [&]() -> std::string {
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--config") options.config = value();
        else if (arg == "--replay") options.replay = value();
        else if (arg == "--trace") options.trace = value();
        else if (arg == "--mode") {
            const auto mode = value();
            if (mode == "single") options.mode = Mode::Single;
            else if (mode == "cyclic") options.mode = Mode::Cyclic;
            else throw std::invalid_argument("mode must be single or cyclic");
        } else if (arg == "--demo") options.demo = true;
        else if (arg == "--json") options.json = true;
        else if (arg == "--realtime") options.realtime = true;
        else if (arg == "--validate-config") options.validate = true;
        else if (arg == "--help" || arg == "-h") options.help = true;
        else if (arg == "--version") options.version = true;
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (options.demo && !options.replay.empty()) throw std::invalid_argument("choose --demo or --replay");
    if (!options.help && !options.version && !options.validate && !options.demo && options.replay.empty())
        throw std::invalid_argument("choose --demo or --replay FILE");
    if (!options.trace.empty()) {
        for (const auto& input : {options.config, options.replay})
            if (!input.empty() && std::filesystem::weakly_canonical(input) == std::filesystem::weakly_canonical(options.trace))
                throw std::invalid_argument("trace must not overwrite an input file");
    }
    return options;
}

class ConsoleEvents final : public EventSink {
public:
    explicit ConsoleEvents(bool json) : json_(json) {}
    void emit(const Event& event) noexcept override {
        try {
            if (json_) std::cout << Json{{"type", "event"}, {"backend", "simulated"}, {"state", to_string(event.state)},
                                        {"cycle", event.cycle}, {"placed_count", event.placed_count},
                                        {"elapsed_s", event.elapsed_s}, {"message", event.message}}.dump() << '\n';
            else std::cout << '[' << to_string(event.state) << "] placed=" << event.placed_count << ' ' << event.message << '\n';
            std::cout.flush();
        } catch (...) { failed_ = true; }
    }
    bool failed() const noexcept { return failed_ || !std::cout; }
private:
    bool json_, failed_ = false;
};

void write_trace(const std::filesystem::path& path, const SimulatedRobot& robot) {
    Json trace = Json::array();
    for (const auto& command : robot.commands()) {
        Json entry{{"operation", command.operation}, {"blocking", command.blocking},
                   {"collision_check", command.collision_check}, {"lock_heading", command.lock_heading}};
        if (command.pose) entry["pose"] = pose_to_array(*command.pose);
        if (command.width) entry["width_m"] = *command.width;
        if (!command.joints.empty()) entry["joints_rad"] = command.joints;
        trace.push_back(std::move(entry));
    }
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open trace output: " + path.string());
    output << trace.dump(2) << '\n';
    output.close();
    if (!output) throw std::runtime_error("failed to write trace output");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.version) { std::cout << "robmove 0.1.0\n"; return 0; }
        if (options.help) {
            std::cout << "robmove: offline C++ pick/place workflow (simulated robot only)\n"
                         "Usage: robmove --demo|--replay FILE [options]\n"
                         "  --mode single|cyclic   Task mode (default: single)\n"
                         "  --config FILE         Strict JSON configuration\n"
                         "  --validate-config     Validate configuration and exit\n"
                         "  --json                Emit JSON Lines events and summary\n"
                         "  --trace FILE          Write simulated commands as JSON\n"
                         "  --realtime            Use real waits instead of a virtual clock\n"
                         "  --help, --version\n";
            return 0;
        }
        auto config = options.config.empty() ? Config{} : load_config(options.config);
        config.validate(options.mode);
        if (options.validate) {
            if (options.json) std::cout << Json{{"type", "config"}, {"valid", true}, {"materials", config.perception.objects.size()}}.dump() << '\n';
            else std::cout << "Configuration valid\n";
            return 0;
        }
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        SimulatedRobot robot;
        ReplayPerception perception = options.demo ? ReplayPerception(demo_frames()) : ReplayPerception(options.replay);
        std::unique_ptr<Clock> clock;
        if (options.realtime) clock = std::make_unique<SteadyClock>();
        else clock = std::make_unique<VirtualClock>();
        ConsoleEvents events(options.json);
        if (!options.json) std::cout << "SIMULATION: no robot connection or model inference\n";
        Workflow workflow(std::move(config), robot, perception, *clock, &events, [] { return interrupted != 0; });
        const auto result = workflow.run(options.mode);
        if (!options.trace.empty()) write_trace(options.trace, robot);
        if (options.json) {
            std::cout << Json{{"type", "summary"}, {"backend", "simulated"}, {"success", result.success},
                              {"placed_count", result.placed_count}, {"held", result.held}, {"cancelled", result.cancelled},
                              {"custody", to_string(result.custody)}, {"reason", result.reason}}.dump() << '\n';
        } else {
            std::cout << (result.success ? "DONE" : "HOLD") << ": placed=" << result.placed_count
                      << ", custody=" << to_string(result.custody) << ", " << result.reason << '\n';
        }
        if (events.failed()) return 1;
        return result.cancelled ? 130 : result.success ? 0 : 2;
    } catch (const std::exception& error) { std::cerr << "robmove: " << error.what() << '\n'; return 1; }
}
