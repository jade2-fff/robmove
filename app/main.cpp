#include "robmove/adapters/replay.hpp"
#include "robmove/config.hpp"
#include "robmove/geometry.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <set>

namespace {
using namespace robmove;
using Json = nlohmann::json;

struct Options {
    std::filesystem::path config, replay;
    bool demo = false, json = false, validate = false, help = false, version = false;
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
        else if (arg == "--demo") options.demo = true;
        else if (arg == "--json") options.json = true;
        else if (arg == "--validate-config") options.validate = true;
        else if (arg == "--help" || arg == "-h") options.help = true;
        else if (arg == "--version") options.version = true;
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (options.demo && !options.replay.empty()) throw std::invalid_argument("choose --demo or --replay");
    if (!options.help && !options.version && !options.validate && !options.demo && options.replay.empty())
        throw std::invalid_argument("choose --demo or --replay FILE");
    return options;
}

const char* status_name(PerceptionStatus status) {
    switch (status) {
        case PerceptionStatus::Found: return "found";
        case PerceptionStatus::Empty: return "empty";
        case PerceptionStatus::Unavailable: return "unavailable";
    }
    throw std::invalid_argument("invalid perception status");
}

Json candidate_json(const Candidate& candidate) {
    Json result{{"class_name", candidate.class_name}, {"confidence", candidate.confidence},
                {"pose_base", pose_to_array(candidate.object_pose)}};
    if (candidate.object_width) result["object_width_m"] = *candidate.object_width;
    if (candidate.ranking_position) {
        const auto& point = *candidate.ranking_position;
        result["ranking_position_base"] = {point.x(), point.y(), point.z()};
    }
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.version) { std::cout << "robmove 0.2.0\n"; return 0; }
        if (options.help) {
            std::cout << "robmove: offline perception and object-pose replay\n"
                         "Usage: robmove --demo|--replay FILE [options]\n"
                         "  --config FILE         Strict perception configuration\n"
                         "  --validate-config     Validate configuration and exit\n"
                         "  --json                Emit JSON Lines observations and summary\n"
                         "  --help, --version\n";
            return 0;
        }
        const auto config = options.config.empty() ? Config{} : load_config(options.config);
        config.validate();
        if (options.validate) {
            if (options.json) std::cout << Json{{"type", "config"}, {"valid", true},
                                               {"materials", config.perception.objects.size()}}.dump() << '\n';
            else std::cout << "Perception configuration valid\n";
            return std::cout ? 0 : 1;
        }
        std::vector<Observation> frames;
        if (options.demo) frames = demo_frames();
        else {
            std::ifstream input(options.replay);
            if (!input) throw std::runtime_error("cannot open replay: " + options.replay.string());
            frames = read_replay(input);
        }
        std::size_t detections = 0, unavailable = 0;
        for (std::size_t index = 0; index < frames.size(); ++index) {
            const auto& frame = frames[index];
            detections += frame.candidates.size();
            if (frame.status == PerceptionStatus::Unavailable) ++unavailable;
            Json candidates = Json::array();
            for (const auto& candidate : frame.candidates) candidates.push_back(candidate_json(candidate));
            Json observation{{"type", "observation"}, {"backend", "replay"}, {"index", index},
                             {"status", status_name(frame.status)}, {"reason", frame.reason},
                             {"candidates", std::move(candidates)}};
            if (options.json) std::cout << observation.dump() << '\n';
            else std::cout << "Frame " << index << ": " << status_name(frame.status)
                           << ", candidates=" << observation["candidates"].dump() << '\n';
        }
        if (options.json) {
            std::cout << Json{{"type", "summary"}, {"backend", "replay"}, {"frames", frames.size()},
                              {"detections", detections}, {"unavailable_frames", unavailable}}.dump() << '\n';
        } else {
            std::cout << "Frames=" << frames.size() << ", detections=" << detections
                      << ", unavailable=" << unavailable << '\n';
        }
        if (!std::cout) return 1;
        return frames.empty() || unavailable > 0 ? 2 : 0;
    } catch (const std::exception& error) { std::cerr << "robmove: " << error.what() << '\n'; return 1; }
}
