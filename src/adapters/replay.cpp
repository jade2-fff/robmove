#include "robmove/adapters/replay.hpp"
#include "../json_io.hpp"

namespace robmove {
std::vector<Observation> read_replay(std::istream& input) {
    const auto document = detail::parse_json(input);
    detail::check_keys(document, {"schema_version", "frames"}, "replay");
    if (document.contains("schema_version") && detail::read<int>(document.at("schema_version")) != 1)
        throw std::invalid_argument("unsupported replay schema_version");
    if (!document.contains("frames") || !document.at("frames").is_array())
        throw std::invalid_argument("replay.frames must be an array");
    std::vector<Observation> frames;
    frames.reserve(document.at("frames").size());
    for (const auto& frame : document.at("frames")) {
        detail::check_keys(frame, {"status", "reason", "candidates"}, "frame");
        const auto status = detail::read<std::string>(frame.at("status"));
        Observation observation;
        if (status == "found") observation.status = PerceptionStatus::Found;
        else if (status == "empty") observation.status = PerceptionStatus::Empty;
        else if (status == "unavailable") observation.status = PerceptionStatus::Unavailable;
        else throw std::invalid_argument("unknown perception status: " + status);
        detail::assign(frame, "reason", observation.reason);
        if (frame.contains("candidates")) {
            if (!frame.at("candidates").is_array()) throw std::invalid_argument("candidates must be an array");
            for (const auto& item : frame.at("candidates")) {
                detail::check_keys(item, {"class_name", "confidence", "pose_base", "object_width_m", "ranking_position_base"}, "candidate");
                Candidate candidate;
                candidate.class_name = detail::read<std::string>(item.at("class_name"));
                candidate.confidence = detail::read<double>(item.at("confidence"));
                candidate.object_pose = detail::read_pose(item.at("pose_base"));
                if (item.contains("object_width_m") && !item.at("object_width_m").is_null())
                    candidate.object_width = detail::read<double>(item.at("object_width_m"));
                if (item.contains("ranking_position_base")) candidate.ranking_position = detail::read_vec3(item.at("ranking_position_base"));
                if (candidate.class_name.empty() || candidate.confidence < 0 || candidate.confidence > 1 ||
                    (candidate.object_width && *candidate.object_width <= 0)) throw std::invalid_argument("invalid candidate metadata");
                observation.candidates.push_back(std::move(candidate));
            }
        }
        if ((observation.status == PerceptionStatus::Found) != !observation.candidates.empty())
            throw std::invalid_argument("only FOUND frames must contain candidates");
        frames.push_back(std::move(observation));
    }
    return frames;
}

std::vector<Observation> demo_frames() {
    return {{PerceptionStatus::Found,
             {{"wulongcha", 0.9, pose_from_array({0.6, -0.1, 0.4, 0, 0, 0, 1}), 0.06, std::nullopt}}, "synthetic demonstration"},
            {PerceptionStatus::Empty, {}, "synthetic empty frame 1"},
            {PerceptionStatus::Empty, {}, "synthetic empty frame 2"},
            {PerceptionStatus::Empty, {}, "synthetic empty frame 3"}};
}

ReplayPerception::ReplayPerception(std::filesystem::path source) : source_(std::move(source)) {}
ReplayPerception::ReplayPerception(std::vector<Observation> frames) : frames_(std::move(frames)) {}
void ReplayPerception::load() {
    if (loaded_) return;
    if (!source_.empty()) {
        std::ifstream input(source_);
        if (!input) throw std::runtime_error("cannot open replay: " + source_.string());
        frames_ = read_replay(input);
    }
    cursor_ = 0;
    loaded_ = true;
}
Observation ReplayPerception::observe() {
    if (!loaded_) throw std::logic_error("replay has not been loaded");
    if (cursor_ == frames_.size()) return {PerceptionStatus::Unavailable, {}, "replay exhausted; no new observation"};
    return frames_[cursor_++];
}
void ReplayPerception::shutdown() noexcept { loaded_ = false; }
}  // namespace robmove
