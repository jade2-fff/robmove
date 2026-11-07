#pragma once

#include "robmove/ports.hpp"
#include <filesystem>
#include <istream>

namespace robmove {
std::vector<Observation> read_replay(std::istream& input);
std::vector<Observation> demo_frames();

class ReplayPerception final : public PerceptionPort {
public:
    explicit ReplayPerception(std::filesystem::path source);
    explicit ReplayPerception(std::vector<Observation> frames);
    void load() override;
    Observation observe() override;
    void shutdown() noexcept override;
private:
    std::filesystem::path source_;
    std::vector<Observation> frames_;
    std::size_t cursor_ = 0;
    bool loaded_ = false;
};
}  // namespace robmove
