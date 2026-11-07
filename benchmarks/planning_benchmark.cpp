#include "robmove/planning.hpp"
#include <nlohmann/json.hpp>
#include <charconv>
#include <chrono>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    try {
        int iterations = 500000;
        if (argc != 1) {
            if (argc != 3 || std::string_view(argv[1]) != "--iterations")
                throw std::invalid_argument("usage: robmove_benchmark [--iterations N]");
            const std::string_view input(argv[2]);
            const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), iterations);
            if (error != std::errc{} || end != input.data() + input.size() || iterations < 1 || iterations > 100000000)
                throw std::invalid_argument("iterations must be in [1,100000000]");
        }
        const robmove::Config config;
        config.validate();
        robmove::Candidate candidate{"wulongcha", 0.9, robmove::pose_from_array({0.6, -0.1, 0.4, 0, 0, 0, 1}), 0.06, {}};
        double checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            candidate.object_pose.position.x() = 0.6 + static_cast<double>(i % 1024) * 1e-6;
            const auto target = robmove::plan_grasp(candidate, config);
            checksum += target.plan.grasp.position.x();
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
#ifdef NDEBUG
        constexpr auto mode = "release";
#else
        constexpr auto mode = "debug";
#endif
        std::cout << nlohmann::json{{"kernel", "foundationpose_grasp_planning"}, {"iterations", iterations},
                                    {"seconds", seconds}, {"ns_per_plan", seconds * 1e9 / iterations},
                                    {"checksum", checksum}, {"build_mode", mode}}.dump() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
