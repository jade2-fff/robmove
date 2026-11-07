#include <robmove/planning.hpp>
#include <cmath>

int main() {
    const robmove::Config config;
    config.validate();
    const robmove::Candidate object{"wulongcha", 0.9,
        robmove::pose_from_array({0.6, -0.1, 0.4, 0, 0, 0, 1}), {}, {}};
    const auto target = robmove::plan_grasp(object, config);
    return std::abs(target.plan.grasp.position.x() - 0.42) < 1e-12 ? 0 : 1;
}
