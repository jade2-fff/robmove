#include <robmove/config.hpp>
#include <robmove/geometry.hpp>
#include <cmath>

int main() {
    const robmove::Config config;
    config.validate();
    const auto point = robmove::back_project(420, 190, 1.5, {500, 500, 320, 240});
    return (point - robmove::Vec3{0.3, -0.15, 1.5}).norm() < 1e-12 ? 0 : 1;
}
