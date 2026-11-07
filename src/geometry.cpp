#include "robmove/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace robmove {
void validate_pose(const Pose& pose) {
    if (!pose.position.allFinite() || !pose.orientation.coeffs().allFinite() ||
        std::abs(pose.orientation.norm() - 1.0) > 1e-6) {
        throw std::invalid_argument("pose must be finite with a unit quaternion");
    }
}

Pose pose_from_array(const std::array<double, 7>& values) {
    Pose pose{{values[0], values[1], values[2]}, {values[6], values[3], values[4], values[5]}};
    const double norm = pose.orientation.coeffs().stableNorm();
    if (!std::isfinite(norm) || norm < 1e-12) throw std::invalid_argument("invalid quaternion");
    pose.orientation.coeffs() /= norm;
    validate_pose(pose);
    return pose;
}

std::array<double, 7> pose_to_array(const Pose& pose) {
    validate_pose(pose);
    return {pose.position.x(), pose.position.y(), pose.position.z(), pose.orientation.x(),
            pose.orientation.y(), pose.orientation.z(), pose.orientation.w()};
}

Vec3 transform_point(const Pose& transform, const Vec3& point) {
    validate_pose(transform);
    if (!point.allFinite()) throw std::invalid_argument("point must be finite");
    const Vec3 result = transform.orientation * point + transform.position;
    if (!result.allFinite()) throw std::invalid_argument("point transform overflow");
    return result;
}

Vec3 back_project(double u, double v, double depth_m, const CameraIntrinsics& k) {
    if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(depth_m) || depth_m <= 0 ||
        !std::isfinite(k.fx) || !std::isfinite(k.fy) || k.fx <= 0 || k.fy <= 0 ||
        !std::isfinite(k.cx) || !std::isfinite(k.cy)) {
        throw std::invalid_argument("invalid pixel, depth or camera intrinsics");
    }
    const Vec3 result{(u - k.cx) * depth_m / k.fx, (v - k.cy) * depth_m / k.fy, depth_m};
    if (!result.allFinite()) throw std::invalid_argument("back projection overflow");
    return result;
}

std::optional<double> robust_depth(std::span<const float> depth, int width, int height,
                                   int u, int v, int radius, std::vector<float>& scratch) {
    if (width <= 0 || height <= 0 || radius < 0 ||
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) != depth.size()) {
        throw std::invalid_argument("invalid depth shape or median radius");
    }
    const auto x = static_cast<std::int64_t>(std::clamp(u, 0, width - 1));
    const auto y = static_cast<std::int64_t>(std::clamp(v, 0, height - 1));
    const auto x0 = std::max<std::int64_t>(0, x - radius);
    const auto x1 = std::min<std::int64_t>(width - 1, x + radius);
    const auto y0 = std::max<std::int64_t>(0, y - radius);
    const auto y1 = std::min<std::int64_t>(height - 1, y + radius);
    scratch.clear();
    scratch.reserve(static_cast<std::size_t>((x1 - x0 + 1) * (y1 - y0 + 1)));
    for (auto row = y0; row <= y1; ++row) {
        for (auto column = x0; column <= x1; ++column) {
            const auto value = depth[static_cast<std::size_t>(row * width + column)];
            if (value > 0.05F && value < 2.0F) scratch.push_back(value);
        }
    }
    if (scratch.empty()) return std::nullopt;
    const auto middle = scratch.begin() + static_cast<std::ptrdiff_t>(scratch.size() / 2);
    std::nth_element(scratch.begin(), middle, scratch.end());
    if (scratch.size() % 2 != 0) return *middle;
    return (static_cast<double>(*middle) + *std::max_element(scratch.begin(), middle)) / 2;
}

double gripper_close_width(double object_width, double margin, double minimum) {
    if (!std::isfinite(object_width) || object_width <= 0 || !std::isfinite(margin) || margin < 0 ||
        !std::isfinite(minimum) || minimum <= 0) throw std::invalid_argument("invalid gripper width");
    return std::max(minimum, object_width - margin);
}
}  // namespace robmove
