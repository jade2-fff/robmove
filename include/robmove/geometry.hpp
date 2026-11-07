#pragma once

#include "robmove/types.hpp"
#include <array>
#include <span>

namespace robmove {
void validate_pose(const Pose& pose);
Pose pose_from_array(const std::array<double, 7>& values);
std::array<double, 7> pose_to_array(const Pose& pose);
Vec3 transform_point(const Pose& transform, const Vec3& point);
Vec3 back_project(double u, double v, double depth_m, const CameraIntrinsics& intrinsics);

// Tightly packed row-major float32 depth in meters. Scratch storage is reusable.
std::optional<double> robust_depth(std::span<const float> depth, int width, int height,
                                   int u, int v, int radius, std::vector<float>& scratch);
}  // namespace robmove
