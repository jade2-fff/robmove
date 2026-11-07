#pragma once

#include "robmove/config.hpp"
#include "robmove/geometry.hpp"

namespace robmove {
// Linear selection, stable on ties. Reject malformed eligible detections instead
// of silently turning sensor failures into an empty scene.
std::optional<Candidate> select_candidate(std::span<const Candidate> candidates, const Config& config);
PickTarget plan_grasp(const Candidate& candidate, const Config& config);
GraspPlan plan_horizontal(const Vec3& point, double yaw_deg, double pre_distance, double ee_distance);
Pose lift_pose(const Pose& pose, double height);
Pose lift_safe_pose(const PickTarget& target, const Config& config);
}  // namespace robmove
