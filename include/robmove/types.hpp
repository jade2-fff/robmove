#pragma once

#include <Eigen/Geometry>
#include <optional>
#include <string>
#include <vector>

namespace robmove {
using Vec3 = Eigen::Vector3d;

// All positions and widths are meters. Quaternions serialize as [qx, qy, qz, qw].
struct Pose {
    Vec3 position = Vec3::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct CameraIntrinsics { double fx, fy, cx, cy; };

struct Candidate {
    std::string class_name;
    double confidence = 0;
    Pose object_pose;  // base_link <- object; inference adapters own camera transforms.
    std::optional<double> object_width;
    // Observed RGB-D centroid before pose registration, if supplied by the adapter.
    std::optional<Vec3> ranking_position;
};

enum class PerceptionStatus { Found, Empty, Unavailable };
struct Observation {
    PerceptionStatus status = PerceptionStatus::Unavailable;
    std::vector<Candidate> candidates;
    std::string reason;
};
}  // namespace robmove
