#include "robmove/planning.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <tuple>

namespace robmove {
namespace {
constexpr double radians = std::numbers::pi / 180.0;

void validate_candidate(const Candidate& candidate) {
    validate_pose(candidate.object_pose);
    if (!std::isfinite(candidate.confidence) || candidate.confidence < 0 || candidate.confidence > 1 ||
        (candidate.object_width && (!std::isfinite(*candidate.object_width) || *candidate.object_width <= 0)) ||
        (candidate.ranking_position && !candidate.ranking_position->allFinite())) {
        throw std::invalid_argument("invalid candidate confidence, width or ranking position");
    }
}

Vec3 horizontal_approach(const Eigen::Matrix3d& rotation, double minimum, double maximum) {
    const auto distance = [](double a, double b) { return std::abs(std::remainder(a - b, 2 * std::numbers::pi)); };
    std::optional<Vec3> best;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 2; ++axis) {
        for (double sign : {1.0, -1.0}) {
            Vec3 horizontal{rotation(0, axis) * sign, rotation(1, axis) * sign, 0};
            if (horizontal.norm() < 0.1) continue;
            horizontal.normalize();
            const double angle = std::atan2(horizontal.y(), horizontal.x());
            if (angle >= minimum && angle <= maximum) return horizontal;
            const double target = distance(angle, minimum) <= distance(angle, maximum) ? minimum : maximum;
            if (const double delta = distance(angle, target); delta < best_distance) {
                best_distance = delta;
                best = Vec3{std::cos(target), std::sin(target), 0};
            }
        }
    }
    if (!best) throw std::invalid_argument("no valid horizontal approach axis");
    return *best;
}
}  // namespace

std::optional<Candidate> select_candidate(std::span<const Candidate> candidates, const Config& config) {
    const Candidate* selected = nullptr;
    std::string selected_name;
    std::tuple<double, double, double> best;
    for (const auto& candidate : candidates) {
        const auto alias = config.perception.aliases.find(candidate.class_name);
        const auto& name = alias == config.perception.aliases.end() ? candidate.class_name : alias->second;
        if (!config.perception.objects.contains(name)) continue;
        validate_candidate(candidate);
        if (candidate.confidence < config.perception.mask_conf_threshold) continue;
        const Vec3& point = candidate.ranking_position ? *candidate.ranking_position : candidate.object_pose.position;
        const auto key = std::tuple{point.x(), config.robot.arm_name == "left_arm" ? -point.y() : point.y(), -candidate.confidence};
        if (!selected || key < best) {
            selected = &candidate;
            selected_name = name;
            best = key;
        }
    }
    if (!selected) return std::nullopt;
    Candidate result = *selected;
    result.class_name = std::move(selected_name);
    return result;
}

PickTarget plan_grasp(const Candidate& candidate, const Config& config) {
    validate_candidate(candidate);
    const auto found = config.perception.objects.find(candidate.class_name);
    if (found == config.perception.objects.end()) throw std::invalid_argument("unconfigured object: " + candidate.class_name);
    const auto& object = found->second;
    const Vec3 center = transform_point(candidate.object_pose, object.grasp_point_object);
    const Eigen::Matrix3d rotation = candidate.object_pose.orientation.toRotationMatrix();
    const Vec3 z = rotation.col(2).normalized();
    const double tilt = std::acos(std::clamp(std::abs(z.z()), 0.0, 1.0)) / radians;
    const bool top_down = tilt > config.robot.foundationpose_lying_angle_threshold_deg;
    const bool left = config.robot.arm_name == "left_arm";
    const double minimum = (left ? -config.robot.work_angle_max_deg : config.robot.work_angle_min_deg) * radians;
    const double maximum = (left ? -config.robot.work_angle_min_deg : config.robot.work_angle_max_deg) * radians;
    const Vec3 seed = top_down ? Vec3{0, 0, -1} : horizontal_approach(rotation, minimum, maximum);
    Vec3 x = seed - z * seed.dot(z);
    if (x.norm() < 0.1) throw std::invalid_argument("approach is parallel to object axis");
    x.normalize();
    const Vec3 y = z.cross(x).normalized();
    x = y.cross(z).normalized();
    Eigen::Matrix3d basis;
    basis.col(0) = x; basis.col(1) = y; basis.col(2) = z;
    const Eigen::Quaterniond orientation{basis};
    const Pose grasp{center - x * object.gripper_ee_distance, orientation.normalized()};
    const Pose pre{grasp.position - x * object.pre_grasp_distance, grasp.orientation};
    validate_pose(grasp);
    validate_pose(pre);
    return {candidate.class_name, candidate.confidence, {pre, grasp, top_down}, candidate.object_width};
}

GraspPlan plan_horizontal(const Vec3& point, double yaw_deg, double pre_distance, double ee_distance) {
    if (!point.allFinite() || !std::isfinite(yaw_deg) || !std::isfinite(pre_distance) || pre_distance <= 0 ||
        !std::isfinite(ee_distance) || ee_distance <= 0) throw std::invalid_argument("invalid horizontal grasp parameters");
    const Eigen::Quaterniond orientation{Eigen::AngleAxisd(yaw_deg * radians, Vec3::UnitZ())};
    const Vec3 approach = orientation * Vec3::UnitX();
    const Pose grasp{point - approach * ee_distance, orientation};
    const Pose pre{grasp.position - approach * pre_distance, orientation};
    validate_pose(grasp);
    validate_pose(pre);
    return {pre, grasp, false};
}

Pose lift_pose(const Pose& pose, double height) {
    validate_pose(pose);
    if (!std::isfinite(height) || height < 0) throw std::invalid_argument("invalid lift height");
    Pose result = pose;
    result.position.z() += height;
    validate_pose(result);
    return result;
}

Pose lift_safe_pose(const PickTarget& target, const Config& config) {
    Pose result = lift_pose(target.plan.grasp, config.robot.lift_height);
    if (target.plan.top_down) result.position.x() -= config.perception.objects.at(target.class_name).pre_grasp_distance;
    else result.position.head<2>() = target.plan.pre_grasp.position.head<2>();
    validate_pose(result);
    return result;
}
}  // namespace robmove
