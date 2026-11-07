#include "test_support.hpp"
#include "robmove/planning.hpp"
#include <limits>
#include <numbers>

using namespace robmove;

static Candidate upright(std::string name = "wulongcha") {
    return {std::move(name), 0.9, pose_from_array({0.6, -0.1, 0.4, 0, 0, 0, 1}), 0.06, std::nullopt};
}

TEST(upright_grasp_preserves_original_offsets) {
    const Config config;
    const auto target = plan_grasp(upright(), config);
    CHECK(!target.plan.top_down);
    CHECK((target.plan.grasp.position - Vec3{0.42, -0.1, 0.5062}).norm() < 1e-10);
    CHECK((target.plan.pre_grasp.position - Vec3{0.34, -0.1, 0.5062}).norm() < 1e-10);
    CHECK(target.plan.grasp.orientation.angularDistance(Eigen::Quaterniond::Identity()) < 1e-10);
    const auto safe = lift_safe_pose(target, config);
    CHECK((safe.position - Vec3{0.34, -0.1, 0.6262}).norm() < 1e-10);
}

TEST(lying_grasp_approaches_down_and_retreats_in_negative_base_x) {
    Config config;
    auto candidate = upright();
    candidate.object_pose.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(std::numbers::pi / 2, Vec3::UnitY()));
    const auto target = plan_grasp(candidate, config);
    CHECK(target.plan.top_down);
    const Vec3 approach = target.plan.grasp.orientation * Vec3::UnitX();
    CHECK((approach - Vec3{0, 0, -1}).norm() < 1e-10);
    CHECK(target.plan.pre_grasp.position.z() > target.plan.grasp.position.z());
    const auto lifted = lift_pose(target.plan.grasp, config.robot.lift_height);
    const auto safe = lift_safe_pose(target, config);
    NEAR(safe.position.x(), lifted.position.x() - 0.08, 1e-10);
    NEAR(safe.position.z(), lifted.position.z(), 1e-10);
}

TEST(grasp_frame_is_orthonormal_and_tracks_tilted_object_axis) {
    Config config;
    auto candidate = upright();
    candidate.object_pose.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(1.2, Vec3{1, 2, 3}.normalized()));
    const auto target = plan_grasp(candidate, config);
    const auto rotation = target.plan.grasp.orientation.toRotationMatrix();
    NEAR(rotation.determinant(), 1.0, 1e-12);
    CHECK((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-12);
    CHECK((rotation.col(2) - candidate.object_pose.orientation * Vec3::UnitZ()).norm() < 1e-12);
}

TEST(candidate_selection_uses_arm_side_before_confidence) {
    Config config;
    auto left = upright(); left.object_pose.position.y() = 0.2; left.confidence = 0.99;
    auto right = upright(); right.object_pose.position.y() = -0.2; right.confidence = 0.6;
    std::vector<Candidate> candidates{left, right};
    CHECK(select_candidate(candidates, config)->object_pose.position.y() < 0);
    config.robot.arm_name = "left_arm";
    CHECK(select_candidate(candidates, config)->object_pose.position.y() > 0);
    candidates[1].ranking_position = Vec3{0.2, -0.2, 0.4};
    CHECK(select_candidate(candidates, config)->confidence == 0.6);
}

TEST(candidate_aliases_are_canonicalized_and_unknown_classes_filtered) {
    Config config;
    std::vector<Candidate> candidates{upright("unknown"), upright("a brown plastic bottle of oolong tea with a white cap")};
    CHECK(select_candidate(candidates, config)->class_name == "wulongcha");
    candidates.resize(1);
    CHECK(!select_candidate(candidates, config));
    candidates = {upright()};
    candidates[0].confidence = std::numeric_limits<double>::quiet_NaN();
    THROWS(select_candidate(candidates, config));
    THROWS(plan_grasp(upright("unknown"), config));
}

TEST(horizontal_yolo_example_matches_original_plan) {
    const auto plan = plan_horizontal({0.6, -0.1, 0.5}, 0, 0.05, 0.15);
    CHECK((plan.grasp.position - Vec3{0.45, -0.1, 0.5}).norm() < 1e-12);
    CHECK((plan.pre_grasp.position - Vec3{0.40, -0.1, 0.5}).norm() < 1e-12);
    NEAR(gripper_close_width(0.01, 0.02), 0.005, 1e-12);
    THROWS(gripper_close_width(-0.1, 0.02));
}
