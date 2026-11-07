#include "test_support.hpp"
#include "robmove/planning.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

using namespace robmove;

TEST(native_planning_matches_200_original_python_reference_cases) {
    std::ifstream input(std::string(ROBMOVE_TEST_DATA) + "/planning_golden.json");
    CHECK(input.good());
    const auto golden = nlohmann::json::parse(input);
    CHECK(golden.at("cases").size() == 200);
    const auto equal_pose = [](const Pose& actual, const nlohmann::json& expected) {
        const auto reference = pose_from_array(expected.get<std::array<double, 7>>());
        CHECK((actual.position - reference.position).norm() < 1e-9);
        CHECK(actual.orientation.angularDistance(reference.orientation) < 1e-9);
    };
    for (const auto& item : golden.at("cases")) {
        Config config;
        config.robot.arm_name = item.at("arm");
        config.robot.gripper_name = config.robot.arm_name == "left_arm" ? "left_gripper" : "right_gripper";
        config.robot.work_angle_min_deg = item.at("min_angle");
        config.robot.work_angle_max_deg = item.at("max_angle");
        config.validate();
        const Candidate candidate{item.at("class_name"), 0.9, pose_from_array(item.at("pose").get<std::array<double, 7>>()), {}, {}};
        const auto& expected = item.at("expected");
        if (expected.is_null()) { THROWS(plan_grasp(candidate, config)); continue; }
        const auto target = plan_grasp(candidate, config);
        equal_pose(target.plan.pre_grasp, expected.at("pre_grasp"));
        equal_pose(target.plan.grasp, expected.at("grasp"));
        equal_pose(lift_safe_pose(target, config), expected.at("lift_safe"));
        CHECK(target.plan.top_down == expected.at("top_down").get<bool>());
    }
}
