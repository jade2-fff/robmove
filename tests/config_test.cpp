#include "test_support.hpp"
#include "robmove/config.hpp"
#include "robmove/geometry.hpp"
#include <chrono>
#include <fstream>
#include <limits>

using namespace robmove;

namespace {
struct ConfigFile {
    std::filesystem::path directory;
    std::filesystem::path file;
    explicit ConfigFile(const std::string& content) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0;; ++attempt) {
            directory = std::filesystem::temp_directory_path() /
                        ("robmove-config-test-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(directory)) break;
            if (attempt >= 100) throw std::runtime_error("cannot create test directory");
        }
        file = directory / "config.json";
        std::ofstream(file) << content;
    }
    ~ConfigFile() { std::error_code error; std::filesystem::remove_all(directory, error); }
};
}

TEST(default_config_retains_five_materials_and_collision_checks) {
    const Config config;
    config.validate();
    CHECK(config.perception.objects.size() == 5);
    CHECK(config.robot.collision_check);
    CHECK(config.robot.arm_name == "right_arm");
    THROWS(config.validate(Mode::Cyclic));
}

TEST(config_rejects_nonfinite_unsafe_widths_and_unbounded_retry_values) {
    Config config;
    config.robot.gripper_close_width = 0.11;
    THROWS(config.validate());
    config = Config{};
    config.robot.joint_speed = std::numeric_limits<double>::quiet_NaN();
    THROWS(config.validate());
    config = Config{};
    config.task.perception_retries = 0;
    THROWS(config.validate());
    config = Config{};
    config.robot.home_joint_positions = std::vector<double>{};
    THROWS(config.validate());
    config = Config{};
    config.robot.gripper_state_poll_interval_s = 0;
    THROWS(config.validate());
}

TEST(config_requires_matching_arm_gripper_and_valid_navigation_poses) {
    Config config;
    config.robot.arm_name = "left_arm";
    THROWS(config.validate());
    config.robot.gripper_name = "left_gripper";
    config.navigation.origin_pose = Pose{};
    config.navigation.pickup_pose = Pose{};
    config.navigation.place_pose = Pose{};
    config.validate(Mode::Cyclic);
    config.navigation.place_pose->orientation.coeffs().setZero();
    THROWS(config.validate(Mode::Cyclic));
}

TEST(config_paths_are_independent_of_working_directory) {
    const ConfigFile file(R"({"schema_version":1,"resource_root":"resources","robot":{"lift_height":0.2},"task":{"max_cycles":4}})");
    const auto config = load_config(file.file);
    NEAR(config.robot.lift_height, 0.2, 1e-12);
    CHECK(config.task.max_cycles == 4);
    CHECK(config.perception.mask_weights == file.directory / "resources/assets/models/yolo26s-seg_finetuned.pt");
}

TEST(config_fails_on_typos_wrong_types_duplicate_keys_and_new_schema) {
    for (const std::string content : {
             R"({"robot":{"lift_heigth":0.2}})", R"({"task":{"max_cycles":1.5}})",
             R"({"robot":{"collision_check":"false"}})", R"({"schema_version":2})",
             R"({"task":{"max_cycles":2,"max_cycles":3}})",
             R"({"task":{"max_cycles":18446744073709551615}})", R"({"task":null})"}) {
        const ConfigFile file(content);
        THROWS(load_config(file.file));
    }
}

TEST(config_allows_material_extension_and_rejects_bad_aliases) {
    const ConfigFile file(R"({"perception":{"objects":{"new_box":{"mesh_path":"box.obj","grasp_point_object":[0,0,0.1],"pre_grasp_distance":0.1,"gripper_ee_distance":0.2}},"aliases":{"new box":"new_box"}}})");
    const auto config = load_config(file.file);
    CHECK(config.perception.objects.size() == 6);
    CHECK(config.perception.aliases.at("new box") == "new_box");
    Config invalid;
    invalid.perception.aliases["bad"] = "missing";
    THROWS(invalid.validate());
}
