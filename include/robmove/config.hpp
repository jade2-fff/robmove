#pragma once

#include "robmove/types.hpp"
#include <filesystem>
#include <map>

namespace robmove {
struct ObjectConfig {
    std::filesystem::path mesh_path;
    Vec3 grasp_point_object = Vec3::Zero();
    double pre_grasp_distance = 0.08;
    double gripper_ee_distance = 0.18;
};

struct RobotConfig {
    std::string arm_name = "right_arm";
    std::string gripper_name = "right_gripper";
    bool collision_check = true;
    bool auto_gripper_close_width = false;
    double gripper_open_width = 0.10;
    double gripper_close_width = 0.04;
    double gripper_width_margin = 0.02;
    double gripper_effort = 25;
    double gripper_speed = 0.1;
    double gripper_grasp_effort_threshold = 8;
    int gripper_grasp_effort_stable_samples = 3;
    double gripper_grasp_width_stable_tolerance_m = 0.003;
    double gripper_grasp_timeout_s = 3;
    double gripper_state_poll_interval_s = 0.1;
    double foundationpose_lying_angle_threshold_deg = 45;
    double lift_height = 0.12;
    double work_angle_min_deg = -10;
    double work_angle_max_deg = 10;
    double move_timeout = 8;
    double joint_speed = 0.2;
    double home_timeout = 25;
    double home_tolerance_rad = 1e-3;
    std::optional<std::vector<double>> home_joint_positions;
};

struct TaskConfig {
    int perception_retries = 3;
    double perception_retry_interval_s = 0.5;
    int empty_confirmations = 3;
    int max_cycles = 20;
};

struct NavigationConfig {
    std::optional<Pose> origin_pose, pickup_pose, place_pose;
    bool lock_heading = true;
    bool collision_check = true;
    bool omni_plan = true;
    int retries = 3;
    double timeout_s = 30;
};

struct PerceptionConfig {
    double mask_conf_threshold = 0.40;
    std::filesystem::path mask_weights = "assets/models/yolo26s-seg_finetuned.pt";
    std::filesystem::path foundationpose_root;
    double input_scale = 1;
    int depth_median_radius = 5;
    std::map<std::string, ObjectConfig, std::less<>> objects{
        {"wulongcha", {"assets/objects/wulongcha/normalized.obj", {0, 0, 0.1062}}},
        {"chouzhi", {"assets/objects/chouzhi/normalized.obj", {0.0822, 0.0747, 0.0789}}},
        {"leshishupian", {"assets/objects/leshishupian/normalized.obj", {0, 0, 0.105}}},
        {"xinlamian", {"assets/objects/xinlamian/normalized.obj", {0, 0, 0.055}}},
        {"kekoukele", {"assets/objects/kekoukele/normalized.obj", {0, 0, 0.0812}}},
    };
    std::map<std::string, std::string, std::less<>> aliases{
        {"a tall green cylindrical can of Lay's Stax potato chips", "leshishupian"},
        {"a red cup of spicy instant noodles", "xinlamian"},
        {"a beige rectangular soft pack of facial tissues", "chouzhi"},
        {"a brown plastic bottle of oolong tea with a white cap", "wulongcha"},
        {"a red plastic bottle of Coca-Cola with a red cap", "kekoukele"},
    };
};

struct Config {
    RobotConfig robot;
    TaskConfig task;
    NavigationConfig navigation;
    PerceptionConfig perception;
    void validate(Mode mode = Mode::Single) const;
};

// Relative asset paths are resolved against the config file, never the process cwd.
Config load_config(const std::filesystem::path& path);
}  // namespace robmove
