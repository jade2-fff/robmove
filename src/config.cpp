#include "robmove/config.hpp"
#include "json_io.hpp"

#include <cmath>
#include <stdexcept>

namespace robmove {
namespace {
using detail::Json;
using detail::assign;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument(message);
}
void positive(double value, const std::string& name) {
    require(std::isfinite(value) && value > 0, name + " must be finite and positive");
}
void nonnegative(double value, const std::string& name) {
    require(std::isfinite(value) && value >= 0, name + " must be finite and nonnegative");
}
void count(int value, const std::string& name) {
    require(value >= 1 && value <= 1'000'000, name + " must be in [1,1000000]");
}

void read_robot(const Json& object, RobotConfig& robot) {
    detail::check_keys(object, {
        "arm_name", "gripper_name", "collision_check", "auto_gripper_close_width", "gripper_open_width",
        "gripper_close_width", "gripper_width_margin", "gripper_effort", "gripper_speed",
        "gripper_grasp_effort_threshold", "gripper_grasp_effort_stable_samples",
        "gripper_grasp_width_stable_tolerance_m", "gripper_grasp_timeout_s", "gripper_state_poll_interval_s",
        "foundationpose_lying_angle_threshold_deg", "lift_height", "work_angle_min_deg", "work_angle_max_deg",
        "move_timeout", "joint_speed", "home_timeout", "home_tolerance_rad", "home_joint_positions"}, "robot");
#define READ(field) assign(object, #field, robot.field)
    READ(arm_name); READ(gripper_name); READ(collision_check); READ(auto_gripper_close_width);
    READ(gripper_open_width); READ(gripper_close_width); READ(gripper_width_margin); READ(gripper_effort);
    READ(gripper_speed); READ(gripper_grasp_effort_threshold); READ(gripper_grasp_effort_stable_samples);
    READ(gripper_grasp_width_stable_tolerance_m); READ(gripper_grasp_timeout_s); READ(gripper_state_poll_interval_s);
    READ(foundationpose_lying_angle_threshold_deg); READ(lift_height); READ(work_angle_min_deg); READ(work_angle_max_deg);
    READ(move_timeout); READ(joint_speed); READ(home_timeout); READ(home_tolerance_rad);
#undef READ
    if (object.contains("home_joint_positions") && !object.at("home_joint_positions").is_null())
        robot.home_joint_positions = detail::read_numbers(object.at("home_joint_positions"));
}

void read_navigation(const Json& object, NavigationConfig& navigation) {
    detail::check_keys(object, {"origin_pose", "pickup_pose", "place_pose", "lock_heading", "collision_check",
                               "omni_plan", "retries", "timeout_s"}, "navigation");
    for (const auto& [key, destination] : {
             std::pair{"origin_pose", &navigation.origin_pose}, {"pickup_pose", &navigation.pickup_pose},
             {"place_pose", &navigation.place_pose}}) {
        if (object.contains(key) && !object.at(key).is_null()) *destination = detail::read_pose(object.at(key));
    }
    assign(object, "lock_heading", navigation.lock_heading);
    assign(object, "collision_check", navigation.collision_check);
    assign(object, "omni_plan", navigation.omni_plan);
    assign(object, "retries", navigation.retries);
     assign(object, "timeout_s", navigation.timeout_s);
}

void read_perception(const Json& object, PerceptionConfig& perception) {
    detail::check_keys(object, {"mask_conf_threshold", "mask_weights", "foundationpose_root", "input_scale",
                               "depth_median_radius", "objects", "aliases"}, "perception");
    assign(object, "mask_conf_threshold", perception.mask_conf_threshold);
    assign(object, "input_scale", perception.input_scale);
    assign(object, "depth_median_radius", perception.depth_median_radius);
    for (const auto& [key, destination] : {
             std::pair{"mask_weights", &perception.mask_weights}, {"foundationpose_root", &perception.foundationpose_root}})
        if (object.contains(key)) *destination = detail::read<std::string>(object.at(key));
    if (object.contains("objects")) {
        const auto& objects = object.at("objects");
        require(objects.is_object(), "perception.objects must be an object");
        for (auto it = objects.begin(); it != objects.end(); ++it) {
            detail::check_keys(it.value(), {"mesh_path", "grasp_point_object", "pre_grasp_distance", "gripper_ee_distance"}, "objects." + it.key());
            auto& settings = perception.objects[it.key()];
            if (it->contains("mesh_path")) settings.mesh_path = detail::read<std::string>(it->at("mesh_path"));
            if (it->contains("grasp_point_object")) settings.grasp_point_object = detail::read_vec3(it->at("grasp_point_object"));
            assign(*it, "pre_grasp_distance", settings.pre_grasp_distance);
            assign(*it, "gripper_ee_distance", settings.gripper_ee_distance);
        }
    }
    if (object.contains("aliases")) {
        const auto& aliases = object.at("aliases");
        require(aliases.is_object(), "perception.aliases must be an object");
        for (auto it = aliases.begin(); it != aliases.end(); ++it)
            perception.aliases[it.key()] = detail::read<std::string>(it.value());
    }
}
}  // namespace

void Config::validate(Mode mode) const {
    require(robot.arm_name == "left_arm" || robot.arm_name == "right_arm", "arm_name must be left_arm or right_arm");
    require(robot.gripper_name == (robot.arm_name == "left_arm" ? "left_gripper" : "right_gripper"), "arm and gripper must match");
    positive(robot.gripper_open_width, "gripper_open_width");
    nonnegative(robot.gripper_close_width, "gripper_close_width");
    require(robot.gripper_close_width < robot.gripper_open_width, "close width must be smaller than open width");
    nonnegative(robot.gripper_width_margin, "gripper_width_margin");
    positive(robot.gripper_effort, "gripper_effort");
    positive(robot.gripper_speed, "gripper_speed");
    positive(robot.gripper_grasp_effort_threshold, "gripper_grasp_effort_threshold");
    require(robot.gripper_grasp_effort_threshold <= robot.gripper_effort, "grasp effort threshold exceeds commanded effort");
    count(robot.gripper_grasp_effort_stable_samples, "gripper_grasp_effort_stable_samples");
    nonnegative(robot.gripper_grasp_width_stable_tolerance_m, "gripper_grasp_width_stable_tolerance_m");
    positive(robot.gripper_grasp_timeout_s, "gripper_grasp_timeout_s");
    positive(robot.gripper_state_poll_interval_s, "gripper_state_poll_interval_s");
    require(robot.gripper_state_poll_interval_s <= robot.gripper_grasp_timeout_s, "gripper polling interval exceeds timeout");
    nonnegative(robot.foundationpose_lying_angle_threshold_deg, "foundationpose_lying_angle_threshold_deg");
    require(robot.foundationpose_lying_angle_threshold_deg <= 90, "lying threshold must be in [0,90]");
    positive(robot.lift_height, "lift_height");
    require(std::isfinite(robot.work_angle_min_deg) && std::isfinite(robot.work_angle_max_deg) &&
            robot.work_angle_min_deg >= -180 && robot.work_angle_max_deg <= 180 &&
            robot.work_angle_min_deg <= robot.work_angle_max_deg, "invalid work angle interval");
    positive(robot.move_timeout, "move_timeout");
    positive(robot.joint_speed, "joint_speed");
    positive(robot.home_timeout, "home_timeout");
    nonnegative(robot.home_tolerance_rad, "home_tolerance_rad");
    if (robot.home_joint_positions) {
        require(!robot.home_joint_positions->empty(), "home_joint_positions cannot be empty");                                    
        for (double joint : *robot.home_joint_positions) require(std::isfinite(joint), "home joints must be finite");
    }                                                                                                                            
    count(task.perception_retries, "perception_retries");                                                                             
    count(task.empty_confirmations, "empty_confirmations");                                   
    count(task.max_cycles, "max_cycles");                               
    nonnegative(task.perception_retry_interval_s, "perception_retry_interval_s");                                              
    positive(navigation.timeout_s, "navigation.timeout_s");                                                                   
    count(navigation.retries, "navigation.retries");                                                                       
    for (const auto* pose : {&navigation.origin_pose, &navigation.pickup_pose, &navigation.place_pose}) {
        if (*pose) validate_pose(**pose);
        else require(mode != Mode::Cyclic, "cyclic mode requires origin, pickup and place poses");
    }
    require(!perception.objects.empty(), "at least one material is required");
    positive(perception.mask_conf_threshold, "mask_conf_threshold");
    require(perception.mask_conf_threshold <= 1, "mask_conf_threshold must be <= 1");
    positive(perception.input_scale, "input_scale");
    require(perception.input_scale <= 1, "input_scale must be <= 1");
    require(perception.depth_median_radius >= 0, "depth_median_radius must be nonnegative");
    require(!perception.mask_weights.empty(), "mask_weights cannot be empty");
    for (const auto& [name, object] : perception.objects) {
        require(!name.empty() && !object.mesh_path.empty(), "material name and mesh_path cannot be empty");
        require(object.grasp_point_object.allFinite(), "grasp_point_object must be finite");
        positive(object.pre_grasp_distance, name + ".pre_grasp_distance");
        positive(object.gripper_ee_distance, name + ".gripper_ee_distance");
    }
    for (const auto& [alias, name] : perception.aliases) {
        require(!alias.empty() && perception.objects.contains(name), "alias must name a configured material");
        require(!perception.objects.contains(alias) || alias == name, "alias cannot remap a canonical material");
    }
}

Config load_config(const std::filesystem::path& path) {
    try {
        std::ifstream input(path);
        if (!input) throw std::invalid_argument("cannot open config");
        const auto document = detail::parse_json(input);
        detail::check_keys(document, {"schema_version", "resource_root", "robot", "task", "navigation", "perception"}, "config");
        if (document.contains("schema_version")) require(detail::read<int>(document.at("schema_version")) == 1, "unsupported schema_version");
        Config config;
        if (document.contains("robot")) read_robot(document.at("robot"), config.robot);
        if (document.contains("navigation")) read_navigation(document.at("navigation"), config.navigation);
        if (document.contains("perception")) read_perception(document.at("perception"), config.perception);
        if (document.contains("task")) {
            const auto& task = document.at("task");
            detail::check_keys(task, {"perception_retries", "perception_retry_interval_s", "empty_confirmations", "max_cycles"}, "task");
            assign(task, "perception_retries", config.task.perception_retries);
            assign(task, "perception_retry_interval_s", config.task.perception_retry_interval_s);
            assign(task, "empty_confirmations", config.task.empty_confirmations);
            assign(task, "max_cycles", config.task.max_cycles);
        }
        std::filesystem::path root = std::filesystem::absolute(path).parent_path();
        if (document.contains("resource_root")) root /= detail::read<std::string>(document.at("resource_root"));
        const auto resolve = [&root](std::filesystem::path& asset) {
            if (!asset.empty()) asset = (asset.is_absolute() ? asset : root / asset).lexically_normal();
        };
        resolve(config.perception.mask_weights);
        resolve(config.perception.foundationpose_root);
        for (auto& [name, object] : config.perception.objects) { (void)name; resolve(object.mesh_path); }
        config.validate();
        return config;
    } catch (const std::exception& error) {
        throw std::invalid_argument(path.string() + ": " + error.what());
    }
}
}  // namespace robmove
