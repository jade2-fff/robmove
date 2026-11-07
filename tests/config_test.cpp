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

TEST(default_config_retains_five_materials_and_model_path) {
    const Config config;
    config.validate();
    CHECK(config.perception.objects.size() == 5);
    CHECK(config.perception.mask_weights == "assets/models/yolo11n-seg_finetuned.pt");
}

TEST(config_rejects_invalid_perception_parameters) {
    Config config;
    config.perception.mask_conf_threshold = 1.1;
    THROWS(config.validate());
    config = Config{};
    config.perception.input_scale = std::numeric_limits<double>::quiet_NaN();
    THROWS(config.validate());
    config = Config{};
    config.perception.depth_median_radius = -1;
    THROWS(config.validate());
    config = Config{};
    config.perception.mask_weights.clear();
    THROWS(config.validate());
    config = Config{};
    config.perception.objects.clear();
    THROWS(config.validate());
}

TEST(config_rejects_motor_navigation_and_grasp_parameters) {
    for (const std::string content : {
             R"({"robot":{}})", R"({"navigation":{}})", R"({"task":{}})",
             R"({"perception":{"objects":{"wulongcha":{"grasp_point_object":[0,0,0.1]}}}})"}) {
        const ConfigFile file(content);
        THROWS(load_config(file.file));
    }
}

TEST(config_paths_are_independent_of_working_directory) {
    const ConfigFile file(R"({"schema_version":1,"resource_root":"resources","perception":{"input_scale":0.5}})");
    const auto config = load_config(file.file);
    NEAR(config.perception.input_scale, 0.5, 1e-12);
    CHECK(config.perception.mask_weights == file.directory / "resources/assets/models/yolo11n-seg_finetuned.pt");
}

TEST(config_fails_on_typos_wrong_types_duplicate_keys_and_new_schema) {
    for (const std::string content : {
             R"({"perception":{"input_sacle":0.2}})", R"({"perception":{"depth_median_radius":1.5}})",
             R"({"perception":{"input_scale":true}})", R"({"schema_version":2})",
             R"({"perception":{"input_scale":0.5,"input_scale":1}})",
             R"({"perception":{"depth_median_radius":18446744073709551615}})", R"({"perception":null})"}) {
        const ConfigFile file(content);
        THROWS(load_config(file.file));
    }
}

TEST(config_allows_material_extension_and_rejects_bad_aliases) {
    const ConfigFile file(R"({"perception":{"objects":{"new_box":{"mesh_path":"box.obj"}},"aliases":{"new box":"new_box"}}})");
    const auto config = load_config(file.file);
    CHECK(config.perception.objects.size() == 6);
    CHECK(config.perception.aliases.at("new box") == "new_box");
    Config invalid;
    invalid.perception.aliases["bad"] = "missing";
    THROWS(invalid.validate());
}
