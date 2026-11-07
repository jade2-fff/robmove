#pragma once

#include "robmove/types.hpp"
#include <filesystem>
#include <map>

namespace robmove {
struct ObjectConfig { std::filesystem::path mesh_path; };

struct PerceptionConfig {
    double mask_conf_threshold = 0.40;
    std::filesystem::path mask_weights = "assets/models/yolo11n-seg_finetuned.pt";
    std::filesystem::path foundationpose_root;
    double input_scale = 1;
    int depth_median_radius = 5;
    std::map<std::string, ObjectConfig, std::less<>> objects{
        {"wulongcha", {"assets/objects/wulongcha/normalized.obj"}},
        {"chouzhi", {"assets/objects/chouzhi/normalized.obj"}},
        {"leshishupian", {"assets/objects/leshishupian/normalized.obj"}},
        {"xinlamian", {"assets/objects/xinlamian/normalized.obj"}},
        {"kekoukele", {"assets/objects/kekoukele/normalized.obj"}},
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
    PerceptionConfig perception;
    void validate() const;
};

// Relative asset paths are resolved against the config file, never the process cwd.
Config load_config(const std::filesystem::path& path);
}  // namespace robmove
