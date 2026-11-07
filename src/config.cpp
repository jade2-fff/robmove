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
            detail::check_keys(it.value(), {"mesh_path"}, "objects." + it.key());
            auto& settings = perception.objects[it.key()];
            if (it->contains("mesh_path")) settings.mesh_path = detail::read<std::string>(it->at("mesh_path"));
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

void Config::validate() const {
    require(!perception.objects.empty(), "at least one material is required");
    positive(perception.mask_conf_threshold, "mask_conf_threshold");
    require(perception.mask_conf_threshold <= 1, "mask_conf_threshold must be <= 1");
    positive(perception.input_scale, "input_scale");
    require(perception.input_scale <= 1, "input_scale must be <= 1");
    require(perception.depth_median_radius >= 0, "depth_median_radius must be nonnegative");
    require(!perception.mask_weights.empty(), "mask_weights cannot be empty");
    for (const auto& [name, object] : perception.objects) {
        require(!name.empty() && !object.mesh_path.empty(), "material name and mesh_path cannot be empty");
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
        detail::check_keys(document, {"schema_version", "resource_root", "perception"}, "config");
        if (document.contains("schema_version")) require(detail::read<int>(document.at("schema_version")) == 1, "unsupported schema_version");
        Config config;
        if (document.contains("perception")) read_perception(document.at("perception"), config.perception);
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
