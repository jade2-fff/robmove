#pragma once

#include "robmove/geometry.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <string_view>
#include <type_traits>

namespace robmove::detail {
using Json = nlohmann::json;

inline Json parse_json(std::istream& input) {
    std::vector<std::set<std::string>> keys;
    return Json::parse(input, [&keys](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        else if (event == Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
            throw std::invalid_argument("duplicate JSON key: " + value.get<std::string>());
        else if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    });
}

inline void check_keys(const Json& object, std::initializer_list<std::string_view> allowed, const std::string& context) {
    if (!object.is_object()) throw std::invalid_argument(context + " must be an object");
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end())
            throw std::invalid_argument("unknown field: " + context + "." + it.key());
    }
}

template<class T> T read(const Json& value) 
{
    if constexpr (std::is_same_v<T, bool>) 
    {
        if (!value.is_boolean()) throw std::invalid_argument("expected a boolean");
    } else if constexpr (std::is_same_v<T, int>) 
    {
        if (!value.is_number_integer()) throw std::invalid_argument("expected an integer");
        if (value.is_number_unsigned()) 
        {
            if (value.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("integer out of range");
        } else
         {
            const auto number = value.get<std::int64_t>();
            if (number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max())
                throw std::invalid_argument("integer out of range");
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (!value.is_number() || !std::isfinite(value.get<double>())) throw std::invalid_argument("expected a finite number");
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!value.is_string()) throw std::invalid_argument("expected a string");
    }
    return value.get<T>();
}

template<class T> void assign(const Json& object, const char* key, T& destination) {
    if (object.contains(key)) {
        try { destination = read<T>(object.at(key)); }
        catch (const std::exception& error) { throw std::invalid_argument(std::string(key) + ": " + error.what()); }
    }
}

inline std::vector<double> read_numbers(const Json& value) {
    if (!value.is_array()) throw std::invalid_argument("expected an array of numbers");
    std::vector<double> numbers;
    numbers.reserve(value.size());
    for (const auto& item : value) numbers.push_back(read<double>(item));
    return numbers;
}

inline Pose read_pose(const Json& value) {
    const auto numbers = read_numbers(value);
    if (numbers.size() != 7) throw std::invalid_argument("pose needs [x,y,z,qx,qy,qz,qw]");
    std::array<double, 7> array{};
    std::copy(numbers.begin(), numbers.end(), array.begin());
    return pose_from_array(array);
}

inline Vec3 read_vec3(const Json& value) {
    const auto numbers = read_numbers(value);
    if (numbers.size() != 3) throw std::invalid_argument("point needs [x,y,z]");
    return {numbers[0], numbers[1], numbers[2]};
}
}  // namespace robmove::detail
