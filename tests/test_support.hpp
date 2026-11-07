#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {
struct Case { std::string name; std::function<void()> run; };
inline std::vector<Case>& cases() { static std::vector<Case> value; return value; }
struct Register {
    Register(std::string name, std::function<void()> run) {
        cases().push_back({std::move(name), std::move(run)});
    }
};
inline void check(bool ok, const char* expression, const char* file, int line) {
    if (!ok) throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expression);
}
template<class F> void throws(F&& action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    check(caught, "expected an exception", __FILE__, __LINE__);
}
}  // namespace test

#define TEST(name) static void name(); static test::Register register_##name(#name, name); static void name()
#define CHECK(...) test::check(static_cast<bool>((__VA_ARGS__)), #__VA_ARGS__, __FILE__, __LINE__)
#define NEAR(a, b, eps) CHECK(std::isfinite(a) && std::abs((a) - (b)) <= (eps))
#define THROWS(...) test::throws([&] { (void)(__VA_ARGS__); })
