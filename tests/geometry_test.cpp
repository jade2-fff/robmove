#include "test_support.hpp"
#include "robmove/geometry.hpp"
#include <limits>
#include <numbers>

using namespace robmove;

TEST(pose_round_trip_preserves_xyzw_convention) {
    const Pose pose = pose_from_array({1, 2, 3, 0, 0, std::sqrt(0.5), std::sqrt(0.5)});
    CHECK((transform_point(pose, Vec3{1, 0, 0}) - Vec3{1, 3, 3}).norm() < 1e-12);
    const auto serialized = pose_to_array(pose);
    NEAR(serialized[5], std::sqrt(0.5), 1e-12);
    NEAR(serialized[6], std::sqrt(0.5), 1e-12);
}

TEST(pose_rejects_zero_and_nonfinite_quaternions) {
    THROWS(pose_from_array({0, 0, 0, 0, 0, 0, 0}));
    THROWS(pose_from_array({0, 0, 0, 0, 0, 0, std::numeric_limits<double>::infinity()}));
    NEAR(pose_from_array({0, 0, 0, 0, 0, 0, 2}).orientation.norm(), 1.0, 1e-12);
}

TEST(back_projection_and_camera_transform_use_meters) {
    const auto point = back_project(420, 190, 1.5, {500, 500, 320, 240});
    CHECK((point - Vec3{0.3, -0.15, 1.5}).norm() < 1e-12);
    THROWS(back_project(1, 1, 0, {500, 500, 320, 240}));
    THROWS(back_project(1, 1, 1, {0, 500, 320, 240}));
}

TEST(depth_median_filters_invalid_pixels_and_handles_even_samples) {
    std::vector<float> pixels{0, 0.1F, 0.2F, 0.3F, 0.4F, 2.0F,
                             std::numeric_limits<float>::quiet_NaN(), 0.05F, -1};
    std::vector<float> scratch;
    const auto depth = robust_depth(pixels, 3, 3, 1, 1, 5, scratch);
    CHECK(depth.has_value());
    NEAR(*depth, 0.25, 1e-6);
    CHECK(!robust_depth(pixels, 3, 3, -10, -10, 0, scratch));
    THROWS(robust_depth(pixels, 4, 3, 0, 0, 1, scratch));
    THROWS(robust_depth(pixels, 3, 3, 0, 0, -1, scratch));
}

TEST(depth_median_clamps_boundary_and_reuses_scratch) {
    std::vector<float> pixels{0.5F, 0.7F, 1.0F, 1.5F};
    std::vector<float> scratch;
    NEAR(*robust_depth(pixels, 2, 2, 100, 100, 0, scratch), 1.5, 1e-6);
    robust_depth(pixels, 2, 2, 0, 0, 5, scratch);
    const auto capacity = scratch.capacity();
    robust_depth(pixels, 2, 2, 0, 0, 5, scratch);
    CHECK(scratch.capacity() == capacity);
}
