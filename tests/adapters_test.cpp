#include "test_support.hpp"
#include "robmove/adapters/replay.hpp"
#include <sstream>

using namespace robmove;

TEST(replay_parses_multiple_candidates_and_reports_eof_as_unavailable) {
    std::istringstream input(R"({"schema_version":1,"frames":[{"status":"found","candidates":[{"class_name":"wulongcha","confidence":0.9,"pose_base":[0.6,-0.1,0.4,0,0,0,1],"object_width_m":0.06}]},{"status":"empty"},{"status":"unavailable","reason":"no depth"}]})");
    ReplayPerception perception(read_replay(input));
    THROWS(perception.observe());
    perception.load();
    perception.load();
    CHECK(perception.observe().candidates[0].class_name == "wulongcha");
    CHECK(perception.observe().status == PerceptionStatus::Empty);
    CHECK(perception.observe().reason == "no depth");
    CHECK(perception.observe().status == PerceptionStatus::Unavailable);
    perception.shutdown(); perception.shutdown();
    THROWS(perception.observe());
}

TEST(replay_rejects_malformed_statuses_unknown_fields_and_poses) {
    for (const std::string value : {
             R"({"frames":[{"status":"found"}]})", R"({"frames":[{"status":"bad"}]})",
             R"({"frames":[{"status":"empty","candidates":[{}]}]})",
             R"({"frames":[{"status":"found","candidates":[{"class_name":"wulongcha","confidence":0.9,"pose_base":[0,0,0,0,0,0,0]}]}]})",
             R"({"framez":[]})", R"({"frames":[],"schema_version":999})"}) {
        std::istringstream input(value); THROWS(read_replay(input));
    }
}
