#include "robmove/ports.hpp"
#include <thread>

namespace robmove {
double SteadyClock::now() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void SteadyClock::sleep_for(double seconds) {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

std::string_view to_string(State state) noexcept {
    switch (state) {
    case State::Init: return "INIT";
    case State::NavToOrigin: return "NAV_TO_ORIGIN";
    case State::NavToPickup: return "NAV_TO_PICKUP";
    case State::ArmHomeForPerception: return "ARM_HOME_FOR_PERCEPTION";
    case State::LoadPerceptionModel: return "LOAD_PERCEPTION_MODEL";
    case State::CameraPerception: return "CAMERA_PERCEPTION";
    case State::Grasp: return "GRASP";
    case State::Lift: return "LIFT";
    case State::LiftSafe: return "LIFT_SAFE";
    case State::ArmHomeWithObject: return "ARM_HOME_WITH_OBJECT";
    case State::NavToPlace: return "NAV_TO_PLACE";
    case State::LocalPlace: return "LOCAL_PLACE";
    case State::RetreatFromPlace: return "RETREAT_FROM_PLACE";
    case State::Release: return "RELEASE";
    case State::RetreatToLiftSafe: return "RETREAT_TO_LIFT_SAFE";
    case State::ArmHomeAfterRelease: return "ARM_HOME_AFTER_RELEASE";
    case State::Hold: return "HOLD";
    case State::Done: return "DONE";
    }
    return "UNKNOWN";
}
std::string_view to_string(Custody custody) noexcept {
    switch (custody) {
    case Custody::Empty: return "empty";
    case Custody::PossiblyHolding: return "possibly_holding";
    case Custody::Holding: return "holding";
    }
    return "unknown";
}
}  // namespace robmove
