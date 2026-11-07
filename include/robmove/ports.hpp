#pragma once

#include "robmove/types.hpp"

namespace robmove {
class PerceptionPort {
public:
    virtual ~PerceptionPort() = default;
    virtual void load() = 0;
    virtual Observation observe() = 0;
    virtual void shutdown() noexcept = 0;
};
}  // namespace robmove
