#include "test_support.hpp"
#include <iostream>

int main() {
    int failures = 0;
    for (const auto& item : test::cases()) {
        try { item.run(); std::cout << "PASS " << item.name << '\n'; }
        catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << item.name << ": " << error.what() << '\n';
        }
    }
    std::cout << test::cases().size() << " cases, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
