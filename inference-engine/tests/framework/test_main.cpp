/*
 * test_main.cpp
 * Runs every test registered via TEST_CASE across the linked translation
 * units and prints a pass/fail summary.
 */
#include <iostream>

#include "framework/test_framework.h"

int main() {
    int passed = 0;
    int failed = 0;

    for (const auto& testCase : test::registry()) {
        try {
            testCase.fn();
            std::cout << "[PASS] " << testCase.name << "\n";
            ++passed;
        } catch (const std::exception& ex) {
            std::cout << "[FAIL] " << testCase.name << ": " << ex.what() << "\n";
            ++failed;
        } catch (...) {
            std::cout << "[FAIL] " << testCase.name << ": unknown exception\n";
            ++failed;
        }
    }

    std::cout << "\n" << passed << " passed, " << failed << " failed, out of " << (passed + failed)
               << " total\n";

    return failed == 0 ? 0 : 1;
}
