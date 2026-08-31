/*
 * test_framework.h
 * Minimal self-registering test framework (no external test dependency):
 * define a test with TEST_CASE(name) { ASSERT_*(...); }, and test_main.cpp
 * runs every registered case.
 */
#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

struct Registrar {
    Registrar(std::string name, std::function<void()> fn) {
        registry().push_back({std::move(name), std::move(fn)});
    }
};

struct AssertionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}

#define TEST_CASE(name)                                                 \
    static void name();                                                 \
    static test::Registrar registrar_##name(#name, name);               \
    static void name()

#define ASSERT_TRUE(cond)                                                                        \
    do {                                                                                         \
        if (!(cond)) {                                                                           \
            throw test::AssertionFailure(std::string("ASSERT_TRUE failed: ") + #cond + " at " +  \
                                          __FILE__ + ":" + std::to_string(__LINE__));             \
        }                                                                                         \
    } while (0)

#define ASSERT_EQ(a, b)                                                                          \
    do {                                                                                         \
        if (!((a) == (b))) {                                                                     \
            throw test::AssertionFailure(std::string("ASSERT_EQ failed: ") + #a + " == " + #b +  \
                                          " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
        }                                                                                         \
    } while (0)

#define ASSERT_NEAR(a, b, eps)                                                                   \
    do {                                                                                         \
        if (std::fabs(static_cast<double>(a) - static_cast<double>(b)) > (eps)) {                \
            throw test::AssertionFailure(std::string("ASSERT_NEAR failed: ") + #a + " ~= " + #b + \
                                          " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
        }                                                                                         \
    } while (0)

#define ASSERT_THROWS(expr)                                                                      \
    do {                                                                                         \
        bool threw = false;                                                                      \
        try {                                                                                     \
            (expr);                                                                              \
        } catch (...) {                                                                          \
            threw = true;                                                                        \
        }                                                                                         \
        if (!threw) {                                                                            \
            throw test::AssertionFailure(std::string("ASSERT_THROWS failed: ") + #expr + " at " + \
                                          __FILE__ + ":" + std::to_string(__LINE__));             \
        }                                                                                         \
    } while (0)
