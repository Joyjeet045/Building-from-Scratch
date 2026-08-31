/*
 * attribute.h
 * Small tagged-union value used to store per-node configuration (e.g. Gemm's
 * transA/transB/alpha/beta) as loaded from the model file.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ie {

enum class AttributeType { Int, Float, Ints };

class Attribute {
public:
    Attribute() = default;

    static Attribute makeInt(int64_t value) {
        Attribute attr;
        attr.type_ = AttributeType::Int;
        attr.intValue_ = value;
        return attr;
    }

    static Attribute makeFloat(float value) {
        Attribute attr;
        attr.type_ = AttributeType::Float;
        attr.floatValue_ = value;
        return attr;
    }

    static Attribute makeInts(std::vector<int64_t> values) {
        Attribute attr;
        attr.type_ = AttributeType::Ints;
        attr.intsValue_ = std::move(values);
        return attr;
    }

    AttributeType type() const { return type_; }

    int64_t asInt() const {
        if (type_ != AttributeType::Int) throw std::runtime_error("Attribute is not an Int");
        return intValue_;
    }

    float asFloat() const {
        if (type_ != AttributeType::Float) throw std::runtime_error("Attribute is not a Float");
        return floatValue_;
    }

    const std::vector<int64_t>& asInts() const {
        if (type_ != AttributeType::Ints) throw std::runtime_error("Attribute is not an Ints list");
        return intsValue_;
    }

private:
    AttributeType type_ = AttributeType::Int;
    int64_t intValue_ = 0;
    float floatValue_ = 0.0f;
    std::vector<int64_t> intsValue_;
};

}
