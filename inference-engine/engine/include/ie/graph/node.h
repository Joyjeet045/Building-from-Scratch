/*
 * node.h
 * Represents a single operation in the computation graph: its op type, the
 * names of its input/output tensors, and any attributes controlling it.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ie/graph/attribute.h"

namespace ie {

enum class OpType { Flatten, Gemm, Relu, Add, GemmRelu, Unknown };

OpType opTypeFromString(const std::string& name);
const char* opTypeToString(OpType op);

class Node {
public:
    Node(std::string name, std::string opTypeStr, std::vector<std::string> inputs,
         std::vector<std::string> outputs, std::unordered_map<std::string, Attribute> attributes);

    const std::string& name() const;
    OpType opType() const;
    const std::string& opTypeString() const;
    const std::vector<std::string>& inputs() const;
    const std::vector<std::string>& outputs() const;

    void setOpType(OpType op);
    void setOutputs(std::vector<std::string> outputs);
    void setAttr(const std::string& key, Attribute value);

    int64_t getAttrInt(const std::string& key, int64_t defaultValue) const;
    float getAttrFloat(const std::string& key, float defaultValue) const;
    std::vector<int64_t> getAttrInts(const std::string& key, std::vector<int64_t> defaultValue) const;
    const std::unordered_map<std::string, Attribute>& attributes() const;

private:
    std::string name_;
    std::string opTypeStr_;
    OpType opType_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, Attribute> attributes_;
};

}
