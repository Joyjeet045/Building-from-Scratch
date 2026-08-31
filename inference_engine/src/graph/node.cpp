/*
 * node.cpp
 * Implementation of Node and the op-type string mapping.
 */
#include "ie/graph/node.h"

namespace ie {

OpType opTypeFromString(const std::string& name) {
    if (name == "Flatten") return OpType::Flatten;
    if (name == "Gemm") return OpType::Gemm;
    if (name == "Relu") return OpType::Relu;
    if (name == "Add") return OpType::Add;
    if (name == "GemmRelu") return OpType::GemmRelu;
    return OpType::Unknown;
}

const char* opTypeToString(OpType op) {
    switch (op) {
        case OpType::Flatten: return "Flatten";
        case OpType::Gemm: return "Gemm";
        case OpType::Relu: return "Relu";
        case OpType::Add: return "Add";
        case OpType::GemmRelu: return "GemmRelu";
        case OpType::Unknown: return "Unknown";
    }
    return "Unknown";
}

Node::Node(std::string name, std::string opTypeStr, std::vector<std::string> inputs,
           std::vector<std::string> outputs, std::unordered_map<std::string, Attribute> attributes)
    : name_(std::move(name)),
      opTypeStr_(std::move(opTypeStr)),
      opType_(opTypeFromString(opTypeStr_)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      attributes_(std::move(attributes)) {}

const std::string& Node::name() const { return name_; }

OpType Node::opType() const { return opType_; }

const std::string& Node::opTypeString() const { return opTypeStr_; }

const std::vector<std::string>& Node::inputs() const { return inputs_; }

const std::vector<std::string>& Node::outputs() const { return outputs_; }

void Node::setOpType(OpType op) {
    opType_ = op;
    opTypeStr_ = opTypeToString(op);
}

void Node::setOutputs(std::vector<std::string> outputs) { outputs_ = std::move(outputs); }

void Node::setAttr(const std::string& key, Attribute value) { attributes_[key] = std::move(value); }

int64_t Node::getAttrInt(const std::string& key, int64_t defaultValue) const {
    auto it = attributes_.find(key);
    if (it == attributes_.end()) return defaultValue;
    return it->second.asInt();
}

float Node::getAttrFloat(const std::string& key, float defaultValue) const {
    auto it = attributes_.find(key);
    if (it == attributes_.end()) return defaultValue;
    return it->second.asFloat();
}

std::vector<int64_t> Node::getAttrInts(const std::string& key, std::vector<int64_t> defaultValue) const {
    auto it = attributes_.find(key);
    if (it == attributes_.end()) return defaultValue;
    return it->second.asInts();
}

const std::unordered_map<std::string, Attribute>& Node::attributes() const { return attributes_; }

}
