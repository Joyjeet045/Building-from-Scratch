/*
 * graph.cpp
 * Implementation of Graph: edge construction by tensor-name matching, a
 * recursive DFS topological sort with cycle detection, and a wavefront
 * levelling that groups mutually independent nodes for parallel execution.
 */
#include "ie/graph/graph.h"

#include <algorithm>
#include <stdexcept>

namespace ie {

void Graph::addNode(std::unique_ptr<Node> node) {
    std::string name = node->name();
    Node* nodePtr = node.get();
    nodeMap_[name].node = std::move(node);
    insertionOrder_.push_back(name);
    updateEdges(nodePtr);
}

void Graph::removeNode(const std::string& name) {
    std::string owned = name;
    if (nodeMap_.erase(owned) == 0) return;
    insertionOrder_.erase(std::remove(insertionOrder_.begin(), insertionOrder_.end(), owned),
                           insertionOrder_.end());
    rebuildEdges();
}

void Graph::rebuildEdges() {
    for (auto& [name, entry] : nodeMap_) {
        entry.parents.clear();
        entry.children.clear();
    }
    for (const auto& name : insertionOrder_) {
        NodeEntry& entry = nodeMap_.at(name);
        Node* node = entry.node.get();
        for (auto& [otherName, other] : nodeMap_) {
            if (other.node.get() == node) continue;
            for (const auto& output : node->outputs()) {
                const auto& otherInputs = other.node->inputs();
                if (std::find(otherInputs.begin(), otherInputs.end(), output) != otherInputs.end()) {
                    entry.children.push_back(other.node.get());
                    other.parents.push_back(node);
                }
            }
        }
    }
}

void Graph::updateEdges(Node* node) {
    NodeEntry& selfEntry = nodeMap_.at(node->name());
    for (auto& [otherName, entry] : nodeMap_) {
        Node* other = entry.node.get();
        if (other == node) continue;

        for (const auto& output : other->outputs()) {
            if (std::find(node->inputs().begin(), node->inputs().end(), output) != node->inputs().end()) {
                entry.children.push_back(node);
                selfEntry.parents.push_back(other);
            }
        }
        for (const auto& output : node->outputs()) {
            if (std::find(other->inputs().begin(), other->inputs().end(), output) != other->inputs().end()) {
                selfEntry.children.push_back(other);
                entry.parents.push_back(node);
            }
        }
    }
}

void Graph::setInputs(std::vector<std::string> inputs) { inputs_ = std::move(inputs); }

void Graph::setOutputs(std::vector<std::string> outputs) { outputs_ = std::move(outputs); }

const std::vector<std::string>& Graph::inputs() const { return inputs_; }

const std::vector<std::string>& Graph::outputs() const { return outputs_; }

Node* Graph::getNode(const std::string& name) const {
    auto it = nodeMap_.find(name);
    if (it == nodeMap_.end()) return nullptr;
    return it->second.node.get();
}

std::size_t Graph::size() const { return nodeMap_.size(); }

std::vector<Node*> Graph::nodesInInsertionOrder() const {
    std::vector<Node*> nodes;
    nodes.reserve(insertionOrder_.size());
    for (const auto& name : insertionOrder_) {
        nodes.push_back(nodeMap_.at(name).node.get());
    }
    return nodes;
}

const std::vector<Node*>& Graph::parentsOf(const std::string& name) const {
    return nodeMap_.at(name).parents;
}

const std::vector<Node*>& Graph::childrenOf(const std::string& name) const {
    return nodeMap_.at(name).children;
}

std::vector<std::vector<Node*>> Graph::executionLevels() const {
    std::vector<Node*> order = topologicalSort();
    std::unordered_map<Node*, std::size_t> level;
    level.reserve(order.size());

    std::size_t maxLevel = 0;
    for (Node* node : order) {
        std::size_t current = 0;
        for (Node* parent : nodeMap_.at(node->name()).parents) {
            auto it = level.find(parent);
            if (it != level.end() && it->second + 1 > current) {
                current = it->second + 1;
            }
        }
        level[node] = current;
        if (current > maxLevel) maxLevel = current;
    }

    std::vector<std::vector<Node*>> levels(order.empty() ? 0 : maxLevel + 1);
    for (Node* node : order) {
        levels[level.at(node)].push_back(node);
    }
    return levels;
}

void Graph::visit(Node* node, std::unordered_set<Node*>& visited, std::unordered_set<Node*>& onStack,
                   std::vector<Node*>& order) const {
    if (visited.count(node)) return;
    if (onStack.count(node)) {
        throw std::runtime_error("Graph contains a cycle at node: " + node->name());
    }
    onStack.insert(node);
    const auto& children = nodeMap_.at(node->name()).children;
    for (Node* child : children) {
        visit(child, visited, onStack, order);
    }
    onStack.erase(node);
    visited.insert(node);
    order.push_back(node);
}

std::vector<Node*> Graph::topologicalSort() const {
    std::unordered_set<Node*> visited;
    std::unordered_set<Node*> onStack;
    std::vector<Node*> order;
    order.reserve(nodeMap_.size());

    for (const auto& name : insertionOrder_) {
        Node* node = nodeMap_.at(name).node.get();
        if (!visited.count(node)) {
            visit(node, visited, onStack, order);
        }
    }
    std::reverse(order.begin(), order.end());
    return order;
}

}
