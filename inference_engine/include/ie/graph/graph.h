/*
 * graph.h
 * Adjacency-list computation graph built incrementally from Nodes. Edges come
 * from matching a node's output tensor names against other nodes' inputs.
 * Provides a DFS topological order for sequential execution and a wavefront
 * decomposition into levels, where every node within a level is independent of
 * its peers and can therefore run concurrently.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ie/graph/node.h"

namespace ie {

struct NodeEntry {
    std::unique_ptr<Node> node;
    std::vector<Node*> parents;
    std::vector<Node*> children;
};

class Graph {
public:
    Graph() = default;

    void addNode(std::unique_ptr<Node> node);

    // Takes the name by value internally: callers commonly pass node->name(),
    // which is destroyed partway through the erase.
    void removeNode(const std::string& name);
    void rebuildEdges();

    void setInputs(std::vector<std::string> inputs);
    void setOutputs(std::vector<std::string> outputs);

    const std::vector<std::string>& inputs() const;
    const std::vector<std::string>& outputs() const;

    Node* getNode(const std::string& name) const;
    std::size_t size() const;

    std::vector<Node*> topologicalSort() const;
    std::vector<std::vector<Node*>> executionLevels() const;
    std::vector<Node*> nodesInInsertionOrder() const;

    const std::vector<Node*>& parentsOf(const std::string& name) const;
    const std::vector<Node*>& childrenOf(const std::string& name) const;

private:
    void updateEdges(Node* node);
    void visit(Node* node, std::unordered_set<Node*>& visited, std::unordered_set<Node*>& onStack,
               std::vector<Node*>& order) const;

    std::unordered_map<std::string, NodeEntry> nodeMap_;
    std::vector<std::string> insertionOrder_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

}
