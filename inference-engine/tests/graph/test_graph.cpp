/*
 * test_graph.cpp
 * Unit tests for Graph: edge construction from tensor-name matching (added in
 * non-topological order) and DFS topological sort, including cycle
 * detection.
 */
#include <algorithm>
#include <unordered_map>

#include "framework/test_framework.h"
#include "ie/graph/graph.h"

namespace {

std::unique_ptr<ie::Node> makeNode(std::string name, std::vector<std::string> inputs,
                                    std::vector<std::string> outputs) {
    return std::make_unique<ie::Node>(std::move(name), "Add", std::move(inputs), std::move(outputs),
                                       std::unordered_map<std::string, ie::Attribute>{});
}

}

TEST_CASE(graph_topological_sort_respects_diamond_dependencies) {
    ie::Graph graph;
    graph.addNode(makeNode("fc3", {"sum", "w3", "b3"}, {"output"}));
    graph.addNode(makeNode("flatten", {"input"}, {"flat"}));
    graph.addNode(makeNode("add", {"left_out", "right_out"}, {"sum"}));
    graph.addNode(makeNode("fc1", {"flat", "w1", "b1"}, {"h"}));
    graph.addNode(makeNode("left2", {"left1_out", "wl2", "bl2"}, {"left_out"}));
    graph.addNode(makeNode("right", {"h", "wr", "br"}, {"right_out"}));
    graph.addNode(makeNode("left1", {"h", "wl1", "bl1"}, {"left1_out"}));

    std::vector<ie::Node*> order = graph.topologicalSort();
    ASSERT_EQ(order.size(), 7u);

    std::unordered_map<std::string, std::size_t> position;
    for (std::size_t i = 0; i < order.size(); ++i) {
        position[order[i]->name()] = i;
    }

    ASSERT_TRUE(position.at("flatten") < position.at("fc1"));
    ASSERT_TRUE(position.at("fc1") < position.at("left1"));
    ASSERT_TRUE(position.at("fc1") < position.at("right"));
    ASSERT_TRUE(position.at("left1") < position.at("left2"));
    ASSERT_TRUE(position.at("left2") < position.at("add"));
    ASSERT_TRUE(position.at("right") < position.at("add"));
    ASSERT_TRUE(position.at("add") < position.at("fc3"));
}

TEST_CASE(graph_get_node_returns_null_for_unknown_name) {
    ie::Graph graph;
    graph.addNode(makeNode("only", {"input"}, {"output"}));
    ASSERT_TRUE(graph.getNode("only") != nullptr);
    ASSERT_TRUE(graph.getNode("missing") == nullptr);
}

TEST_CASE(graph_topological_sort_detects_cycles) {
    ie::Graph graph;
    graph.addNode(makeNode("a", {"y"}, {"x"}));
    graph.addNode(makeNode("b", {"x"}, {"y"}));
    ASSERT_THROWS(graph.topologicalSort());
}
