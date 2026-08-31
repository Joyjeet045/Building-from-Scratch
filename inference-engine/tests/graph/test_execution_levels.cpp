/*
 * test_execution_levels.cpp
 * Tests that the wavefront decomposition places mutually independent nodes in
 * the same level, which is what makes inter-op parallelism safe.
 */
#include <vector>

#include "framework/model_fixtures.h"
#include "framework/test_framework.h"
#include "ie/graph/graph.h"

TEST_CASE(graph_levels_group_independent_branches_together) {
    ie::Model model = fixtures::buildBranchingModel();
    std::vector<std::vector<ie::Node*>> levels = model.graph().executionLevels();

    ASSERT_EQ(levels.size(), 5u);
    ASSERT_EQ(levels[0].size(), 1u);
    ASSERT_EQ(levels[0][0]->name(), std::string("flatten"));
    ASSERT_EQ(levels[1][0]->name(), std::string("fc1"));
    ASSERT_EQ(levels[2][0]->name(), std::string("relu1"));

    ASSERT_EQ(levels[3].size(), 2u);
    bool hasLeft = levels[3][0]->name() == "left" || levels[3][1]->name() == "left";
    bool hasRight = levels[3][0]->name() == "right" || levels[3][1]->name() == "right";
    ASSERT_TRUE(hasLeft);
    ASSERT_TRUE(hasRight);

    ASSERT_EQ(levels[4].size(), 1u);
    ASSERT_EQ(levels[4][0]->name(), std::string("add"));
}

TEST_CASE(graph_levels_cover_every_node_exactly_once) {
    ie::Model model = fixtures::buildBranchingModel();
    std::vector<std::vector<ie::Node*>> levels = model.graph().executionLevels();

    std::size_t total = 0;
    for (const auto& level : levels) total += level.size();
    ASSERT_EQ(total, model.graph().size());
}

TEST_CASE(graph_levels_never_place_a_node_before_its_parent) {
    ie::Model model = fixtures::buildBranchingModel();
    std::vector<std::vector<ie::Node*>> levels = model.graph().executionLevels();

    std::unordered_map<std::string, std::size_t> levelOf;
    for (std::size_t i = 0; i < levels.size(); ++i) {
        for (ie::Node* node : levels[i]) levelOf[node->name()] = i;
    }

    for (const auto& [name, level] : levelOf) {
        for (ie::Node* parent : model.graph().parentsOf(name)) {
            ASSERT_TRUE(levelOf.at(parent->name()) < level);
        }
    }
}
