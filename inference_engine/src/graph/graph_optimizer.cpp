/*
 * graph_optimizer.cpp
 * Implementation of the load-time graph rewrites. Passes run to a fixpoint in
 * the order fuse, fold, eliminate, since each can expose opportunities for the
 * others.
 */
#include "ie/graph/graph_optimizer.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace ie {

namespace {

bool isGemmLike(OpType op) { return op == OpType::Gemm; }

}

std::string OptimizationReport::toString() const {
    std::ostringstream oss;
    oss << "nodes " << nodesBefore << " -> " << nodesAfter << " (fused " << fusedActivations
        << ", folded " << foldedConstants << ", eliminated " << eliminatedNodes << ")";
    return oss.str();
}

GraphOptimizer::GraphOptimizer(OptimizerOptions options) : options_(options) {}

std::size_t GraphOptimizer::fuseActivations(Model& model) const {
    Graph& graph = model.graph();
    std::size_t fused = 0;

    for (;;) {
        Node* target = nullptr;
        Node* producer = nullptr;

        for (Node* node : graph.nodesInInsertionOrder()) {
            if (node->opType() != OpType::Relu) continue;
            if (node->inputs().size() != 1) continue;

            const auto& parents = graph.parentsOf(node->name());
            if (parents.size() != 1) continue;

            Node* parent = parents.front();
            if (!isGemmLike(parent->opType())) continue;
            if (parent->outputs().size() != 1) continue;
            if (graph.childrenOf(parent->name()).size() != 1) continue;

            const auto& graphOutputs = graph.outputs();
            if (std::find(graphOutputs.begin(), graphOutputs.end(), parent->outputs().front()) !=
                graphOutputs.end()) {
                continue;
            }

            target = node;
            producer = parent;
            break;
        }

        if (target == nullptr) break;

        producer->setOpType(OpType::GemmRelu);
        producer->setOutputs(target->outputs());
        graph.removeNode(target->name());
        ++fused;
    }

    return fused;
}

std::size_t GraphOptimizer::foldConstants(Model& model, ExecutionProvider& provider) const {
    Graph& graph = model.graph();
    std::size_t folded = 0;

    for (;;) {
        Node* candidate = nullptr;
        for (Node* node : graph.nodesInInsertionOrder()) {
            if (node->outputs().size() != 1) continue;
            if (!provider.supports(node->opType())) continue;

            bool allConstant = !node->inputs().empty();
            for (const auto& input : node->inputs()) {
                if (!model.hasInitializer(input)) {
                    allConstant = false;
                    break;
                }
            }
            if (allConstant) {
                candidate = node;
                break;
            }
        }

        if (candidate == nullptr) break;

        std::vector<const Tensor<float>*> inputs;
        inputs.reserve(candidate->inputs().size());
        for (const auto& input : candidate->inputs()) {
            inputs.push_back(model.getInitializer(input));
        }

        Tensor<float> output;
        provider.compute(*candidate, inputs, output);

        model.addInitializer(candidate->outputs().front(), std::move(output));
        graph.removeNode(candidate->name());
        ++folded;
    }

    return folded;
}

std::size_t GraphOptimizer::eliminateDeadNodes(Model& model) const {
    Graph& graph = model.graph();
    std::size_t eliminated = 0;

    for (;;) {
        std::unordered_set<std::string> consumed;
        for (Node* node : graph.nodesInInsertionOrder()) {
            for (const auto& input : node->inputs()) consumed.insert(input);
        }
        for (const auto& output : graph.outputs()) consumed.insert(output);

        Node* dead = nullptr;
        for (Node* node : graph.nodesInInsertionOrder()) {
            bool anyUsed = false;
            for (const auto& output : node->outputs()) {
                if (consumed.count(output)) {
                    anyUsed = true;
                    break;
                }
            }
            if (!anyUsed) {
                dead = node;
                break;
            }
        }

        if (dead == nullptr) break;

        graph.removeNode(dead->name());
        ++eliminated;
    }

    return eliminated;
}

OptimizationReport GraphOptimizer::run(Model& model, ExecutionProvider& provider) const {
    OptimizationReport report;
    report.nodesBefore = model.graph().size();

    if (options_.foldConstants) {
        report.foldedConstants = foldConstants(model, provider);
    }
    if (options_.fuseActivations) {
        report.fusedActivations = fuseActivations(model);
    }
    if (options_.eliminateDeadNodes) {
        report.eliminatedNodes = eliminateDeadNodes(model);
    }

    report.nodesAfter = model.graph().size();
    return report;
}

}
