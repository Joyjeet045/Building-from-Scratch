/*
 * graph_optimizer.h
 * Ahead-of-time rewrites applied once at load time: fusing an activation into
 * the producer that feeds it, folding subgraphs whose inputs are all
 * initializers into new initializers, and dropping nodes whose results nobody
 * consumes. Each pass reports how many rewrites it made.
 */
#pragma once

#include <string>
#include <vector>

#include "ie/model/model.h"
#include "ie/providers/execution_provider.h"

namespace ie {

struct OptimizationReport {
    std::size_t fusedActivations = 0;
    std::size_t foldedConstants = 0;
    std::size_t eliminatedNodes = 0;
    std::size_t nodesBefore = 0;
    std::size_t nodesAfter = 0;

    std::string toString() const;
    std::size_t total() const { return fusedActivations + foldedConstants + eliminatedNodes; }
};

struct OptimizerOptions {
    bool fuseActivations = true;
    bool foldConstants = true;
    bool eliminateDeadNodes = true;
};

class GraphOptimizer {
public:
    explicit GraphOptimizer(OptimizerOptions options = {});

    OptimizationReport run(Model& model, ExecutionProvider& provider) const;

private:
    std::size_t fuseActivations(Model& model) const;
    std::size_t foldConstants(Model& model, ExecutionProvider& provider) const;
    std::size_t eliminateDeadNodes(Model& model) const;

    OptimizerOptions options_;
};

}
