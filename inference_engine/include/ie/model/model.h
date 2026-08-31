/*
 * model.h
 * Bundles a Graph together with its initializer tensors (trained weights),
 * mirroring the parsed contents of a model file. Initializers always live in
 * default CPU memory: a session copies them into provider-owned memory when it
 * loads, so a Model never outlives an allocator it depends on.
 */
#pragma once

#include <string>
#include <unordered_map>

#include "ie/core/tensor.h"
#include "ie/graph/graph.h"

namespace ie {

class Model {
public:
    Model() = default;

    Graph& graph() { return graph_; }
    const Graph& graph() const { return graph_; }

    void addInitializer(std::string name, Tensor<float> tensor) {
        initializers_.insert_or_assign(std::move(name), std::move(tensor));
    }

    bool hasInitializer(const std::string& name) const { return initializers_.count(name) != 0; }

    const Tensor<float>* getInitializer(const std::string& name) const {
        auto it = initializers_.find(name);
        return it == initializers_.end() ? nullptr : &it->second;
    }

    void removeInitializer(const std::string& name) { initializers_.erase(name); }

    const std::unordered_map<std::string, Tensor<float>>& initializers() const { return initializers_; }

private:
    Graph graph_;
    std::unordered_map<std::string, Tensor<float>> initializers_;
};

}
