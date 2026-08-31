/*
 * provider_factory.cpp
 * Backend registry. A CUDA provider would be added here behind a build flag
 * without any other file changing.
 */
#include "ie/providers/provider_factory.h"

#include <algorithm>
#include <stdexcept>

namespace ie {

std::vector<std::string> availableExecutionProviders() { return {"cpu"}; }

bool isExecutionProviderAvailable(const std::string& name) {
    std::vector<std::string> providers = availableExecutionProviders();
    return std::find(providers.begin(), providers.end(), name) != providers.end();
}

std::unique_ptr<ExecutionProvider> makeExecutionProvider(const std::string& name,
                                                          const CpuProviderOptions& cpuOptions) {
    if (name == "cpu") {
        return std::make_unique<CpuExecutionProvider>(cpuOptions);
    }
    throw std::invalid_argument("Unknown execution provider: " + name);
}

}
