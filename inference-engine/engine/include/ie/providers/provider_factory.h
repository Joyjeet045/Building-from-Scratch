/*
 * provider_factory.h
 * Single place that maps a provider name from the config to a concrete
 * backend, so registering a new one touches only this translation unit.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ie/providers/cpu_execution_provider.h"
#include "ie/providers/execution_provider.h"

namespace ie {

std::unique_ptr<ExecutionProvider> makeExecutionProvider(const std::string& name,
                                                          const CpuProviderOptions& cpuOptions);

std::vector<std::string> availableExecutionProviders();

bool isExecutionProviderAvailable(const std::string& name);

}
