/*
 * model_loader.h
 * Reads and writes the engine's own compact binary model format (magic
 * "OIEN"), which plays the same role ONNX/protobuf plays in the original
 * design: a serialized graph plus its trained weight tensors. Layout:
 *   [4]   magic "OIEN"
 *   [u32] version
 *   [u32] initializer count, each: name, ndim, dims[u64]*, data[f32]*
 *   [u32] node count, each: name, op_type, inputs[string]*, outputs[string]*,
 *         attr count, each: name, tag(0=int64,1=float,2=int64 list), value
 *   [u32] graph input count, each: name
 *   [u32] graph output count, each: name
 * All strings are length-prefixed (u32) UTF-8 bytes; all integers/floats are
 * little-endian.
 */
#pragma once

#include <string>

#include "ie/model/model.h"

namespace ie {

class ModelLoader {
public:
    static Model load(const std::string& path);
    static void save(const Model& model, const std::string& path);
};

}
