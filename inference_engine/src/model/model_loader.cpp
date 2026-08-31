/*
 * model_loader.cpp
 * Binary reader/writer for the "OIEN" model format described in
 * model_loader.h.
 */
#include "ie/model/model_loader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace ie {

namespace {

constexpr char kMagic[4] = {'O', 'I', 'E', 'N'};
constexpr uint32_t kVersion = 1;

constexpr uint8_t kAttrInt = 0;
constexpr uint8_t kAttrFloat = 1;
constexpr uint8_t kAttrInts = 2;

void writeU32(std::ostream& os, uint32_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeU64(std::ostream& os, uint64_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeI64(std::ostream& os, int64_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeF32(std::ostream& os, float value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeString(std::ostream& os, const std::string& s) {
    writeU32(os, static_cast<uint32_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

uint32_t readU32(std::istream& is) {
    uint32_t value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) throw std::runtime_error("Unexpected end of model file while reading u32");
    return value;
}

uint64_t readU64(std::istream& is) {
    uint64_t value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) throw std::runtime_error("Unexpected end of model file while reading u64");
    return value;
}

int64_t readI64(std::istream& is) {
    int64_t value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) throw std::runtime_error("Unexpected end of model file while reading i64");
    return value;
}

float readF32(std::istream& is) {
    float value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) throw std::runtime_error("Unexpected end of model file while reading f32");
    return value;
}

std::string readString(std::istream& is) {
    uint32_t len = readU32(is);
    std::string s(len, '\0');
    if (len > 0) {
        is.read(s.data(), len);
        if (!is) throw std::runtime_error("Unexpected end of model file while reading string");
    }
    return s;
}

}

Model ModelLoader::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open model file: " + path);

    char magic[4];
    file.read(magic, 4);
    if (!file || magic[0] != kMagic[0] || magic[1] != kMagic[1] || magic[2] != kMagic[2] ||
        magic[3] != kMagic[3]) {
        throw std::runtime_error("Not a valid OIEN model file: " + path);
    }

    uint32_t version = readU32(file);
    if (version != kVersion) {
        throw std::runtime_error("Unsupported OIEN model version: " + std::to_string(version));
    }

    Model model;

    uint32_t numInitializers = readU32(file);
    for (uint32_t i = 0; i < numInitializers; ++i) {
        std::string name = readString(file);
        uint32_t ndim = readU32(file);
        std::vector<uint64_t> dims(ndim);
        uint64_t total = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            dims[d] = readU64(file);
            total *= dims[d];
        }
        std::vector<float> data(total);
        for (uint64_t e = 0; e < total; ++e) {
            data[e] = readF32(file);
        }
        model.addInitializer(name, Tensor<float>(dims, data));
    }

    uint32_t numNodes = readU32(file);
    for (uint32_t i = 0; i < numNodes; ++i) {
        std::string name = readString(file);
        std::string opType = readString(file);

        uint32_t numInputs = readU32(file);
        std::vector<std::string> inputs(numInputs);
        for (uint32_t j = 0; j < numInputs; ++j) inputs[j] = readString(file);

        uint32_t numOutputs = readU32(file);
        std::vector<std::string> outputs(numOutputs);
        for (uint32_t j = 0; j < numOutputs; ++j) outputs[j] = readString(file);

        uint32_t numAttrs = readU32(file);
        std::unordered_map<std::string, Attribute> attrs;
        for (uint32_t j = 0; j < numAttrs; ++j) {
            std::string attrName = readString(file);
            uint8_t tag = 0;
            file.read(reinterpret_cast<char*>(&tag), sizeof(tag));
            if (!file) throw std::runtime_error("Unexpected end of model file while reading attribute tag");

            if (tag == kAttrInt) {
                attrs.emplace(attrName, Attribute::makeInt(readI64(file)));
            } else if (tag == kAttrFloat) {
                attrs.emplace(attrName, Attribute::makeFloat(readF32(file)));
            } else if (tag == kAttrInts) {
                uint32_t count = readU32(file);
                std::vector<int64_t> values(count);
                for (uint32_t k = 0; k < count; ++k) values[k] = readI64(file);
                attrs.emplace(attrName, Attribute::makeInts(std::move(values)));
            } else {
                throw std::runtime_error("Unknown attribute tag in model file: " + std::to_string(tag));
            }
        }

        model.graph().addNode(
            std::make_unique<Node>(name, opType, std::move(inputs), std::move(outputs), std::move(attrs)));
    }

    uint32_t numGraphInputs = readU32(file);
    std::vector<std::string> graphInputs(numGraphInputs);
    for (uint32_t i = 0; i < numGraphInputs; ++i) graphInputs[i] = readString(file);
    model.graph().setInputs(std::move(graphInputs));

    uint32_t numGraphOutputs = readU32(file);
    std::vector<std::string> graphOutputs(numGraphOutputs);
    for (uint32_t i = 0; i < numGraphOutputs; ++i) graphOutputs[i] = readString(file);
    model.graph().setOutputs(std::move(graphOutputs));

    return model;
}

void ModelLoader::save(const Model& model, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file for writing: " + path);

    file.write(kMagic, 4);
    writeU32(file, kVersion);

    const auto& initializers = model.initializers();
    writeU32(file, static_cast<uint32_t>(initializers.size()));
    for (const auto& [name, tensor] : initializers) {
        writeString(file, name);
        writeU32(file, static_cast<uint32_t>(tensor.shape().size()));
        for (uint64_t d : tensor.shape()) writeU64(file, d);
        for (float v : tensor) writeF32(file, v);
    }

    std::vector<Node*> nodes = model.graph().nodesInInsertionOrder();
    writeU32(file, static_cast<uint32_t>(nodes.size()));
    for (Node* node : nodes) {
        writeString(file, node->name());
        writeString(file, node->opTypeString());

        writeU32(file, static_cast<uint32_t>(node->inputs().size()));
        for (const auto& input : node->inputs()) writeString(file, input);

        writeU32(file, static_cast<uint32_t>(node->outputs().size()));
        for (const auto& output : node->outputs()) writeString(file, output);

        const auto& attrs = node->attributes();
        writeU32(file, static_cast<uint32_t>(attrs.size()));
        for (const auto& [attrName, attr] : attrs) {
            writeString(file, attrName);
            if (attr.type() == AttributeType::Int) {
                file.put(static_cast<char>(kAttrInt));
                writeI64(file, attr.asInt());
            } else if (attr.type() == AttributeType::Float) {
                file.put(static_cast<char>(kAttrFloat));
                writeF32(file, attr.asFloat());
            } else {
                file.put(static_cast<char>(kAttrInts));
                const auto& values = attr.asInts();
                writeU32(file, static_cast<uint32_t>(values.size()));
                for (int64_t v : values) writeI64(file, v);
            }
        }
    }

    const auto& graphInputs = model.graph().inputs();
    writeU32(file, static_cast<uint32_t>(graphInputs.size()));
    for (const auto& name : graphInputs) writeString(file, name);

    const auto& graphOutputs = model.graph().outputs();
    writeU32(file, static_cast<uint32_t>(graphOutputs.size()));
    for (const auto& name : graphOutputs) writeString(file, name);
}

}
