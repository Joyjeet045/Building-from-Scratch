"""
train_mnist.py

Trains the branching MLP used by the inference engine (Flatten -> Gemm ->
Relu -> Gemm/Gemm branches -> Add -> Gemm), then exports the trained weights
and the graph topology to the engine's own ".oien" binary format so that no
protobuf/ONNX toolchain is required to run inference in C++.

OIEN binary layout (all integers/floats little-endian, strings length-
prefixed UTF-8):
  magic "OIEN", u32 version
  u32 initializer_count, each: name, u32 ndim, u64 dims[ndim], f32 data[prod(dims)]
  u32 node_count, each: name, op_type, u32 n_in, string[n_in], u32 n_out,
      string[n_out], u32 n_attr, each: name, u8 tag(0=int64,1=float,2=int64 list), value
  u32 graph_input_count, string[*]
  u32 graph_output_count, string[*]

Also writes a handful of raw 28x28 uint8 sample images (one per digit) under
inputs/, each paired with its true label recorded in inputs/labels.txt, for
exercising the compiled C++ engine end to end.
"""
import argparse
import os
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

OIEN_MAGIC = b"OIEN"
OIEN_VERSION = 1
ATTR_INT = 0
ATTR_FLOAT = 1
ATTR_INTS = 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2_left = nn.Linear(512, 200)
        self.fc2_left2 = nn.Linear(200, 100)
        self.fc2_right = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        left = F.relu(self.fc2_left(h))
        left = self.fc2_left2(left)
        right = self.fc2_right(h)
        combined = left + right
        return self.fc3(combined)


def train(model, device, epochs, batch_size, lr, data_dir):
    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                test_loss += F.cross_entropy(output, labels, reduction="sum").item()
                correct += (output.argmax(dim=1) == labels).sum().item()

        test_loss /= len(test_set)
        accuracy = 100.0 * correct / len(test_set)
        print(f"Epoch {epoch} - Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return test_set


def write_str(f, s):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<I", len(encoded)))
    f.write(encoded)


def write_tensor(f, name, tensor):
    write_str(f, name)
    data = tensor.detach().cpu().contiguous().numpy()
    f.write(struct.pack("<I", data.ndim))
    for dim in data.shape:
        f.write(struct.pack("<Q", dim))
    f.write(data.astype("<f4").tobytes())


def write_attr_int(name, value):
    return ("int", name, value)


def write_attr_float(name, value):
    return ("float", name, value)


def write_node(f, name, op_type, inputs, outputs, attrs):
    write_str(f, name)
    write_str(f, op_type)
    f.write(struct.pack("<I", len(inputs)))
    for i in inputs:
        write_str(f, i)
    f.write(struct.pack("<I", len(outputs)))
    for o in outputs:
        write_str(f, o)
    f.write(struct.pack("<I", len(attrs)))
    for kind, attr_name, value in attrs:
        write_str(f, attr_name)
        if kind == "int":
            f.write(struct.pack("<B", ATTR_INT))
            f.write(struct.pack("<q", value))
        else:
            f.write(struct.pack("<B", ATTR_FLOAT))
            f.write(struct.pack("<f", value))


def export_oien(model, path):
    state = model.state_dict()
    initializers = [
        ("fc1.weight", state["fc1.weight"]),
        ("fc1.bias", state["fc1.bias"]),
        ("fc2_left.weight", state["fc2_left.weight"]),
        ("fc2_left.bias", state["fc2_left.bias"]),
        ("fc2_left2.weight", state["fc2_left2.weight"]),
        ("fc2_left2.bias", state["fc2_left2.bias"]),
        ("fc2_right.weight", state["fc2_right.weight"]),
        ("fc2_right.bias", state["fc2_right.bias"]),
        ("fc3.weight", state["fc3.weight"]),
        ("fc3.bias", state["fc3.bias"]),
    ]

    gemm_attrs = [
        write_attr_int("transA", 0),
        write_attr_int("transB", 1),
        write_attr_float("alpha", 1.0),
        write_attr_float("beta", 1.0),
    ]

    nodes = [
        ("flatten", "Flatten", ["input"], ["flatten_out"], [write_attr_int("axis", 1)]),
        ("fc1", "Gemm", ["flatten_out", "fc1.weight", "fc1.bias"], ["fc1_out"], gemm_attrs),
        ("relu1", "Relu", ["fc1_out"], ["relu1_out"], []),
        ("fc2_left", "Gemm", ["relu1_out", "fc2_left.weight", "fc2_left.bias"], ["fc2_left_out"], gemm_attrs),
        ("relu2", "Relu", ["fc2_left_out"], ["relu2_out"], []),
        ("fc2_left2", "Gemm", ["relu2_out", "fc2_left2.weight", "fc2_left2.bias"], ["left_out"], gemm_attrs),
        ("fc2_right", "Gemm", ["relu1_out", "fc2_right.weight", "fc2_right.bias"], ["right_out"], gemm_attrs),
        ("add", "Add", ["left_out", "right_out"], ["add_out"], []),
        ("fc3", "Gemm", ["add_out", "fc3.weight", "fc3.bias"], ["output"], gemm_attrs),
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(OIEN_MAGIC)
        f.write(struct.pack("<I", OIEN_VERSION))

        f.write(struct.pack("<I", len(initializers)))
        for name, tensor in initializers:
            write_tensor(f, name, tensor)

        f.write(struct.pack("<I", len(nodes)))
        for name, op_type, inputs, outputs, attrs in nodes:
            write_node(f, name, op_type, inputs, outputs, attrs)

        f.write(struct.pack("<I", 1))
        write_str(f, "input")

        f.write(struct.pack("<I", 1))
        write_str(f, "output")


def export_sample_images(data_dir, inputs_dir):
    raw_test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
    os.makedirs(inputs_dir, exist_ok=True)

    seen_labels = set()
    label_lines = []
    for idx in range(len(raw_test_set)):
        image, label = raw_test_set[idx]
        if label in seen_labels:
            continue
        seen_labels.add(label)

        filename = f"image_{label}.ubyte"
        with open(os.path.join(inputs_dir, filename), "wb") as f:
            f.write(image.tobytes())
        label_lines.append(f"{filename} {label}")

        if len(seen_labels) == 10:
            break

    with open(os.path.join(inputs_dir, "labels.txt"), "w") as f:
        f.write("\n".join(label_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train the branching MNIST MLP and export it to .oien")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    print(model)

    data_dir = os.path.join(args.root, "data")
    train(model, device, args.epochs, args.batch_size, args.lr, data_dir)

    model_path = os.path.join(args.root, "models", "mnist_ffn.oien")
    export_oien(model, model_path)
    print(f"Model saved as {model_path}")

    inputs_dir = os.path.join(args.root, "inputs")
    export_sample_images(data_dir, inputs_dir)
    print(f"Sample images saved under {inputs_dir}")


if __name__ == "__main__":
    main()
