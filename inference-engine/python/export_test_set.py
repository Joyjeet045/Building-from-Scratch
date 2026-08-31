"""
export_test_set.py

Dumps the full MNIST test set into a single flat binary file consumable by
the C++ engine's `eval` subcommand, so end-to-end accuracy can be measured
with the real inference path rather than only spot-checking single images.

Layout: u32 count, then `count` records of [784 uint8 pixels][1 uint8 label].
"""
import argparse
import os
import struct

import torchvision


def main():
    parser = argparse.ArgumentParser(description="Export the MNIST test set to a flat binary file")
    parser.add_argument("--root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()

    data_dir = os.path.join(args.root, "data")
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=None)

    out_dir = os.path.join(args.root, "inputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mnist_test.bin")

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", len(test_set)))
        for image, label in test_set:
            f.write(image.tobytes())
            f.write(struct.pack("<B", label))

    print(f"Wrote {len(test_set)} samples to {out_path}")


if __name__ == "__main__":
    main()
