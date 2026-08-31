"""
make_corpus.py - generates a synthetic text corpus for the word-count example

Writes N files of random sentences drawn from a fixed vocabulary, so the
expected word counts can be computed independently and compared against what
the MapReduce job produces.

    python python/make_corpus.py --files 40 --lines 5000 --out data/input
"""
import argparse
import collections
import json
import os
import random

VOCAB = [
    "map", "reduce", "shuffle", "partition", "worker", "coordinator", "task",
    "cluster", "node", "record", "key", "value", "merge", "sort", "spill",
    "combiner", "counter", "heartbeat", "straggler", "backup", "fault",
    "tolerance", "distributed", "throughput", "latency", "storage", "stream",
    "the", "a", "of", "and", "to", "in", "is", "it", "that", "with",
]


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic corpus")
    parser.add_argument("--files", type=int, default=40)
    parser.add_argument("--lines", type=int, default=5000)
    parser.add_argument("--words-per-line", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/input")
    parser.add_argument("--expected", default="data/expected.json")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    counts = collections.Counter()
    for i in range(args.files):
        path = os.path.join(args.out, f"doc-{i:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(args.lines):
                words = [random.choice(VOCAB) for _ in range(args.words_per_line)]
                counts.update(words)
                f.write(" ".join(words) + "\n")

    os.makedirs(os.path.dirname(args.expected) or ".", exist_ok=True)
    with open(args.expected, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(counts.items())), f, indent=2)

    total = sum(counts.values())
    print(f"wrote {args.files} files to {args.out}")
    print(f"{total:,} words, {len(counts)} distinct")
    print(f"expected counts: {args.expected}")


if __name__ == "__main__":
    main()
