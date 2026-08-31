"""
verify.py - checks a job's output against independently computed counts

Reads every out/part-* file, unescapes the tab-separated records, and compares
the totals with the expected.json written by make_corpus.py. Also asserts the
partitioning invariant: each key must appear in exactly one output file.

    python python/verify.py --out work/<job-id>/out
"""
import argparse
import glob
import json
import os
import sys


def unescape(s):
    if "\\" not in s:
        return s
    out = []
    i = 0
    while i < len(s):
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        i += 1
        mapping = {"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}
        out.append(mapping[s[i]])
        i += 1
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="Verify MapReduce output")
    parser.add_argument("--out", required=True, help="job output directory")
    parser.add_argument("--expected", default="data/expected.json")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.out, "part-*")))
    if not files:
        sys.exit(f"no part files in {args.out}")

    actual = {}
    owner = {}
    duplicates = []

    for path in files:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                key, _, value = line.partition("\t")
                key = unescape(key)
                if key in owner and owner[key] != name:
                    duplicates.append((key, owner[key], name))
                owner[key] = name
                actual[key] = int(unescape(value))

    with open(args.expected, encoding="utf-8") as f:
        expected = json.load(f)

    print(f"output files: {len(files)}")
    print(f"distinct keys: {len(actual)} (expected {len(expected)})")

    problems = []
    if duplicates:
        problems.append(f"{len(duplicates)} key(s) span multiple partitions: {duplicates[:3]}")

    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    if missing:
        problems.append(f"missing keys: {sorted(missing)[:5]}")
    if extra:
        problems.append(f"unexpected keys: {sorted(extra)[:5]}")

    mismatched = [(k, expected[k], actual[k]) for k in sorted(set(expected) & set(actual)) if expected[k] != actual[k]]
    if mismatched:
        problems.append(f"{len(mismatched)} count mismatch(es): {mismatched[:5]}")

    if problems:
        for p in problems:
            print("FAIL:", p)
        sys.exit(1)

    total = sum(actual.values())
    print(f"total words: {total:,}")
    print("every key in exactly one partition: yes")
    print("all counts match expected: yes")


if __name__ == "__main__":
    main()
