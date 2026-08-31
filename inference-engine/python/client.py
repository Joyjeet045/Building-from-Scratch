"""
client.py

Exercises the running inference server: sends every sample image, then fires a
burst of concurrent requests to demonstrate that the server's dynamic batcher
coalesces them into far fewer forward passes than there were requests.

Usage:
    python python/client.py [--host 127.0.0.1] [--port 8080] [--burst 200]
"""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request


def infer(url, pixels):
    request = urllib.request.Request(
        url, data=pixels, headers={"Content-Type": "application/octet-stream"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description="Send inference requests to the engine server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--burst", type=int, default=200)
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    inputs_dir = os.path.join(args.root, "inputs")

    print("health:", get(f"{base}/health"))

    labels_path = os.path.join(inputs_dir, "labels.txt")
    samples = []
    with open(labels_path) as f:
        for line in f:
            filename, label = line.split()
            with open(os.path.join(inputs_dir, filename), "rb") as image:
                samples.append((image.read(), int(label)))

    print("\nSingle-request accuracy check:")
    correct = 0
    for pixels, label in samples:
        result = infer(f"{base}/infer", pixels)
        ok = result["prediction"] == label
        correct += ok
        print(f"  true={label} predicted={result['prediction']} {'OK' if ok else 'MISMATCH'}")
    print(f"  {correct}/{len(samples)} correct")

    print(f"\nFiring {args.burst} concurrent requests to trigger dynamic batching...")
    payload = samples[0][0]
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(infer, f"{base}/infer", payload) for _ in range(args.burst)]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start

    print(f"  {len(results)} responses in {elapsed:.3f}s ({len(results) / elapsed:.0f} req/s end-to-end)")

    metrics = get(f"{base}/metrics")
    print("\nmetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
