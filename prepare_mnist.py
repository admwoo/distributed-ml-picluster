"""Fetch MNIST and write pre-split per-node shard CSVs for the datastore.

Downloads MNIST via torchvision, takes a subset, normalizes pixels to [0, 1], and
writes one shard file per node: data/mnist_shard_<i>.csv. Each file has a header row
then rows of 784 pixel values followed by an integer label (0-9). Sharding is strided
(row i -> node i % N), matching the datastore's round-robin balance so every shard
sees all ten classes.

Pre-split couples the data to N: regenerate if you change the node count.

Usage:  python prepare_mnist.py [--nodes 3] [--samples 10000] [--out data]
"""

import argparse
import csv
import os

import numpy as np
from torchvision import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=3, help="number of shard files to write")
    ap.add_argument("--samples", type=int, default=10000, help="total training rows to use")
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--prefix", default="mnist", help="shard file prefix")
    args = ap.parse_args()

    print("loading MNIST via torchvision (downloads on first run)…")
    ds = datasets.MNIST(root="/tmp/mnist_data", train=True, download=True)
    # ds.data: (60000, 28, 28) uint8; flatten to 784 and normalize to [0, 1]
    X = ds.data.numpy().reshape(len(ds), -1).astype(np.float32) / 255.0
    y = ds.targets.numpy().astype(int)

    n = min(args.samples, len(X))
    X, y = X[:n], y[:n]
    print(f"using {n} rows, {X.shape[1]} features, {len(set(y.tolist()))} classes")

    os.makedirs(args.out, exist_ok=True)
    header = [f"px{i}" for i in range(X.shape[1])] + ["label"]

    files, writers = [], []
    for node in range(args.nodes):
        f = open(os.path.join(args.out, f"{args.prefix}_shard_{node}.csv"), "w", newline="")
        w = csv.writer(f)
        w.writerow(header)
        files.append(f)
        writers.append(w)

    counts = [0] * args.nodes
    for i in range(n):
        node = i % args.nodes  # strided: keeps class balance across shards
        writers[node].writerow([f"{v:.6f}" for v in X[i]] + [int(y[i])])
        counts[node] += 1

    for f in files:
        f.close()
    for node in range(args.nodes):
        print(f"  shard {node}: {counts[node]} rows -> {args.prefix}_shard_{node}.csv")


if __name__ == "__main__":
    main()
