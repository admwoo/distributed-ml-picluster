"""ML sidecar (PyTorch MLP) behind the model-agnostic flat-param protocol.

This is the neural-net counterpart to sidecar.py (numpy softmax logistic regression).
Both speak the same HTTP contract to the Go layer, but this one adds an /init_params
endpoint and swaps the wire format from {weights, bias} to a single flat {params}
vector, because a multi-layer net has no meaningful weights/bias split.

The Go layer knows nothing about the model's shape. Params cross the wire as one flat
float vector; this sidecar owns all knowledge of how to unflatten it into the network's
tensors and how to flatten gradients back. Swapping the model (layers, sizes, even the
task) is a change to THIS FILE ONLY — as long as the flat vector round-trips, the
coordinator's generic SGD keeps working.

Endpoints:
  GET  /init_params  -> {"params": [...]}   initial (randomly initialized) flat params
  POST /gradient     <- {"params": [...]}   -> {"gradients": [...], "row_count", "loss"}
  POST /evaluate     <- {"params": [...]}   -> {"correct", "total"}
  POST /load_shard   <- {"rows": [{Features, Label}]}
  GET  /health
"""

import argparse
from flask import Flask, request, jsonify
import torch
import torch.nn as nn

# --- Architecture. Keep INPUT_DIM/NUM_CLASSES in sync with the dataset being served
# (iris: 4/3, MNIST: 784/10). The coordinator stays agnostic to all of this. ---
INPUT_DIM = 4
HIDDEN1 = 16
HIDDEN2 = 8
NUM_CLASSES = 3
L2_LAMBDA = 0.01  # weight decay; applied to weight matrices only (not biases)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

torch.manual_seed(0)  # reproducible init for the one sidecar that seeds /init_params
model = MLP()
criterion = nn.CrossEntropyLoss()

X = None  # (n_samples, INPUT_DIM) float32
y = None  # (n_samples,) int64 labels


def flatten_params():
    """Model params as one flat list, in model.parameters() order."""
    return torch.cat([p.detach().flatten() for p in model.parameters()]).tolist()


def load_flat_params(flat):
    """Copy a flat vector back into the model tensors, in the same order."""
    vec = torch.tensor(flat, dtype=torch.float32)
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n


@app.route("/health")
def health():
    shard_size = 0 if X is None else X.shape[0]
    return jsonify({"status": "ok", "shard_size": shard_size})


@app.route("/init_params")
def init_params():
    # The coordinator calls this once to seed the cluster with a shared starting point.
    return jsonify({"params": flatten_params()})


@app.route("/load_shard", methods=["POST"])
def load_shard():
    global X, y
    rows = request.json["rows"]
    X = torch.tensor([r["Features"] for r in rows], dtype=torch.float32)
    y = torch.tensor([r["Label"] for r in rows], dtype=torch.int64)
    return jsonify({"ok": True})


@app.route("/gradient", methods=["POST"])
def gradient():
    if X is None:
        return jsonify({"error": "shard not loaded"}), 400

    load_flat_params(request.json["params"])
    model.zero_grad()

    logits = model(X)
    loss = criterion(logits, y)
    # L2 penalty on weight matrices only (2-D params); biases (1-D) are left alone.
    l2 = sum(p.pow(2).sum() for p in model.parameters() if p.dim() > 1)
    loss = loss + (L2_LAMBDA / 2) * l2
    loss.backward()

    grads = torch.cat([p.grad.flatten() for p in model.parameters()]).tolist()
    return jsonify({"gradients": grads, "row_count": int(X.shape[0]), "loss": float(loss)})


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if X is None:
        return jsonify({"error": "shard not loaded"}), 400
    load_flat_params(request.json["params"])
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        correct = int((preds == y).sum())
    return jsonify({"correct": correct, "total": int(X.shape[0])})


@app.route("/reload_shard", methods=["POST"])
def reload_shard():
    global X, y
    if X is None:
        return load_shard()
    rows = request.json["rows"]
    X_new = torch.tensor([r["Features"] for r in rows], dtype=torch.float32)
    y_new = torch.tensor([r["Label"] for r in rows], dtype=torch.int64)
    X = torch.cat([X, X_new])
    y = torch.cat([y, y_new])
    return jsonify({"ok": True, "shard_size": int(X.shape[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
