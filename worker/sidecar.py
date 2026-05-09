import argparse
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

X = None  # (n_samples, num_features)
y = None  # (n_samples,) int labels

@app.route("/health")
def health():
    shard_size = len(X) if X is not None else 0
    return jsonify({"status": "ok", "shard_size": shard_size})

@app.route("/load_shard", methods=["POST"])
def load_shard():
    global X, y
    rows = request.json["rows"]
    X = np.array([r["Features"] for r in rows], dtype=np.float64)
    y = np.array([r["Label"] for r in rows], dtype=np.int32)
    return jsonify({"ok": True})

@app.route("/gradient", methods=["POST"])
def gradient():
    if X is None:
        return jsonify({"error": "shard not loaded"}), 400

    data = request.json
    W = np.array(data["weights"], dtype=np.float64)  # (num_classes, num_features)
    b = np.array(data["bias"],    dtype=np.float64)  # (num_classes,)

    n = len(X)
    # softmax
    logits = X @ W.T + b                                        # (n, num_classes)
    logits -= logits.max(axis=1, keepdims=True)                 # numerical stability
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)            # (n, num_classes)

    # one-hot targets
    num_classes = W.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1.0

    # gradients
    diff = probs - one_hot                                       # (n, num_classes)
    dW = (diff.T @ X) / n                                       # (num_classes, num_features)
    db = diff.mean(axis=0)                                       # (num_classes,)

    # flatten: weights first, then bias — matches config.ParamSize convention
    flat = np.concatenate([dW.flatten(), db])
    return jsonify({"gradients": flat.tolist()})

@app.route("/evaluate", methods=["POST"])
def evaluate():
    if X is None:
        return jsonify({"error": "shard not loaded"}), 400
    data = request.json
    W = np.array(data["weights"], dtype=np.float64)
    b = np.array(data["bias"],    dtype=np.float64)
    logits = X @ W.T + b
    preds = logits.argmax(axis=1)
    correct = int((preds == y).sum())
    total = int(len(y))
    return jsonify({"correct": correct, "total": total})

@app.route("/reload_shard", methods=["POST"])
def reload_shard():
    global X, y
    if X is None:
        return load_shard()
    rows = request.json["rows"]
    X_new = np.array([r["Features"] for r in rows], dtype=np.float64)
    y_new = np.array([r["Label"] for r in rows], dtype=np.int32)
    X = np.vstack([X, X_new])
    y = np.concatenate([y, y_new])
    return jsonify({"ok": True, "shard_size": len(X)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
