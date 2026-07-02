"""Layer 0 — Observability monitor.

A passive observer that is not part of any quorum. Its only job is to receive
epoch checkpoint pushes from the coordinator and surface training progress.

It does not poll, vote, or call any other node — it only listens. The coordinator
posts to /checkpoint each time an epoch starts or finishes (see
coordinator.postCheckpoint). Statuses sent today: started, complete, converged,
max_epochs_reached.

Endpoints:
  POST /checkpoint   record an epoch update (in-memory + append to log file)
  GET  /             live HTML dashboard (auto-refreshing, with loss/accuracy charts)
  GET  /plot.png     loss & accuracy vs epoch, rendered server-side with matplotlib
  GET  /checkpoints  full history as JSON
  GET  /health       liveness probe
"""

import argparse
import io
import json
import threading
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless PNG rendering — no GUI backend, safe in a server thread
from matplotlib.figure import Figure

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8084)  # matches config.MonitorPort
parser.add_argument("--log", default="monitor/checkpoints.log",
                    help="append-only JSON-lines log of every checkpoint received")
args = parser.parse_args()

# in-memory history powers the dashboard; the log file is the durable record.
# guarded by a lock because Flask's dev server handles requests on multiple threads.
_lock = threading.Lock()
_history = []  # list of {epoch, status, accuracy, loss, received_at}


@app.route("/checkpoint", methods=["POST"])
def checkpoint():
    data = request.get_json(force=True)
    record = {
        "epoch": int(data.get("epoch", 0)),
        "status": str(data.get("status", "")),
        "accuracy": float(data.get("accuracy", 0.0)),
        "loss": float(data.get("loss", 0.0)),
        "received_at": datetime.now().isoformat(timespec="seconds"),
    }
    with _lock:
        _history.append(record)
        # durable append so checkpoints survive a monitor restart / allow later analysis
        with open(args.log, "a") as f:
            f.write(json.dumps(record) + "\n")
    return jsonify({"ok": True})


@app.route("/checkpoints")
def checkpoints():
    with _lock:
        return jsonify(list(_history))


@app.route("/health")
def health():
    with _lock:
        return jsonify({"status": "ok", "checkpoints": len(_history)})


@app.route("/plot.png")
def plot_png():
    # Only "started" carries no metrics (accuracy/loss 0); every other status is a
    # real datapoint. A fresh Figure per request keeps this thread-safe under Flask.
    with _lock:
        pts = [r for r in _history if r["status"] != "started"]

    fig = Figure(figsize=(7.2, 5.2), facecolor="#0d1117")
    ax_loss, ax_acc = fig.subplots(2, 1, sharex=True)
    for ax in (ax_loss, ax_acc):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.grid(True, color="#21262d", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("#21262d")

    if pts:
        epochs = [r["epoch"] for r in pts]
        ax_loss.plot(epochs, [r["loss"] for r in pts], color="#58a6ff", linewidth=1.5)
        ax_acc.plot(epochs, [r["accuracy"] for r in pts], color="#3fb950", linewidth=1.5)

    ax_loss.set_ylabel("loss", color="#58a6ff", fontsize=9)
    ax_acc.set_ylabel("accuracy", color="#3fb950", fontsize=9)
    ax_acc.set_xlabel("epoch", color="#8b949e", fontsize=9)
    ax_acc.set_ylim(0, 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=100)
    return Response(buf.getvalue(), mimetype="image/png")


# Static shell. All live content is filled in by polling JS below — no full-page
# reload, so the chart never blanks between refreshes. Not an f-string, so the JS
# braces/${} are literal.
_DASHBOARD = """<!doctype html>
<html><head>
  <meta charset="utf-8">
  <title>picluster monitor</title>
  <style>
    body { font-family: ui-monospace, monospace; max-width: 760px; margin: 2rem auto;
           background: #0d1117; color: #c9d1d9; }
    h1 { font-size: 1.1rem; color: #58a6ff; }
    .big { font-size: 1.4rem; margin: .4rem 0; }
    .muted { color: #8b949e; }
    .banner { background: #1f6feb22; border: 1px solid #1f6feb; padding: .6rem .8rem;
              border-radius: 6px; margin: 1rem 0; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: .85rem; }
    th, td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid #21262d; }
    th { color: #8b949e; font-weight: normal; }
    .chart { width: 100%; margin-top: 1rem; border: 1px solid #21262d; border-radius: 6px;
             display: block; min-height: 260px; background: #0d1117; }
    .s-started { color: #8b949e; }
    .s-complete { color: #c9d1d9; }
    .s-converged { color: #3fb950; }
    .s-max_epochs_reached { color: #d29922; }
  </style>
</head><body>
  <h1>distributed-ml-picluster &middot; observability</h1>
  <div id="summary"><div class="big muted">waiting for coordinator…</div></div>
  <div id="banner"></div>
  <img id="chart" class="chart" src="/plot.png?n=0" alt="loss and accuracy over epochs">
  <table>
    <tr><th>epoch</th><th>status</th><th>accuracy</th><th>loss</th><th>received</th></tr>
    <tbody id="tbody"></tbody>
  </table>
  <script>
    async function refresh() {
      let data;
      try {
        data = await (await fetch("/checkpoints")).json();
      } catch (e) {
        setTimeout(refresh, 2000);
        return;
      }

      const summary = document.getElementById("summary");
      if (data.length === 0) {
        summary.innerHTML = "<div class='big muted'>waiting for coordinator…</div>";
      } else {
        const latest = data[data.length - 1];
        const bestAcc = Math.max.apply(null, data.map(r => r.accuracy));
        summary.innerHTML =
          `<div class='big'>epoch ${latest.epoch} &middot; <span class='s-${latest.status}'>${latest.status}</span></div>` +
          `<div>accuracy <b>${latest.accuracy.toFixed(4)}</b> &middot; loss <b>${latest.loss.toFixed(4)}</b> ` +
          `&middot; best acc <b>${bestAcc.toFixed(4)}</b> &middot; ${data.length} checkpoints</div>`;
      }

      const done = data.slice().reverse().find(
        r => r.status === "converged" || r.status === "max_epochs_reached");
      document.getElementById("banner").innerHTML = done
        ? `<div class='banner'>training finished: <b>${done.status}</b> at epoch ${done.epoch} (accuracy ${done.accuracy.toFixed(4)})</div>`
        : "";

      const recent = data.slice(-50).reverse();
      document.getElementById("tbody").innerHTML = recent.length
        ? recent.map(r => `<tr><td>${r.epoch}</td><td class='s-${r.status}'>${r.status}</td>` +
            `<td>${r.accuracy.toFixed(4)}</td><td>${r.loss.toFixed(4)}</td><td>${r.received_at}</td></tr>`).join("")
        : "<tr><td colspan='5' class='muted'>no checkpoints yet</td></tr>";

      // Double-buffer the chart: load the new PNG off-screen and swap it in only once
      // it has fully decoded, so the visible chart never goes blank.
      const buf = new Image();
      buf.onload = () => { document.getElementById("chart").src = buf.src; };
      buf.src = "/plot.png?n=" + data.length;

      setTimeout(refresh, 500);
    }
    refresh();
  </script>
</body></html>"""


@app.route("/")
def dashboard():
    return Response(_DASHBOARD, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
