"""Layer 0 — Observability monitor.

A passive observer that is not part of any quorum. Its only job is to receive
epoch checkpoint pushes from the coordinator and surface training progress.

It does not poll, vote, or call any other node — it only listens. The coordinator
posts to /checkpoint each time an epoch starts or finishes (see
coordinator.postCheckpoint). Statuses sent today: started, complete, converged,
max_epochs_reached.

Endpoints:
  POST /checkpoint   record an epoch update (in-memory + append to log file)
  GET  /             live HTML dashboard (auto-refreshing)
  GET  /checkpoints  full history as JSON
  GET  /health       liveness probe
"""

import argparse
import json
import threading
from datetime import datetime

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
_history = []  # list of {epoch, status, accuracy, received_at}


@app.route("/checkpoint", methods=["POST"])
def checkpoint():
    data = request.get_json(force=True)
    record = {
        "epoch": int(data.get("epoch", 0)),
        "status": str(data.get("status", "")),
        "accuracy": float(data.get("accuracy", 0.0)),
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


@app.route("/")
def dashboard():
    with _lock:
        history = list(_history)

    latest = history[-1] if history else None
    best_acc = max((r["accuracy"] for r in history), default=0.0)
    # training is "done" once a terminal status arrives
    done = next((r for r in reversed(history)
                 if r["status"] in ("converged", "max_epochs_reached")), None)

    # most recent first, capped so the page stays small
    rows = "".join(
        f"<tr><td>{r['epoch']}</td><td class='s-{r['status']}'>{r['status']}</td>"
        f"<td>{r['accuracy']:.4f}</td><td>{r['received_at']}</td></tr>"
        for r in reversed(history[-50:])
    ) or "<tr><td colspan='4' class='muted'>no checkpoints yet</td></tr>"

    if latest:
        summary = (
            f"<div class='big'>epoch {latest['epoch']} &middot; "
            f"<span class='s-{latest['status']}'>{latest['status']}</span></div>"
            f"<div>accuracy <b>{latest['accuracy']:.4f}</b> &middot; "
            f"best <b>{best_acc:.4f}</b> &middot; "
            f"{len(history)} checkpoints</div>"
        )
    else:
        summary = "<div class='big muted'>waiting for coordinator…</div>"

    banner = ""
    if done:
        banner = (f"<div class='banner'>training finished: "
                  f"<b>{done['status']}</b> at epoch {done['epoch']} "
                  f"(accuracy {done['accuracy']:.4f})</div>")

    html = f"""<!doctype html>
<html><head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="2">
  <title>picluster monitor</title>
  <style>
    body {{ font-family: ui-monospace, monospace; max-width: 760px; margin: 2rem auto;
            background: #0d1117; color: #c9d1d9; }}
    h1 {{ font-size: 1.1rem; color: #58a6ff; }}
    .big {{ font-size: 1.4rem; margin: .4rem 0; }}
    .muted {{ color: #8b949e; }}
    .banner {{ background: #1f6feb22; border: 1px solid #1f6feb; padding: .6rem .8rem;
               border-radius: 6px; margin: 1rem 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: .85rem; }}
    th, td {{ text-align: left; padding: .3rem .5rem; border-bottom: 1px solid #21262d; }}
    th {{ color: #8b949e; font-weight: normal; }}
    .s-started {{ color: #8b949e; }}
    .s-complete {{ color: #c9d1d9; }}
    .s-converged {{ color: #3fb950; }}
    .s-max_epochs_reached {{ color: #d29922; }}
  </style>
</head><body>
  <h1>distributed-ml-picluster &middot; observability</h1>
  {summary}
  {banner}
  <table>
    <tr><th>epoch</th><th>status</th><th>accuracy</th><th>received</th></tr>
    {rows}
  </table>
</body></html>"""
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
