"""Read-only forecast API and analyst dashboard (prototype of Section 4.4).

Run:  A2_KEYS="<key>:analyst,<key2>:viewer" python3 server.py --port 8080
Endpoints (all need header X-API-Key, except /v1/health):
  GET /v1/health
  GET /v1/forecasts?period=<n>&top_k=<1..157>&grid=<grid version>
  GET /v1/cells/<grid_id>/history          (analyst role only)
  GET /dashboard?period=<n>                (HTML ranked list + map, analyst or viewer)
Safeguards: schema-hash check at start-up, input validation, role-based access, append-only audit log.
Only area-level aggregates are served; no individual-level data exist in the store.
"""
import argparse
import hashlib
import html
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import prep  # noqa: E402  (only for the feature list used in the schema hash)

DISCLAIMER = ("Area-level forecast of recorded incidents (not victims). Not a measure of any individual's risk. "
              "Decision support only; a human analyst must review every use. "
              "This site provides applications using data that has been modified for use from its original source, "
              "www.cityofchicago.org, the official website of the City of Chicago.")
GRID_VERSION = "utm16n-2200m-rook3000-v1"


def schema_hash(feats, grid_version):
    return hashlib.sha256(json.dumps([list(feats), grid_version]).encode()).hexdigest()[:16]


class Store:
    def __init__(self, path):
        self.d = json.loads(Path(path).read_text())
        expected = schema_hash(prep.BASE_FEATS, GRID_VERSION)
        if self.d["feature_schema_hash"] != expected or self.d["grid_version"] != GRID_VERSION:
            raise SystemExit(f"REFUSING TO START: store schema {self.d['feature_schema_hash']} != code schema {expected}")
        self.history = None

    def periods(self):
        return sorted(int(k) for k in self.d["periods"])


def parse_keys():
    out = {}
    for item in os.environ.get("A2_KEYS", "").split(","):
        if ":" in item:
            k, role = item.split(":", 1)
            if k and role in ("analyst", "viewer"):
                out[k] = role
    return out


class Handler(BaseHTTPRequestHandler):
    store = None
    keys = {}
    audit_path = HERE / "audit.log"

    def log_message(self, *a):  # silence default logging
        pass

    def _audit(self, status, role):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "path": self.path, "status": status}
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _send(self, status, body, ctype="application/json", role="-"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
        self._audit(status, role)

    def _err(self, status, msg, role="-"):
        self._send(status, json.dumps({"error": msg}), role=role)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/v1/health":
            return self._send(200, json.dumps({"status": "ok", "model": self.store.d["model_version"],
                                               "grid": self.store.d["grid_version"],
                                               "schema": self.store.d["feature_schema_hash"]}))
        role = self.keys.get(self.headers.get("X-API-Key", ""))
        if role is None:
            return self._err(401, "missing or invalid API key")
        if u.path == "/v1/forecasts":
            return self._forecasts(q, role)
        if u.path.startswith("/v1/cells/") and u.path.endswith("/history"):
            if role != "analyst":
                return self._err(403, "history requires the analyst role", role)
            return self._history(u.path.split("/")[3], role)
        if u.path == "/dashboard":
            return self._dashboard(q, role)
        return self._err(404, "not found", role)

    def _period(self, q, role):
        try:
            t = int(q.get("period", str(self.store.periods()[-1])))
        except ValueError:
            self._err(400, "period must be an integer", role)
            return None
        if t not in self.store.periods():
            self._err(400, f"unknown period {t}; available {self.store.periods()}", role)
            return None
        return t

    def _forecasts(self, q, role):
        if q.get("grid", GRID_VERSION) != GRID_VERSION:
            return self._err(400, f"unknown grid version; supported: {GRID_VERSION}", role)
        t = self._period(q, role)
        if t is None:
            return
        try:
            k = int(q.get("top_k", "32"))
        except ValueError:
            return self._err(400, "top_k must be an integer", role)
        if not 1 <= k <= len(self.store.d["cells"]):
            return self._err(400, "top_k out of range", role)
        P = self.store.d["periods"][str(t)]
        cells = sorted(P["cells"], key=lambda c: c["rank"])[:k]
        body = {"period": t, "period_first_day": P["first_day"], "period_days": 28, "period_closed": P["closed"],
                "model": {"name": "lr-victimisation", "version": self.store.d["model_version"],
                          "feature_schema": self.store.d["feature_schema_hash"], "grid": self.store.d["grid_version"],
                          "trained_on_target_periods": self.store.d["trained_on_target_periods"]},
                "cells": cells, "disclaimer": DISCLAIMER}
        self._send(200, json.dumps(body), role=role)

    def _history(self, cid, role):
        if cid not in self.store.d["cells"]:
            return self._err(404, "unknown grid_id", role)
        rows = []
        for t in self.store.periods():
            for c in self.store.d["periods"][str(t)]["cells"]:
                if c["grid_id"] == cid:
                    rows.append({"period": t, "rank": c["rank"], "score": c["score"], "actual_high": c["actual_high"]})
        self._send(200, json.dumps({"grid_id": cid, "location": self.store.d["cells"][cid], "history": rows,
                                    "disclaimer": DISCLAIMER}), role=role)

    def _dashboard(self, q, role):
        t = self._period(q, role)
        if t is None:
            return
        P = self.store.d["periods"][str(t)]
        cells = sorted(P["cells"], key=lambda c: c["rank"])
        geo = self.store.d["cells"]
        lons = [geo[c["grid_id"]]["lon"] for c in cells]
        lats = [geo[c["grid_id"]]["lat"] for c in cells]
        x0, x1, y0, y1 = min(lons), max(lons), min(lats), max(lats)
        dots = []
        for c in cells:
            g = geo[c["grid_id"]]
            x = 10 + 280 * (g["lon"] - x0) / (x1 - x0)
            y = 390 - 380 * (g["lat"] - y0) / (y1 - y0)
            col = {"top10": "#b2182b", "top20": "#ef8a62", "other": "#d9d9d9"}[c["band"]]
            dots.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{col}"><title>{html.escape(c["grid_id"])} rank {c["rank"]}</title></rect>')
        rows = "".join(
            f'<tr><td>{c["rank"]}</td><td>{html.escape(c["grid_id"])}</td><td>{c["score"]:.3f}</td>'
            f'<td>{c["rank_interval"][0]}–{c["rank_interval"][1]}</td><td>{"yes" if c["near_cutoff"] else ""}</td>'
            f'<td>{c["history_victim_counts"]}</td><td>{html.escape(", ".join(c["top_factors"]))}</td></tr>'
            for c in cells[:40])
        page = (f"<!doctype html><meta charset=utf-8><title>Forecast dashboard</title>"
                f"<h2>Recorded-victimisation forecast, period {t} (from {P['first_day']}, "
                f"{'closed' if P['closed'] else 'forecast – period not closed'})</h2>"
                f"<p><b>{html.escape(DISCLAIMER)}</b></p><svg width=300 height=400 style='float:left;margin-right:20px'>{''.join(dots)}</svg>"
                f"<table border=1 cellpadding=3><tr><th>Rank</th><th>Cell</th><th>Score</th><th>Rank 90% interval</th>"
                f"<th>Near cut-off</th><th>History (t-3..t-1)</th><th>Top factors</th></tr>{rows}</table>")
        self._send(200, page, ctype="text/html; charset=utf-8", role=role)


def make_server(port=0, store_path=HERE / "artifacts" / "forecasts.json", audit=None):
    Handler.store = Store(store_path)
    Handler.keys = parse_keys()
    if audit:
        Handler.audit_path = Path(audit)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    srv = make_server(args.port)
    print(f"serving on http://127.0.0.1:{srv.server_address[1]}  (keys loaded: {len(Handler.keys)})")
    srv.serve_forever()
