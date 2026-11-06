"""Automated tests and latency measurement for the prototype (writes results/E20_prototype_test.json)."""
import copy
import json
import secrets
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
analyst, viewer = secrets.token_hex(8), secrets.token_hex(8)       # throw-away test keys, never stored
import os  # noqa: E402
os.environ["A2_KEYS"] = f"{analyst}:analyst,{viewer}:viewer"
import server  # noqa: E402

audit = Path(tempfile.mkdtemp()) / "audit.log"
srv = server.make_server(0, audit=audit)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{port}"
results, checks = {}, []


def get(path, key=None):
    req = urllib.request.Request(BASE + path, headers={"X-API-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def check(name, cond, detail=""):
    checks.append({"test": name, "passed": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name, detail)


s, b = get("/v1/health")
check("health endpoint open and reports versions", s == 200 and json.loads(b)["grid"] == server.GRID_VERSION)
check("missing API key rejected (401)", get("/v1/forecasts")[0] == 401)
check("wrong API key rejected (401)", get("/v1/forecasts", "nope")[0] == 401)
check("viewer cannot read cell history (403)", get("/v1/cells/5_8/history", viewer)[0] == 403)
s, b = get("/v1/cells/5_8/history", analyst)
check("analyst can read cell history", s == 200 and len(json.loads(b)["history"]) == 11)
check("unknown cell -> 404", get("/v1/cells/zzz/history", analyst)[0] == 404)
check("non-integer period -> 400", get("/v1/forecasts?period=abc", analyst)[0] == 400)
check("unknown period -> 400", get("/v1/forecasts?period=99", analyst)[0] == 400)
check("top_k=0 -> 400", get("/v1/forecasts?top_k=0", analyst)[0] == 400)
check("top_k=1000 -> 400", get("/v1/forecasts?top_k=1000", analyst)[0] == 400)
check("unknown grid version -> 400", get("/v1/forecasts?grid=other-v9", analyst)[0] == 400)
s, b = get("/v1/forecasts?period=13&top_k=32", analyst)
j = json.loads(b)
check("forecast for unclosed period flagged period_closed=false", s == 200 and j["period_closed"] is False and len(j["cells"]) == 32)
check("response carries disclaimer, model and schema version", "individual" in j["disclaimer"] and j["model"]["feature_schema"])
check("response carries the City of Chicago data-modification notice", "modified for use from its original source" in j["disclaimer"])
check("ranks are 1..32 with no gaps", [c["rank"] for c in j["cells"]] == list(range(1, 33)))

# parity: served scores for the two test periods must reproduce the offline LR test result exactly
y, sc, per = [], [], []
topk_tp = topk_fp = topk_fn = 0
for t in (11, 12):
    s, b = get(f"/v1/forecasts?period={t}&top_k=157", analyst)
    cells = json.loads(b)["cells"]
    for c in cells:
        y.append(c["actual_high"]); sc.append(c["score"]); per.append(t)
    flagged = {c["grid_id"] for c in cells if c["rank"] <= 32}
    for c in cells:
        a, f = c["actual_high"] == 1, c["grid_id"] in flagged
        topk_tp += a and f; topk_fp += (not a) and f; topk_fn += a and (not f)
ap_served = average_precision_score(y, sc)
offline = json.loads(json.dumps(__import__("pandas").read_csv(HERE.parent / "results" / "E2_p2_test_runs.csv").query("model == 'LR'").iloc[0][["avg_precision", "topk_precision", "topk_recall"]].to_dict()))
check("serving parity: AP on test periods equals offline experiment E2",
      abs(ap_served - offline["avg_precision"]) < 1e-3, f"served {ap_served:.4f} vs offline {offline['avg_precision']:.4f} (served scores rounded to 4 d.p.)")
prec, rec = topk_tp / (topk_tp + topk_fp), topk_tp / (topk_tp + topk_fn)
check("serving parity: top-20% precision/recall equal offline", abs(prec - offline["topk_precision"]) < 0.005 and abs(rec - offline["topk_recall"]) < 0.005,
      f"served {prec:.3f}/{rec:.3f} vs offline {offline['topk_precision']:.3f}/{offline['topk_recall']:.3f}")
s, b = get("/dashboard?period=13", viewer)
check("dashboard renders for viewer with disclaimer", s == 200 and b"individual" in b and b"<svg" in b)

# schema-hash guard: a store built with a different feature schema must refuse to load
bad = copy.deepcopy(server.Handler.store.d)
bad["feature_schema_hash"] = "deadbeefdeadbeef"
tmp = Path(tempfile.mkdtemp()) / "bad.json"
tmp.write_text(json.dumps(bad))
try:
    server.Store(tmp)
    refused = False
except SystemExit:
    refused = True
check("store with mismatched feature schema is refused at start-up", refused)

# interval coverage
all_cells = [c for t in (11, 12) for c in json.loads(get(f"/v1/forecasts?period={t}&top_k=157", analyst)[1])["cells"]]
cov = np.mean([c["rank_interval"][0] <= c["rank"] <= c["rank_interval"][1] for c in all_cells])
results["rank_interval_contains_point_rank_share"] = float(cov)
near = [c for c in all_cells if c["near_cutoff"]]
results["near_cutoff_cells_test"] = len(near)
results["near_cutoff_error_rate"] = float(np.mean([(c["rank"] <= 32) != (c["actual_high"] == 1) for c in near]))
results["non_near_cutoff_error_rate"] = float(np.mean([(c["rank"] <= 32) != (c["actual_high"] == 1) for c in all_cells if not c["near_cutoff"]]))

# latency: 600 requests mixed
paths = ["/v1/forecasts?period=13&top_k=32", "/v1/forecasts?period=12&top_k=157", "/v1/cells/12_7/history", "/dashboard?period=13"]
lat = []
for i in range(600):
    t0 = time.perf_counter()
    st, _ = get(paths[i % 4], analyst)
    lat.append((time.perf_counter() - t0) * 1000)
    assert st == 200
lat_by = {p: [] for p in paths}
for i, l in enumerate(lat):
    lat_by[paths[i % 4]].append(l)
results["latency_ms_all"] = {"p50": statistics.median(lat), "p95": float(np.percentile(lat, 95)), "max": max(lat), "n": len(lat)}
results["latency_ms_by_endpoint_p50"] = {p: statistics.median(v) for p, v in lat_by.items()}
n_audit = sum(1 for _ in open(audit))
check("every request written to the audit log", n_audit >= 600, f"{n_audit} lines")
results["store_bytes"] = (HERE / "artifacts" / "forecasts.json").stat().st_size
results["checks"] = checks
results["n_passed"] = sum(c["passed"] for c in checks)
results["n_checks"] = len(checks)
(HERE.parent / "results" / "E20_prototype_test.json").write_text(json.dumps(results, indent=2))
print(json.dumps({k: v for k, v in results.items() if k != "checks"}, indent=1))
sys.exit(0 if results["n_passed"] == results["n_checks"] else 1)
