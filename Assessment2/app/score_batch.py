"""Batch scoring job for the prototype (design Section 4.4 'batch inference').

Fits the champion model exactly as evaluated in E2 (logistic regression, training target periods 3-8, C = 10,
balanced weights), scores every cell for every target period 3-13 using ONLY earlier periods, and writes a
versioned prediction store. Period 13 is the forecast for the period that has not closed yet.
"""
import hashlib
import warnings
import json
import sys
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import models as M  # noqa: E402
import prep  # noqa: E402

ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)
GRID_VERSION = "utm16n-2200m-rook3000-v1"
MODEL_VERSION = "lr-victimisation-1.0.0"
TRAIN = list(range(3, 9))
TOPK = 32
B_BOOT = 200


def schema_hash(feats, grid_version):
    return hashlib.sha256(json.dumps([feats, grid_version]).encode()).hexdigest()[:16]


def main():
    prep.N_PERIODS = 14
    D = prep.build_all()
    snap, cells, df = D["snap"].copy(), D["cells"], D["df"]
    snap["HIGH"] = prep.label_high(snap, 80)
    feats = prep.BASE_FEATS
    best_c = json.loads((HERE.parent / "results" / "E1_p2_best_params.json").read_text())["LR"]["C"]
    tr = snap[snap.PERIOD.isin(TRAIN)].sort_values(["PERIOD", "NODE"])
    pr = M.Prepper(feats).fit(tr[feats].values)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=best_c, max_iter=5000, class_weight="balanced").fit(pr(tr[feats].values), tr.HIGH.values)
    # bootstrap refits (resampling training rows) give a rank interval per cell
    rng = np.random.default_rng(0)
    boots = []
    Xtr, ytr = pr(tr[feats].values), tr.HIGH.values
    for _ in range(B_BOOT):
        idx = rng.integers(0, len(ytr), len(ytr))
        boots.append(LogisticRegression(C=best_c, max_iter=5000, class_weight="balanced").fit(Xtr[idx], ytr[idx]))
    coef = lr.coef_[0]
    names = list(feats)
    store = {"model_version": MODEL_VERSION, "grid_version": GRID_VERSION, "feature_schema_hash": schema_hash(names, GRID_VERSION),
             "trained_on_target_periods": TRAIN, "c": best_c, "top_k": TOPK, "periods": {}, "cells": {}}
    p0 = df.groupby("PERIOD").DATE.min()
    for t in range(prep.N_LAGS, 14):
        s = snap[snap.PERIOD == t].sort_values("NODE")
        Z = pr(s[feats].values)
        score = lr.predict_proba(Z)[:, 1]
        ranks_b = np.array([(-b.predict_proba(Z)[:, 1]).argsort().argsort() + 1 for b in boots])
        order = (-score).argsort().argsort() + 1
        contrib = Z * coef
        lo, hi = np.percentile(ranks_b, 5, axis=0), np.percentile(ranks_b, 95, axis=0)
        entries = []
        for i, cid in enumerate(s.GRID_ID.values):
            top = np.argsort(-np.abs(contrib[i]))[:3]
            entries.append({
                "grid_id": cid, "rank": int(order[i]), "score": round(float(score[i]), 4),
                "band": "top10" if order[i] <= 16 else ("top20" if order[i] <= TOPK else "other"),
                "rank_interval": [int(lo[i]), int(hi[i])],
                "near_cutoff": bool(abs(order[i] - TOPK) <= 2),
                "history_victim_counts": [int(s.VICTIM_L3.values[i]), int(s.VICTIM_L2.values[i]), int(s.VICTIM_L1.values[i])],
                "top_factors": [names[j] for j in top],
                "actual_high": (int(s.HIGH.values[i]) if t < 13 else None),
            })
        store["periods"][str(t)] = {"first_day": str(p0[t].date()), "closed": t < 13, "cells": entries}
    for _, r in cells.iterrows():
        store["cells"][r.GRID_ID] = {"lat": round(float(r.LAT), 5), "lon": round(float(r.LON), 5), "incidents_12m": int(r.N)}
    (ART / "forecasts.json").write_text(json.dumps(store))
    joblib.dump({"model": lr, "prepper": pr, "features": names, "version": MODEL_VERSION,
                 "schema_hash": store["feature_schema_hash"]}, ART / "model.joblib")
    print("wrote", ART / "forecasts.json", "periods", list(store["periods"]), "bytes", (ART / "forecasts.json").stat().st_size)


if __name__ == "__main__":
    main()
