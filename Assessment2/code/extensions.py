"""Extension experiments E22-E24 (added after the main report draft; all choices on validation data only).

E22  Narrower label: FBI index violent crime only (01A homicide, 02 criminal sexual assault, 03 robbery,
     04A aggravated assault, 04B aggravated battery) instead of the 11 person-directed categories.
E23  Error-disparity audit: Phase 2 / Phase 1 test errors by community-area hardship tercile
     (City of Chicago census indicators 2008-2012, data set kn9c-c2s2; boundaries igwz-8jzy).
E24  Additional algorithm family: histogram gradient boosting for both phases.

Usage: python3 extensions.py [E22 E23 E24]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

import models as M
import prep
import run_experiments as R

EXT = prep.ROOT / "machinelearningcode/Coding/Dataset/external"
VIOLENT_FBI = ["01A", "02", "03", "04A", "04B"]


# ================================================================== E22 violent-only label
def violent_loader():
    df, log = prep.load_clean()
    df["VICTIM"] = df.FBI_CD.astype(str).str.strip().isin(VIOLENT_FBI).astype(int)
    return df, log


def E22():
    saved = (R.SNAP, R.DF)
    DD = prep.build_all(loader=violent_loader)
    assert len(DD["cells"]) == R.N, "grid must be unchanged"
    R.SNAP, R.DF = DD["snap"].copy(), DD["df"]
    R.SNAP["HIGH"] = prep.label_high(R.SNAP, 80)
    n_viol = int(R.DF.VICTIM.sum())
    Xtr, ytr, _ = R.xy(R.TRAIN)
    Xva, yva, _ = R.xy(R.VAL)
    Xte, yte, rte = R.xy(R.TEST)
    per = rte.PERIOD.values
    out, best = [], {}
    # validation-only tuning with the same grids as E1
    for name in ["LR", "RF"]:
        best[name], _ = R.tune_tabular_p2(name)
    for kind in ["MLP", "GCN", "ResGCN"]:
        best[kind], _ = R.tune_nn(kind, "binary", ei=R.EI, ew=R.EW)
    kd = {bw: average_precision_score(yva, R.kde_scores(R.VAL, bw)) for bw in [300, 500, 800, 1200, 2000, 3000]}
    best["KDE"] = max(kd, key=kd.get)
    out.append(dict(model="Persistence", **R.p2_eval(yte, R.persistence_score(rte), per)))
    out.append(dict(model="KDE", **R.p2_eval(yte, R.kde_scores(R.TEST, best["KDE"]), per)))
    for name in ["LR", "RF"]:
        ss = []
        for sd in (R.SEEDS[:5] if name == "RF" else [0]):
            m, pr, _ = R.fit_tabular(name, best[name], Xtr, ytr, seed=sd)
            ss.append(R.p2_eval(yte, R.tab_score(m, pr, Xte), per))
        out.append(dict(model=name, **pd.DataFrame(ss).mean().to_dict()))
    for kind in ["MLP", "GCN", "ResGCN"]:
        ps, _, _, _ = R.run_nn_seeds(kind, best[kind], "binary", ei=R.EI, ew=R.EW, seeds=R.SEEDS[:5])
        d = pd.DataFrame([R.p2_eval(yte, p, per) for p in ps])
        out.append(dict(model=kind, ap_sd=d.avg_precision.std(), **d.mean().to_dict()))
    res = pd.DataFrame(out)
    res["n_violent_incidents"] = n_viol
    res["share_of_records"] = n_viol / len(R.DF)
    res["test_prevalence"] = yte.mean()
    # how much do the two label definitions agree on the test periods?
    main_high = saved[0][saved[0].PERIOD.isin(R.TEST)].sort_values(["PERIOD", "NODE"]).HIGH.values
    res["label_agreement_with_main"] = float((main_high == yte).mean())
    res["label_jaccard_with_main"] = float(((main_high == 1) & (yte == 1)).sum() / ((main_high == 1) | (yte == 1)).sum())
    R.save("E22_violent_only", res)
    R.save("E22_best_params", best)
    R.SNAP, R.DF = saved
    print(res.round(3).to_string())


# ================================================================== E23 error-disparity audit
def _cell_hardship():
    gj = json.loads((EXT / "community_areas.geojson").read_text())
    census = pd.read_csv(EXT / "census_socioeconomic_2008_2012.csv")
    census.columns = [c.strip() for c in census.columns]
    census = census.dropna(subset=["Community Area Number"])
    census["CA"] = census["Community Area Number"].astype(int)
    pts = R.DF[["LONGITUDE", "LATITUDE"]].values
    ca = np.zeros(len(pts), int)
    for f in gj["features"]:
        num = int(f["properties"]["area_numbe"])
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            inside = MPath(np.array(poly[0])[:, :2]).contains_points(pts)
            ca[inside & (ca == 0)] = num
    d = pd.DataFrame({"GRID_ID": R.DF.GRID_ID.values, "CA": ca})
    d = d[d.CA > 0].merge(census[["CA", "HARDSHIP INDEX", "PER CAPITA INCOME", "PERCENT HOUSEHOLDS BELOW POVERTY"]],
                          on="CA")
    cell = d.groupby("GRID_ID").agg(hardship=("HARDSHIP INDEX", "mean"),
                                    income=("PER CAPITA INCOME", "mean"),
                                    poverty=("PERCENT HOUSEHOLDS BELOW POVERTY", "mean"),
                                    main_ca=("CA", lambda s: s.mode().iat[0]),
                                    n=("CA", "size"))
    cell["matched_share"] = cell.n / R.DF.groupby("GRID_ID").size().reindex(cell.index)
    return cell.reset_index(), float((ca > 0).mean())


def _group_rates(y, pred, score):
    tp, fp = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum()
    fn, tn = ((pred == 0) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    return dict(n=len(y), prevalence=y.mean(), flag_rate=pred.mean(),
                recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                precision=tp / max(tp + fp, 1), error_rate=(fp + fn) / len(y),
                ap=average_precision_score(y, score) if 0 < y.sum() < len(y) else np.nan)


def E23():
    cell, matched = _cell_hardship()
    cell["tercile"] = pd.qcut(cell.hardship.rank(method="first"), 3, labels=["low hardship", "middle", "high hardship"])
    p2 = pd.read_csv(R.RES / "E2_p2_test_preds.csv").merge(cell[["GRID_ID", "tercile", "hardship"]], on="GRID_ID")
    p1 = pd.read_csv(R.RES / "E3_p1_test_preds.csv").merge(cell[["GRID_ID", "tercile"]], on="GRID_ID")
    rng = np.random.default_rng(0)
    rows = []
    for model in ["LR", "Persistence", "RF", "ResGCN", "GCN"]:
        pred = R.topk_pred(p2[model].values, p2.PERIOD.values)
        p2["_pred"] = pred
        for g, sub in p2.groupby("tercile", observed=True):
            r = _group_rates(sub.y.values, sub._pred.values, sub[model].values)
            # cell-cluster bootstrap CI for recall and FPR (resample cells, keep both periods together)
            ids = sub.GRID_ID.unique()
            rec, fpr = [], []
            for _ in range(2000):
                s = sub.set_index("GRID_ID").loc[rng.choice(ids, len(ids))]
                q = _group_rates(s.y.values, s._pred.values, s[model].values)
                rec.append(q["recall"]), fpr.append(q["fpr"])
            rows.append(dict(phase=2, model=model, group=g, n_cells=len(ids), **r,
                             recall_lo=np.percentile(rec, 2.5), recall_hi=np.percentile(rec, 97.5),
                             fpr_lo=np.percentile(fpr, 2.5), fpr_hi=np.percentile(fpr, 97.5)))
    for model in ["LR", "Persistence", "MLP", "GCN"]:
        for g, sub in p1.groupby("tercile", observed=True):
            rows.append(dict(phase=1, model=model, group=g, n_cells=sub.GRID_ID.nunique(), n=len(sub),
                             error_rate=(sub[model] != sub.y).mean(),
                             macro_f1=f1_score(sub.y, sub[model], average="macro")))
    res = pd.DataFrame(rows)
    # Spearman correlation between hardship and test victimisation counts (context for group base rates)
    rho = p2.groupby("GRID_ID").agg(h=("hardship", "first"), v=("Y_VICTIM", "mean")).corr("spearman").iloc[0, 1]
    meta = dict(incidents_matched_to_community_area=matched, cells=int(len(cell)),
                hardship_tercile_cutoffs=[float(cell.hardship.quantile(q)) for q in (1 / 3, 2 / 3)],
                spearman_hardship_vs_test_victimisation=float(rho),
                census_source="City of Chicago, Census Data - Selected socioeconomic indicators 2008-2012 (kn9c-c2s2)")
    R.save("E23_disparity_audit", res)
    R.save("E23_meta", meta)
    cell.to_csv(R.RES / "E23_cell_hardship.csv", index=False)
    print(json.dumps(meta, indent=1))
    print(res.round(3).to_string())


# ================================================================== E24 gradient boosting
HGB_GRID = [dict(learning_rate=lr, max_depth=d, max_iter=it)
            for lr in [0.03, 0.1] for d in [3, None] for it in [100, 300]]


def _hgb(p, seed, cw="balanced"):
    return HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                                          max_iter=p["max_iter"], l2_regularization=1.0,
                                          class_weight=cw, random_state=seed)


def E24():
    out, tune = [], []
    # Phase 2
    Xtr, ytr, _ = R.xy(R.TRAIN)
    Xva, yva, _ = R.xy(R.VAL)
    Xte, yte, rte = R.xy(R.TEST)
    pr = M.Prepper(R.FEATS, R.PREP_MODE).fit(Xtr)
    for p in HGB_GRID:
        m = _hgb(p, 0).fit(pr(Xtr), ytr)
        tune.append(dict(phase=2, params=json.dumps(p), val=average_precision_score(yva, m.predict_proba(pr(Xva))[:, 1])))
    t2 = pd.DataFrame(tune)
    b2 = json.loads(t2.loc[t2.val.idxmax(), "params"])
    for sd in R.SEEDS[:5]:
        t0 = time.perf_counter()
        m = _hgb(b2, sd).fit(pr(Xtr), ytr)
        fit_s = time.perf_counter() - t0
        out.append(dict(phase=2, model="HGB", seed=sd, fit_seconds=fit_s, val_best=t2.val.max(),
                        **R.p2_eval(yte, m.predict_proba(pr(Xte))[:, 1], rte.PERIOD.values)))
    # Phase 1
    tune = []
    Xtr, ytr, rtr = R.xy(R.TRAIN, target="Y_P1")
    Xva, yva, rva = R.xy(R.VAL, target="Y_P1")
    Xte, yte, rte = R.xy(R.TEST, target="Y_P1")
    mtr, mva, mte = (r.Y_TOTAL.values > 0 for r in (rtr, rva, rte))
    pr = M.Prepper(R.FEATS, R.PREP_MODE).fit(Xtr[mtr])
    for p in HGB_GRID:
        m = _hgb(p, 0).fit(pr(Xtr[mtr]), ytr[mtr])
        tune.append(dict(phase=1, params=json.dumps(p), val=f1_score(yva[mva], m.predict(pr(Xva[mva])), average="macro")))
    t1 = pd.DataFrame(tune)
    b1 = json.loads(t1.loc[t1.val.idxmax(), "params"])
    for sd in R.SEEDS[:5]:
        m = _hgb(b1, sd).fit(pr(Xtr[mtr]), ytr[mtr])
        out.append(dict(phase=1, model="HGB", seed=sd, val_best=t1.val.max(),
                        **R.p1_eval(yte[mte], m.predict(pr(Xte[mte])))))
    res = pd.DataFrame(out)
    R.save("E24_hgb", res)
    R.save("E24_hgb_tuning", pd.concat([t2, t1], ignore_index=True))
    R.save("E24_best_params", {"phase2": b2, "phase1": b1})
    print(res.groupby("phase").mean(numeric_only=True).round(4).T.to_string())


ALL = {"E22": E22, "E23": E23, "E24": E24}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(ALL)):
        t0 = time.time()
        print(f"\n===== {k}")
        ALL[k]()
        print(f"[{k} done in {time.time() - t0:.1f}s]")
