"""All Assessment 2 experiments (E0-E13). Usage: python3 run_experiments.py [E0 E1 ...]

Chronological protocol (28-day periods, 0-12):
  features for period t use periods < t only; targets are counts in period t.
  train = target periods 3-8, validation = 9-10, test = 11-12 (never used for any choice).
"""
import itertools
import json
import os
import platform
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import models as M
import prep

warnings.filterwarnings("ignore")
MODE = os.environ.get("A2_DATA", "one")      # "one" = one-year extract (main study); "multi" = three-year data
if MODE == "multi":
    import multiyear as MY
    MY.configure()
    RES = prep.OUT / "results_multi"
    TRAIN, VAL, TEST, ROLL, LOADER = MY.TRAIN, MY.VAL, MY.TEST, MY.ROLL, MY.loader
else:
    RES = prep.OUT / "results"
    TRAIN, VAL, TEST, ROLL, LOADER = list(range(3, 9)), [9, 10], [11, 12], [8, 9, 10, 11, 12], None
if os.environ.get("A2_ROLL"):                  # e.g. A2_ROLL=15:39:2 -> rolling-origin windows 15,17,...,37
    _a, _b, _c = (int(v) for v in os.environ["A2_ROLL"].split(":"))
    ROLL = list(range(_a, _b, _c))
RES.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(10))
TUNE_SEEDS = [0, 1, 2]
TOPK_SHARE = 0.20
PREP_MODE = "main"

D = prep.build_all(loader=LOADER)
if MODE == "multi":
    # Phase 1 class set re-derived from the multi-year TRAINING periods only (same 5% rule), then rebuild labels
    _share, _keep = prep.p1_class_rule(D["panel"], TRAIN, 0.05)
    prep.P1_KEEP = sorted(_keep)
    prep.P1_CLASSES = prep.P1_KEEP + ["OTHER_TYPE"]
    D = prep.build_all(loader=LOADER)
CELLS, EI, EW, SNAP, DF = D["cells"], D["ei"], D["ew"], D["snap"].copy(), D["df"]
N = len(CELLS)
SNAP["HIGH"] = prep.label_high(SNAP, 80)
FEATS = prep.BASE_FEATS
NCLS = len(prep.P1_CLASSES)


def save(name, obj):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(RES / f"{name}.csv", index=False)
    else:
        (RES / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))


def rows(periods, snap=None):
    s = SNAP if snap is None else snap
    return s[s.PERIOD.isin(periods)].sort_values(["PERIOD", "NODE"])


def xy(periods, feats=FEATS, target="HIGH", snap=None):
    r = rows(periods, snap)
    return r[feats].values, r[target].values, r


def topk_pred(score, periods):
    """Flag the top 20% of cells in each period (operational rule; no held-out tuning)."""
    pred = np.zeros(len(score), int)
    k = int(np.ceil(TOPK_SHARE * N))
    for i, _ in enumerate(np.unique(periods)):
        idx = np.where(periods == np.unique(periods)[i])[0]
        pred[idx[np.argsort(-score[idx])[:k]]] = 1
    return pred


def p2_eval(y, score, periods, thr=None):
    pred = topk_pred(score, periods)
    m = M.binary_metrics(y, pred.astype(float), 0.5)
    rank = M.binary_metrics(y, score, np.inf if thr is None else thr)
    out = {f"topk_{k}": v for k, v in m.items() if k in ("pos_f1", "macro_f1", "precision", "recall", "accuracy")}
    out.update(roc_auc=rank["roc_auc"], avg_precision=rank["avg_precision"])
    if thr is not None:
        out.update({f"thr_{k}": v for k, v in rank.items() if k in ("pos_f1", "macro_f1", "precision", "recall")})
        if "brier" in rank:
            out["brier"] = rank["brier"]
    return out


# ------------------------------------------------------------------ baselines for Phase 2
def persistence_score(r):
    return r.VICTIM_L1.values + 0.5 * r.VICTIM_L2.values + 0.25 * r.VICTIM_L3.values


def kde_scores(periods, bandwidth, victim_only=True):
    out = []
    cxy = CELLS[["CX", "CY"]].values
    for t in periods:
        pts = DF[(DF.PERIOD >= t - 3) & (DF.PERIOD < t)]
        if victim_only:
            pts = pts[pts.VICTIM == 1]
        out.append(M.kde_cell_scores(pts[["X_M", "Y_M"]].values, cxy, bandwidth))
    return np.concatenate(out)


# ------------------------------------------------------------------ generic fitting helpers
def fit_tabular(name, params, Xtr, ytr, seed=0, cw="balanced", smote=False):
    prepper = M.Prepper(FEATS_CUR, PREP_MODE).fit(Xtr)
    Xt = prepper(Xtr)
    if smote:
        Xt, ytr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(Xt, ytr)
    m = M.make_tabular(name, params, "binary", seed, cw)
    t0 = time.perf_counter()
    m.fit(Xt, ytr)
    return m, prepper, time.perf_counter() - t0


def tab_score(m, prepper, X, multi=False):
    s = M.tabular_scores(m, prepper(X))
    if multi:
        return s
    return s[:, 1] if s.ndim == 2 else s


FEATS_CUR = FEATS


def grid_nn():
    return [dict(hidden=h, dropout=d, lr=lr, wd=wd, layers=L)
            for h, d, lr, wd, L in itertools.product([32, 64, 128], [0.3, 0.5], [0.01, 0.003],
                                                     [5e-4, 5e-3], [2])]


TAB_GRIDS = {
    "LR": [dict(C=c) for c in [0.01, 0.1, 1, 10, 100]],
    "SVM": [dict(C=c, gamma=g) for c in [0.1, 1, 10, 100] for g in ["scale", 0.01, 0.1]],
    "RF": [dict(max_depth=d, min_samples_leaf=l) for d in [3, 6, None] for l in [1, 5]],
}


# ================================================================== E0 pilot audit
def E0():
    out = {"cleaning_log": D["log"]}
    pilot = prep.ROOT / "machinelearningcode/Coding/Notebooks/data"
    nf = pd.read_csv(pilot / "node_features.csv")
    type_cols = ["ASSAULT", "BATTERY", "BURGLARY", "CRIMINAL DAMAGE", "CRIMINAL SEXUAL ASSAULT",
                 "CRIMINAL TRESPASS", "DECEPTIVE PRACTICE", "MOTOR VEHICLE THEFT", "NARCOTICS",
                 "OFFENSE INVOLVING CHILDREN", "OTHER", "OTHER OFFENSE", "PUBLIC PEACE VIOLATION",
                 "ROBBERY", "SEX OFFENSE", "THEFT", "WEAPONS VIOLATION"]
    present = [c for c in type_cols if c in nf.columns]
    if "TARGET_CRIME_TYPE" in nf.columns:
        target = nf.TARGET_CRIME_TYPE
    else:
        target = nf[present].idxmax(axis=1)
    rule = nf[present].idxmax(axis=1)
    out["p1_pilot_argmax_rule_accuracy_all_cells"] = float((rule == target).mean())
    out["p1_pilot_label_counts"] = target.value_counts().to_dict()
    out["p1_pilot_has_target_col"] = "TARGET_CRIME_TYPE" in nf.columns
    out["graph"] = dict(nodes=N, directed_edges=int(EI.shape[1]), undirected=int(EI.shape[1] // 2),
                        unique_edge_lengths_km=sorted(set(np.round(D["dist"], 3).tolist())),
                        unique_weights=sorted(set(np.round(EW, 4).tolist())),
                        isolated=int((np.bincount(EI[0], minlength=N) == 0).sum()),
                        mean_degree=float(np.bincount(EI[0], minlength=N).mean()),
                        diagonal_neighbour_distance_km=float(2.2 * np.sqrt(2)))
    # Phase 2 pilot design reproduced: same file, same stratified random split (seed 42)
    from sklearn.model_selection import train_test_split
    p2 = pd.read_csv(pilot / "phase2_victimisation_dataset.csv")
    fc = ["GRID_CENTER_LAT", "GRID_CENTER_LON", "HIST_TOTAL_CRIME", "HIST_VICTIMISATION", "HIST_DOMESTIC",
          "HIST_ARREST", "HIST_UNIQUE_CRIMES", "HIST_UNIQUE_LOCATIONS", "HIST_VICTIMISATION_RATE",
          "HIST_VICTIMISATION_PER_DAY"]
    # the saved pilot file stores HIST_DOMESTIC / HIST_ARREST as concatenated "Y"/"N" strings
    # (a pilot bug: string columns were summed). Recompute them from the cleaned data, same window.
    out["p2_pilot_file_string_bug"] = bool(p2.HIST_DOMESTIC.dtype == object)
    h = DF[DF.DATE < "2026-06-12"].groupby("GRID_ID")[["DOMESTIC_F", "ARREST_F"]].sum()
    p2["HIST_DOMESTIC"] = p2.GRID_ID.map(h.DOMESTIC_F).fillna(0)
    p2["HIST_ARREST"] = p2.GRID_ID.map(h.ARREST_F).fillna(0)
    X, y = p2[fc], p2.HIGH_VICTIMISATION_RISK.astype(int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=42)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    s_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
    s_pers = Xte.HIST_VICTIMISATION_PER_DAY.values
    thr_pers = np.percentile(Xtr.HIST_VICTIMISATION_PER_DAY, 80)
    out["p2_pilot_split"] = dict(train=len(ytr), val=len(yva), test=len(yte), test_pos=int(yte.sum()),
                                 total_pos=int(y.sum()))
    out["p2_pilot_LR"] = M.binary_metrics(yte.values, s_lr, 0.5)
    out["p2_pilot_persistence_trainthr"] = M.binary_metrics(yte.values, s_pers, thr_pers)
    out["p2_pilot_GCN_reported"] = json.loads(
        (prep.ROOT / "machinelearningcode/Coding/Notebooks/results/phase2_victimisation_metrics.json").read_text())
    # correlation between historical and future rate in pilot data: how predictable by persistence
    out["p2_pilot_spearman_hist_future"] = float(
        p2.HIST_VICTIMISATION_PER_DAY.rank().corr(p2.FUTURE_VICTIMISATION_PER_DAY.rank()))
    save("E0_pilot_audit", out)
    print(json.dumps({k: v for k, v in out.items() if k != "cleaning_log"}, indent=1, default=float)[:3000])


# ================================================================== E1/E2 Phase 2 tuning + test
def tune_tabular_p2(name, feats=None):
    global FEATS_CUR
    FEATS_CUR = feats or FEATS
    Xtr, ytr, _ = xy(TRAIN, FEATS_CUR)
    Xva, yva, rva = xy(VAL, FEATS_CUR)
    res = []
    for p in TAB_GRIDS[name]:
        m, pr, _ = fit_tabular(name, p, Xtr, ytr)
        s = tab_score(m, pr, Xva)
        res.append(dict(model=name, params=json.dumps(p), val_ap=M.average_precision_score(yva, s)))
    df = pd.DataFrame(res)
    best = json.loads(df.loc[df.val_ap.idxmax(), "params"])
    FEATS_CUR = FEATS
    return best, df


def tune_nn(kind, task, feats=FEATS, target="HIGH", ei=EI, ew=EW, grid=None, snap=None, cw=True):
    global FEATS_CUR
    FEATS_CUR = feats
    Xtr, ytr, rtr = xy(TRAIN, feats, target, snap)
    Xva, yva, rva = xy(VAL, feats, target, snap)
    pr = M.Prepper(feats, PREP_MODE).fit(Xtr[mask_rows(rtr, task)])
    res = []
    for p in grid or grid_nn():
        scores = []
        for sd in TUNE_SEEDS:
            _, info = M.train_graph_model(kind, pr(Xtr), ytr, pr(Xva), yva, ei, ew, N, task,
                                          n_classes=NCLS if task == "multi" else 2, seed=sd,
                                          train_mask=mask_rows(rtr, task), val_mask=mask_rows(rva, task),
                                          class_weight=cw, **p)
            scores.append(info["val_score"])
        res.append(dict(model=kind, params=json.dumps(p), val=np.mean(scores), val_sd=np.std(scores)))
    df = pd.DataFrame(res)
    FEATS_CUR = FEATS
    return json.loads(df.loc[df.val.idxmax(), "params"]), df


def mask_rows(r, task):
    if task == "multi":
        return (r.Y_TOTAL.values > 0)
    return np.ones(len(r), bool)


def run_nn_seeds(kind, params, task, feats=FEATS, target="HIGH", ei=EI, ew=EW, train=TRAIN, val=VAL,
                 test=TEST, seeds=SEEDS, snap=None, cw=True, train_mask_fn=None, curve=False):
    Xtr, ytr, rtr = xy(train, feats, target, snap)
    Xva, yva, rva = xy(val, feats, target, snap)
    Xte, yte, rte = xy(test, feats, target, snap)
    mtr, mva = mask_rows(rtr, task), mask_rows(rva, task)
    if train_mask_fn is not None:
        mtr = mtr & train_mask_fn(rtr)
        mva = mva & train_mask_fn(rva)
    pr = M.Prepper(feats, PREP_MODE).fit(Xtr[mtr])
    preds, infos = [], []
    for sd in seeds:
        model, info = M.train_graph_model(kind, pr(Xtr), ytr, pr(Xva), yva, ei, ew, N, task,
                                          n_classes=NCLS if task == "multi" else 2, seed=sd,
                                          train_mask=mtr, val_mask=mva, class_weight=cw,
                                          return_curve=curve, **params)
        t0 = time.perf_counter()
        preds.append(M.predict_graph(model, pr(Xte), ei, ew, N, task))
        info["infer_seconds"] = time.perf_counter() - t0
        if task == "binary":
            info["val_thr"] = M.best_threshold(yva[mva], M.predict_graph(model, pr(Xva), ei, ew, N, task)[mva])
        infos.append(info)
    return preds, infos, yte, rte


def E1_E2():
    tuning, best = [], {}
    for name in ["LR", "SVM", "RF"]:
        b, df = tune_tabular_p2(name)
        best[name] = b
        tuning.append(df.rename(columns={"val_ap": "val"}))
    for kind in ["MLP", "GCN", "ResGCN", "GAT"]:
        b, df = tune_nn(kind, "binary")
        best[kind] = b
        tuning.append(df)
    # KDE bandwidth on validation AP
    _, yva, rva = xy(VAL)
    kd = []
    for bw in [300, 500, 800, 1200, 2000, 3000]:
        kd.append(dict(model="KDE", params=json.dumps({"bandwidth_m": bw}),
                       val=M.average_precision_score(yva, kde_scores(VAL, bw))))
    kd = pd.DataFrame(kd)
    best["KDE"] = json.loads(kd.loc[kd.val.idxmax(), "params"])
    tuning.append(kd)
    save("E1_p2_tuning", pd.concat(tuning, ignore_index=True))
    save("E1_p2_best_params", best)
    print("best", best)

    # ---- test
    Xtr, ytr, rtr = xy(TRAIN)
    Xva, yva, rva = xy(VAL)
    Xte, yte, rte = xy(TEST)
    per = rte.PERIOD.values
    results, preds = [], {"GRID_ID": rte.GRID_ID.values, "PERIOD": per, "y": yte, "Y_VICTIM": rte.Y_VICTIM.values}
    # majority / prior
    results.append(dict(model="Majority (all low)", seed=0, **{"topk_pos_f1": 0.0}, roc_auc=0.5,
                        avg_precision=yte.mean(), thr_pos_f1=0.0,
                        thr_macro_f1=M.f1_score(yte, np.zeros_like(yte), average="macro")))
    s = persistence_score(rte)
    results.append(dict(model="Persistence (lagged counts)", seed=0, **p2_eval(yte, s, per)))
    preds["Persistence"] = s
    s = kde_scores(TEST, best["KDE"]["bandwidth_m"])
    results.append(dict(model="KDE (victimisation, prior 3 periods)", seed=0, **p2_eval(yte, s, per)))
    preds["KDE"] = s
    for name in ["LR", "SVM", "RF"]:
        seeds = SEEDS if name == "RF" else [0]
        ss = []
        for sd in seeds:
            m, pr, fit_s = fit_tabular(name, best[name], Xtr, ytr, seed=sd)
            thr = M.best_threshold(yva, tab_score(m, pr, Xva))
            t0 = time.perf_counter()
            s = tab_score(m, pr, Xte)
            inf = time.perf_counter() - t0
            results.append(dict(model=name, seed=sd, fit_seconds=fit_s, infer_seconds=inf, **p2_eval(yte, s, per, thr)))
            ss.append(s)
        preds[name] = np.mean(ss, 0)
    for kind in ["MLP", "GCN", "ResGCN", "GAT"]:
        ps, infos, _, _ = run_nn_seeds(kind, best[kind], "binary")
        for sd, (p, info) in enumerate(zip(ps, infos)):
            results.append(dict(model=kind, seed=sd, fit_seconds=info["train_seconds"],
                                infer_seconds=info["infer_seconds"], best_epoch=info["best_epoch"],
                                n_params=info["n_params"], **p2_eval(yte, p, per, info["val_thr"])))
        preds[kind] = np.mean(ps, 0)
        np.save(RES / f"E2_{kind}_seed_preds.npy", np.array(ps))
    res = pd.DataFrame(results)
    save("E2_p2_test_runs", res)
    save("E2_p2_test_preds", pd.DataFrame(preds))
    summ = res.groupby("model", sort=False).agg(["mean", "std"]).round(4)
    print(summ[[("topk_macro_f1", "mean"), ("topk_pos_f1", "mean"), ("roc_auc", "mean"), ("avg_precision", "mean"),
                ("avg_precision", "std"), ("thr_macro_f1", "mean")]])


# ================================================================== E3 Phase 1
def p1_eval(y, pred):
    return M.multiclass_metrics(y, pred, NCLS)


def kde_p1(periods, bw):
    cats = [c for c in DF.CAT.value_counts().index]
    cxy = CELLS[["CX", "CY"]].values
    out = []
    for t in periods:
        pts = DF[(DF.PERIOD >= t - 3) & (DF.PERIOD < t)]
        dens = np.stack([len(pts[pts.CAT == c]) * M.kde_cell_scores(pts[pts.CAT == c][["X_M", "Y_M"]].values, cxy, bw)
                         for c in cats], 1)
        out.append(prep.to_p1_class(np.array(cats)[dens.argmax(1)]))
    return np.concatenate(out)


def E3():
    global FEATS_CUR
    tgt = "Y_P1"
    Xtr, ytr, rtr = xy(TRAIN, target=tgt)
    Xva, yva, rva = xy(VAL, target=tgt)
    Xte, yte, rte = xy(TEST, target=tgt)
    mtr, mva, mte = (r.Y_TOTAL.values > 0 for r in (rtr, rva, rte))
    tuning, best = [], {}
    FEATS_CUR = FEATS
    for name in ["LR", "SVM", "RF"]:
        res = []
        for p in TAB_GRIDS[name]:
            m, pr, _ = fit_tabular(name, p, Xtr[mtr], ytr[mtr])
            pred = m.predict(pr(Xva[mva]))
            res.append(dict(model=name, params=json.dumps(p), val=M.f1_score(yva[mva], pred, average="macro")))
        df = pd.DataFrame(res)
        best[name] = json.loads(df.loc[df.val.idxmax(), "params"])
        tuning.append(df)
    for kind in ["MLP", "GCN", "ResGCN", "GAT"]:
        b, df = tune_nn(kind, "multi", target=tgt)
        best[kind] = b
        tuning.append(df)
    kd = []
    for bw in [300, 500, 800, 1200, 2000]:
        kd.append(dict(model="KDE", params=json.dumps({"bandwidth_m": bw}),
                       val=M.f1_score(yva[mva], kde_p1(VAL, bw)[mva], average="macro")))
    kd = pd.DataFrame(kd)
    best["KDE"] = json.loads(kd.loc[kd.val.idxmax(), "params"])
    tuning.append(kd)
    save("E3_p1_tuning", pd.concat(tuning, ignore_index=True))
    save("E3_p1_best_params", best)

    yt = yte[mte]
    results, preds = [], {"GRID_ID": rte.GRID_ID.values[mte], "PERIOD": rte.PERIOD.values[mte], "y": yt}
    maj = np.bincount(ytr[mtr]).argmax()
    results.append(dict(model="Majority class", seed=0, **p1_eval(yt, np.full_like(yt, maj))))
    results.append(dict(model="Persistence (dominant type, prior 3 periods)", seed=0,
                        **p1_eval(yt, rte.PERSIST_P1.values[mte])))
    preds["Persistence"] = rte.PERSIST_P1.values[mte]
    kp = kde_p1(TEST, best["KDE"]["bandwidth_m"])[mte]
    results.append(dict(model="KDE (per-category density)", seed=0, **p1_eval(yt, kp)))
    preds["KDE"] = kp
    for name in ["LR", "SVM", "RF"]:
        for sd in (SEEDS if name == "RF" else [0]):
            m, pr, fs = fit_tabular(name, best[name], Xtr[mtr], ytr[mtr], seed=sd)
            pred = m.predict(pr(Xte[mte]))
            results.append(dict(model=name, seed=sd, fit_seconds=fs, **p1_eval(yt, pred)))
            if sd == 0:
                preds[name] = pred
    for kind in ["MLP", "GCN", "ResGCN", "GAT"]:
        ps, infos, _, _ = run_nn_seeds(kind, best[kind], "multi", target=tgt)
        for sd, (p, info) in enumerate(zip(ps, infos)):
            results.append(dict(model=kind, seed=sd, fit_seconds=info["train_seconds"], best_epoch=info["best_epoch"],
                                n_params=info["n_params"], **p1_eval(yt, p[mte].argmax(1))))
        preds[kind] = np.mean(ps, 0)[mte].argmax(1)
    res = pd.DataFrame(results)
    save("E3_p1_test_runs", res)
    save("E3_p1_test_preds", pd.DataFrame(preds))
    print(res.groupby("model", sort=False)[["accuracy", "macro_f1", "weighted_f1", "kappa", "f1_c0", "f1_c1", "f1_c2", "f1_c3"]]
          .agg(["mean", "std"]).round(3))
    # contemporaneous (leaky) variant: same-period composition as features -> label is argmax of features
    cur = D["panel"][D["panel"].PERIOD.isin(TEST) & (D["panel"].TOTAL > 0)]
    ccols = [c for c in cur.columns if c.startswith("C_")]
    leak = prep.to_p1_class([c[2:] for c in np.array(ccols)[cur[ccols].values.argmax(1)]])
    save("E3_p1_leakage_check", {"argmax_rule_accuracy_same_window": 1.0,
                                 "n": int(len(leak)), "note": "label is by construction argmax of same-window counts"})


# ================================================================== E4 imbalance strategies
def E4():
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    Xtr, ytr, _ = xy(TRAIN)
    Xva, yva, _ = xy(VAL)
    Xte, yte, rte = xy(TEST)
    per = rte.PERIOD.values
    out = []
    for label, cw, sm in [("LR: no weighting", None, False), ("LR: class weights", "balanced", False),
                          ("LR: SMOTE (train only)", None, True)]:
        m, pr, _ = fit_tabular("LR", best["LR"], Xtr, ytr, cw=cw, smote=sm)
        thr_default = 0.5
        thr_val = M.best_threshold(yva, tab_score(m, pr, Xva))
        s = tab_score(m, pr, Xte)
        r = p2_eval(yte, s, per, thr_default)
        r2 = p2_eval(yte, s, per, thr_val)
        out.append(dict(strategy=label, decision="threshold 0.5", **{k: v for k, v in r.items() if k.startswith("thr") or k in ("avg_precision", "roc_auc")}))
        out.append(dict(strategy=label, decision="validation-tuned threshold", **{k: v for k, v in r2.items() if k.startswith("thr") or k in ("avg_precision", "roc_auc")}))
    for label, cw in [("GCN: unweighted CE", False), ("GCN: weighted CE", True)]:
        ps, infos, _, _ = run_nn_seeds("GCN", best["GCN"], "binary", cw=cw)
        for decision in ["threshold 0.5", "validation-tuned threshold"]:
            rr = [p2_eval(yte, p, per, 0.5 if decision == "threshold 0.5" else i["val_thr"]) for p, i in zip(ps, infos)]
            df = pd.DataFrame(rr)
            out.append(dict(strategy=label, decision=decision,
                            **{k: df[k].mean() for k in df.columns if k.startswith("thr") or k in ("avg_precision", "roc_auc")},
                            thr_macro_f1_sd=df.thr_macro_f1.std()))
    res = pd.DataFrame(out)
    save("E4_imbalance", res)
    print(res.round(3))


# ================================================================== E5 graph ablations
def E5():
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    best1 = json.loads((RES / "E3_p1_best_params.json").read_text())
    Xte, yte, rte = xy(TEST)
    per = rte.PERIOD.values
    variants = []
    for label, radius, wt in [("Rook 3.0 km, inverse-distance (pilot)", 3000, "inverse"),
                              ("Rook 3.0 km, unweighted", 3000, "unit"),
                              ("Queen 3.2 km, inverse-distance", 3200, "inverse"),
                              ("Radius 4.5 km, inverse-distance", 4500, "inverse"),
                              ("Radius 6.6 km, inverse-distance", 6600, "inverse")]:
        ei, ew, _ = prep.build_graph(CELLS, radius, wt)
        variants.append((label, ei, ew))
    out = []
    for kind in ["GCN", "ResGCN"]:
        for label, ei, ew in variants:
            deg = np.bincount(ei[0], minlength=N).mean()
            ps, infos, _, _ = run_nn_seeds(kind, best[kind], "binary", ei=ei, ew=ew)
            df = pd.DataFrame([p2_eval(yte, p, per) for p in ps])
            p1s, _, y1, r1 = run_nn_seeds(kind, best1[kind], "multi", target="Y_P1", ei=ei, ew=ew)
            mk = r1.Y_TOTAL.values > 0
            d1 = pd.DataFrame([p1_eval(y1[mk], p[mk].argmax(1)) for p in p1s])
            out.append(dict(model=kind, variant=label, mean_degree=deg, edges=ei.shape[1] // 2,
                            p2_ap=df.avg_precision.mean(), p2_ap_sd=df.avg_precision.std(),
                            p2_topk_macro_f1=df.topk_macro_f1.mean(), p2_auc=df.roc_auc.mean(),
                            p1_macro_f1=d1.macro_f1.mean(), p1_macro_f1_sd=d1.macro_f1.std(), p1_acc=d1.accuracy.mean()))
        # depth ablation on the pilot graph
        for L in [1, 2, 3]:
            p = dict(best[kind], layers=L)
            ps, _, _, _ = run_nn_seeds(kind, p, "binary")
            df = pd.DataFrame([p2_eval(yte, q, per) for q in ps])
            p1 = dict(best1[kind], layers=L)
            p1s, _, y1, r1 = run_nn_seeds(kind, p1, "multi", target="Y_P1")
            mk = r1.Y_TOTAL.values > 0
            d1 = pd.DataFrame([p1_eval(y1[mk], q[mk].argmax(1)) for q in p1s])
            out.append(dict(model=kind, variant=f"Depth {L} (rook graph)", mean_degree=np.nan, edges=np.nan,
                            p2_ap=df.avg_precision.mean(), p2_ap_sd=df.avg_precision.std(),
                            p2_topk_macro_f1=df.topk_macro_f1.mean(), p2_auc=df.roc_auc.mean(),
                            p1_macro_f1=d1.macro_f1.mean(), p1_macro_f1_sd=d1.macro_f1.std(), p1_acc=d1.accuracy.mean()))
    # hand-crafted spatial lag in a non-graph model
    global FEATS_CUR
    feats = FEATS + prep.SLAG_FEATS
    FEATS_CUR = feats
    Xtr, ytr, _ = xy(TRAIN, feats)
    Xte2, _, _ = xy(TEST, feats)
    m, pr, _ = fit_tabular("LR", best["LR"], Xtr, ytr)
    r = p2_eval(yte, tab_score(m, pr, Xte2), per)
    FEATS_CUR = FEATS
    Xtr0, _, _ = xy(TRAIN)
    m0, pr0, _ = fit_tabular("LR", best["LR"], Xtr0, ytr)
    r0 = p2_eval(yte, tab_score(m0, pr0, Xte), per)
    out.append(dict(model="LR", variant="Without spatial-lag features", p2_ap=r0["avg_precision"], p2_topk_macro_f1=r0["topk_macro_f1"], p2_auc=r0["roc_auc"]))
    out.append(dict(model="LR", variant="With spatial-lag features", p2_ap=r["avg_precision"], p2_topk_macro_f1=r["topk_macro_f1"], p2_auc=r["roc_auc"]))
    res = pd.DataFrame(out)
    save("E5_graph_ablation", res)
    print(res.round(4).to_string())


# ================================================================== E6 feature ablation
def E6():
    global FEATS_CUR
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    best1 = json.loads((RES / "E3_p1_best_params.json").read_text())
    share = [f for f in FEATS if f.startswith("SHARE3_")]
    vict = ["VICTIM_L1", "VICTIM_L2", "VICTIM_L3", "VICTIM_MEAN_ALL", "VICTIM_SHARE", "VICTIM_TREND"]
    sets = {
        "All features": FEATS,
        "Without victimisation history": [f for f in FEATS if f not in vict],
        "Without composition shares": [f for f in FEATS if f not in share],
        "Without coordinates": [f for f in FEATS if f not in ("LAT", "LON")],
        "Victimisation lags only": ["VICTIM_L1", "VICTIM_L2", "VICTIM_L3", "VICTIM_MEAN_ALL"],
    }
    out = []
    for name, feats in sets.items():
        FEATS_CUR = feats
        Xtr, ytr, rtr = xy(TRAIN, feats)
        Xte, yte, rte = xy(TEST, feats)
        m, pr, _ = fit_tabular("LR", best["LR"], Xtr, ytr)
        r = p2_eval(yte, tab_score(m, pr, Xte), rte.PERIOD.values)
        ps, _, _, _ = run_nn_seeds("GCN", best["GCN"], "binary", feats=feats, seeds=SEEDS[:5])
        g = pd.DataFrame([p2_eval(yte, p, rte.PERIOD.values) for p in ps])
        # phase 1
        X1, y1, r1 = xy(TRAIN, feats, "Y_P1")
        X1t, y1t, r1t = xy(TEST, feats, "Y_P1")
        mk, mkt = r1.Y_TOTAL.values > 0, r1t.Y_TOTAL.values > 0
        m1, pr1, _ = fit_tabular("LR", json.loads((RES / "E3_p1_best_params.json").read_text())["LR"], X1[mk], y1[mk])
        lr1 = p1_eval(y1t[mkt], m1.predict(pr1(X1t[mkt])))
        p1s, _, _, _ = run_nn_seeds("GCN", best1["GCN"], "multi", feats=feats, target="Y_P1", seeds=SEEDS[:5])
        g1 = pd.DataFrame([p1_eval(y1t[mkt], p[mkt].argmax(1)) for p in p1s])
        out.append(dict(feature_set=name, n_features=len(feats), p2_LR_ap=r["avg_precision"], p2_GCN_ap=g.avg_precision.mean(),
                        p2_LR_topk_macro_f1=r["topk_macro_f1"], p2_GCN_topk_macro_f1=g.topk_macro_f1.mean(),
                        p1_LR_macro_f1=lr1["macro_f1"], p1_GCN_macro_f1=g1.macro_f1.mean()))
    FEATS_CUR = FEATS
    res = pd.DataFrame(out)
    save("E6_feature_ablation", res)
    print(res.round(3).to_string())


# ================================================================== E7 sensitivity (threshold, grid size)
def E7():
    global CELLS, EI, EW, SNAP, DF, N
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    out = []
    for pct in [70, 80, 90]:
        SNAP["HIGH"] = prep.label_high(SNAP, pct)
        Xtr, ytr, _ = xy(TRAIN)
        Xte, yte, rte = xy(TEST)
        m, pr, _ = fit_tabular("LR", best["LR"], Xtr, ytr)
        s = tab_score(m, pr, Xte)
        ps, _, _, _ = run_nn_seeds("GCN", best["GCN"], "binary", seeds=SEEDS[:5], ei=EI, ew=EW)
        pe = persistence_score(rte)
        rs, _, _, _ = run_nn_seeds("ResGCN", best["ResGCN"], "binary", seeds=SEEDS[:5], ei=EI, ew=EW)
        for name, sc in [("Persistence", [pe]), ("LR", [s]), ("GCN", ps), ("ResGCN", rs)]:
            out.append(dict(experiment="label percentile", setting=pct, model=name, prevalence=yte.mean(),
                            ap=np.mean([M.average_precision_score(yte, q) for q in sc]),
                            auc=np.mean([M.roc_auc_score(yte, q) for q in sc])))
    SNAP["HIGH"] = prep.label_high(SNAP, 80)
    saved = (CELLS, EI, EW, SNAP, DF, N)
    for cell in [1500, 2200, 3000]:
        DD = prep.build_all(cell_m=cell, radius_m=cell * 1.37)
        CELLS, EI, EW, SNAP, DF = DD["cells"], DD["ei"], DD["ew"], DD["snap"].copy(), DD["df"]
        N = len(CELLS)
        SNAP["HIGH"] = prep.label_high(SNAP, 80)
        Xtr, ytr, _ = xy(TRAIN)
        Xte, yte, rte = xy(TEST)
        m, pr, _ = fit_tabular("LR", best["LR"], Xtr, ytr)
        s = tab_score(m, pr, Xte)
        ps, infos, _, _ = run_nn_seeds("GCN", best["GCN"], "binary", seeds=SEEDS[:5], ei=EI, ew=EW)
        pe = persistence_score(rte)
        rs, _, _, _ = run_nn_seeds("ResGCN", best["ResGCN"], "binary", seeds=SEEDS[:5], ei=EI, ew=EW)
        for name, sc in [("Persistence", [pe]), ("LR", [s]), ("GCN", ps), ("ResGCN", rs)]:
            out.append(dict(experiment="grid size (m)", setting=cell, model=name, n_cells=N, prevalence=yte.mean(),
                            ap=np.mean([M.average_precision_score(yte, q) for q in sc]),
                            auc=np.mean([M.roc_auc_score(yte, q) for q in sc]),
                            gcn_train_s=np.mean([i["train_seconds"] for i in infos]) if name == "GCN" else np.nan))
    CELLS, EI, EW, SNAP, DF, N = saved
    res = pd.DataFrame(out)
    save("E7_sensitivity", res)
    print(res.round(3).to_string())


# ================================================================== E8 rolling-origin temporal evaluation
def E8():
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    best1 = json.loads((RES / "E3_p1_best_params.json").read_text())
    global FEATS_CUR
    FEATS_CUR = FEATS
    out = []
    for T in ROLL:
        tr, va, te = list(range(3, T - 1)), [T - 1], [T]
        Xtr, ytr, _ = xy(tr)
        Xte, yte, rte = xy(te)
        per = rte.PERIOD.values
        m, pr, _ = fit_tabular("LR", best["LR"], Xtr, ytr)
        r = dict(test_period=T, n_train_periods=len(tr))
        res_lr = p2_eval(yte, tab_score(m, pr, Xte), per)
        res_p = p2_eval(yte, persistence_score(rte), per)
        res_k = p2_eval(yte, kde_scores(te, best["KDE"]["bandwidth_m"]), per)
        ps, _, _, _ = run_nn_seeds("GCN", best["GCN"], "binary", train=tr, val=va, test=te, seeds=SEEDS[:5])
        gg = pd.DataFrame([p2_eval(yte, p, per) for p in ps])
        rs, _, _, _ = run_nn_seeds("ResGCN", best["ResGCN"], "binary", train=tr, val=va, test=te, seeds=SEEDS[:5])
        rg = pd.DataFrame([p2_eval(yte, p, per) for p in rs])
        out.append(dict(r, task="P2", model="ResGCN", ap=rg.avg_precision.mean(), topk_macro_f1=rg.topk_macro_f1.mean(),
                        auc=rg.roc_auc.mean(), ap_sd=rg.avg_precision.std()))
        for name, rr in [("Persistence", res_p), ("KDE", res_k), ("LR", res_lr)]:
            out.append(dict(r, task="P2", model=name, ap=rr["avg_precision"], topk_macro_f1=rr["topk_macro_f1"], auc=rr["roc_auc"]))
        out.append(dict(r, task="P2", model="GCN", ap=gg.avg_precision.mean(), topk_macro_f1=gg.topk_macro_f1.mean(),
                        auc=gg.roc_auc.mean(), ap_sd=gg.avg_precision.std()))
        # phase 1
        X1, y1, r1 = xy(tr, target="Y_P1")
        X1t, y1t, r1t = xy(te, target="Y_P1")
        mk, mkt = r1.Y_TOTAL.values > 0, r1t.Y_TOTAL.values > 0
        m1, pr1, _ = fit_tabular("LR", best1["LR"], X1[mk], y1[mk])
        out.append(dict(r, task="P1", model="LR", macro_f1=p1_eval(y1t[mkt], m1.predict(pr1(X1t[mkt])))["macro_f1"]))
        out.append(dict(r, task="P1", model="Persistence", macro_f1=p1_eval(y1t[mkt], r1t.PERSIST_P1.values[mkt])["macro_f1"]))
        p1s, _, _, _ = run_nn_seeds("GCN", best1["GCN"], "multi", target="Y_P1", train=tr, val=va, test=te, seeds=SEEDS[:5])
        out.append(dict(r, task="P1", model="GCN", macro_f1=np.mean([p1_eval(y1t[mkt], p[mkt].argmax(1))["macro_f1"] for p in p1s])))
    res = pd.DataFrame(out)
    save("E8_rolling_origin" + os.environ.get("A2_SUFFIX", ""), res)
    print(res.round(3).to_string())


# ================================================================== E9 spatial-block CV
def E9():
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    global FEATS_CUR
    FEATS_CUR = FEATS
    blocks = KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(CELLS[["CX", "CY"]].values)
    save("E9_blocks", pd.DataFrame({"GRID_ID": CELLS.GRID_ID, "BLOCK": blocks}))
    Xtr, ytr, rtr = xy(TRAIN)
    Xte, yte, rte = xy(TEST)
    out = []
    for b in range(5):
        held = set(CELLS.GRID_ID[blocks == b])
        trm = ~rtr.GRID_ID.isin(held).values
        tem = rte.GRID_ID.isin(held).values
        m, pr, _ = fit_tabular("LR", best["LR"], Xtr[trm], ytr[trm])
        s = tab_score(m, pr, Xte)[tem]
        pe = persistence_score(rte)[tem]
        fn = lambda r: ~r.GRID_ID.isin(held).values
        ps, _, _, _ = run_nn_seeds("GCN", best["GCN"], "binary", seeds=SEEDS[:5], train_mask_fn=fn)
        ms, _, _, _ = run_nn_seeds("MLP", best["MLP"], "binary", seeds=SEEDS[:5], train_mask_fn=fn)
        rs, _, _, _ = run_nn_seeds("ResGCN", best["ResGCN"], "binary", seeds=SEEDS[:5], train_mask_fn=fn)
        yb = yte[tem]
        row = dict(block=b, n_cells=len(held), n_test=int(tem.sum()), n_pos=int(yb.sum()))
        if yb.sum() == 0 or yb.sum() == len(yb):
            out.append(dict(row, note="single class in block; AP/AUC undefined"))
            continue
        out.append(dict(row, persistence_ap=M.average_precision_score(yb, pe), lr_ap=M.average_precision_score(yb, s),
                        mlp_ap=np.mean([M.average_precision_score(yb, p[tem]) for p in ms]),
                        gcn_ap=np.mean([M.average_precision_score(yb, p[tem]) for p in ps]),
                        resgcn_ap=np.mean([M.average_precision_score(yb, p[tem]) for p in rs]),
                        lr_auc=M.roc_auc_score(yb, s), gcn_auc=np.mean([M.roc_auc_score(yb, p[tem]) for p in ps])))
    res = pd.DataFrame(out)
    save("E9_spatial_block_cv", res)
    print(res.round(3).to_string())


# ================================================================== E10 uncertainty (block bootstrap)
def E10():
    pr = pd.read_csv(RES / "E2_p2_test_preds.csv")
    cells = CELLS.set_index("GRID_ID")
    sb = (cells.ROW // 3).astype(str) + "_" + (cells.COL // 3).astype(str)
    pr["SB"] = pr.GRID_ID.map(sb)
    blocks = pr.SB.unique()
    rng = np.random.default_rng(0)
    models_ = ["Persistence", "KDE", "LR", "SVM", "RF", "MLP", "GCN", "ResGCN", "GAT"]
    B = 2000
    by_block = {b: np.where(pr.SB.values == b)[0] for b in blocks}
    stats = {m: [] for m in models_}
    diffs = {"GCN-LR": [], "GCN-Persistence": [], "LR-Persistence": [], "GCN-MLP": [], "ResGCN-MLP": [],
             "ResGCN-GCN": [], "RF-LR": [], "ResGCN-LR": [], "GAT-MLP": [], "GAT-GCN": []}
    y = pr.y.values
    for _ in range(B):
        idx = np.concatenate([by_block[b] for b in rng.choice(blocks, len(blocks), replace=True)])
        yy = y[idx]
        if yy.sum() == 0:
            continue
        ap = {m: M.average_precision_score(yy, pr[m].values[idx]) for m in models_}
        for m in models_:
            stats[m].append(ap[m])
        diffs["GCN-LR"].append(ap["GCN"] - ap["LR"])
        diffs["GCN-Persistence"].append(ap["GCN"] - ap["Persistence"])
        diffs["LR-Persistence"].append(ap["LR"] - ap["Persistence"])
        diffs["GCN-MLP"].append(ap["GCN"] - ap["MLP"])
        diffs["ResGCN-MLP"].append(ap["ResGCN"] - ap["MLP"])
        diffs["ResGCN-GCN"].append(ap["ResGCN"] - ap["GCN"])
        diffs["RF-LR"].append(ap["RF"] - ap["LR"])
        diffs["ResGCN-LR"].append(ap["ResGCN"] - ap["LR"])
        diffs["GAT-MLP"].append(ap["GAT"] - ap["MLP"])
        diffs["GAT-GCN"].append(ap["GAT"] - ap["GCN"])
    out = []
    for m in models_:
        a = np.array(stats[m])
        out.append(dict(quantity=f"AP {m}", point=M.average_precision_score(y, pr[m].values),
                        lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5)))
    for k, v in diffs.items():
        a = np.array(v)
        out.append(dict(quantity=f"AP difference {k}", point=np.mean(a), lo=np.percentile(a, 2.5),
                        hi=np.percentile(a, 97.5), share_positive=(a > 0).mean()))
    # naive (iid) bootstrap for comparison of interval width
    a = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        a.append(M.average_precision_score(y[idx], pr.GCN.values[idx]))
    out.append(dict(quantity="AP GCN (naive iid bootstrap)", point=M.average_precision_score(y, pr.GCN.values),
                    lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5)))
    res = pd.DataFrame(out)
    res["n_blocks"] = len(blocks)
    save("E10_bootstrap", res)
    # Phase 1 bootstrap on macro F1
    p1 = pd.read_csv(RES / "E3_p1_test_preds.csv")
    p1["SB"] = p1.GRID_ID.map(sb)
    bb = {b: np.where(p1.SB.values == b)[0] for b in p1.SB.unique()}
    o1 = {m: [] for m in ["Persistence", "KDE", "LR", "SVM", "RF", "MLP", "GCN", "ResGCN", "GAT"]}
    d1, d2 = [], []
    for _ in range(B):
        idx = np.concatenate([bb[b] for b in rng.choice(list(bb), len(bb), replace=True)])
        for m in o1:
            o1[m].append(M.f1_score(p1.y.values[idx], p1[m].values[idx], average="macro", labels=range(NCLS), zero_division=0))
        d1.append(o1["GCN"][-1] - o1["Persistence"][-1])
        d2.append(o1["LR"][-1] - o1["Persistence"][-1])
    r1 = [dict(quantity=f"P1 macro-F1 {m}", point=M.f1_score(p1.y, p1[m], average="macro", labels=range(NCLS)),
               lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)) for m, v in o1.items()]
    r1.append(dict(quantity="P1 macro-F1 difference GCN-Persistence", point=np.mean(d1), lo=np.percentile(d1, 2.5),
                   hi=np.percentile(d1, 97.5), share_positive=(np.array(d1) > 0).mean()))
    r1.append(dict(quantity="P1 macro-F1 difference LR-Persistence", point=np.mean(d2), lo=np.percentile(d2, 2.5),
                   hi=np.percentile(d2, 97.5), share_positive=(np.array(d2) > 0).mean()))
    res = pd.concat([res, pd.DataFrame(r1)], ignore_index=True)
    save("E10_bootstrap", res)
    print(res.round(3).to_string())


# ================================================================== E11 continuous count prediction
def E11():
    global FEATS_CUR
    FEATS_CUR = FEATS
    Xtr, ytr, rtr = xy(TRAIN, target="Y_VICTIM")
    Xva, yva, _ = xy(VAL, target="Y_VICTIM")
    Xte, yte, rte = xy(TEST, target="Y_VICTIM")
    pr = M.Prepper(FEATS).fit(Xtr)
    out = []
    pers = rte.VICTIM_L1.values
    pers3 = (rte.VICTIM_L1 + rte.VICTIM_L2 + rte.VICTIM_L3).values / 3
    best_a, best_v = None, np.inf
    for a in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
        g = PoissonRegressor(alpha=a, max_iter=3000).fit(pr(Xtr), ytr)
        v = mean_absolute_error(yva, g.predict(pr(Xva)))
        if v < best_v:
            best_a, best_v = a, v
    g = PoissonRegressor(alpha=best_a, max_iter=3000).fit(pr(Xtr), ytr)
    preds = {"Persistence (last period)": pers, "Persistence (3-period mean)": pers3, "Poisson GLM": g.predict(pr(Xte))}
    gp = dict(json.loads((RES / "E1_p2_best_params.json").read_text())["GCN"])
    ps, _, _, _ = run_nn_seeds("GCN", gp, "count", target="Y_VICTIM", seeds=SEEDS[:5])
    preds["GCN (Poisson loss)"] = np.mean(ps, 0)
    ms, _, _, _ = run_nn_seeds("MLP", gp, "count", target="Y_VICTIM", seeds=SEEDS[:5])
    preds["MLP (Poisson loss)"] = np.mean(ms, 0)
    for k, p in preds.items():
        out.append(dict(model=k, MAE=mean_absolute_error(yte, p), RMSE=np.sqrt(mean_squared_error(yte, p)),
                        R2=r2_score(yte, p), spearman=pd.Series(p).rank().corr(pd.Series(yte).rank())))
    res = pd.DataFrame(out)
    res["poisson_alpha"] = best_a
    save("E11_count_regression", res)
    print(res.round(3).to_string())


# ================================================================== E12 unsupervised DBSCAN hotspots
def E12():
    out = {}
    hist = DF[(DF.PERIOD >= 8) & (DF.PERIOD <= 10) & (DF.VICTIM == 1)]
    test = DF[DF.PERIOD.isin(TEST) & (DF.VICTIM == 1)]
    X = hist[["X_M", "Y_M"]].values
    # k-distance (k = min_samples) curve for eps selection
    from sklearn.neighbors import NearestNeighbors
    nn_ = NearestNeighbors(n_neighbors=30).fit(X)
    kd = np.sort(nn_.kneighbors(X)[0][:, -1])
    save("E12_kdist", pd.DataFrame({"kdist_m": kd[:: max(1, len(kd) // 2000)]}))
    rows_ = []
    val = DF[DF.PERIOD.isin(VAL) & (DF.VICTIM == 1)]
    for eps in [100, 150, 200, 300]:
        for ms in [30, 60, 100]:
            t0 = time.perf_counter()
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            secs = time.perf_counter() - t0
            hc = hist.assign(L=lab)
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            noise = (lab == -1).mean()
            # cells whose clustered points are >= 50% of the cell's victimisation incidents
            frac = hc.groupby("GRID_ID").L.apply(lambda s: (s >= 0).mean())
            flagged = set(frac[frac >= 0.5].index)
            area = len(flagged) / N
            hit = test.GRID_ID.isin(flagged).mean()
            sil = np.nan
            if n_cl > 1:
                from sklearn.metrics import silhouette_score
                core = lab >= 0
                idx = np.random.default_rng(0).choice(np.where(core)[0], min(5000, core.sum()), replace=False)
                sil = silhouette_score(X[idx], lab[idx])
            rows_.append(dict(eps_m=eps, min_samples=ms, clusters=n_cl, noise_share=noise, flagged_cells=len(flagged),
                              area_share=area, test_hit_rate=hit, PAI=hit / area if area > 0 else np.nan,
                              silhouette_clustered=sil, seconds=secs))
    res = pd.DataFrame(rows_)
    save("E12_dbscan_grid", res)
    # stability: two disjoint windows with the chosen setting (chosen on PAI computed for validation periods)
    sel = []
    for _, r in res.iterrows():
        lab = DBSCAN(eps=r.eps_m, min_samples=int(r.min_samples)).fit_predict(
            DF[(DF.PERIOD >= 5) & (DF.PERIOD <= 7) & (DF.VICTIM == 1)][["X_M", "Y_M"]].values)
        hc = DF[(DF.PERIOD >= 5) & (DF.PERIOD <= 7) & (DF.VICTIM == 1)].assign(L=lab)
        frac = hc.groupby("GRID_ID").L.apply(lambda s: (s >= 0).mean())
        fl = set(frac[frac >= 0.5].index)
        a = len(fl) / N
        sel.append(val.GRID_ID.isin(fl).mean() / a if a > 0 else 0)
    res["val_PAI_(fit_5-7,eval_9-10)"] = sel
    save("E12_dbscan_grid", res)
    b = res.iloc[int(np.argmax(sel))]
    out["selected"] = b.to_dict()

    def flagged_for(win):
        h = DF[DF.PERIOD.isin(win) & (DF.VICTIM == 1)]
        lab = DBSCAN(eps=b.eps_m, min_samples=int(b.min_samples)).fit_predict(h[["X_M", "Y_M"]].values)
        frac = h.assign(L=lab).groupby("GRID_ID").L.apply(lambda s: (s >= 0).mean())
        return set(frac[frac >= 0.5].index), lab, h
    f1_, _, _ = flagged_for([2, 3, 4])
    f2_, _, _ = flagged_for([5, 6, 7])
    out["stability_jaccard_windows_2-4_vs_5-7"] = len(f1_ & f2_) / max(1, len(f1_ | f2_))
    fsel, lab, h = flagged_for([8, 9, 10])
    h.assign(L=lab)[["X_M", "Y_M", "L"]].to_csv(RES / "E12_dbscan_points.csv", index=False)
    pd.Series(sorted(fsel), name="GRID_ID").to_csv(RES / "E12_dbscan_flagged_cells.csv", index=False)
    # PAI comparison at matched area (top-20% cells) for supervised/KDE rankings on test periods
    pr = pd.read_csv(RES / "E2_p2_test_preds.csv")
    k = int(np.ceil(TOPK_SHARE * N))
    comp = []
    for m in ["Persistence", "KDE", "LR", "RF", "GCN", "ResGCN"]:
        hits = []
        for t in TEST:
            sub = pr[pr.PERIOD == t]
            top = set(sub.nlargest(k, m).GRID_ID)
            hits.append(test[test.PERIOD == t].GRID_ID.isin(top).mean())
        comp.append(dict(method=m, area_share=k / N, hit_rate=np.mean(hits), PAI=np.mean(hits) / (k / N)))
    comp.append(dict(method=f"DBSCAN (eps={b.eps_m:.0f} m, minPts={int(b.min_samples)})", area_share=len(fsel) / N,
                     hit_rate=test.GRID_ID.isin(fsel).mean(), PAI=test.GRID_ID.isin(fsel).mean() / (len(fsel) / N)))
    save("E12_pai_comparison", pd.DataFrame(comp))
    save("E12_summary", out)
    print(res.round(3).to_string())
    print(pd.DataFrame(comp).round(3))
    print(out)


# ================================================================== E13 environment / runtime
def E13():
    env = dict(python=platform.python_version(), platform=platform.platform(), processor="Apple M5 (10 cores), 16 GB RAM",
               numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, torch=torch.__version__,
               torch_geometric=torch_geometric.__version__, seeds=SEEDS,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               periods={int(p): [str(g.DATE.min().date()), str(g.DATE.max().date()), int(len(g))]
                        for p, g in DF.groupby("PERIOD")},
               split=dict(train=TRAIN, val=VAL, test=TEST))
    save("E13_environment", env)
    print(env)


# ================================================================== training curves for figure
def curves():
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    _, infos, _, _ = run_nn_seeds("GCN", dict(best["GCN"]), "binary", seeds=[0], curve=True)
    pd.DataFrame(infos[0]["curve"], columns=["epoch", "train_loss", "val_loss", "val_ap"]).to_csv(RES / "curve_gcn_p2.csv", index=False)
    _, infos, _, _ = run_nn_seeds("MLP", dict(best["MLP"]), "binary", seeds=[0], curve=True)
    pd.DataFrame(infos[0]["curve"], columns=["epoch", "train_loss", "val_loss", "val_ap"]).to_csv(RES / "curve_mlp_p2.csv", index=False)


ALL = {"E0": E0, "E1": E1_E2, "E3": E3, "E4": E4, "E5": E5, "E6": E6, "E7": E7, "E8": E8, "E9": E9,
       "E10": E10, "E11": E11, "E12": E12, "E13": E13, "curves": curves}

# ================================================================== E14 feature importance (validation data only)
def E14():
    from sklearn.inspection import permutation_importance
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    Xtr, ytr, _ = xy(TRAIN)
    Xva, yva, _ = xy(VAL)
    m, pr, _ = fit_tabular("LR", best["LR"], Xtr, ytr)
    rf, prf, _ = fit_tabular("RF", best["RF"], Xtr, ytr)
    pi = permutation_importance(rf, prf(Xva), yva, scoring="average_precision", n_repeats=20, random_state=0)
    pl = permutation_importance(m, pr(Xva), yva, scoring="average_precision", n_repeats=20, random_state=0)
    res = pd.DataFrame({"feature": FEATS, "lr_std_coef": m.coef_[0], "lr_perm_ap_drop": pl.importances_mean,
                        "rf_perm_ap_drop": pi.importances_mean, "rf_perm_sd": pi.importances_std})
    res = res.reindex(res.lr_std_coef.abs().sort_values(ascending=False).index)
    save("E14_feature_importance", res)
    print(res.round(4).to_string())


ALL["E14"] = E14



# ================================================================== E15 preprocessing ablation (post-hoc extension)
def E15():
    """Measured effect of each preprocessing choice (hyperparameters re-tuned only for LR; others as in E1/E3)."""
    global PREP_MODE, FEATS_CUR
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    best1 = json.loads((RES / "E3_p1_best_params.json").read_text())
    Xtr, ytr, rtr = xy(TRAIN)
    Xva, yva, _ = xy(VAL)
    Xte, yte, rte = xy(TEST)
    X1tr, y1tr, r1tr = xy(TRAIN, target="Y_P1")
    X1te, y1te, r1te = xy(TEST, target="Y_P1")
    m1tr, m1te = r1tr.Y_TOTAL.values > 0, r1te.Y_TOTAL.values > 0
    low_cells = set(CELLS.GRID_ID[CELLS.N < 50 * (3 if MODE == "multi" else 1)])
    out = []
    for mode in ["main", "raw", "std", "log", "robust"]:
        PREP_MODE = mode
        FEATS_CUR = FEATS
        # LR with C re-tuned on validation for this preprocessing
        bv, bc = -1, None
        for C in TAB_GRIDS["LR"]:
            m, pr, _ = fit_tabular("LR", C, Xtr, ytr)
            v = M.average_precision_score(yva, tab_score(m, pr, Xva))
            if v > bv:
                bv, bc = v, C
        m, pr, _ = fit_tabular("LR", bc, Xtr, ytr)
        lr = p2_eval(yte, tab_score(m, pr, Xte), rte.PERIOD.values)
        row = dict(variant=mode, p2_LR_val_ap=bv, p2_LR_ap=lr["avg_precision"], p2_LR_topk_macro_f1=lr["topk_macro_f1"], LR_C=bc["C"])
        bv1, bc1 = -1, None
        for C in TAB_GRIDS["LR"]:
            mm, pp, _ = fit_tabular("LR", C, X1tr[m1tr], y1tr[m1tr])
            Xv1, yv1, rv1 = xy(VAL, target="Y_P1")
            v = M.f1_score(yv1[rv1.Y_TOTAL.values > 0], mm.predict(pp(Xv1[rv1.Y_TOTAL.values > 0])), average="macro")
            if v > bv1:
                bv1, bc1 = v, C
        mm, pp, _ = fit_tabular("LR", bc1, X1tr[m1tr], y1tr[m1tr])
        row["p1_LR_val_macro_f1"] = bv1
        row["p1_LR_macro_f1"] = p1_eval(y1te[m1te], mm.predict(pp(X1te[m1te])))["macro_f1"]
        for kind in ["MLP", "ResGCN", "GCN"]:
            ps, _, _, _ = run_nn_seeds(kind, best[kind], "binary", seeds=SEEDS[:5])
            row[f"p2_{kind}_ap"] = float(np.mean([M.average_precision_score(yte, p) for p in ps]))
        ps1, _, _, _ = run_nn_seeds("MLP", best1["MLP"], "multi", target="Y_P1", seeds=SEEDS[:5])
        row["p1_MLP_macro_f1"] = float(np.mean([p1_eval(y1te[m1te], p[m1te].argmax(1))["macro_f1"] for p in ps1]))
        out.append(row)
    # data-selection variant: drop the low-volume (<50 incidents/yr) cells from training only
    PREP_MODE = "main"
    keep = ~rtr.GRID_ID.isin(low_cells).values
    m, pr, _ = fit_tabular("LR", best["LR"], Xtr[keep], ytr[keep])
    lr = p2_eval(yte, tab_score(m, pr, Xte), rte.PERIOD.values)
    fn = lambda r: ~r.GRID_ID.isin(low_cells).values
    row = dict(variant=f"drop {len(low_cells)} low-volume cells from training", p2_LR_ap=lr["avg_precision"],
               p2_LR_topk_macro_f1=lr["topk_macro_f1"])
    for kind in ["MLP", "ResGCN", "GCN"]:
        ps, _, _, _ = run_nn_seeds(kind, best[kind], "binary", seeds=SEEDS[:5], train_mask_fn=fn)
        row[f"p2_{kind}_ap"] = float(np.mean([M.average_precision_score(yte, p) for p in ps]))
    out.append(row)
    res = pd.DataFrame(out)
    save("E15_preprocessing_ablation", res)
    print(res.round(4).to_string())


# ================================================================== E16 block permutation tests (post-hoc extension)
def _block_ids(grid_ids):
    cells = CELLS.set_index("GRID_ID")
    sb = (cells.ROW // 3).astype(str) + "_" + (cells.COL // 3).astype(str)
    b = pd.Series(grid_ids).map(sb)
    codes, uniq = pd.factorize(b)
    return codes, len(uniq)


def E16():
    """Paired spatial-block permutation (sign-flip) tests; scores are percentile-ranked within period so
    that models on different scales can be exchanged within a block."""
    pr = pd.read_csv(RES / "E2_p2_test_preds.csv")
    names = ["Persistence", "KDE", "LR", "SVM", "RF", "MLP", "GCN", "ResGCN", "GAT"]
    R = {m: pr.groupby("PERIOD")[m].rank(pct=True).values for m in names}
    y = pr.y.values
    codes, nb = _block_ids(pr.GRID_ID.values)
    rng = np.random.default_rng(1)
    B = 4000
    comps = [("LR", "Persistence"), ("RF", "LR"), ("MLP", "LR"), ("SVM", "LR"), ("GCN", "MLP"), ("ResGCN", "MLP"),
             ("ResGCN", "GCN"), ("GAT", "MLP"), ("GAT", "GCN"), ("KDE", "Persistence")]
    rows_ = []
    flips = rng.integers(0, 2, size=(B, nb)).astype(bool)
    for a, b in comps:
        obs = M.average_precision_score(y, R[a]) - M.average_precision_score(y, R[b])
        d = np.empty(B)
        for i in range(B):
            f = flips[i][codes]
            d[i] = M.average_precision_score(y, np.where(f, R[b], R[a])) - M.average_precision_score(y, np.where(f, R[a], R[b]))
        p = (np.sum(np.abs(d) >= abs(obs) - 1e-12) + 1) / (B + 1)
        rows_.append(dict(task="P2 AP", comparison=f"{a} - {b}", observed_diff=obs, p_raw=p))
    q = pd.read_csv(RES / "E3_p1_test_preds.csv")
    y1 = q.y.values
    codes1, nb1 = _block_ids(q.GRID_ID.values)
    flips1 = rng.integers(0, 2, size=(B, nb1)).astype(bool)
    c1 = [("LR", "Persistence"), ("GCN", "SVM"), ("GCN", "MLP"), ("LR", "GCN"), ("ResGCN", "GCN"), ("GAT", "GCN"),
          ("MLP", "LR")]
    for a, b in c1:
        f1 = lambda pa, pb: M.f1_score(y1, pa, average="macro", labels=range(NCLS), zero_division=0)
        obs = f1(q[a].values, None) - f1(q[b].values, None)
        d = np.empty(B)
        for i in range(B):
            f = flips1[i][codes1]
            d[i] = f1(np.where(f, q[b].values, q[a].values), None) - f1(np.where(f, q[a].values, q[b].values), None)
        p = (np.sum(np.abs(d) >= abs(obs) - 1e-12) + 1) / (B + 1)
        rows_.append(dict(task="P1 macro-F1", comparison=f"{a} - {b}", observed_diff=obs, p_raw=p))
    res = pd.DataFrame(rows_)
    for t in res.task.unique():           # Holm correction within each task family
        sub = res[res.task == t].sort_values("p_raw")
        m = len(sub)
        adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(sub.p_raw)])
        res.loc[sub.index, "p_holm"] = np.minimum(adj, 1)
    save("E16_permutation_tests", res)
    print(res.round(4).to_string())


# ================================================================== E17 Phase 1 feature extension (post-hoc extension)
def add_ext_features():
    """History-only extras for Phase 1: previous-period group shares, dominant-class one-hots, 6-period shares,
    dominance margin and mix entropy."""
    panel = D["panel"]
    ccols = [c for c in panel.columns if c.startswith("C_")]
    cats = [c[2:] for c in ccols]
    gidx = np.array([prep.P1_CLASSES.index(c) if c in prep.P1_KEEP else len(prep.P1_KEEP) for c in cats])
    order = CELLS.GRID_ID.values
    P = panel.set_index(["GRID_ID", "PERIOD"]).sort_index()
    A = np.stack([P.xs(t, level="PERIOD").reindex(order)[ccols].values for t in range(prep.N_PERIODS)], 1)  # cell, period, cat
    G = np.zeros((A.shape[0], A.shape[1], len(prep.P1_CLASSES)))
    for k in range(len(prep.P1_CLASSES)):
        G[:, :, k] = A[:, :, gidx == k].sum(2)
    rows_ = []
    K = len(prep.P1_CLASSES)
    for t in range(prep.N_LAGS, prep.N_PERIODS):
        g1 = G[:, t - 1, :]
        g3 = G[:, max(0, t - 3):t, :].sum(1)
        g6 = G[:, max(0, t - 6):t, :].sum(1)
        a6 = A[:, max(0, t - 6):t, :].sum(1)
        f = {"PERIOD": t, "NODE": np.arange(A.shape[0])}
        s1 = g1 / np.maximum(g1.sum(1, keepdims=True), 1)
        s6 = g6 / np.maximum(g6.sum(1, keepdims=True), 1)
        s3 = g3 / np.maximum(g3.sum(1, keepdims=True), 1)
        for k in range(K):
            f[f"X_SHARE1_{k}"] = s1[:, k]
            f[f"X_SHARE6_{k}"] = s6[:, k]
            f[f"X_DOM1_{k}"] = ((g1.argmax(1) == k) & (g1.sum(1) > 0)).astype(float)
            f[f"X_DOM3_{k}"] = ((g3.argmax(1) == k) & (g3.sum(1) > 0)).astype(float)
        srt = np.sort(s3, 1)
        f["X_MARGIN3"] = srt[:, -1] - srt[:, -2]
        pa = a6 / np.maximum(a6.sum(1, keepdims=True), 1)
        f["X_ENTROPY6"] = -(pa * np.log(pa + 1e-12)).sum(1)
        rows_.append(pd.DataFrame(f))
    ext = pd.concat(rows_, ignore_index=True)
    cols = [c for c in ext.columns if c.startswith("X_")]
    global SNAP
    SNAP = SNAP.drop(columns=[c for c in cols if c in SNAP.columns]).merge(ext, on=["PERIOD", "NODE"], how="left")
    return cols


def E17():
    global FEATS_CUR
    ext = add_ext_features()
    feats_sets = {"base (23 features)": FEATS, f"extended (+{len(ext)} history features)": FEATS + ext}
    out = []
    for name, feats in feats_sets.items():
        FEATS_CUR = feats
        Xtr, ytr, rtr = xy(TRAIN, feats, "Y_P1")
        Xva, yva, rva = xy(VAL, feats, "Y_P1")
        Xte, yte, rte = xy(TEST, feats, "Y_P1")
        mtr, mva, mte = (r.Y_TOTAL.values > 0 for r in (rtr, rva, rte))
        for model in ["LR", "RF", "MLP", "ResGCN"]:
            if model in ("LR", "RF"):
                bv, bp = -1, None
                for p in TAB_GRIDS[model]:
                    m, pr, _ = fit_tabular(model, p, Xtr[mtr], ytr[mtr])
                    v = M.f1_score(yva[mva], m.predict(pr(Xva[mva])), average="macro")
                    if v > bv:
                        bv, bp = v, p
                m, pr, _ = fit_tabular(model, bp, Xtr[mtr], ytr[mtr])
                te = p1_eval(yte[mte], m.predict(pr(Xte[mte])))
                out.append(dict(features=name, model=model, val_macro_f1=bv, test_macro_f1=te["macro_f1"],
                                test_macro_f1_sd=0.0, test_acc=te["accuracy"], test_kappa=te["kappa"]))
            else:
                b, tune = tune_nn(model, "multi", feats=feats, target="Y_P1")
                ps, infos, _, _ = run_nn_seeds(model, b, "multi", feats=feats, target="Y_P1", seeds=SEEDS[:5])
                te = [p1_eval(yte[mte], p[mte].argmax(1)) for p in ps]
                out.append(dict(features=name, model=model, val_macro_f1=float(tune.val.max()),
                                test_macro_f1=float(np.mean([t["macro_f1"] for t in te])),
                                test_macro_f1_sd=float(np.std([t["macro_f1"] for t in te], ddof=1)),
                                test_acc=float(np.mean([t["accuracy"] for t in te])),
                                test_kappa=float(np.mean([t["kappa"] for t in te]))))
    FEATS_CUR = FEATS
    res = pd.DataFrame(out)
    save("E17_phase1_features", res)
    print(res.round(3).to_string())


# ================================================================== E18 tuned count regression (post-hoc extension)
def E18():
    Xtr, ytr, _ = xy(TRAIN, target="Y_VICTIM")
    Xva, yva, _ = xy(VAL, target="Y_VICTIM")
    Xte, yte, rte = xy(TEST, target="Y_VICTIM")
    pr = M.Prepper(FEATS).fit(Xtr)
    preds, notes = {}, {}
    preds["Persistence (last period)"] = rte.VICTIM_L1.values
    preds["Persistence (3-period mean)"] = (rte.VICTIM_L1 + rte.VICTIM_L2 + rte.VICTIM_L3).values / 3
    best_a, best_v = None, np.inf
    for a in [1e-4, 1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100]:
        g = PoissonRegressor(alpha=a, max_iter=5000).fit(pr(Xtr), ytr)
        v = mean_absolute_error(yva, g.predict(pr(Xva)))
        if v < best_v:
            best_a, best_v = a, v
    g = PoissonRegressor(alpha=best_a, max_iter=5000).fit(pr(Xtr), ytr)
    preds["Poisson GLM (tuned alpha)"] = g.predict(pr(Xte))
    notes["Poisson GLM (tuned alpha)"] = f"alpha={best_a}"
    tun = []
    for kind in ["MLP", "GCN", "ResGCN", "GAT"]:
        b, df = tune_nn(kind, "count", target="Y_VICTIM")
        ps, _, _, _ = run_nn_seeds(kind, b, "count", target="Y_VICTIM", seeds=SEEDS[:5])
        preds[f"{kind} (Poisson loss, tuned)"] = np.mean(ps, 0)
        notes[f"{kind} (Poisson loss, tuned)"] = json.dumps(b)
        tun.append(df)
    save("E18_count_tuning", pd.concat(tun, ignore_index=True))
    out = [dict(model=k, MAE=mean_absolute_error(yte, p), RMSE=np.sqrt(mean_squared_error(yte, p)), R2=r2_score(yte, p),
                spearman=pd.Series(p).rank().corr(pd.Series(yte).rank()), config=notes.get(k, "")) for k, p in preds.items()]
    res = pd.DataFrame(out)
    save("E18_count_tuned", res)
    print(res.round(3).to_string())


# ================================================================== E19 effective sample size and centroid graph (post-hoc extension)
def morans_i(z, W):
    z = z - z.mean()
    return len(z) / W.sum() * (z @ W @ z) / (z @ z)


def E19():
    out = {}
    n = len(CELLS)
    W = np.zeros((n, n))
    W[EI[0], EI[1]] = 1.0
    ys = []
    for t in sorted(SNAP.PERIOD.unique()):
        ys.append(np.log1p(SNAP[SNAP.PERIOD == t].sort_values("NODE").Y_VICTIM.values))
    ys = np.array(ys)
    I = np.array([morans_i(y, W) for y in ys])
    test_I = np.array([morans_i(ys[i], W) for i, t in enumerate(sorted(SNAP.PERIOD.unique())) if t in TEST])
    rho_t = np.array([np.corrcoef(ys[i - 1], ys[i])[0, 1] for i in range(1, len(ys))])
    mI, mR = float(I.mean()), float(rho_t.mean())
    n_test_periods = len(TEST)
    n_eff_space = n * (1 - mI) / (1 + mI)
    n_eff_time = max(1.0, n_test_periods * (1 - mR) / (1 + mR))
    out.update(morans_I_mean=mI, morans_I_test=float(test_I.mean()), temporal_lag1_corr_log_counts=mR,
               n_cells=n, n_test_periods=n_test_periods, n_test_rows=n * n_test_periods,
               n_eff_space=n_eff_space, n_eff_time_per_cell=n_eff_time, n_eff_total_approx=n_eff_space * n_eff_time,
               n_incidents=int(len(DF)))
    # pilot-style comparison: Moran's I on binary HIGH labels
    hi = np.array([morans_i(SNAP[SNAP.PERIOD == t].sort_values("NODE").HIGH.values.astype(float), W) for t in TEST])
    out["morans_I_high_label_test"] = float(hi.mean())
    # centroid-based graph (distances between incident centroids from periods before the test window, as in the base paper)
    pre = DF[DF.PERIOD < min(TEST)]
    cen = pre.groupby("GRID_ID")[["X_M", "Y_M"]].mean().reindex(CELLS.GRID_ID)
    cen = cen.fillna(CELLS.set_index("GRID_ID")[["CX", "CY"]].rename(columns={"CX": "X_M", "CY": "Y_M"}))
    cc = CELLS.copy()
    cc["CX"], cc["CY"] = cen.X_M.values, cen.Y_M.values
    ei_c, ew_c, dist_c = prep.build_graph(cc, 3000.0, "inverse")
    deg = np.bincount(ei_c[0], minlength=n)
    out.update(centroid_graph_undirected_edges=int(ei_c.shape[1] // 2), centroid_graph_mean_degree=float(deg.mean()),
               centroid_graph_isolated=int((deg == 0).sum()), centroid_edge_len_km_min=float(dist_c.min()),
               centroid_edge_len_km_max=float(dist_c.max()), centroid_edge_len_km_cv=float(dist_c.std() / dist_c.mean()))
    best = json.loads((RES / "E1_p2_best_params.json").read_text())
    best1 = json.loads((RES / "E3_p1_best_params.json").read_text())
    Xte, yte, rte = xy(TEST)
    per = rte.PERIOD.values
    for kind in ["GCN", "ResGCN", "GAT"]:
        ps, _, _, _ = run_nn_seeds(kind, best[kind], "binary", ei=ei_c, ew=ew_c)
        d2 = pd.DataFrame([p2_eval(yte, p, per) for p in ps])
        p1s, _, y1, r1 = run_nn_seeds(kind, best1[kind], "multi", target="Y_P1", ei=ei_c, ew=ew_c)
        mk = r1.Y_TOTAL.values > 0
        d1 = pd.DataFrame([p1_eval(y1[mk], p[mk].argmax(1)) for p in p1s])
        out[f"centroid_{kind}_p2_ap"] = float(d2.avg_precision.mean())
        out[f"centroid_{kind}_p1_macro_f1"] = float(d1.macro_f1.mean())
    save("E19_effective_n_and_centroid_graph", out)
    print(json.dumps(out, indent=1))


ALL.update({"E15": E15, "E16": E16, "E17": E17, "E18": E18, "E19": E19})

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(ALL)):
        t0 = time.time()
        print(f"\n===== {k}")
        ALL[k]()
        print(f"[{k} done in {time.time() - t0:.1f}s]")


