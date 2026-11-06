"""Figures for the Assessment 2 report (reads results/*.csv produced by run_experiments.py)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import prep

RES = prep.OUT / "results"
FIG = prep.OUT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY, INK, MUTED = "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9c8c3",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ecebe7", "grid.linewidth": 0.6, "axes.titleweight": "bold",
                     "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 150})
MODEL_COLOR = {"Persistence": GREY, "KDE": YELLOW, "LR": BLUE, "SVM": VIOLET, "RF": AQUA,
               "MLP": MAGENTA, "GCN": RED, "ResGCN": ORANGE, "GAT": GREEN}

D = prep.build_all()
CELLS, EI, DF = D["cells"], D["ei"], D["df"]


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------- pipeline / design diagrams
def box(ax, x, y, w, h, text, fc="#eef4fc", ec=BLUE, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    steps = [
        (0.1, "Raw CPD extract\n229,275 incidents\n(24 Aug 2025 – 23 Aug 2026)", "RO1"),
        (2.1, "Cleaning\n−2 duplicates, −40 no coords,\n−638 outside bounds\n→ 228,595", "RO1"),
        (4.1, "UTM 16N projection\n2.2 km grid → 157 cells\nrook graph (272 edges)", "RO1"),
        (6.1, "13 × 28-day periods\nlagged features (t−1…t−3)\n+ targets in period t", "RO2–RO3"),
        (8.1, "Chronological split\ntrain 3–8 · val 9–10\ntest 11–12", "RO4"),
    ]
    for x, t, ro in steps:
        box(ax, x, 3.7, 1.8, 1.25, t, fs=7)
        ax.text(x + 0.9, 3.55, ro, ha="center", va="top", fontsize=7, color=MUTED)
    for i in range(4):
        arrow(ax, steps[i][0] + 1.8, 4.32, steps[i + 1][0], 4.32)
    lanes = [
        (2.35, "Phase 1 (supervised)\ndominant crime type, 4 classes", "Persistence · KDE (per category)\nLR · SVM · RF · MLP · GCN · ResGCN", "#fdf1ea", ORANGE),
        (1.25, "Phase 2 (supervised)\nhigh recorded victimisation, top 20%", "Persistence · KDE · LR · SVM · RF\nMLP · GCN · ResGCN\n+ count regression (Poisson GLM, GCN)", "#e9f7f1", AQUA),
        (0.15, "Unsupervised\nDBSCAN hotspot zones", "k-distance ε choice · validation PAI\nstability (Jaccard)", "#f1effc", VIOLET),
    ]
    for y, title, body, fc, ec in lanes:
        box(ax, 0.1, y, 3.6, 0.95, title, fc=fc, ec=ec, bold=True, fs=7.5)
        box(ax, 3.9, y, 3.9, 0.95, body, fc="white", ec=ec, fs=7)
        arrow(ax, 3.7, y + 0.47, 3.9, y + 0.47)
    box(ax, 8.1, 0.15, 1.8, 3.15,
        "Evaluation\n\nvalidation-only tuning\n10 seeds\nspatial block bootstrap\nrolling origin (5 windows)\nspatial-block CV\nablations\n→ model selection\n→ application design",
        fc="#f7f7f5", ec=MUTED, fs=7)
    for y in (2.82, 1.72, 0.62):
        arrow(ax, 7.8, y, 8.1, y)
    arrow(ax, 9.0, 3.7, 9.0, 3.3)
    savefig(fig, "F01_pipeline")


def fig_periods():
    fig, ax = plt.subplots(figsize=(10, 1.9))
    cols = {"burn-in (features only)": "#e6e5e1", "train targets": BLUE, "validation": YELLOW, "test": RED}
    lab = ["burn-in (features only)"] * 3 + ["train targets"] * 6 + ["validation"] * 2 + ["test"] * 2
    env = json.loads((RES / "E13_environment.json").read_text())["periods"]
    for p in range(13):
        ax.barh(0, 0.94, left=p, color=cols[lab[p]], height=0.6)
        ax.text(p + 0.47, 0, str(p), ha="center", va="center", color="white" if lab[p] != "burn-in (features only)" else INK, fontsize=8, fontweight="bold")
        if p in (0, 3, 9, 11, 12):
            ax.text(p + 0.47, -0.55, env[str(p)][0][5:], ha="center", fontsize=7, color=MUTED)
    ax.annotate("", xy=(5, 0.45), xytext=(2, 0.45), arrowprops=dict(arrowstyle="<->", color=MUTED))
    ax.text(3.5, 0.62, "features for target period 5 use periods 2–4 only (+ all earlier totals)", ha="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.8, 0.9)
    ax.axis("off")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in cols.items()], ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.55), fontsize=8)
    savefig(fig, "F02_temporal_design")


# ------------------------------------------------------------- data figures
def fig_eda():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    p = DF[DF.PERIOD < 13].groupby("PERIOD").agg(total=("VICTIM", "size"), victim=("VICTIM", "sum"))
    ax = axs[0]
    ax.plot(p.index, p.total / 1000, color=BLUE, lw=2, marker="o", ms=4, label="all incidents")
    ax.plot(p.index, p.victim / 1000, color=ORANGE, lw=2, marker="o", ms=4, label="victimisation-related")
    ax.set_xlabel("28-day period")
    ax.set_ylabel("incidents (thousands)")
    ax.set_title("A. Incidents per period")
    ax.legend(fontsize=7)
    ax.set_ylim(0, None)
    c = DF.groupby("GRID_ID").size().sort_values(ascending=False).values
    ax = axs[1]
    ax.plot(np.arange(1, len(c) + 1) / len(c) * 100, np.cumsum(c) / c.sum() * 100, color=BLUE, lw=2)
    ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--")
    for q in (10, 20):
        k = int(round(q / 100 * len(c)))
        v = c[:k].sum() / c.sum() * 100
        ax.scatter([q], [v], color=BLUE, s=20, zorder=3)
        ax.text(q + 3, v - 6, f"top {q}% of cells\n= {v:.0f}% of incidents", fontsize=7, color=INK)
    ax.set_xlabel("% of occupied cells (ranked)")
    ax.set_ylabel("cumulative % of incidents")
    ax.set_title("B. Spatial concentration")
    s = D["snap"]
    ax = axs[2]
    vals = []
    for t in range(4, 13):
        a = s[s.PERIOD == t].set_index("GRID_ID").Y_VICTIM
        b = s[s.PERIOD == t - 1].set_index("GRID_ID").Y_VICTIM
        vals.append(pd.Series(a).rank().corr(pd.Series(b).reindex(a.index).rank()))
    ax.plot(range(4, 13), vals, color=AQUA, lw=2, marker="o", ms=5)
    ax.set_ylim(0.9, 1)
    ax.set_xlabel("period t")
    ax.set_ylabel("Spearman ρ (t vs t−1)")
    ax.set_title("C. Cell-rank persistence of victimisation")
    for x, v in zip(range(4, 13), vals):
        ax.text(x, v + 0.003, f"{v:.2f}", ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    savefig(fig, "F03_eda")
    return vals


def fig_graph():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ax = axs[0]
    x, y = CELLS.LON.values, CELLS.LAT.values
    for a, b in EI.T[: EI.shape[1] // 2]:
        ax.plot([x[a], x[b]], [y[a], y[b]], color="#c9c8c3", lw=0.7, zorder=1)
    sc = ax.scatter(x, y, c=np.log10(CELLS.N), cmap="Blues", s=38, edgecolor="white", lw=0.5, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("log10 incidents per cell (12 months)")
    ax.set_title("A. 157 occupied 2.2 km cells and rook graph (3 km radius)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.33)
    ax = axs[1]
    deg = np.bincount(EI[0], minlength=len(CELLS))
    ax.bar(*np.unique(deg, return_counts=True), color=BLUE, width=0.6)
    ax.set_xlabel("node degree (neighbours within 3 km)")
    ax.set_ylabel("cells")
    ax.set_title("B. Degree distribution (mean 3.46, all edges = 2.2 km)")
    ax.text(0.5, 0.95, "diagonal neighbours lie 3.11 km away → excluded;\ninverse-distance weights are therefore all 1/2.2",
            transform=ax.transAxes, fontsize=7.5, va="top", color=MUTED)
    fig.tight_layout()
    savefig(fig, "F04_grid_graph")


# ------------------------------------------------------------- results figures
def fig_p2_compare():
    r = pd.read_csv(RES / "E2_p2_test_runs.csv")
    b = pd.read_csv(RES / "E10_bootstrap.csv")
    names = {"Persistence (lagged counts)": "Persistence", "KDE (victimisation, prior 3 periods)": "KDE"}
    r["m"] = r.model.replace(names)
    order = ["Persistence", "KDE", "LR", "SVM", "RF", "MLP", "GCN", "ResGCN", "GAT"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axs[0]
    for i, m in enumerate(order):
        row = b[b.quantity == f"AP {m}"].iloc[0]
        ax.plot([row.lo, row.hi], [i, i], color=MODEL_COLOR[m], lw=2, solid_capstyle="round")
        ax.scatter(row.point, i, color=MODEL_COLOR[m], s=40, zorder=3, edgecolor="white")
        sd = r[r.m == m].avg_precision
        if len(sd) > 1:
            ax.scatter(sd, [i + 0.22] * len(sd), color=MODEL_COLOR[m], s=8, alpha=0.6)
        ax.text(row.hi + 0.004, i, f"{row.point:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0.8, 1.02)
    ax.invert_yaxis()
    ax.set_xlabel("average precision (test periods 11–12, n = 314; prevalence 0.21)\nlarge dot = point estimate, small dots = individual seeds")
    ax.set_title("A. Ranking quality with 95% spatial-block bootstrap CI")
    ax = axs[1]
    g = r.groupby("m").topk_macro_f1.agg(["mean", "std"]).reindex(order)
    ax.barh(range(len(order)), g["mean"], xerr=g["std"].fillna(0), color=[MODEL_COLOR[m] for m in order], height=0.6,
            error_kw=dict(lw=1, ecolor=MUTED))
    for i, v in enumerate(g["mean"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.axvline(0.70, color=RED, ls="--", lw=1)
    ax.text(0.705, 8.6, "RO3 criterion 0.70", fontsize=7, color=RED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0.6, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 (top-20% flagging rule per period)")
    ax.set_title("B. Decision quality (mean ± sd over seeds)")
    fig.tight_layout()
    savefig(fig, "F05_p2_comparison")


def fig_roc_pr():
    p = pd.read_csv(RES / "E2_p2_test_preds.csv")
    y = p.y.values
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for m in ["Persistence", "KDE", "LR", "RF", "GCN", "ResGCN"]:
        fpr, tpr, _ = roc_curve(y, p[m])
        axs[0].plot(fpr, tpr, color=MODEL_COLOR[m], lw=1.6, label=m)
        pr_, rc, _ = precision_recall_curve(y, p[m])
        axs[1].plot(rc, pr_, color=MODEL_COLOR[m], lw=1.6, label=m)
    axs[0].plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    axs[0].set_xlim(0, 0.4)
    axs[0].set_ylim(0.6, 1.01)
    axs[0].set_xlabel("false-positive rate (zoomed)")
    axs[0].set_ylabel("true-positive rate")
    axs[0].set_title("A. ROC (test)")
    axs[0].legend(fontsize=7, loc="lower right")
    axs[1].axhline(y.mean(), color=GREY, ls=":", lw=1)
    axs[1].set_ylim(0.5, 1.01)
    axs[1].set_xlabel("recall")
    axs[1].set_ylabel("precision")
    axs[1].set_title("B. Precision–recall (test)")
    for m in ["LR", "RF", "MLP", "GCN", "ResGCN"]:
        ft, mp = calibration_curve(y, p[m], n_bins=8, strategy="quantile")
        axs[2].plot(mp, ft, color=MODEL_COLOR[m], marker="o", ms=4, lw=1.4, label=m)
    axs[2].plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    axs[2].set_xlabel("mean predicted probability")
    axs[2].set_ylabel("observed share high")
    axs[2].set_title("C. Reliability (8 quantile bins)")
    axs[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    savefig(fig, "F06_roc_pr_calibration")


def cm_plot(ax, cm, labels, title, cmap="Blues"):
    ax.imshow(cm, cmap=cmap)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=9)


def topk(score, per, n=157):
    pred = np.zeros(len(score), int)
    k = int(np.ceil(0.2 * n))
    for t in np.unique(per):
        idx = np.where(per == t)[0]
        pred[idx[np.argsort(-score[idx])[:k]]] = 1
    return pred


def fig_confusions():
    p = pd.read_csv(RES / "E2_p2_test_preds.csv")
    q = pd.read_csv(RES / "E3_p1_test_preds.csv")
    lab1 = ["BATTERY", "CRIM. DAMAGE", "THEFT", "OTHER TYPE"]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.3))
    for ax, m in zip(axs[:2], ["LR", "ResGCN"]):
        cm = confusion_matrix(p.y, topk(p[m].values, p.PERIOD.values))
        cm_plot(ax, cm, ["low", "high"], f"Phase 2 · {m} (top-20%)")
    for ax, m in zip(axs[2:], ["Persistence", "LR"]):
        cm = confusion_matrix(q.y, q[m], labels=range(4))
        cm_plot(ax, cm, lab1, f"Phase 1 · {m}", cmap="Oranges")
    fig.tight_layout()
    savefig(fig, "F07_confusion")


def fig_curves():
    g = pd.read_csv(RES / "curve_gcn_p2.csv")
    m = pd.read_csv(RES / "curve_mlp_p2.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for d, name, c in [(g, "GCN", RED), (m, "MLP", MAGENTA)]:
        axs[0].plot(d.epoch, d.train_loss, color=c, lw=1.4, label=f"{name} train")
        axs[0].plot(d.epoch, d.val_loss, color=c, lw=1.4, ls="--", label=f"{name} validation")
        axs[1].plot(d.epoch, d.val_ap, color=c, lw=1.4, label=name)
        be = d.val_ap.idxmax()
        axs[1].scatter(d.epoch[be], d.val_ap[be], color=c, s=25, zorder=3)
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("weighted cross-entropy")
    axs[0].set_title("A. Loss (Phase 2, seed 0)")
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("validation average precision")
    axs[1].set_title("B. Checkpoint selection (dot = restored epoch)")
    axs[1].set_ylim(0.8, 1)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "F08_training_curves")


def fig_ablation():
    a = pd.read_csv(RES / "E5_graph_ablation.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, col, sdcol, title, xl in [(axs[0], "p2_ap", "p2_ap_sd", "A. Phase 2 average precision", (0.7, 1.0)),
                                       (axs[1], "p1_macro_f1", "p1_macro_f1_sd", "B. Phase 1 macro-F1", (0.3, 0.55))]:
        sub = a[a.model.isin(["GCN", "ResGCN"])].copy()
        labels = sub.variant.unique()
        for k, (kind, c) in enumerate([("GCN", RED), ("ResGCN", ORANGE)]):
            s = sub[sub.model == kind].set_index("variant").reindex(labels)
            ax.barh(np.arange(len(labels)) + (k - 0.5) * 0.36, s[col], xerr=s[sdcol], height=0.34, color=c,
                    label=kind, error_kw=dict(lw=0.8, ecolor=MUTED))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(*xl)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
    lr = a[a.model == "LR"]
    for _, row in lr.iterrows():
        pass
    mlp_ap = pd.read_csv(RES / "E2_p2_test_runs.csv").query("model == 'MLP'").avg_precision.mean()
    axs[0].axvline(mlp_ap, color=MAGENTA, ls="--", lw=1)
    axs[0].text(mlp_ap - 0.002, -0.7, "MLP (no graph)", fontsize=7, color=MAGENTA, ha="right")
    mlp_f1 = pd.read_csv(RES / "E3_p1_test_runs.csv").query("model == 'MLP'").macro_f1.mean()
    axs[1].axvline(mlp_f1, color=MAGENTA, ls="--", lw=1)
    axs[1].text(mlp_f1 + 0.002, -0.7, "MLP (no graph)", fontsize=7, color=MAGENTA)
    fig.tight_layout()
    savefig(fig, "F09_graph_ablation")


def fig_rolling():
    r = pd.read_csv(RES / "E8_rolling_origin.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for m in ["Persistence", "KDE", "LR", "GCN", "ResGCN"]:
        s = r[(r.task == "P2") & (r.model == m)]
        axs[0].plot(s.test_period, s.ap, marker="o", ms=4, lw=1.6, color=MODEL_COLOR[m], label=m)
    axs[0].set_xlabel("test period (train on all earlier periods)")
    axs[0].set_ylabel("average precision")
    axs[0].set_title("A. Phase 2 rolling-origin evaluation")
    axs[0].legend(fontsize=7)
    for m in ["Persistence", "LR", "GCN"]:
        s = r[(r.task == "P1") & (r.model == m)]
        axs[1].plot(s.test_period, s.macro_f1, marker="o", ms=4, lw=1.6, color=MODEL_COLOR[m], label=m)
    axs[1].set_xlabel("test period")
    axs[1].set_ylabel("macro-F1")
    axs[1].set_title("B. Phase 1 rolling-origin evaluation")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "F10_rolling_origin")


def fig_error_map():
    p = pd.read_csv(RES / "E2_p2_test_preds.csv")
    q = pd.read_csv(RES / "E3_p1_test_preds.csv")
    c = CELLS.set_index("GRID_ID")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.2))
    sub = p[p.PERIOD == 12].copy()
    sub["pred"] = topk(sub.LR.values, sub.PERIOD.values)
    kind = np.select([(sub.y == 1) & (sub.pred == 1), (sub.y == 0) & (sub.pred == 1), (sub.y == 1) & (sub.pred == 0)],
                     ["true positive", "false positive", "false negative"], "true negative")
    cmap = {"true positive": BLUE, "false positive": YELLOW, "false negative": RED, "true negative": "#dcdbd6"}
    ax = axs[0]
    for k, col in cmap.items():
        g = sub[kind == k]
        ax.scatter(c.loc[g.GRID_ID].LON, c.loc[g.GRID_ID].LAT, s=46, color=col, label=f"{k} ({len(g)})",
                   edgecolor="white", lw=0.5, marker="s")
    ax.set_title("A. Phase 2, LR, test period 12 (top-20% rule)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_aspect(1.33)
    ax = axs[1]
    err = q.assign(err=(q.LR != q.y).astype(int)).groupby("GRID_ID").err.mean()
    sc = ax.scatter(c.loc[err.index].LON, c.loc[err.index].LAT, c=err.values, cmap="Oranges", s=46, marker="s",
                    edgecolor="white", lw=0.5, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, shrink=0.7).set_label("share of test periods misclassified")
    ax.set_title("B. Phase 1, LR, error rate per cell (periods 11–12)")
    ax.set_aspect(1.33)
    for a in axs:
        a.set_xlabel("longitude")
        a.set_ylabel("latitude")
    fig.tight_layout()
    savefig(fig, "F11_error_maps")


def fig_dbscan():
    pts = pd.read_csv(RES / "E12_dbscan_points.csv")
    pai = pd.read_csv(RES / "E12_pai_comparison.csv")
    fl = set(pd.read_csv(RES / "E12_dbscan_flagged_cells.csv").GRID_ID)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    noise = pts[pts.L < 0]
    ax.scatter(noise.X_M / 1000, noise.Y_M / 1000, s=0.3, color="#d4d3ce", rasterized=True)
    cl = pts[pts.L >= 0]
    ax.scatter(cl.X_M / 1000, cl.Y_M / 1000, s=0.4, color=VIOLET, rasterized=True)
    for _, r in CELLS[CELLS.GRID_ID.isin(fl)].iterrows():
        ax.add_patch(plt.Rectangle(((r.CX - 1100) / 1000, (r.CY - 1100) / 1000), 2.2, 2.2, fill=False, ec=ORANGE, lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlabel("UTM easting (km)")
    ax.set_ylabel("UTM northing (km)")
    ax.set_title("A. DBSCAN clusters, periods 8–10 (orange = flagged cells)")
    ax = axs[1]
    pai = pai.sort_values("PAI")
    cols = [VIOLET if "DBSCAN" in m else MODEL_COLOR.get(m, GREY) for m in pai.method]
    ax.barh(range(len(pai)), pai.PAI, color=cols, height=0.6)
    ax.set_yticks(range(len(pai)))
    ax.set_yticklabels([m.replace(" (", "\n(") for m in pai.method], fontsize=7)
    for i, (v, h, a) in enumerate(zip(pai.PAI, pai.hit_rate, pai.area_share)):
        ax.text(v + 0.03, i, f"PAI {v:.2f}  (hit {h:.0%}, area {a:.0%})", va="center", fontsize=7)
    ax.set_xlim(0, pai.PAI.max() * 1.45)
    ax.set_xlabel("Predictive Accuracy Index on test periods 11–12")
    ax.set_title("B. Hotspot efficiency")
    fig.tight_layout()
    savefig(fig, "F12_dbscan_pai")


def fig_sensitivity():
    s = pd.read_csv(RES / "E7_sensitivity.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.1))
    for ax, exp, xl in [(axs[0], "label percentile", "label percentile"), (axs[1], "grid size (m)", "grid cell size (m)")]:
        sub = s[s.experiment == exp]
        for m in ["Persistence", "LR", "GCN", "ResGCN"]:
            q = sub[sub.model == m]
            ax.plot(q.setting, q.ap, marker="o", ms=4, lw=1.6, color=MODEL_COLOR[m], label=m)
        ax.set_xlabel(xl)
        ax.set_ylabel("average precision (test)")
        ax.legend(fontsize=7)
        ax.set_xticks(sub.setting.unique())
    axs[0].set_title("A. High-risk threshold")
    n = s[s.experiment == "grid size (m)"].drop_duplicates("setting")
    axs[1].set_title("B. Grid resolution (" + ", ".join(f"{int(a)} m: {int(b)} cells" for a, b in zip(n.setting, n.n_cells)) + ")", fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "F13_sensitivity")


def fig_multi():
    RM = prep.OUT / "results_multi"
    r = pd.read_csv(RM / "E2_p2_test_runs.csv")
    r["m"] = r.model.replace({"Persistence (lagged counts)": "Persistence", "KDE (victimisation, prior 3 periods)": "KDE"})
    b = pd.read_csv(RM / "E10_bootstrap.csv")
    r1 = pd.read_csv(RES / "E2_p2_test_runs.csv")
    r1["m"] = r1.model.replace({"Persistence (lagged counts)": "Persistence", "KDE (victimisation, prior 3 periods)": "KDE"})
    order = ["Persistence", "KDE", "LR", "SVM", "RF", "MLP", "GCN", "ResGCN", "GAT"]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axs[0]
    for i, m in enumerate(order):
        row = b[b.quantity == f"AP {m}"].iloc[0]
        ax.plot([row.lo, row.hi], [i, i], color=MODEL_COLOR[m], lw=2, solid_capstyle="round")
        ax.scatter(row.point, i, color=MODEL_COLOR[m], s=40, zorder=3, edgecolor="white")
        ax.text(row.hi + 0.004, i, f"{row.point:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0.78, 1.02)
    ax.set_xlabel("average precision, 6 test periods (95% block-bootstrap CI)")
    ax.set_title("A. Three-year evaluation")
    ax = axs[1]
    x = r1.groupby("m").avg_precision.mean()
    y = r.groupby("m").avg_precision.mean()
    ax.plot([0.8, 1], [0.8, 1], color=GREY, ls=":", lw=1)
    for m in order:
        ax.scatter(x[m], y[m], color=MODEL_COLOR[m], s=45, zorder=3, edgecolor="white", label=m)
    ax.set_xlim(0.84, 1.0)
    ax.set_ylim(0.84, 1.0)
    ax.set_xlabel("AP, one-year study (2 test periods)")
    ax.set_ylabel("AP, three-year study (6 test periods)")
    ax.set_title("B. Does the ranking replicate?")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax = axs[2]
    rol = pd.read_csv(RM / "E8_rolling_origin.csv")
    for m in ["Persistence", "KDE", "LR", "GCN", "ResGCN"]:
        sub = rol[(rol.task == "P2") & (rol.model == m)]
        ax.plot(sub.test_period, sub.ap, marker="o", ms=4, lw=1.6, color=MODEL_COLOR[m], label=m)
    ax.set_xlabel("test period (retrained on all earlier periods)")
    ax.set_ylabel("average precision")
    ax.set_title("C. Rolling origin, three-year data")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "F15_multiyear")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    box(ax, 0.1, 3.3, 1.9, 1.0, "Chicago Data Portal\n(Socrata API)\nnightly pull", fc="#f7f7f5", ec=MUTED)
    box(ax, 2.4, 3.3, 2.2, 1.0, "Ingest & validate\nschema · bounds · dedup\n(versioned preprocessing)", fs=7.5)
    box(ax, 5.0, 3.3, 2.2, 1.0, "Feature store\ncell × 28-day period\nlags · shares · graph v1")
    box(ax, 7.6, 3.3, 2.3, 1.0, "Batch scoring (every 28 days)\nLR champion · RF / ResGCN\nchallengers · DBSCAN layer", fc="#e9f7f1", ec=AQUA, fs=7.5)
    for x1, x2 in [(2.0, 2.4), (4.6, 5.0), (7.2, 7.6)]:
        arrow(ax, x1, 3.8, x2, 3.8)
    box(ax, 7.6, 1.75, 2.3, 1.1, "Prediction store\nscore · rank · CI band\nmodel + data version", fc="#eef4fc")
    arrow(ax, 8.75, 3.3, 8.75, 2.85)
    box(ax, 5.0, 1.75, 2.2, 1.1, "REST API (read-only)\nGET /v1/cells/risk\nGET /v1/cells/{id}/history", fc="#eef4fc")
    arrow(ax, 7.6, 2.3, 7.2, 2.3)
    box(ax, 2.4, 1.75, 2.2, 1.1, "Analyst dashboard\nmap · ranked list · history\nuncertainty · explanations", fc="#fdf1ea", ec=ORANGE)
    arrow(ax, 5.0, 2.3, 4.6, 2.3)
    box(ax, 0.1, 1.75, 1.9, 1.1, "Human review\nplanning meeting\n(decision support\nonly)", fc="#f7f7f5", ec=MUTED, fs=7.5)
    arrow(ax, 2.4, 2.3, 2.0, 2.3)
    box(ax, 2.4, 0.2, 7.5, 1.05, "Governance & MLOps\nregistry (model, scaler, feature schema, graph version) · drift monitors (feature PSI, label prevalence)\n"
        "delayed-label evaluation · retrain every 3 periods or on drift alert · champion/challenger gate · rollback\n"
        "role-based access + SSO · audit log of queries and overrides · quarterly bias / feedback-loop review", fc="#f1effc", ec=VIOLET, fs=6.8)
    arrow(ax, 6.0, 1.25, 6.0, 1.75)
    box(ax, 0.1, 0.2, 1.9, 1.05, "Feedback\nanalyst flags,\noverrides", fc="#f7f7f5", ec=MUTED)
    arrow(ax, 2.0, 0.72, 2.4, 0.72)
    savefig(fig, "F14_architecture")


if __name__ == "__main__":
    import sys
    fns = {k: v for k, v in globals().items() if k.startswith("fig_")}
    for k in (sys.argv[1:] or fns):
        print(k)
        out = fns[k]()
        if out is not None:
            print(out)
