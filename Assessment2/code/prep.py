"""Data preparation for IT8105 Assessment 2.

Rebuilds the cleaned incident table, the 2.2 km grid, the spatial graph and
the leakage-safe cell-period panels used by every experiment.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "machinelearningcode/Coding/Dataset/Crimes_-_One_year_prior_to_present_20260831.csv"
OUT = ROOT / "Assessment2"

LAT_RANGE = (41.6, 42.1)
LON_RANGE = (-87.9, -87.5)
PERIOD_DAYS = 28
N_PERIODS = 13
N_LAGS = 3

# Person-directed offence categories (recorded incidents, not unique victims).
VICTIM_CATS = [
    "ASSAULT", "BATTERY", "ROBBERY", "CRIMINAL SEXUAL ASSAULT", "SEX OFFENSE",
    "STALKING", "HOMICIDE", "KIDNAPPING", "OFFENSE INVOLVING CHILDREN",
    "INTIMIDATION", "HUMAN TRAFFICKING",
]
# Crime categories with < 1,000 incidents are grouped into OTHER (pilot rule, Zubair et al.).
# Phase 1 label classes: a category is kept as its own class only if it is the dominant
# category in >= 5% of TRAINING cell-periods (periods 3-8); all others become OTHER_TYPE.
# The rule was applied once to the training periods (see p1_class_rule()).
RARE_MIN = 1000
P1_KEEP = ["BATTERY", "CRIMINAL DAMAGE", "THEFT"]
P1_CLASSES = P1_KEEP + ["OTHER_TYPE"]


def read_raw():
    """Read the one-year CPD extract with normalised column names."""
    df = pd.read_csv(RAW)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["DATE_OF_OCCURRENCE"] = pd.to_datetime(df.DATE_OF_OCCURRENCE, format="%m/%d/%Y %I:%M:%S %p")
    return df


def clean(df):
    """Clean a raw incident frame (DATE_OF_OCCURRENCE already parsed); returns (frame, count log)."""
    log = [("Raw records", len(df))]
    df = df.drop_duplicates()
    log.append(("After exact-duplicate removal", len(df)))
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    log.append(("After missing-coordinate removal", len(df)))
    inside = df.LATITUDE.between(*LAT_RANGE) & df.LONGITUDE.between(*LON_RANGE)
    df = df[inside].copy()
    log.append(("After study-boundary filter", len(df)))
    df["DATE"] = df.DATE_OF_OCCURRENCE
    df["LOCATION_DESCRIPTION"] = df.LOCATION_DESCRIPTION.fillna("UNKNOWN")
    df["VICTIM"] = df.PRIMARY_DESCRIPTION.isin(VICTIM_CATS).astype(int)
    df["ARREST_F"] = df.ARREST.astype(str).str.upper().eq("Y").astype(int)
    df["DOMESTIC_F"] = df.DOMESTIC.astype(str).str.upper().eq("Y").astype(int)
    vc = df.PRIMARY_DESCRIPTION.value_counts()
    df["CAT"] = df.PRIMARY_DESCRIPTION.where(df.PRIMARY_DESCRIPTION.isin(vc[vc >= RARE_MIN].index), "OTHER")
    tr = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)
    df["X_M"], df["Y_M"] = tr.transform(df.LONGITUDE.values, df.LATITUDE.values)
    start = df.DATE.min().normalize()
    df["PERIOD"] = ((df.DATE.dt.normalize() - start).dt.days // PERIOD_DAYS).astype(int)
    return df, log


def load_clean():
    """Return the cleaned incident table and a stage-by-stage count log (one-year extract)."""
    return clean(read_raw())


def build_grid(df, cell_m=2200.0):
    """Assign incidents to square cells; returns cells table (occupied cells only)."""
    min_x, min_y = df.X_M.min(), df.Y_M.min()
    df = df.copy()
    df["COL"] = np.floor((df.X_M - min_x) / cell_m).astype(int)
    df["ROW"] = np.floor((df.Y_M - min_y) / cell_m).astype(int)
    df["GRID_ID"] = df.ROW.astype(str) + "_" + df.COL.astype(str)
    cells = (df.groupby(["GRID_ID", "ROW", "COL"]).size().rename("N").reset_index()
             .sort_values(["ROW", "COL"]).reset_index(drop=True))
    cells["CX"] = min_x + (cells.COL + 0.5) * cell_m
    cells["CY"] = min_y + (cells.ROW + 0.5) * cell_m
    back = Transformer.from_crs("EPSG:26916", "EPSG:4326", always_xy=True)
    cells["LON"], cells["LAT"] = back.transform(cells.CX.values, cells.CY.values)
    cells["NODE"] = np.arange(len(cells))
    return df, cells, (min_x, min_y)


def build_graph(cells, radius_m=3000.0, weighting="inverse"):
    """Undirected radius graph between cell centres. Returns (edge_index, edge_weight, dist)."""
    xy = cells[["CX", "CY"]].values
    pairs = cKDTree(xy).query_pairs(r=radius_m, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((2, 0), int), np.zeros(0), np.zeros(0)
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1) / 1000.0
    w = 1.0 / d if weighting == "inverse" else np.ones_like(d)
    ei = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)
    return ei, np.concatenate([w, w]), np.concatenate([d, d])


def period_panel(df, cells):
    """Cell x period aggregates (dense: zero-filled for empty cell-periods)."""
    df = df[df.PERIOD < N_PERIODS]
    idx = pd.MultiIndex.from_product([cells.GRID_ID, range(N_PERIODS)], names=["GRID_ID", "PERIOD"])
    g = df.groupby(["GRID_ID", "PERIOD"])
    agg = pd.DataFrame({
        "TOTAL": g.size(),
        "VICTIM": g.VICTIM.sum(),
        "DOMESTIC": g.DOMESTIC_F.sum(),
        "ARREST": g.ARREST_F.sum(),
        "NIGHT": g.DATE.apply(lambda s: ((s.dt.hour >= 20) | (s.dt.hour < 6)).sum()),
        "UNIQ_TYPES": g.PRIMARY_DESCRIPTION.nunique(),
    }).reindex(idx, fill_value=0)
    comp = (df.groupby(["GRID_ID", "PERIOD", "CAT"]).size().unstack(fill_value=0)
            .reindex(idx, fill_value=0))
    # order categories by overall frequency so argmax ties resolve to the more common category
    comp = comp[comp.sum().sort_values(ascending=False).index]
    comp.columns = ["C_" + c for c in comp.columns]
    return agg.join(comp).reset_index()


def to_p1_class(cat_names):
    return np.array([P1_CLASSES.index(c) if c in P1_KEEP else len(P1_KEEP) for c in cat_names])


def p1_class_rule(panel, train_periods=range(3, 9), min_share=0.05):
    """Training-only rule that produced P1_KEEP: dominant-category share per training cell-period."""
    ccols = [c for c in panel.columns if c.startswith("C_")]
    p = panel[panel.PERIOD.isin(train_periods) & (panel.TOTAL > 0)]
    dom = pd.Series(np.array(ccols)[p[ccols].values.argmax(1)]).str[2:]
    share = dom.value_counts(normalize=True)
    return share, sorted(share[share >= min_share].index)


def feature_snapshots(panel, cells, edge_index, edge_weight):
    """Leakage-safe features for target period t using only periods < t.

    Returns a long table with one row per (cell, target period t >= N_LAGS).
    """
    P = panel.set_index(["GRID_ID", "PERIOD"]).sort_index()
    order = cells.GRID_ID.values
    n = len(order)
    # row-normalised neighbour averaging operator for the hand-crafted spatial-lag variant
    W = np.zeros((n, n))
    W[edge_index[0], edge_index[1]] = edge_weight
    rs = W.sum(1, keepdims=True)
    W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    rows = []
    for t in range(N_LAGS, N_PERIODS):
        f = pd.DataFrame({"GRID_ID": order, "PERIOD": t})
        for k in range(1, N_LAGS + 1):
            sl = P.xs(t - k, level="PERIOD").reindex(order)
            f[f"VICTIM_L{k}"] = sl.VICTIM.values
            f[f"TOTAL_L{k}"] = sl.TOTAL.values
        hist = P[P.index.get_level_values("PERIOD") < t].groupby(level="GRID_ID").sum().reindex(order)
        nh = t
        f["VICTIM_MEAN_ALL"] = hist.VICTIM.values / nh
        f["TOTAL_MEAN_ALL"] = hist.TOTAL.values / nh
        f["VICTIM_SHARE"] = hist.VICTIM.values / np.maximum(hist.TOTAL.values, 1)
        f["DOMESTIC_SHARE"] = hist.DOMESTIC.values / np.maximum(hist.TOTAL.values, 1)
        f["ARREST_SHARE"] = hist.ARREST.values / np.maximum(hist.TOTAL.values, 1)
        f["NIGHT_SHARE"] = hist.NIGHT.values / np.maximum(hist.TOTAL.values, 1)
        last3 = P[(P.index.get_level_values("PERIOD") < t) &
                  (P.index.get_level_values("PERIOD") >= t - N_LAGS)].groupby(level="GRID_ID").sum().reindex(order)
        for c in SHARE_CATS:
            f[f"SHARE3_{c.replace(' ', '_')}"] = last3[f"C_{c}"].values / np.maximum(last3.TOTAL.values, 1)
        f["VICTIM_TREND"] = f.VICTIM_L1 - f.VICTIM_L3
        f["LAT"] = cells.LAT.values
        f["LON"] = cells.LON.values
        # spatial lag of recent victimisation (used only by the "+spatial lag" tabular variant)
        f["SLAG_VICTIM_L1"] = (W * f.VICTIM_L1.values[None, :].astype(float)).sum(1)
        f["SLAG_VICTIM_MEAN"] = (W * f.VICTIM_MEAN_ALL.values[None, :]).sum(1)
        # persistence (historical dominant category over last 3 periods, mapped to P1 classes)
        ccols = [c for c in P.columns if c.startswith("C_")]
        f["PERSIST_P1"] = to_p1_class([c[2:] for c in np.array(ccols)[last3[ccols].values.argmax(1)]])
        # targets for period t
        cur = P.xs(t, level="PERIOD").reindex(order)
        f["Y_VICTIM"] = cur.VICTIM.values
        f["Y_TOTAL"] = cur.TOTAL.values
        f["Y_P1"] = to_p1_class([c[2:] for c in np.array(ccols)[cur[ccols].values.argmax(1)]])
        f["NODE"] = np.arange(n)
        rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    return out


SHARE_CATS = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "MOTOR VEHICLE THEFT",
              "OTHER OFFENSE", "DECEPTIVE PRACTICE", "BURGLARY"]
BASE_FEATS = (["VICTIM_L1", "VICTIM_L2", "VICTIM_L3", "TOTAL_L1", "TOTAL_L2", "TOTAL_L3",
               "VICTIM_MEAN_ALL", "TOTAL_MEAN_ALL", "VICTIM_SHARE", "DOMESTIC_SHARE",
               "ARREST_SHARE", "NIGHT_SHARE", "VICTIM_TREND", "LAT", "LON"]
              + [f"SHARE3_{c.replace(' ', '_')}" for c in SHARE_CATS])
SLAG_FEATS = ["SLAG_VICTIM_L1", "SLAG_VICTIM_MEAN"]


def label_high(snap, pct=80):
    """Per-period relative label: cell is 'high' if its period count >= the period's pct-th percentile."""
    thr = snap.groupby("PERIOD").Y_VICTIM.transform(lambda s: np.percentile(s, pct))
    return (snap.Y_VICTIM >= thr).astype(int).values


def build_all(cell_m=2200.0, radius_m=3000.0, weighting="inverse", loader=None):
    df, log = (loader or load_clean)()
    df, cells, origin = build_grid(df, cell_m)
    ei, ew, dist = build_graph(cells, radius_m, weighting)
    panel = period_panel(df, cells)
    snap = feature_snapshots(panel, cells, ei, ew)
    return dict(df=df, log=log, cells=cells, origin=origin, ei=ei, ew=ew, dist=dist,
                panel=panel, snap=snap)


if __name__ == "__main__":
    D = build_all()
    for k, v in D["log"]:
        print(f"{k:35s}{v:>9,}")
    c, ei = D["cells"], D["ei"]
    print("cells", len(c), "directed edges", ei.shape[1], "undirected", ei.shape[1] // 2)
    deg = np.bincount(ei[0], minlength=len(c))
    print("isolated", (deg == 0).sum(), "deg mean", deg.mean(), "unique dists", np.unique(D["dist"].round(3)))
    s = D["snap"]
    print(s.shape)
    print(prep_rule := p1_class_rule(D["panel"]))
    print(pd.crosstab(s.PERIOD, s.Y_P1[s.Y_TOTAL > 0]))
    print(s.groupby("PERIOD").Y_VICTIM.describe())
    s["H"] = label_high(s)
    print(s.groupby("PERIOD").H.mean())
    print("persistence P1 acc", (s.PERSIST_P1 == s.Y_P1).mean())
    print("period range", D["df"].groupby("PERIOD").DATE.agg(["min", "max", "size"]))
