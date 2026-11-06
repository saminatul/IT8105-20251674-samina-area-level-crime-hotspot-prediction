"""Model wrappers shared by all experiments (tabular baselines, KDE, MLP and GCN)."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, cohen_kappa_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from torch_geometric.nn import GATConv, GCNConv

torch.set_num_threads(4)

COUNT_FEATS = ["VICTIM_L1", "VICTIM_L2", "VICTIM_L3", "TOTAL_L1", "TOTAL_L2", "TOTAL_L3",
               "VICTIM_MEAN_ALL", "TOTAL_MEAN_ALL", "SLAG_VICTIM_L1", "SLAG_VICTIM_MEAN"]


def transform(X, cols):
    """log1p for skewed counts, signed log for the trend; shares and coordinates unchanged."""
    X = X.astype(float).copy()
    for j, c in enumerate(cols):
        if c in COUNT_FEATS:
            X[:, j] = np.log1p(X[:, j])
        elif c == "VICTIM_TREND":
            X[:, j] = np.sign(X[:, j]) * np.log1p(np.abs(X[:, j]))
    return X


class Prepper:
    """Fits the transform + scaler on training rows only.

    mode: 'main' = log transform + z-score (used in all main experiments); 'raw' = no processing;
    'std' = z-score only; 'log' = log transform only; 'robust' = log transform + median/IQR scaling.
    """

    def __init__(self, cols, mode="main"):
        self.cols = cols
        self.mode = mode
        self.sc = {"main": StandardScaler, "std": StandardScaler, "robust": RobustScaler}.get(mode, lambda: None)()

    def _t(self, X):
        return transform(X, self.cols) if self.mode in ("main", "log", "robust") else np.asarray(X, dtype=float)

    def fit(self, X):
        if self.sc is not None:
            self.sc.fit(self._t(X))
        return self

    def __call__(self, X):
        Z = self._t(X)
        return self.sc.transform(Z) if self.sc is not None else Z


# ---------------------------------------------------------------- metrics
def binary_metrics(y, score, thr):
    pred = (score >= thr).astype(int)
    k = max(1, int(round(y.mean() * len(y))))
    top = np.argsort(-score)[:k]
    out = dict(
        pos_f1=f1_score(y, pred, zero_division=0),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        accuracy=accuracy_score(y, pred),
        balanced_acc=balanced_accuracy_score(y, pred),
        roc_auc=roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        avg_precision=average_precision_score(y, score) if y.sum() > 0 else np.nan,
        prec_at_k=y[top].mean(),
        n=len(y), n_pos=int(y.sum()), n_flagged=int(pred.sum()),
    )
    if score.min() >= 0 and score.max() <= 1:
        out["brier"] = brier_score_loss(y, score)
    return out


def multiclass_metrics(y, pred, n_classes):
    out = dict(
        accuracy=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0),
        weighted_f1=f1_score(y, pred, average="weighted", labels=range(n_classes), zero_division=0),
        balanced_acc=balanced_accuracy_score(y, pred),
        kappa=cohen_kappa_score(y, pred),
        n=len(y),
    )
    f1s = f1_score(y, pred, average=None, labels=range(n_classes), zero_division=0)
    for c, v in enumerate(f1s):
        out[f"f1_c{c}"] = v
    return out


def best_threshold(y, score):
    """Threshold maximising positive-class F1 on validation data."""
    cands = np.unique(score)
    best, bt = -1, 0.5
    for t in cands:
        f = f1_score(y, (score >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, t
    return bt


# ---------------------------------------------------------------- tabular
def make_tabular(name, params, task, seed=0, class_weight="balanced"):
    if name == "LR":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=5000, class_weight=class_weight)
    if name == "SVM":
        return SVC(C=params.get("C", 1.0), gamma=params.get("gamma", "scale"), kernel="rbf",
                   class_weight=class_weight, probability=False, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, max_depth=params.get("max_depth"),
                                      min_samples_leaf=params.get("min_samples_leaf", 1),
                                      class_weight="balanced_subsample" if class_weight else None,
                                      random_state=seed, n_jobs=4)
    raise ValueError(name)


def tabular_scores(model, X):
    if hasattr(model, "predict_proba") and not isinstance(model, SVC):
        p = model.predict_proba(X)
        return p
    return model.decision_function(X)


# ---------------------------------------------------------------- KDE
def kde_cell_scores(points_xy, cell_xy, bandwidth_m):
    """Gaussian KDE fitted on incident coordinates (metres), evaluated at cell centres."""
    if len(points_xy) == 0:
        return np.zeros(len(cell_xy))
    kde = KernelDensity(bandwidth=bandwidth_m, kernel="gaussian").fit(points_xy)
    return np.exp(kde.score_samples(cell_xy))


# ---------------------------------------------------------------- graph models
class GCN(nn.Module):
    def __init__(self, d_in, hidden, d_out, layers=2, dropout=0.5):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [d_out]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, ei, ew):
        for i, conv in enumerate(self.convs):
            x = conv(x, ei, ew)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, self.training)
        return x


class ResGCN(nn.Module):
    """GCN layer plus a separate linear self path: H' = relu(A_hat H W_n + H W_s).

    Motivated by the normalised propagation weight on a node's own features, 1/(d_i+1),
    which is ~0.22 on the rook graph; the self path lets the model keep cell-specific signal.
    """

    def __init__(self, d_in, hidden, d_out, layers=2, dropout=0.5):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [d_out]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.selfs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, ei, ew):
        for i, (conv, lin) in enumerate(zip(self.convs, self.selfs)):
            x = conv(x, ei, ew) + lin(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, self.training)
        return x


class GAT(nn.Module):
    """Two-layer graph attention network (learned neighbour weights; edge weights are not used)."""

    def __init__(self, d_in, hidden, d_out, layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.c1 = GATConv(d_in, max(hidden // heads, 4), heads=heads, dropout=dropout)
        self.c2 = GATConv(max(hidden // heads, 4) * heads, d_out, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, ei, ew=None):
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.c1(x, ei))
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, ei)


class MLP(nn.Module):
    """Same depth/width as the GCN but without neighbourhood aggregation (graph ablation)."""

    def __init__(self, d_in, hidden, d_out, layers=2, dropout=0.5):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [d_out]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, ei=None, ew=None):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, self.training)
        return x


def batch_graph(n_nodes, n_snap, ei, ew):
    """Block-diagonal graph: one copy of the spatial graph per period snapshot."""
    eis = [ei + k * n_nodes for k in range(n_snap)]
    return (torch.tensor(np.concatenate(eis, 1), dtype=torch.long),
            torch.tensor(np.tile(ew, n_snap), dtype=torch.float32))


def train_graph_model(kind, Xtr, ytr, Xva, yva, ei, ew, n_nodes, task, n_classes=2,
                      hidden=64, layers=2, dropout=0.5, lr=0.01, wd=5e-4, epochs=400,
                      patience=60, seed=0, class_weight=True, train_mask=None, val_mask=None,
                      return_curve=False):
    """Train GCN/MLP on stacked period snapshots. X arrays are (n_snap*n_nodes, d), ordered by
    period then node. Optional masks hide node labels (spatial-block experiments)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ntr, nva = len(Xtr) // n_nodes, len(Xva) // n_nodes
    eitr, ewtr = batch_graph(n_nodes, ntr, ei, ew)
    eiva, ewva = batch_graph(n_nodes, nva, ei, ew)
    xtr, xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    mtr = torch.ones(len(Xtr), dtype=torch.bool) if train_mask is None else torch.tensor(train_mask)
    mva = torch.ones(len(Xva), dtype=torch.bool) if val_mask is None else torch.tensor(val_mask)
    if task == "count":
        d_out = 1
        ttr = torch.tensor(ytr, dtype=torch.float32)
        loss_fn = lambda out, t: F.poisson_nll_loss(out.squeeze(-1), t, log_input=True)
    else:
        d_out = n_classes
        ttr = torch.tensor(ytr, dtype=torch.long)
        w = None
        if class_weight:
            cnt = np.bincount(ytr[mtr.numpy()], minlength=n_classes).astype(float)
            w = torch.tensor(cnt.sum() / (n_classes * np.maximum(cnt, 1)), dtype=torch.float32)
        loss_fn = lambda out, t: F.cross_entropy(out, t, weight=w)
    Net = {"GCN": GCN, "ResGCN": ResGCN, "MLP": MLP, "GAT": GAT}[kind]
    model = Net(Xtr.shape[1], hidden, d_out, layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, best_ep, bad = -np.inf, None, 0, 0
    curve = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(xtr, eitr, ewtr)
        loss = loss_fn(out[mtr], ttr[mtr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            ov = model(xva, eiva, ewva)[mva]
        sv = val_score(task, ov, yva[mva.numpy()], n_classes)
        if return_curve:
            with torch.no_grad():
                vl = loss_fn(ov, torch.tensor(yva[mva.numpy()], dtype=ttr.dtype)).item()
            curve.append((ep, loss.item(), vl, sv))
        if sv > best + 1e-6:
            best, best_ep, bad = sv, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    info = dict(best_epoch=best_ep, val_score=best, train_seconds=time.perf_counter() - t0,
                n_params=sum(p.numel() for p in model.parameters()))
    if return_curve:
        info["curve"] = curve
    return model, info


def val_score(task, out, y, n_classes):
    if task == "binary":
        s = torch.softmax(out, 1)[:, 1].numpy()
        return average_precision_score(y, s) if y.sum() > 0 else 0.0
    if task == "multi":
        return f1_score(y, out.argmax(1).numpy(), average="macro", labels=range(n_classes), zero_division=0)
    # count: negative Poisson deviance proxy
    mu = out.squeeze(-1).exp().numpy()
    return -np.mean(mu - y * np.log(mu + 1e-9))


def predict_graph(model, X, ei, ew, n_nodes, task):
    n = len(X) // n_nodes
    eib, ewb = batch_graph(n_nodes, n, ei, ew)
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32), eib, ewb)
    if task == "binary":
        return torch.softmax(out, 1)[:, 1].numpy()
    if task == "multi":
        return torch.softmax(out, 1).numpy()
    return out.squeeze(-1).exp().numpy()
