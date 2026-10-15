"""
mor_order_selection.py  -  ONE file: find the smallest ROM order with NRMSE <= 1 %.

Task
    For every (System_ID, Method) the candidate ROM orders (10%..90% of N) form a ladder.
    A candidate PASSES when its actual step-response NRMSE <= 1 % and the ROM is stable
    (RelDC / RelHinf are not used any more).  Every model predicts P(pass) per candidate;
    the chosen ROM is the smallest candidate whose (monotone) probability >= threshold.
    The threshold is tuned automatically on a VALIDATION split, never on the test set.

Models (identical features, splits and threshold rule for all)
    Heuristic_Eff99, LogReg, ExtraTrees, HistGB, MLP (sklearn)         + RNN, GRU, LSTM (torch)

Scenarios
    ID   : 5 random grouped splits by System_ID (70/15/15)
    OOD  : train FOM 20,40,100 | validation 60 | test 30,50,80  (whole orders unseen)

Reported per model (test set only, ladders that have at least one passing candidate)
    Success_pct       % of chosen ROMs whose actual NRMSE <= 1 % (and stable)
    ROM_NRMSE_pct     mean actual NRMSE (%) of the chosen ROMs        <- the "< 1 %" check
    Order_NRMSE_pct   RMSE(chosen order - true minimum order) / range of true orders
    Order_R2          R2 of chosen vs true minimum order
    Under_pct         % chosen order below the true minimum (ROM would fail)
    Mean_Order / Oracle_Order

Run:  python mor_order_selection.py --data ai_guided_multi_order_mor_dataset.csv
      add --fast for a quick pass, --no-dl to skip the torch models
Needs: numpy pandas scikit-learn  (+ torch for RNN/GRU/LSTM)
"""
import argparse
import copy
import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
DEV = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu") if torch else None

NRMSE_LIMIT = 0.01
TARGET_SUCCESS = 0.98                      # threshold rule: smallest order with >= 98 % validation success
OOD_TRAIN, OOD_VAL, OOD_TEST = [20, 40, 100], [60], [30, 50, 80]
THR_GRID = np.linspace(0.02, 0.98, 97)

STATIC = ["R_min", "R_max", "R_mean", "R_std", "R_cv", "L_min", "L_max", "L_mean", "L_std", "L_cv",
          "C_min", "C_max", "C_mean", "C_std", "C_cv",
          "log_neg_pole", "FOM_DC", "log_hinf", "eff90", "eff95", "eff99", "hsv_peak", "hsv_mean_max",
          "hsv_cv", "hsv_min_log", "entropy_norm", "log_hsv_max", "is_matchdc"]
STEP = ["Candidate_Fraction", "Retained_HSV_Fraction", "Discarded_HSV_Fraction", "log_disc_frac",
        "log_bound", "log_bound_dc", "log_bound_hinf", "log_disc_max", "gap90", "gap95", "gap99"]


# ------------------------------------------------------------------------ data
def build(path):
    df = pd.read_csv(path).sort_values(["System_ID", "Method", "Candidate_Order"]).reset_index(drop=True)
    df["key"] = df.groupby(["System_ID", "Method"], sort=False).ngroup()
    df["Pass"] = ((df["NRMSE"] <= NRMSE_LIMIT) & (df["Stable_ROM"].astype(int) == 1)).astype(int)
    N = df["FOM_Order"].astype(float)
    L = lambda s, e=1e-12: np.log10(np.clip(s, e, None))
    df["is_matchdc"] = (df["Method"] == "BT_MatchDC").astype(int)
    df["log_neg_pole"] = L(-df["Max_Pole"])
    df["log_hinf"] = L(df["FOM_Hinf"])
    for p in (90, 95, 99):
        df[f"eff{p}"] = df[f"Effective_Order_{p}"] / N
        df[f"gap{p}"] = df["Candidate_Fraction"] - df[f"eff{p}"]
    df["hsv_peak"] = df["HSV_Max"] / df["HSV_Total"]
    df["hsv_mean_max"] = df["HSV_Mean"] / df["HSV_Max"]
    df["hsv_cv"] = df["HSV_Std"] / df["HSV_Mean"]
    df["hsv_min_log"] = L(df["HSV_Min"] / df["HSV_Max"], 1e-30)
    df["entropy_norm"] = df["HSV_Entropy"] / np.log(N)
    df["log_hsv_max"] = L(df["HSV_Max"], 1e-30)
    df["log_disc_frac"] = L(df["Discarded_HSV_Fraction"] + 1e-12)
    df["log_bound"] = L(df["BT_Bound"] + 1e-12)
    df["log_bound_dc"] = L(df["BT_Bound"] / np.clip(df["FOM_DC"].abs(), 1e-12, None) + 1e-12)
    df["log_bound_hinf"] = L(df["BT_Bound"] / np.clip(df["FOM_Hinf"], 1e-12, None) + 1e-12)
    df["log_disc_max"] = L(df["Discarded_HSV_Sum"] / df["HSV_Max"] + 1e-12)

    G, K = df["key"].nunique(), int(df.groupby("key").size().max())
    pos = df.groupby("key").cumcount().values
    idx = -np.ones((G, K), int)
    idx[df["key"].values, pos] = np.arange(len(df))
    valid = idx >= 0
    safe = np.clip(idx, 0, None)
    grab = lambda c: np.where(valid, df[c].values[safe], 0)
    D = dict(df=df, G=G, K=K, idx=idx, valid=valid, O=grab("Candidate_Order"), Y=grab("Pass"),
             E=grab("NRMSE"))
    first = df.groupby("key").first()
    D["N"] = first["FOM_Order"].values.astype(float)
    D["sid"] = first["System_ID"].values
    D["S"] = first[STATIC].values.astype(np.float32)
    Q = np.zeros((G, K, len(STEP)), np.float32)
    Q[valid] = df[STEP].values[idx[valid]]
    D["Q"] = Q
    D["Xrows"] = df[STATIC + STEP].values.astype(float)
    D["yrows"] = df["Pass"].values
    D["true"] = np.array([D["O"][g][valid[g] & (D["Y"][g] == 1)].min() if (valid[g] & (D["Y"][g] == 1)).any()
                          else np.nan for g in range(G)])
    print(f"[data] ladders={G}  candidates/ladder={K}  pass rate={df['Pass'].mean():.1%}  "
          f"unreachable ladders={int(np.isnan(D['true']).sum())}")
    return D


def rows_of(D, sel):
    return D["idx"][sel][D["valid"][sel]]


def to_mat(D, vec, sel):
    P = np.zeros((len(sel), D["K"]))
    P[D["valid"][sel]] = vec
    return P


# ---------------------------------------------------------- selection + metrics
def choose(D, P, sel, thr):
    v = D["valid"][sel]
    Pm = np.maximum.accumulate(np.where(v, P, 0.0), axis=1)          # monotone pass probability
    ok = (Pm >= thr) & v
    last = v.shape[1] - 1 - np.argmax(v[:, ::-1], axis=1)             # fallback: largest candidate
    return np.where(ok.any(1), np.argmax(ok, 1), last)


def metrics(D, P, sel, thr):
    ci = choose(D, P, sel, thr)
    r = np.arange(len(sel))
    order, ok = D["O"][sel][r, ci], D["Y"][sel][r, ci]
    nr = D["E"][sel][r, ci]
    true = D["true"][sel]
    m = ~np.isnan(true)
    order, ok, nr, true = order[m], ok[m], nr[m], true[m]
    rmse = np.sqrt(np.mean((order - true) ** 2))
    return {"Success_pct": 100 * ok.mean(), "ROM_NRMSE_pct": 100 * nr.mean(), "ROM_NRMSE_max_pct": 100 * nr.max(),
            "Order_NRMSE_pct": 100 * rmse / max(true.max() - true.min(), 1e-12),
            "Order_R2": r2_score(true, order), "Under_pct": 100 * np.mean(order < true),
            "Exact_pct": 100 * np.mean(order == true), "Mean_Order": order.mean(), "Oracle_Order": true.mean()}


def tune_threshold(D, P_val, va):
    """Validation-only rule: smallest mean order among thresholds with success >= TARGET_SUCCESS;
    if none reaches it, the threshold with the highest success."""
    best, key = 0.5, None
    for t in THR_GRID:
        ci = choose(D, P_val, va, t)
        r = np.arange(len(va))
        m = ~np.isnan(D["true"][va])
        succ = D["Y"][va][r, ci][m].mean()
        order = D["O"][va][r, ci][m].mean()
        k = (0, order, t) if succ >= TARGET_SUCCESS else (1, -succ, order)
        if key is None or k < key:
            key, best = k, float(t)
    return best


# ---------------------------------------------------------------------- splits
def id_split(D, seed):
    ids = np.unique(D["sid"])
    np.random.default_rng(seed).shuffle(ids)
    a, b = int(round(0.70 * len(ids))), int(round(0.85 * len(ids)))
    f = lambda s: np.where(np.isin(D["sid"], ids[s]))[0]
    return f(slice(0, a)), f(slice(a, b)), f(slice(b, None))


def ood_split(D):
    f = lambda o: np.where(np.isin(D["N"], o))[0]
    return f(OOD_TRAIN), f(OOD_VAL), f(OOD_TEST)


# ---------------------------------------------------------------------- models
def sk_model(make):
    def run(D, tr, va, te):
        rt = rows_of(D, tr)
        m = make()
        m.fit(D["Xrows"][rt], D["yrows"][rt])
        return [to_mat(D, m.predict_proba(D["Xrows"][rows_of(D, s)])[:, 1], s) for s in (va, te)]
    return run


def heuristic(D, tr, va, te):
    out = []
    for s in (va, te):
        cf = D["Q"][s][:, :, STEP.index("Candidate_Fraction")]
        e = D["S"][s][:, STATIC.index("eff99")][:, None]
        out.append((cf >= e).astype(float))
    return out


class SeqNet(nn.Module if torch else object):
    def __init__(self, d, cell, hidden=64, drop=0.1):
        super().__init__()
        self.rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell](d, hidden, batch_first=True,
                                                                         bidirectional=True)
        self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h).squeeze(-1)


def seq_inputs(D, fit):
    S = D["S"]
    S = (S - S[fit].mean(0)) / (S[fit].std(0) + 1e-6)
    Q, v = D["Q"], D["valid"]
    fv = Q[fit][v[fit]]
    Q = np.where(v[..., None], (Q - fv.mean(0)) / (fv.std(0) + 1e-6), 0.0)
    X = np.concatenate([np.repeat(S[:, None], D["K"], 1), Q], -1)
    return np.clip(np.nan_to_num(X), -8, 8).astype(np.float32)


def rnn_model(cell, seeds, epochs, patience):
    def run(D, tr, va, te):
        X = torch.from_numpy(seq_inputs(D, tr)).to(DEV)
        Y = torch.from_numpy(D["Y"].astype(np.float32)).to(DEV)
        M = torch.from_numpy(D["valid"].astype(np.float32)).to(DEV)
        Pva, Pte = [], []
        for seed in range(seeds):
            torch.manual_seed(seed)
            rng = np.random.RandomState(seed)
            net = SeqNet(X.shape[-1], cell).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)

            def loss(idx):
                i = torch.from_numpy(idx).to(DEV)
                l = nn.functional.binary_cross_entropy_with_logits(net(X[i]), Y[i], reduction="none")
                return (l * M[i]).sum() / M[i].sum()

            best, state, wait = 1e18, None, 0
            for _ in range(epochs):
                net.train()
                perm = rng.permutation(tr)
                for k in range(0, len(perm), 64):
                    opt.zero_grad()
                    loss(perm[k:k + 64]).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = loss(va).item()
                if vl < best - 1e-5:
                    best, state, wait = vl, copy.deepcopy(net.state_dict()), 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
            net.load_state_dict(state)
            net.eval()
            with torch.no_grad():
                Pva.append(torch.sigmoid(net(X[torch.from_numpy(va).to(DEV)])).cpu().numpy())
                Pte.append(torch.sigmoid(net(X[torch.from_numpy(te).to(DEV)])).cpu().numpy())
        return np.mean(Pva, 0), np.mean(Pte, 0)
    return run


class RidgeProb:
    """Ridge regression on the 0/1 pass label, clipped to [0,1] and used as a pass probability."""
    def __init__(self, alpha=1.0):
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = np.clip(self.m.predict(X), 0, 1)
        return np.c_[1 - p, p]


FAMILY = {"Heuristic_Eff99": "Baseline", "Ridge": "Linear", "LogReg": "Linear",
          "RandomForest": "Trees", "ExtraTrees": "Trees",
          "HistGB": "Boosting", "XGBoost": "Boosting", "LightGBM": "Boosting",
          "MLP": "Neural", "RNN": "Recurrent", "GRU": "Recurrent", "LSTM": "Recurrent"}
COLORS = {"Baseline": "#9aa0a6", "Linear": "#4c72b0", "Trees": "#dd8452", "Boosting": "#8172b3",
          "Neural": "#55a868", "Recurrent": "#c44e52"}


def model_zoo(fast, seeds, no_dl):
    n = 100 if fast else 400
    z = {
        "Heuristic_Eff99": heuristic,
        "Ridge": sk_model(lambda: RidgeProb(1.0)),
        "LogReg": sk_model(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
        "RandomForest": sk_model(lambda: RandomForestClassifier(n, min_samples_leaf=2, n_jobs=-1, random_state=0)),
        "ExtraTrees": sk_model(lambda: ExtraTreesClassifier(n, min_samples_leaf=2, n_jobs=-1, random_state=0)),
        "HistGB": sk_model(lambda: HistGradientBoostingClassifier(random_state=0)),
        "MLP": sk_model(lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, random_state=0))),
    }
    try:
        from xgboost import XGBClassifier
        z["XGBoost"] = sk_model(lambda: XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.05,
                                                      subsample=0.9, tree_method="hist", n_jobs=-1,
                                                      random_state=0, verbosity=0))
    except ImportError:
        print("[note] xgboost not installed -> skipped (pip install xgboost)")
    try:
        from lightgbm import LGBMClassifier
        z["LightGBM"] = sk_model(lambda: LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=15,
                                                        random_state=0, verbose=-1, n_jobs=-1))
    except ImportError:
        print("[note] lightgbm not installed -> skipped (pip install lightgbm)")
    if torch is None and not no_dl:
        print("[note] PyTorch not installed -> RNN/GRU/LSTM skipped.  pip install torch")
    elif not no_dl:
        ep, pat = (60, 10) if fast else (250, 25)
        for cell in ("RNN", "GRU", "LSTM"):
            z[cell] = rnn_model(cell, 1 if fast else seeds, ep, pat)
    return z


# --------------------------------------------------------------------- reporting
def plot_table(res, order, out):
    import matplotlib.pyplot as plt
    cols = [("Success_pct", "Success %", True, "{:.1f}"), ("ROM_NRMSE_pct", "ROM NRMSE %", False, "{:.3f}"),
            ("Order_NRMSE_pct", "Order NRMSE %", False, "{:.2f}"), ("Order_R2", "Order R\u00b2", True, "{:.3f}")]
    scens = ["ID", "OOD"]
    header = ["Rank", "Model"] + [f"{sc}\n{c[1]}" for sc in scens for c in cols]
    d = {sc: res[res.Scenario == sc].set_index("Model").reindex(order) for sc in scens}
    cells = [[str(i + 1), m] + [c[3].format(d[sc].loc[m, c[0]]) for sc in scens for c in cols]
             for i, m in enumerate(order)]
    fig, ax = plt.subplots(figsize=(15, 0.55 * len(order) + 2.2))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.8)
    for j in range(len(header)):
        t[(0, j)].set_facecolor("#263238")
        t[(0, j)].set_text_props(color="white", fontweight="bold")
    learned = [i for i, m in enumerate(order) if m != "Heuristic_Eff99"]
    for k, (sc, c) in enumerate([(sc, c) for sc in scens for c in cols]):
        vals = d[sc][c[0]].values
        best = max(learned, key=lambda i: vals[i] if c[2] else -vals[i])
        t[(best + 1, 2 + k)].set_facecolor("#c8e6c9")
        t[(best + 1, 2 + k)].set_text_props(fontweight="bold")
    for i, m in enumerate(order):
        t[(i + 1, 1)].set_text_props(color=COLORS[FAMILY[m]], fontweight="bold")
    ax.set_title("MOR order selection - all models (green = best learned model per column)\n"
                 "Rows sorted by mean Order NRMSE over ID and OOD", fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)



def export_readable_results(res, out_dir):
    """Save readable CSVs and a compact overall ID+OOD comparison table."""
    import matplotlib.pyplot as plt

    readable_names = {
        "Scenario": "Evaluation Scenario",
        "Model": "ML Model",
        "Family": "Model Family",
        "Success_pct": "Successful ROM Selection (%)",
        "ROM_NRMSE_pct": "Mean Actual ROM NRMSE (%)",
        "ROM_NRMSE_max_pct": "Worst-Case ROM NRMSE (%)",
        "Order_NRMSE_pct": "ROM Order Selection Error (%)",
        "Order_R2": "ROM Order Prediction R-squared",
        "Under_pct": "Unsafe Under-Selection (%)",
        "Exact_pct": "Exact Minimum-Order Match (%)",
        "Mean_Order": "Mean Selected ROM Order",
        "Oracle_Order": "Mean True Minimum ROM Order",
        "Threshold": "Pass-Probability Threshold",
        "n_splits": "Number of Evaluation Splits",
    }

    # Keep the original internal dataframe unchanged for all calculations/plots.
    readable = res.rename(columns=readable_names).copy()
    readable.to_csv(os.path.join(out_dir, "results_readable.csv"), index=False)

    metrics_to_average = [
        "Success_pct", "ROM_NRMSE_pct", "ROM_NRMSE_max_pct",
        "Order_NRMSE_pct", "Order_R2", "Under_pct", "Exact_pct",
        "Mean_Order", "Oracle_Order", "Threshold"
    ]
    overall = res.groupby("Model")[metrics_to_average].mean()
    overall["Family"] = res.groupby("Model")["Family"].first()
    overall["n_splits"] = res.groupby("Model")["n_splits"].sum()
    overall = overall.sort_values("Order_NRMSE_pct")

    overall_names = {
        "Model": "ML Model",
        "Family": "Model Family",
        "Success_pct": "Successful ROM Selection (%)",
        "ROM_NRMSE_pct": "Mean Actual ROM NRMSE (%)",
        "ROM_NRMSE_max_pct": "Worst-Case ROM NRMSE (%)",
        "Order_NRMSE_pct": "ROM Order Selection Error (%)",
        "Order_R2": "ROM Order Prediction R-squared",
        "Under_pct": "Unsafe Under-Selection (%)",
        "Exact_pct": "Exact Minimum-Order Match (%)",
        "Mean_Order": "Mean Selected ROM Order",
        "Oracle_Order": "Mean True Minimum ROM Order",
        "Threshold": "Pass-Probability Threshold",
        "n_splits": "Total Scenario-Splits",
    }
    overall_export = overall.reset_index().rename(columns=overall_names)
    overall_export.to_csv(os.path.join(out_dir, "overall_model_comparison.csv"), index=False)

    # Compact visual summary: only the two requested overall comparison metrics.
    # These values are already the mean across the ID and OOD scenario results.
    models = list(overall.index)
    show_cols = [
        ("Order_NRMSE_pct", "Mean Order NRMSE\n(ID + OOD) (%)", "{:.2f}"),
        ("Order_R2", "Mean Order R²\n(ID + OOD)", "{:.3f}"),
    ]
    headers = ["ML Model"] + [label for _, label, _ in show_cols]
    cell_text = []
    for model in models:
        row = [model]
        row += [fmt.format(overall.loc[model, key]) for key, _, fmt in show_cols]
        cell_text.append(row)

    fig_width = 10
    fig_height = max(3.0, 0.42 * len(models) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#263238")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, model in enumerate(models, start=1):
        table[(i, 0)].set_text_props(fontweight="bold")
        table[(i, 0)].set_facecolor("#f0f0f0")
    ax.set_title(
        "Overall MOR Order-Selection Comparison (ID + OOD averages)\n"
        "Lower Mean Order NRMSE is better; higher Mean Order R² is better",
        fontsize=12, pad=16
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "0_overall_model_comparison.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bars(res, order, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    metrics_ = [("Success_pct", "Success %  (chosen ROM has NRMSE <= 1 %)"),
                ("ROM_NRMSE_pct", "Mean NRMSE of chosen ROM (%)   [red line = 1 % limit]"),
                ("Order_NRMSE_pct", "Order NRMSE (%)   [lower = closer to true minimum order]"),
                ("Order_R2", "Order R\u00b2")]
    fig, axes = plt.subplots(len(metrics_), 2, figsize=(14, 3.0 * len(metrics_)))
    for i, (col, title) in enumerate(metrics_):
        for j, sc in enumerate(["ID", "OOD"]):
            ax = axes[i, j]
            d = res[res.Scenario == sc].set_index("Model").reindex(order)
            vals = d[col].values
            ax.barh(d.index, vals, color=[COLORS[FAMILY[m]] for m in d.index])
            ax.invert_yaxis()
            if col == "ROM_NRMSE_pct":
                ax.axvline(1.0, color="crimson", ls="--", lw=1.5)
                ax.set_xlim(0, max(1.6, 1.4 * np.nanmax(np.where(d.index == "Heuristic_Eff99", 0, vals))))
            span = np.nanmax(np.abs(ax.get_xlim()))
            for y, v in enumerate(vals):
                ax.text(min(v, ax.get_xlim()[1]) + 0.01 * span, y, f"{v:.2f}", va="center", fontsize=8)
            ax.set_title(f"{sc}: {title}", fontsize=10, loc="left")
            ax.grid(axis="x", alpha=0.25)
    fig.legend(handles=[Patch(color=c, label=f) for f, c in COLORS.items()], loc="lower center",
               ncol=len(COLORS), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_scatter(scatter, order, scen, out):
    import matplotlib.pyplot as plt
    names = [m for m in order if (scen, m) in scatter]
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes.ravel(), names):
        t, c = scatter[(scen, m)]
        bad = c < t - 1e-9
        j = lambda v: v + rng.normal(0, 0.006, len(v))
        ax.scatter(j(t[~bad]), j(c[~bad]), s=14, color="#4c72b0", alpha=0.6, label="ok")
        ax.scatter(j(t[bad]), j(c[bad]), s=18, color="crimson", alpha=0.9, label="too small (fails)")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{m}   fails={100 * bad.mean():.1f}%", fontsize=10, color=COLORS[FAMILY[m]])
        ax.set_xlabel("true minimum order / N")
        ax.set_ylabel("chosen order / N")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{scen} test: chosen vs true minimum ROM order (on the line = perfect; below = fails)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="mor_output")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-dl", action="store_true")
    a = ap.parse_args()
    path = a.data or os.environ.get("MOR_DATA") or (glob.glob("*mor_dataset.csv") or [None])[0]
    if not path:
        raise SystemExit("Pass --data <csv>")
    os.makedirs(a.out, exist_ok=True)
    D = build(path)
    if torch is not None and not a.no_dl:
        print(f"[torch] {torch.__version__}  device = {DEV}" +
              (f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else "  (no CUDA found -> running on CPU)"))
    zoo = model_zoo(a.fast, a.seeds, a.no_dl)

    scenarios = {"ID": [id_split(D, s) for s in range(1 if a.fast else a.repeats)], "OOD": [ood_split(D)]}
    rows, scatter = [], {}
    for scen, splits in scenarios.items():
        for name, run in zoo.items():
            per = []
            for k, (tr, va, te) in enumerate(splits):
                Pva, Pte = run(D, tr, va, te)
                thr = tune_threshold(D, Pva, va)
                per.append({**metrics(D, Pte, te, thr), "Threshold": thr})
                if k == 0:                                        # keep first split for the scatter plot
                    ci = choose(D, Pte, te, thr)
                    r = np.arange(len(te))
                    m = ~np.isnan(D["true"][te])
                    scatter[(scen, name)] = (D["true"][te][m] / D["N"][te][m], D["O"][te][r, ci][m] / D["N"][te][m])
            r_ = {k_: float(np.mean([p[k_] for p in per])) for k_ in per[0]}
            r_.update(Scenario=scen, Model=name, Family=FAMILY.get(name, "Other"), n_splits=len(per))
            rows.append(r_)
            print(f"{scen:3s} {name:16s} success={r_['Success_pct']:6.1f}%  ROM NRMSE={r_['ROM_NRMSE_pct']:.3f}%  "
                  f"order NRMSE={r_['Order_NRMSE_pct']:5.2f}%  R2={r_['Order_R2']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    # Keep results.csv with original machine-readable columns for compatibility.
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    # Also create human-readable CSVs and a full summary table with the extra metrics.
    export_readable_results(res, a.out)

    overall = res.groupby("Model")[[
        "Order_NRMSE_pct", "Order_R2", "Success_pct", "ROM_NRMSE_pct",
        "ROM_NRMSE_max_pct", "Under_pct", "Exact_pct", "Mean_Order",
        "Oracle_Order", "Threshold"
    ]].mean()
    order = list(overall.sort_values("Order_NRMSE_pct").index)
    with pd.option_context("display.width", 260, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print("\n=== OVERALL (mean of ID and OOD, sorted by ROM Order Selection Error) ===")
        print(overall.loc[order].to_string())
    try:
        plot_table(res, order, os.path.join(a.out, "1_results_table.png"))
        plot_bars(res, order, os.path.join(a.out, "2_comparison_bars.png"))
        plot_scatter(scatter, order, "OOD", os.path.join(a.out, "3_chosen_vs_true_OOD.png"))
        plot_scatter(scatter, order, "ID", os.path.join(a.out, "4_chosen_vs_true_ID.png"))
    except ImportError:
        print("matplotlib missing -> no PNGs (pip install matplotlib)")
    print(f"\nBest overall by order NRMSE: {order[0]}")
    print(f"Saved results.csv and PNGs in ./{a.out}/")


if __name__ == "__main__":
    main()
