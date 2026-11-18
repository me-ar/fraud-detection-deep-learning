"""Run the full benchmark: classical ML vs deep learning vs unsupervised anomaly detection.

Usage: python -m fraud.experiment
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from .data import ROOT, load_split
from .nets import predict_proba, train_autoencoder, train_mlp

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
ALERT_RATE = 0.002  # analysts can review 0.2% of transactions (≈ 2 in 1,000)
COLORS = {"Logistic Regression": "#94a3b8", "XGBoost": "#2563eb", "MLP (focal loss)": "#dc2626",
          "Autoencoder (unsupervised)": "#16a34a", "Ensemble (XGB + MLP)": "#7c3aed"}


def at_alert_budget(y, score, amount, rate=ALERT_RATE) -> dict:
    """Business view: if we flag the top `rate` of transactions, what do we catch?"""
    k = max(1, int(round(len(score) * rate)))
    top = np.argsort(-score)[:k]
    caught = y[top] == 1
    return {"alerts": k, "precision": float(caught.mean()),
            "recall": float(caught.sum() / y.sum()),
            "fraud_value_caught_pct": float(amount[top][caught].sum() / amount[y == 1].sum())}


def bootstrap_ci(y, score, n=500, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() > 0:
            vals.append(average_precision_score(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    s = load_split()
    print(f"train={len(s.y_train):,} ({int(s.y_train.sum())} fraud) | val={len(s.y_val):,} "
          f"({int(s.y_val.sum())}) | test={len(s.y_test):,} ({int(s.y_test.sum())})")

    scores_val, scores_test, timings = {}, {}, {}

    t = time.time()
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1).fit(s.X_train, s.y_train)
    scores_val["Logistic Regression"] = lr.predict_proba(s.X_val)[:, 1]
    scores_test["Logistic Regression"] = lr.predict_proba(s.X_test)[:, 1]
    timings["Logistic Regression"] = time.time() - t

    t = time.time()
    xgb = XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.7, min_child_weight=3, eval_metric="aucpr",
                        early_stopping_rounds=100, random_state=42, n_jobs=-1)
    xgb.fit(s.X_train, s.y_train, eval_set=[(s.X_val, s.y_val)], verbose=False)
    scores_val["XGBoost"] = xgb.predict_proba(s.X_val)[:, 1]
    scores_test["XGBoost"] = xgb.predict_proba(s.X_test)[:, 1]
    timings["XGBoost"] = time.time() - t
    print(f"XGBoost best iteration: {xgb.best_iteration}")

    t = time.time()
    mlp, mlp_hist = train_mlp(s.X_train, s.y_train, s.X_val, s.y_val)
    scores_val["MLP (focal loss)"] = predict_proba(mlp, s.X_val)
    scores_test["MLP (focal loss)"] = predict_proba(mlp, s.X_test)
    timings["MLP (focal loss)"] = time.time() - t

    t = time.time()
    ae, ae_hist = train_autoencoder(s.X_train[s.y_train == 0], s.X_val, s.y_val)
    scores_val["Autoencoder (unsupervised)"] = ae.anomaly_score(s.X_val)
    scores_test["Autoencoder (unsupervised)"] = ae.anomaly_score(s.X_test)
    timings["Autoencoder (unsupervised)"] = time.time() - t

    def rank_avg(d, keys):
        return np.mean([rankdata(d[k]) / len(d[k]) for k in keys], axis=0)

    ens = ["XGBoost", "MLP (focal loss)"]
    scores_val["Ensemble (XGB + MLP)"] = rank_avg(scores_val, ens)
    scores_test["Ensemble (XGB + MLP)"] = rank_avg(scores_test, ens)

    rows = []
    for name, sc in scores_test.items():
        lo, hi = bootstrap_ci(s.y_test, sc)
        budget = at_alert_budget(s.y_test, sc, s.amount_test)
        rows.append({"model": name,
                     "val_pr_auc": round(average_precision_score(s.y_val, scores_val[name]), 4),
                     "test_pr_auc": round(average_precision_score(s.y_test, sc), 4),
                     "test_pr_auc_95ci": f"[{lo:.3f}, {hi:.3f}]",
                     "test_roc_auc": round(roc_auc_score(s.y_test, sc), 4),
                     "precision@0.2%": round(budget["precision"], 3),
                     "recall@0.2%": round(budget["recall"], 3),
                     "fraud_value_caught@0.2%": round(budget["fraud_value_caught_pct"], 3),
                     "train_seconds": round(timings.get(name, 0.0), 1)})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv(REPORTS / "results.csv", index=False)
    (REPORTS / "results.md").write_text(results.to_markdown(index=False))
    json.dump({"mlp": mlp_hist, "autoencoder": ae_hist}, open(REPORTS / "training_history.json", "w"),
              indent=2)
    torch.save(mlp.state_dict(), REPORTS / "mlp_state_dict.pt")

    # ---- figures ----
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, sc in scores_test.items():
        p, r, _ = precision_recall_curve(s.y_test, sc)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(s.y_test, sc):.3f})",
                color=COLORS[name], lw=2 if "Ensemble" in name else 1.4)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall on future transactions (test)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "pr_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    h = pd.DataFrame(mlp_hist)
    axes[0].plot(h.epoch, h.train_loss, color="#dc2626", label="train focal loss")
    ax2 = axes[0].twinx()
    ax2.plot(h.epoch, h.val_pr_auc, color="#2563eb", label="val PR-AUC")
    axes[0].set(title="MLP training", xlabel="epoch", ylabel="focal loss")
    ax2.set_ylabel("val PR-AUC")
    sc = scores_test["Autoencoder (unsupervised)"]
    bins = np.logspace(np.log10(sc.min() + 1e-6), np.log10(sc.max()), 60)
    axes[1].hist(sc[s.y_test == 0], bins=bins, alpha=0.6, density=True, label="legit", color="#94a3b8")
    axes[1].hist(sc[s.y_test == 1], bins=bins, alpha=0.7, density=True, label="fraud", color="#dc2626")
    axes[1].set_xscale("log")
    axes[1].set(title="Autoencoder reconstruction error (test)", xlabel="MSE (log scale)",
                ylabel="density")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "training_and_anomaly.png")
    print(f"Saved results to {REPORTS}")


if __name__ == "__main__":
    main()
