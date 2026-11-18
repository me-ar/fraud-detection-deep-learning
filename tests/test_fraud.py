import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score

from fraud.data import temporal_split
from fraud.experiment import at_alert_budget
from fraud.nets import AutoEncoder, FocalLoss, predict_proba, train_autoencoder, train_mlp


def synthetic(n=4000, fraud_rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(np.float32)
    X = rng.normal(size=(n, 10)).astype(np.float32)
    X[y == 1, :3] += 3.0  # fraud is shifted on 3 features
    return X, y


def test_focal_loss_reduces_to_weighted_bce():
    logits, targets = torch.randn(64), (torch.rand(64) > 0.5).float()
    fl = FocalLoss(alpha=0.5, gamma=0.0)(logits, targets)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.isclose(fl, 0.5 * bce, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    targets = torch.tensor([1.0])
    easy, hard = torch.tensor([5.0]), torch.tensor([-1.0])
    fl, bce = FocalLoss(gamma=2.0), torch.nn.functional.binary_cross_entropy_with_logits
    assert fl(easy, targets) / bce(easy, targets) < fl(hard, targets) / bce(hard, targets)


def test_temporal_split_is_ordered_and_scaled_on_train_only():
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = rng.permutation(n) * 10.0
    df["Amount"] = rng.exponential(50, n)
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    s = temporal_split(df)
    assert len(s.X_train) == 700 and len(s.X_val) == 150 and len(s.X_test) == 150
    assert np.allclose(s.X_train.mean(0), 0, atol=1e-5)  # scaler fit on train
    assert "log_amount" in s.feature_names and "Time" not in s.feature_names
    # test amounts must correspond to the latest transactions
    assert np.allclose(np.sort(s.amount_test),
                       np.sort(df.sort_values("Time")["Amount"].to_numpy()[850:]))


def test_mlp_learns_separable_signal():
    X, y = synthetic()
    model, hist = train_mlp(X[:3000], y[:3000], X[3000:], y[3000:], epochs=5, batch=256,
                            verbose=False)
    assert average_precision_score(y[3000:], predict_proba(model, X[3000:])) > 0.8
    assert len(hist) >= 1


def test_autoencoder_scores_anomalies_higher():
    X, y = synthetic()
    model, _ = train_autoencoder(X[:3000][y[:3000] == 0], X[3000:], y[3000:], epochs=5,
                                 verbose=False)
    s = model.anomaly_score(X[3000:])
    assert s[y[3000:] == 1].mean() > s[y[3000:] == 0].mean()
    assert isinstance(model, AutoEncoder)


@pytest.mark.parametrize("rate", [0.01, 0.1])
def test_alert_budget(rate):
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 10)
    score = y + np.linspace(0, 0.1, 100)  # perfect ranker
    amount = np.ones(100)
    out = at_alert_budget(y, score, amount, rate=rate)
    assert out["alerts"] == int(round(100 * rate))
    assert out["precision"] == 1.0
