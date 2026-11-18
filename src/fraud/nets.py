"""PyTorch models: a supervised MLP (focal loss) and an unsupervised autoencoder."""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): down-weights easy negatives so the network
    concentrates on the rare, hard fraud cases instead of the 99.8% easy majority."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class FraudMLP(nn.Module):
    def __init__(self, n_in: int, hidden=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AutoEncoder(nn.Module):
    def __init__(self, n_in: int, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU(),
                                     nn.Linear(16, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, 24),
                                     nn.ReLU(), nn.Linear(24, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        x = torch.from_numpy(X)
        return ((self(x) - x) ** 2).mean(dim=1).numpy()


@torch.no_grad()
def predict_proba(model: FraudMLP, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch]))) for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


def train_mlp(X_tr, y_tr, X_va, y_va, epochs=30, lr=1e-3, batch=2048, patience=5,
              seed=42, verbose=True) -> tuple[FraudMLP, list[dict]]:
    """Train with early stopping on validation PR-AUC (the metric we actually care about)."""
    set_seed(seed)
    model = FraudMLP(X_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = FocalLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                    batch_size=batch, shuffle=True)
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_ap = average_precision_score(y_va, predict_proba(model, X_va))
        sched.step(val_ap)
        history.append({"epoch": epoch, "train_loss": total / len(X_tr), "val_pr_auc": val_ap})
        if verbose:
            print(f"  [MLP] epoch {epoch:2d} loss={total / len(X_tr):.5f} val PR-AUC={val_ap:.4f}")
        if val_ap > best:
            best, best_state, wait = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def train_autoencoder(X_normal, X_va, y_va, epochs=20, lr=1e-3, batch=1024, seed=42,
                      verbose=True) -> tuple[AutoEncoder, list[dict]]:
    """Train only on legitimate transactions; fraud should reconstruct poorly."""
    set_seed(seed)
    model = AutoEncoder(X_normal.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(torch.from_numpy(X_normal)), batch_size=batch, shuffle=True)
    best, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_ap = average_precision_score(y_va, model.anomaly_score(X_va))
        history.append({"epoch": epoch, "recon_loss": total / len(X_normal), "val_pr_auc": val_ap})
        if verbose:
            print(f"  [AE ] epoch {epoch:2d} recon={total / len(X_normal):.5f} val PR-AUC={val_ap:.4f}")
        if val_ap > best:
            best, best_state = val_ap, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history
