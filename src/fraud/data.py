"""Load the credit-card fraud dataset and split it *by time*.

A random split leaks future fraud patterns into training and overstates performance;
in production a model is always trained on the past and scored on the future.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "creditcard.csv"
DATA_URL = ("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/"
            "master/creditcard.csv")
TARGET = "Class"


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    amount_test: np.ndarray
    feature_names: list[str]


def download() -> None:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading creditcard.csv (~100 MB)...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    hour = (df["Time"] // 3600) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df.drop(columns=["Time", "Amount"])


def temporal_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15) -> Split:
    df = df.sort_values("Time").reset_index(drop=True)
    amount = df["Amount"].to_numpy()
    feats = add_features(df)
    y = feats.pop(TARGET).to_numpy().astype(np.float32)
    names = list(feats.columns)
    X = feats.to_numpy(dtype=np.float32)

    n = len(df)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
    scaler = StandardScaler().fit(X[:i_tr])  # fit on the past only
    X = scaler.transform(X).astype(np.float32)
    return Split(X[:i_tr], y[:i_tr], X[i_tr:i_va], y[i_tr:i_va], X[i_va:], y[i_va:],
                 amount[i_va:], names)


def load_split() -> Split:
    download()
    return temporal_split(pd.read_csv(DATA_PATH))
