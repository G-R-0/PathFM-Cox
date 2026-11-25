"""Step 3: train and evaluate H&E-only PathFM-Cox with Lunit features.

This script is self-contained and does not depend on pipeline.py or data.py.
It reads cached patch features from Step 2, trains the coordinate-aware
attention module plus Cox head, and reports survival metrics.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from lifelines.utils import concordance_index


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_col(col):
    return re.sub(r"[^a-z0-9]", "", str(col).strip().lower())


def read_table(path):
    return pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)


def load_clinical(path, os_event_value=1):
    df = read_table(path)
    cmap = {norm_col(c): c for c in df.columns}

    def pick(*names):
        for name in names:
            if name in cmap:
                return cmap[name]
        return None

    pid = pick("patientid", "patient_id", "submitterid", "bcrpatientbarcode")
    os_col = pick("os", "overallsurvival", "vitalstatus")
    time_col = pick("ostime", "os_time", "overallsurvivaltime")
    if pid is None or os_col is None or time_col is None:
        raise ValueError(f"clinical file needs Patient ID, OS, OS.time columns; got {df.columns.tolist()}")

    df = df.rename(columns={pid: "Patient ID", os_col: "OS", time_col: "OS.time"})
    df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS.time"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df = df.dropna(subset=["Patient ID", "OS", "OS.time"]).copy()
    unique_os = set(df["OS"].unique().tolist())
    if not unique_os.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"OS must be binary 0/1 after numeric conversion; got {sorted(unique_os)}")
    df["OS"] = df["OS"].astype(int)
    if os_event_value not in {0, 1}:
        raise ValueError(f"os_event_value must be 0 or 1; got {os_event_value}")
    if os_event_value == 0:
        df["OS"] = 1 - df["OS"]
    if df["Patient ID"].duplicated().any():
        raise ValueError("Duplicate patient IDs in clinical table")
    if not np.isfinite(df["OS.time"]).all() or (df["OS.time"] < 0).any():
        raise ValueError("Invalid survival times")
    return df


def encoder_cache_name(encoder):
    return re.sub(r"[^a-z0-9._-]", "_", str(encoder).strip().lower())


def prefix_metrics(metrics, prefix):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def concordance_index_simple(times, risks, events):
    """Harrell's C-index; larger Cox scores indicate higher risk."""
    try:
        return float(concordance_index(times, -np.asarray(risks), events))
    except ZeroDivisionError:
        return float("nan")


def compute_survival_metrics(df, split_name, result_dir, risk_threshold=None, write_csv=True):
