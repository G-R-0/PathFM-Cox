"""Step 2: extract frozen Lunit pathology foundation features from H&E patches.

This script is self-contained and does not depend on pipeline.py or data.py.
It reads tiled patch folders, filters tissue patches, extracts frozen Lunit
embeddings, and caches patch features plus slide-local coordinates.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

_IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


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
    return df


def patient_id_from_slide(slide_id):
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(slide_id), flags=re.I)
    return match.group(1).upper() if match else str(slide_id)[:12]


def patch_index(name):
    match = re.search(r"patch_(\d+)\.", str(name), flags=re.I)
    return int(match.group(1)) if match else None


def grid_pos(idx, cols):
    zero_based = int(idx) - 1
    return zero_based // cols, zero_based % cols


def resize_and_normalize(pil_image, size):
    resample = getattr(Image, "Resampling", Image).BILINEAR
    image = pil_image.convert("RGB").resize((size, size), resample=resample)
