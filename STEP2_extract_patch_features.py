"""Step 2: extract frozen Lunit pathology foundation features from H&E patches.

This script is self-contained and does not depend on pipeline.py or data.py.
It reads tiled patch folders, filters tissue patches, extracts frozen Lunit
embeddings, and caches patch features plus slide-local coordinates.
"""

import argparse
import hashlib
import json
import os
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
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - _IMAGE_MEAN) / _IMAGE_STD


def tissue_ratio_rgb(pil_image, sat_threshold=20, white_threshold=235):
    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(maxc == 0, 0.0, (maxc - minc) / maxc * 255.0)
    gray = rgb.mean(axis=2)
    white = (gray >= float(white_threshold)) & (sat <= 30.0)
    tissue = (sat >= float(sat_threshold)) & (maxc >= 25.0) & (maxc <= 250.0) & (~white)
    return float(np.mean(tissue))


class PatchDataset(Dataset):
    def __init__(self, tiles, image_size):
        self.tiles = list(tiles)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        with Image.open(tile["file_path"]) as img:
            tensor = resize_and_normalize(img, self.image_size)
        return tensor, idx


class LunitFeatureEncoder(nn.Module):
    def __init__(self, foundation_root=None, encoder_weight_path=None, feature_dim=512, device=None):
        super().__init__()
        self.foundation_root = Path(foundation_root) if foundation_root else None
        self.encoder_weight_path = encoder_weight_path
        self.feature_dim = int(feature_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone, backbone_dim = self._build_backbone()
        self.proj = nn.Identity() if backbone_dim == self.feature_dim else nn.Sequential(
            nn.Linear(backbone_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        for param in self.parameters():
            param.requires_grad = False
        self.to(self.device).eval()

    def _resolve_source(self):
        if self.encoder_weight_path:
            return self.encoder_weight_path
        if self.foundation_root is not None:
            return str(self.foundation_root / "Lunit")
        return "lunit"

    def _build_backbone(self):
        from transformers import AutoModel

        source = self._resolve_source()
        local_only = Path(source).exists()
        model = AutoModel.from_pretrained(source, trust_remote_code=True, local_files_only=local_only)
        config = getattr(model, "config", None)
        backbone_dim = getattr(config, "hidden_size", None) or getattr(config, "projection_dim", None)
        if backbone_dim is None:
            backbone_dim = 768
        return model, int(backbone_dim)

    @staticmethod
    def _hf_to_tensor(output):
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)):
            first = output[0]
            return first[:, 0] if getattr(first, "dim", lambda: 0)() == 3 else first
        raise TypeError(f"unsupported HF output: {type(output)}")

    def forward(self, x):
        with torch.no_grad():
            try:
                output = self.backbone(pixel_values=x)
            except TypeError:
                output = self.backbone(x)
            feat = self._hf_to_tensor(output)
            if feat.dim() == 3:
                feat = feat[:, 0]
            feat = self.proj(feat)
            return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


class Step2FeatureExtractor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clinical_df = load_clinical(args.clinical_total_path, os_event_value=args.os_event_value)
        self.valid_patients = self._valid_patients()
        cache_root = args.feature_cache_root or (Path(args.output_dir) / "feature_cache" / f"he_{encoder_cache_name(args.encoder)}_spot_cache")
        self.cache_dir = Path(cache_root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.slide_grid_meta = self._load_stitch_metadata(args.stitch_csv_path)
        self.encoder = None
        print(f"Encoder: {args.encoder}")
        print(f"Valid patients: {len(self.valid_patients)}")
        print(f"Cache dir: {self.cache_dir}")

    def _valid_patients(self):
        patients = self.clinical_df["Patient ID"].astype(str).tolist()
        if self.args.patients:
            keep = {x.strip() for x in self.args.patients.replace(",", " ").split() if x.strip()}
            patients = [pid for pid in patients if pid in keep]
        if self.args.max_patients:
            patients = patients[: self.args.max_patients]
        return patients

    def _load_stitch_metadata(self, stitch_csv_path):
        if not stitch_csv_path:
            return {}
        path = Path(stitch_csv_path)
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if "patient_id" not in df.columns:
            return {}
        meta = {}
        for _, row in df.iterrows():
            patient_id = str(row.get("patient_id", "")).strip()
            if not patient_id:
                continue
            try:
                meta[patient_id] = {
                    "stitch_rows": int(row.get("stitch_rows")),
                    "stitch_cols": int(row.get("stitch_cols")),
                }
            except Exception:
                continue
        print(f"Loaded stitch metadata for {len(meta)} slides")
        return meta

    def discover_slide_dirs(self):
        root = Path(self.args.tcga_slide_dir)
        if not root.exists():
            raise FileNotFoundError(root)
        valid = set(self.valid_patients)
        slide_dirs = []
        for path in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name):
            patient_id = patient_id_from_slide(path.name)
            if patient_id not in valid:
                continue
            if not any(path.glob("patch_*.*")):
                continue
            slide_dirs.append(path)
        print(f"Found {len(slide_dirs)} slide dirs")
        return slide_dirs

    def load_tiles(self, slide_dir):
        files = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            files.extend(slide_dir.glob(pattern))
        pairs = [(patch_index(p.name), p) for p in files]
        pairs = [(idx, p) for idx, p in pairs if idx is not None]
        if not pairs:
            return []
        patient_id = patient_id_from_slide(slide_dir.name)
        meta = self.slide_grid_meta.get(patient_id, {})
        total = max(idx for idx, _ in pairs)
        cols = int(meta.get("stitch_cols") or np.ceil(np.sqrt(total)))
        rows = int(meta.get("stitch_rows") or np.ceil(total / cols))
        tiles = []
        for idx, path in sorted(pairs, key=lambda x: x[0]):
            row, col = grid_pos(idx, cols)
            if self.args.skip_tissue_filter:
                ratio = 1.0
            else:
                with Image.open(path) as img:
                    ratio = tissue_ratio_rgb(img, self.args.tissue_sat_threshold, self.args.tissue_white_threshold)
            if ratio >= self.args.tissue_threshold:
                tiles.append({
                    "file_path": str(path),
                    "tile_name": path.name,
                    "patch_index": int(idx),
                    "grid_i": int(row),
                    "grid_j": int(col),
                    "coords": (float(row), float(col)),
                    "tissue_ratio": float(ratio),
                    "stitch_rows": int(rows),
                    "stitch_cols": int(cols),
                    "total_patches": int(total),
                })
        return tiles

    def ensure_encoder(self):
        if self.encoder is None:
            print(f"Loading frozen encoder: {self.args.encoder}")
            self.encoder = LunitFeatureEncoder(
                foundation_root=self.args.foundation_root,
                encoder_weight_path=self.args.encoder_weight_path,
                feature_dim=self.args.feature_dim,
                device=self.device,
            )

    def slide_ids(self, patient_id):
        path = self.cache_dir / patient_id / "slide_ids.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                return [str(x) for x in json.load(handle)]
        return []

    def encode_slide(self, slide_dir):
        slide_id = slide_dir.name
        patient_id = patient_id_from_slide(slide_id)
        patient_root = self.cache_dir / patient_id
        done = patient_root / f"done_{slide_id}.json"
        if self.args.resume and done.exists():
            print(f"Skip cached {slide_id}")
            return
        tiles = self.load_tiles(slide_dir)
        if not tiles:
            print(f"No tissue tiles: {slide_id}")
            return

        loader = DataLoader(
            PatchDataset(tiles, self.args.image_size),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        features = np.zeros((len(tiles), self.args.feature_dim), dtype=np.float32)
        for images, indices in tqdm(loader, desc=f"Encode {slide_id}"):
            feat = self.encoder(images.to(self.device)).detach().cpu().numpy().astype(np.float32)
            features[indices.numpy()] = feat

        coords = np.asarray([t["coords"] for t in tiles], dtype=np.float32)
        details = pd.DataFrame([
            {
                "patient_id": patient_id,
                "slide_id": slide_id,
                "tile_name": t["tile_name"],
                "patch_index": t["patch_index"],
                "grid_row": t["grid_i"],
                "grid_col": t["grid_j"],
                "tissue_ratio": t["tissue_ratio"],
                "stitch_rows": t["stitch_rows"],
                "stitch_cols": t["stitch_cols"],
                "total_patches": t["total_patches"],
            }
            for t in tiles
        ])

        patient_root.mkdir(parents=True, exist_ok=True)
        np.save(patient_root / f"{slide_id}_spot_features.npy", features)
        np.save(patient_root / f"{slide_id}_spot_coords.npy", coords)
        details.to_csv(patient_root / f"{slide_id}_spot_details.csv", index=False)

        slide_ids = self.slide_ids(patient_id)
        if slide_id not in slide_ids:
            slide_ids.append(slide_id)
        with open(patient_root / "slide_ids.json", "w", encoding="utf-8") as handle:
            json.dump(sorted(slide_ids), handle, ensure_ascii=False, indent=2)
        with open(done, "w", encoding="utf-8") as handle:
            json.dump({"patient_id": patient_id, "slide_id": slide_id, "n_spots": len(tiles)}, handle, indent=2)
        print(f"Saved {len(tiles)} spot features for {patient_id}/{slide_id}")

    def run(self):
        self.ensure_encoder()
        for slide_dir in self.discover_slide_dirs():
            self.encode_slide(slide_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 2: extract frozen Lunit patch features for PathFM-Cox.")
    parser.add_argument("--tcga_slide_dir", required=True, help="Root directory of pre-cut slide patch folders.")
    parser.add_argument("--clinical_total_path", required=True, help="Clinical file with Patient ID, OS, and OS.time.")
    parser.add_argument("--output_dir", required=True, help="Output root for feature cache and results.")
    parser.add_argument("--stitch_csv_path", default=None, help="Optional stitch CSV from Step 1.")
    parser.add_argument("--foundation_root", default=None, help="Local root directory for foundation models.")
    parser.add_argument("--encoder_weight_path", default=None, help="Optional Lunit model directory/checkpoint override.")
    parser.add_argument("--feature_cache_root", default=None, help="Optional explicit feature-cache directory.")
    parser.add_argument("--encoder", default="lunit", help="Paper default: lunit.")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--tissue_threshold", type=float, default=0.01)
    parser.add_argument("--tissue_sat_threshold", type=int, default=20)
    parser.add_argument("--tissue_white_threshold", type=int, default=235)
    parser.add_argument("--os_event_value", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--patients", default=None, help="Optional comma- or space-separated patient whitelist.")
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--max_spots_per_patient", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_tissue_filter", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Skip already encoded slides.")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Re-encode all slides.")
    parser.set_defaults(resume=True)
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    extractor = Step2FeatureExtractor(args)
    extractor.run()


if __name__ == "__main__":
    main()

