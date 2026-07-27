"""Step 3: train and evaluate H&E-only PathFM-Cox with Lunit features.

This script is self-contained and does not depend on pipeline.py or data.py.
It reads cached patch features from Step 2, trains the coordinate-aware
attention module plus Cox head, and reports survival metrics.
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
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings("ignore")


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


def encoder_cache_name(encoder):
    return re.sub(r"[^a-z0-9._-]", "_", str(encoder).strip().lower())


def prefix_metrics(metrics, prefix):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def concordance_index_simple(times, risks, events):
    times = np.asarray(times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    events = np.asarray(events, dtype=int)
    concordant = 0.0
    permissible = 0.0
    tied = 0.0
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] == times[j] and events[i] == 0 and events[j] == 0:
                continue
            if times[i] == times[j] and events[i] == 1 and events[j] == 1:
                permissible += 1
                if risks[i] == risks[j]:
                    tied += 1
                elif risks[i] > risks[j]:
                    concordant += 1
                else:
                    concordant += 0
                    tied += 0
                continue
            if times[i] < times[j] and events[i] == 1:
                permissible += 1
                if risks[i] == risks[j]:
                    tied += 1
                elif risks[i] > risks[j]:
                    concordant += 1
            elif times[j] < times[i] and events[j] == 1:
                permissible += 1
                if risks[i] == risks[j]:
                    tied += 1
                elif risks[j] > risks[i]:
                    concordant += 1
    if permissible == 0:
        return float("nan")
    return float((concordant + 0.5 * tied) / permissible)


def compute_survival_metrics(df, split_name, result_dir, risk_threshold=None, write_csv=True):
    metrics = {
        "cindex": float("nan"),
        "risk_threshold": float("nan"),
        "logrank_p": float("nan"),
        "hr": float("nan"),
        "hr_ci_lower": float("nan"),
        "hr_ci_upper": float("nan"),
    }
    if df.empty:
        return metrics

    metrics["cindex"] = concordance_index_simple(df["OS.time"], df["risk_score"], df["OS"])
    threshold = float(np.nanmedian(df["risk_score"])) if risk_threshold is None else float(risk_threshold)
    metrics["risk_threshold"] = threshold
    df = df.copy()
    df["risk_group"] = np.where(df["risk_score"] >= threshold, "high", "low")

    if write_csv:
        df.to_csv(result_dir / f"predictions_{split_name}.csv", index=False)

    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import logrank_test

        high = df[df["risk_group"] == "high"]
        low = df[df["risk_group"] == "low"]
        if len(high) > 0 and len(low) > 0:
            metrics["logrank_p"] = float(logrank_test(high["OS.time"], low["OS.time"], high["OS"], low["OS"]).p_value)
            cox_df = df[["OS.time", "OS"]].copy()
            cox_df["high_risk"] = (df["risk_group"] == "high").astype(int).values
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="OS.time", event_col="OS", show_progress=False)
            row = cph.summary.iloc[0]
            metrics["hr"] = float(row.get("exp(coef)", np.nan))
            metrics["hr_ci_lower"] = float(row.get("exp(coef) lower 95%", np.nan))
            metrics["hr_ci_upper"] = float(row.get("exp(coef) upper 95%", np.nan))
            cph.summary.to_csv(result_dir / f"cox_summary_{split_name}.csv")

            if write_csv:
                import matplotlib.pyplot as plt

                kmf = KaplanMeierFitter()
                fig, ax = plt.subplots(figsize=(6, 5))
                if len(low) > 0:
                    kmf.fit(low["OS.time"], low["OS"], label="Low risk")
                    kmf.plot(ax=ax)
                if len(high) > 0:
                    kmf.fit(high["OS.time"], high["OS"], label="High risk")
                    kmf.plot(ax=ax)
                ax.set_title(f"{split_name} KM | log-rank p={metrics['logrank_p']:.3g}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Survival probability")
                fig.tight_layout()
                fig.savefig(result_dir / f"km_{split_name}.png", dpi=300)
                plt.close(fig)
    except Exception:
        pass

    return metrics


class DirectAttentionPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.1, pooling="attention", use_coord=True):
        super().__init__()
        self.pooling = str(pooling).lower()
        self.use_coord = bool(use_coord)
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.coord = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_features, spot_coords=None):
        x = torch.nan_to_num(patch_features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.pooling == "mean":
            weights = torch.full((x.shape[0],), 1.0 / max(1, x.shape[0]), dtype=x.dtype, device=x.device)
            return torch.mean(x, dim=0), weights
        scores = self.attn(x)
        if self.use_coord and spot_coords is not None:
            c = torch.nan_to_num(spot_coords.float(), nan=0.0, posinf=0.0, neginf=0.0)
            c = (c - c.min(0)[0]) / (c.max(0)[0] - c.min(0)[0] + 1e-8)
            scores = scores + self.coord(torch.cat([torch.sin(c), torch.cos(c)], dim=-1))
        weights = F.softmax(scores, dim=0)
        patient_feat = torch.sum(x * weights, dim=0)
        return patient_feat, weights.squeeze(-1)


class SurvivalMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(128,), dropout=0.2):
        super().__init__()
        layers = []
        last = int(in_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers += [nn.Linear(last, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CoxPHLoss(nn.Module):
    def forward(self, risk_scores, times, events):
        order = torch.argsort(times, descending=True)
        risk = risk_scores.reshape(-1)[order]
        event = events.reshape(-1)[order]
        loss = -(risk - torch.logcumsumexp(risk, dim=0)) * event
        return loss.sum() / (event.sum() + 1e-8)


class PathFMCOXPipeline:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_name = encoder_cache_name(args.encoder)
        self.clinical_df = load_clinical(args.clinical_total_path, os_event_value=args.os_event_value)
        self.cache_dir = Path(args.feature_cache_root or (Path(args.output_dir) / "feature_cache" / f"he_{self.encoder_name}_spot_cache"))
        self.result_dir = Path(args.result_dir or (Path(args.output_dir) / args.experiment_name / f"result_{self.encoder_name}"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.valid_patients = self._valid_patients()
        self.direct_pooling = None
        self.survival_mlp = None
        self.optimizer = None
        self.cox_loss = CoxPHLoss().to(self.device)
        print(f"Encoder: {self.encoder_name}")
        print(f"Valid patients: {len(self.valid_patients)}")
        print(f"Cache dir: {self.cache_dir}")
        print(f"Result dir: {self.result_dir}")

    def _valid_patients(self):
        patients = self.clinical_df["Patient ID"].astype(str).tolist()
        if self.args.patients:
            keep = {x.strip() for x in self.args.patients.replace(",", " ").split() if x.strip()}
            patients = [pid for pid in patients if pid in keep]
        if self.args.max_patients:
            patients = patients[: self.args.max_patients]
        return patients

    def slide_ids(self, patient_id):
        path = self.cache_dir / patient_id / "slide_ids.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                return [str(x) for x in json.load(handle)]
        return []

    def load_patient(self, patient_id):
        root = self.cache_dir / patient_id
        if not root.exists():
            return None, None
        slide_ids = self.slide_ids(patient_id)
        if not slide_ids:
            slide_ids = [p.name.replace("_spot_features.npy", "") for p in root.glob("*_spot_features.npy")]
        feats, coords = [], []
        for sid in slide_ids:
            feat_path = root / f"{sid}_spot_features.npy"
            coord_path = root / f"{sid}_spot_coords.npy"
            if feat_path.exists() and coord_path.exists():
                feats.append(np.load(feat_path).astype(np.float32))
                coords.append(np.load(coord_path).astype(np.float32))
        if not feats:
            return None, None
        feat = np.concatenate(feats, axis=0)
        coord = np.concatenate(coords, axis=0)
        if self.args.max_spots_per_patient and feat.shape[0] > self.args.max_spots_per_patient:
            stable_seed = int(hashlib.md5(str(patient_id).encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(stable_seed)
            keep = np.sort(rng.choice(feat.shape[0], self.args.max_spots_per_patient, replace=False))
            feat, coord = feat[keep], coord[keep]
        return feat, coord

    def has_cached_features(self, patient_id):
        root = self.cache_dir / patient_id
        if not root.is_dir():
            return False
        slide_ids = self.slide_ids(patient_id)
        if not slide_ids:
            slide_ids = [p.name.replace("_spot_features.npy", "") for p in root.glob("*_spot_features.npy")]
        for sid in slide_ids:
            if (root / f"{sid}_spot_features.npy").exists() and (root / f"{sid}_spot_coords.npy").exists():
                return True
        return False

    def cached_patients(self):
        return [p for p in self.valid_patients if self.has_cached_features(p)]

    def cache_status_message(self, cached_count):
        patient_dirs = [p for p in self.valid_patients if (self.cache_dir / p).is_dir()]
        any_patient_dirs = [p.name for p in self.cache_dir.iterdir() if p.is_dir()] if self.cache_dir.exists() else []
        examples = any_patient_dirs[:5]
        return (
            f"cached_features={cached_count}, matching_patient_dirs={len(patient_dirs)}, "
            f"valid_patients={len(self.valid_patients)}, cache_dir={self.cache_dir}, example_cache_dirs={examples}"
        )

    def reset_downstream_modules(self):
        self.direct_pooling = DirectAttentionPooling(
            self.args.feature_dim,
            self.args.attention_hidden_dim,
            self.args.attention_dropout,
            pooling=self.args.pooling,
            use_coord=not self.args.no_coord,
        ).to(self.device)
        self.survival_mlp = SurvivalMLP(self.args.feature_dim, tuple(self.args.mlp_hidden_dims), self.args.mlp_dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.direct_pooling.parameters()) + list(self.survival_mlp.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def split_dataset(self):
        from sklearn.model_selection import train_test_split

        patients = self.cached_patients()
        if len(patients) < 5:
            if not patients:
                print(f"[WARN] No cached patients found: {self.cache_status_message(len(patients))}")
            return {"train": patients, "val": [], "test": []}
        if self.args.test_split_mode == "from_file" and self.args.test_patient_path and Path(self.args.test_patient_path).exists():
            tdf = pd.read_csv(self.args.test_patient_path)
            cmap = {norm_col(c): c for c in tdf.columns}
            col = cmap.get("patientid") or cmap.get("patient_id") or tdf.columns[0]
            test = [p for p in tdf[col].astype(str).str.strip().tolist() if p in patients]
            train_val = [p for p in patients if p not in set(test)]
            train, val = train_test_split(train_val, test_size=0.2, random_state=self.args.seed) if len(train_val) > 1 else (train_val, [])
            return {"train": list(train), "val": list(val), "test": test}
        clinical_index = self.clinical_df.set_index("Patient ID")
        labels = clinical_index.loc[patients, "OS"].astype(int).to_numpy()
        train_val, test = train_test_split(patients, test_size=self.args.test_size, random_state=self.args.seed, stratify=labels)
        train_val_labels = clinical_index.loc[train_val, "OS"].astype(int).to_numpy()
        train, val = train_test_split(train_val, test_size=self.args.val_size, random_state=self.args.seed, stratify=train_val_labels) if len(train_val) > 1 else (train_val, [])
        return {"train": list(train), "val": list(val), "test": list(test)}

    def make_cv_splits(self):
        from sklearn.model_selection import StratifiedKFold, train_test_split

        patients = self.cached_patients()
        if len(patients) < self.args.cv_folds:
            raise ValueError(f"Need at least cv_folds patients with cached features; got {len(patients)}. {self.cache_status_message(len(patients))}")
        patients = np.asarray(patients)
        clinical_index = self.clinical_df.set_index("Patient ID")
        strata = clinical_index.loc[patients, "OS"].astype(int).to_numpy()
        for repeat in range(self.args.cv_repeats):
            kfold = StratifiedKFold(n_splits=self.args.cv_folds, shuffle=True, random_state=self.args.seed + repeat)
            for fold, (train_val_idx, test_idx) in enumerate(kfold.split(patients, strata)):
                if self.args.fold_id >= 0 and fold != self.args.fold_id:
                    continue
                train_val = patients[train_val_idx].tolist()
                test = patients[test_idx].tolist()
                if len(train_val) > 1 and self.args.val_size > 0:
                    train_val_strata = clinical_index.loc[train_val, "OS"].astype(int).to_numpy()
                    _, class_counts = np.unique(train_val_strata, return_counts=True)
                    stratify = train_val_strata if len(class_counts) > 1 and class_counts.min() >= 2 else None
                    train, val = train_test_split(train_val, test_size=self.args.val_size, random_state=self.args.seed + repeat * 100 + fold, stratify=stratify)
                else:
                    train, val = train_val, []
                yield repeat, fold, {"train": list(train), "val": list(val), "test": list(test)}

    def patient_expression_target(self, patient_id):
        row = self.clinical_df.loc[self.clinical_df["Patient ID"] == patient_id].iloc[0]
        return float(row["OS.time"]), int(row["OS"])

    def patient_forward(self, patient_id):
        feat, coord = self.load_patient(patient_id)
        if feat is None:
            return None
        x = torch.from_numpy(feat).float().to(self.device)
        c = torch.from_numpy(coord).float().to(self.device)
        patient_feat, weights = self.direct_pooling(x, c)
        risk = self.survival_mlp(patient_feat.unsqueeze(0)).squeeze()
        return {"patient_feat": patient_feat, "risk": risk, "weights": weights}

    def train_downstream(self, split):
        train_patients = list(split.get("train", []))
        val_patients = list(split.get("val", [])) or list(train_patients)
        if not train_patients:
            raise ValueError("No training patients available")
        self.reset_downstream_modules()
        clinical_index = self.clinical_df.set_index("Patient ID")
        best_val = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state = None
        history = []

        for epoch in range(self.args.epochs):
            self.direct_pooling.train()
            self.survival_mlp.train()
            epoch_losses = []
            epoch_cox_losses = []
            permutation = np.random.permutation(len(train_patients))
            shuffled = [train_patients[i] for i in permutation]
            for start in range(0, len(shuffled), self.args.patient_batch_size):
                batch_patients = shuffled[start:start + self.args.patient_batch_size]
                risks, times, events = [], [], []
                for patient_id in batch_patients:
                    output = self.patient_forward(patient_id)
                    if output is None:
                        continue
                    risks.append(output["risk"])
                    times.append(torch.tensor(float(clinical_index.loc[patient_id, "OS.time"]), device=self.device))
                    events.append(torch.tensor(float(clinical_index.loc[patient_id, "OS"]), device=self.device))
                if not risks:
                    continue
                risk_scores = torch.stack(risks)
                time_tensor = torch.stack(times)
                event_tensor = torch.stack(events)
                loss = self.args.lambda_cox * self.cox_loss(risk_scores, time_tensor, event_tensor)
                if self.args.lambda_l1 > 0:
                    first_layer = next(self.survival_mlp.mlp.parameters())
                    loss = loss + self.args.lambda_l1 * first_layer.abs().sum()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.args.grad_clip > 0:
                    params = list(self.direct_pooling.parameters()) + list(self.survival_mlp.parameters())
                    torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip)
                self.optimizer.step()
                epoch_losses.append(float(loss.item()))
                epoch_cox_losses.append(float((loss / self.args.lambda_cox).item() if self.args.lambda_cox != 0 else loss.item()))

            val_metrics = self.evaluate_survival_only(val_patients)
            val_cindex = val_metrics["cindex"]
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            avg_cox = float(np.mean(epoch_cox_losses)) if epoch_cox_losses else np.nan
            history.append({"epoch": epoch + 1, "loss": avg_loss, "cox_loss": avg_cox, "val_cindex": val_cindex})
            print(f"Epoch {epoch + 1:03d}/{self.args.epochs} loss={avg_loss:.4f} cox={avg_cox:.4f} val_cindex={val_cindex}")

            improved = np.isfinite(val_cindex) and val_cindex > best_val + self.args.early_stopping_min_delta
            if improved:
                best_val = val_cindex
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_state = {
                    "direct_pooling": self.direct_pooling.state_dict(),
                    "mlp": self.survival_mlp.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_cindex": best_val,
                }
            else:
                epochs_without_improvement += 1
            if self.args.early_stopping_patience > 0 and epochs_without_improvement >= self.args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}: best_val_cindex={best_val} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.direct_pooling.load_state_dict(best_state["direct_pooling"])
            self.survival_mlp.load_state_dict(best_state["mlp"])
        checkpoint = {
            "direct_pooling": self.direct_pooling.state_dict(),
            "mlp": self.survival_mlp.state_dict(),
            "args": vars(self.args),
            "best_val_cindex": best_val,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, self.result_dir / "ge_survival_model.pt")
        pd.DataFrame(history).to_csv(self.result_dir / "ge_survival_training_history.csv", index=False)

    @torch.no_grad()
    def predict_patients(self, patient_ids):
        records = []
        clinical_index = self.clinical_df.set_index("Patient ID")
        self.direct_pooling.eval()
        self.survival_mlp.eval()
        for patient_id in patient_ids:
            output = self.patient_forward(patient_id)
            if output is None:
                continue
            records.append({
                "patient_id": patient_id,
                "OS.time": float(clinical_index.loc[patient_id, "OS.time"]),
                "OS": int(clinical_index.loc[patient_id, "OS"]),
                "risk_score": float(output["risk"].detach().cpu().item()),
            })
        return pd.DataFrame(records)

    def evaluate(self, patient_ids, split_name, write_csv=True, risk_threshold=None):
        df = self.predict_patients(patient_ids)
        metrics = compute_survival_metrics(df, split_name, self.result_dir, risk_threshold=risk_threshold, write_csv=write_csv)
        return df, metrics

    def evaluate_survival_only(self, patient_ids):
        df = self.predict_patients(patient_ids)
        return compute_survival_metrics(df, "_tmp", self.result_dir, write_csv=False)

    def evaluate_split_bundle(self, split):
        train_df, train_metrics = self.evaluate(split.get("train", []), "train", write_csv=True)
        threshold = train_metrics["risk_threshold"]
        val_df, val_metrics = self.evaluate(split.get("val", []), "val", write_csv=True, risk_threshold=threshold)
        test_df, test_metrics = self.evaluate(split.get("test", []), "test", write_csv=True, risk_threshold=threshold)
        summary = {}
        summary.update(prefix_metrics(train_metrics, "train"))
        summary.update(prefix_metrics(val_metrics, "val"))
        summary.update(prefix_metrics(test_metrics, "test"))
        summary["train_n"] = len(train_df)
        summary["val_n"] = len(val_df)
        summary["test_n"] = len(test_df)
        return summary

    def run_single_split(self):
        split = self.split_dataset()
        with open(self.result_dir / "ge_survival_dataset_split.json", "w", encoding="utf-8") as handle:
            json.dump(split, handle, ensure_ascii=False, indent=2)
        print(f"Split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
        self.train_downstream(split)
        summary = self.evaluate_split_bundle(split)
        pd.DataFrame([{**{"experiment": self.args.experiment_name, "encoder": self.encoder_name, "seed": self.args.seed}, **summary}]).to_csv(self.result_dir / "single_split_summary.csv", index=False)

    def run_cross_validation(self):
        base_result_dir = self.result_dir
        rows = []
        for repeat, fold, split in self.make_cv_splits():
            fold_name = f"repeat_{repeat}_fold_{fold}" if self.args.cv_repeats > 1 else f"fold_{fold}"
            self.result_dir = base_result_dir / fold_name
            self.result_dir.mkdir(parents=True, exist_ok=True)
            with open(self.result_dir / "ge_survival_dataset_split.json", "w", encoding="utf-8") as handle:
                json.dump(split, handle, ensure_ascii=False, indent=2)
            set_seed(self.args.seed + repeat * 100 + fold)
            print(f"CV repeat={repeat} fold={fold}: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
            self.train_downstream(split)
            metrics = self.evaluate_split_bundle(split)
            rows.append({"experiment": self.args.experiment_name, "encoder": self.encoder_name, "repeat": repeat, "fold": fold, "seed": self.args.seed, **metrics})
        self.result_dir = base_result_dir
        if rows:
            summary = pd.DataFrame(rows)
            summary.to_csv(base_result_dir / "cv_summary.csv", index=False)
            numeric = summary.select_dtypes(include=[np.number])
            pd.DataFrame({"mean": numeric.mean(), "std": numeric.std()}).to_csv(base_result_dir / "cv_summary_mean_std.csv")
            print(f"Saved CV summary: {base_result_dir / 'cv_summary.csv'}")

    def run(self):
        if self.args.cv_mode == "kfold":
            self.run_cross_validation()
        else:
            self.run_single_split()


def build_parser():
    parser = argparse.ArgumentParser(description="Step 3: train and evaluate H&E-only PathFM-Cox.")
    parser.add_argument("--tcga_slide_dir", default=None, help="Optional pre-cut slide directory; kept for compatibility.")
    parser.add_argument("--clinical_total_path", required=True, help="Clinical file with Patient ID, OS, and OS.time.")
    parser.add_argument("--output_dir", required=True, help="Output root containing the Step 2 feature cache.")
    parser.add_argument("--feature_cache_root", default=None, help="Optional explicit feature-cache directory from Step 2.")
    parser.add_argument("--result_dir", default=None, help="Optional explicit result directory.")
    parser.add_argument("--foundation_root", default=None, help="Kept for CLI compatibility; not used in Step 3.")
    parser.add_argument("--encoder", default="lunit", help="Paper default: lunit.")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--experiment_name", default="PathFM_Cox_Lunit")
    parser.add_argument("--attention_hidden_dim", type=int, default=256)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--no_coord", action="store_true", help="Disable coordinate branch for ablation.")
    parser.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=[128])
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--lambda_cox", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--patient_batch_size", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--cv_mode", choices=["single", "kfold"], default="kfold")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_repeats", type=int, default=5)
    parser.add_argument("--fold_id", type=int, default=-1)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_split_mode", choices=["random", "from_file"], default="random")
    parser.add_argument("--test_patient_path", default=None)
    parser.add_argument("--os_event_value", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--patients", default=None, help="Optional comma- or space-separated patient whitelist.")
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--max_spots_per_patient", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    pipeline = PathFMCOXPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
