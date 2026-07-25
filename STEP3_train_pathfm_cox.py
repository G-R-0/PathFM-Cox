"""Step 3: train and evaluate H&E-only PathFM-Cox with Lunit features.

This script launches pipeline.py in direct survival mode, matching the paper's
basic PathFM-Cox workflow: H&E patch features -> coordinate-aware attention ->
Cox risk prediction.
"""

import argparse



def build_parser():
    parser = argparse.ArgumentParser(
        description="Step 3: train and evaluate H&E-only PathFM-Cox."
    )
    parser.add_argument("--tcga_slide_dir", required=True, help="Root directory of pre-cut slide patch folders.")
    parser.add_argument("--clinical_total_path", required=True, help="Clinical file with Patient ID, OS, and OS.time.")
    parser.add_argument("--output_dir", required=True, help="Output root containing the Step 2 feature cache.")
    parser.add_argument("--foundation_root", default=None, help="Local root directory for foundation models.")
    parser.add_argument("--feature_cache_root", default=None, help="Optional explicit feature-cache directory from Step 2.")
    parser.add_argument("--encoder", default="lunit", help="Paper default: lunit.")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--experiment_name", default="PathFM_Cox_Lunit")
    parser.add_argument("--attention_hidden_dim", type=int, default=256)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
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
    parser.add_argument("--no_coord", action="store_true", help="Disable coordinate branch for ablation.")
    return parser


def add_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def to_pipeline_args(args):
    from pipeline import build_parser as build_pipeline_parser
    cmd = [
        "--stage", "downstream",
        "--survival_mode", "direct",
        "--tcga_slide_dir", args.tcga_slide_dir,
        "--clinical_total_path", args.clinical_total_path,
        "--os_event_value", str(args.os_event_value),
        "--output_dir", args.output_dir,
        "--encoder", args.encoder,
        "--feature_dim", str(args.feature_dim),
        "--experiment_name", args.experiment_name,
        "--attention_hidden_dim", str(args.attention_hidden_dim),
        "--attention_dropout", str(args.attention_dropout),
        "--mlp_dropout", str(args.mlp_dropout),
        "--lambda_cox", str(args.lambda_cox),
        "--lambda_l1", str(args.lambda_l1),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--epochs", str(args.epochs),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_min_delta", str(args.early_stopping_min_delta),
        "--patient_batch_size", str(args.patient_batch_size),
        "--grad_clip", str(args.grad_clip),
        "--cv_mode", args.cv_mode,
        "--cv_folds", str(args.cv_folds),
        "--cv_repeats", str(args.cv_repeats),
        "--fold_id", str(args.fold_id),
        "--test_size", str(args.test_size),
        "--val_size", str(args.val_size),
        "--test_split_mode", args.test_split_mode,
        "--seed", str(args.seed),
    ]
    cmd.append("--mlp_hidden_dims")
    cmd.extend(str(hidden_dim) for hidden_dim in args.mlp_hidden_dims)
    add_optional(cmd, "--foundation_root", args.foundation_root)
    add_optional(cmd, "--feature_cache_root", args.feature_cache_root)
    add_optional(cmd, "--test_patient_path", args.test_patient_path)
    add_optional(cmd, "--patients", args.patients)
    add_optional(cmd, "--max_patients", args.max_patients)
    add_optional(cmd, "--max_spots_per_patient", args.max_spots_per_patient)
    if args.cpu:
        cmd.append("--cpu")
    if args.no_coord:
        cmd.append("--no_coord")
    return build_pipeline_parser().parse_args(cmd)


def main():
    args = build_parser().parse_args()
    from data import set_seed
    from pipeline import HEGESurvivalPipeline

    pipeline_args = to_pipeline_args(args)
    set_seed(pipeline_args.seed)
    HEGESurvivalPipeline(pipeline_args).run()


if __name__ == "__main__":
    main()


