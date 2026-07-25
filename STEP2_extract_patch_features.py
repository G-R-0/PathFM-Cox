"""Step 2: extract frozen Lunit pathology foundation features from H&E patches.

This script launches pipeline.py in H&E-only feature-extraction mode. It keeps the
foundation encoder frozen and writes reusable patch-feature caches.
"""

import argparse



def build_parser():
    parser = argparse.ArgumentParser(
        description="Step 2: extract frozen Lunit patch features for PathFM-Cox."
    )
    parser.add_argument("--tcga_slide_dir", required=True, help="Root directory of pre-cut slide patch folders.")
    parser.add_argument("--clinical_total_path", required=True, help="Clinical file with Patient ID, OS, and OS.time.")
    parser.add_argument("--output_dir", required=True, help="Output root for feature cache and results.")
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
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_tissue_filter", action="store_true")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume; by default cached slides are skipped.")
    return parser


def add_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def to_pipeline_args(args):
    from pipeline import build_parser as build_pipeline_parser
    cmd = [
        "--stage", "encode",
        "--survival_mode", "direct",
        "--tcga_slide_dir", args.tcga_slide_dir,
        "--clinical_total_path", args.clinical_total_path,
        "--os_event_value", str(args.os_event_value),
        "--output_dir", args.output_dir,
        "--encoder", args.encoder,
        "--feature_dim", str(args.feature_dim),
        "--image_size", str(args.image_size),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--tissue_threshold", str(args.tissue_threshold),
        "--tissue_sat_threshold", str(args.tissue_sat_threshold),
        "--tissue_white_threshold", str(args.tissue_white_threshold),
        "--seed", str(args.seed),
    ]
    add_optional(cmd, "--foundation_root", args.foundation_root)
    add_optional(cmd, "--encoder_weight_path", args.encoder_weight_path)
    add_optional(cmd, "--feature_cache_root", args.feature_cache_root)
    add_optional(cmd, "--patients", args.patients)
    add_optional(cmd, "--max_patients", args.max_patients)
    if args.cpu:
        cmd.append("--cpu")
    if args.skip_tissue_filter:
        cmd.append("--skip_tissue_filter")
    if not args.no_resume:
        cmd.append("--resume")
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

