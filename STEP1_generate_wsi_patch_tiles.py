"""Step 1: tile H&E SVS whole-slide images into non-overlapping patches.

This script is a paper-level wrapper around data_cut_svs.py. It does not modify
or depend on the hard-coded paths in data_cut_svs.py; all paths are provided by
command-line arguments.
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm



def build_parser():
    parser = argparse.ArgumentParser(
        description="Step 1: generate H&E WSI patch tiles from SVS files."
    )
    parser.add_argument("--svs_root_dir", required=True, help="Directory containing raw .svs files.")
    parser.add_argument("--save_root_dir", required=True, help="Output directory for per-slide patch folders.")
    parser.add_argument("--stitch_csv_path", required=True, help="CSV path for recording tiling metadata.")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch width/height. Paper default: 224.")
    parser.add_argument("--level", type=int, default=0, help="OpenSlide pyramid level. Paper default: 0.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: min(SLURM_NTASKS_PER_NODE or CPU count, CPU count).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing stitch CSV instead of recreating it.",
    )
    return parser


def resolve_num_workers(value):
    if value is not None:
        return max(1, min(int(value), cpu_count()))
    slurm_workers = os.environ.get("SLURM_NTASKS_PER_NODE")
    if slurm_workers:
        return max(1, min(int(slurm_workers), cpu_count()))
    return cpu_count()


def ensure_stitch_csv(csv_path, append=False):
    if append and os.path.exists(csv_path):
        return
    init_stitch_csv(csv_path)


def main():
    args = build_parser().parse_args()
    from data_cut_svs import append_to_stitch_csv, init_stitch_csv, process_single_svs

    svs_root_dir = os.path.abspath(args.svs_root_dir)
    save_root_dir = os.path.abspath(args.save_root_dir)
    stitch_csv_path = os.path.abspath(args.stitch_csv_path)
    patch_size = (int(args.patch_size), int(args.patch_size))
    num_workers = resolve_num_workers(args.num_workers)

    if not os.path.isdir(svs_root_dir):
        raise FileNotFoundError(f"SVS directory not found: {svs_root_dir}")

    os.makedirs(save_root_dir, exist_ok=True)
    ensure_stitch_csv(stitch_csv_path, append=args.append)

    svs_files = sorted(f for f in os.listdir(svs_root_dir) if f.lower().endswith(".svs"))
    if not svs_files:
        raise FileNotFoundError(f"No .svs files found in: {svs_root_dir}")

    print(f"SVS directory: {svs_root_dir}")
    print(f"Patch output: {save_root_dir}")
    print(f"Stitch CSV: {stitch_csv_path}")
    print(f"Found {len(svs_files)} SVS files")
    print(f"Patch size: {patch_size[0]} x {patch_size[1]}, level={args.level}, workers={num_workers}")

    process_args = [
        (svs_filename, svs_root_dir, save_root_dir, patch_size, args.level)
        for svs_filename in svs_files
    ]

    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_svs, process_args), total=len(process_args)):
            append_to_stitch_csv(stitch_csv_path, result)
            results.append(result)

    summary = pd.DataFrame(results)
    success_count = int((summary["stitch_rows"] != "failed").sum()) if not summary.empty else 0
    print("Step 1 finished")
    print(f"Slides processed successfully: {success_count}/{len(svs_files)}")


if __name__ == "__main__":
    main()


