"""Step 1: tile H&E SVS whole-slide images into non-overlapping patches.

This script is self-contained and does not depend on data_cut_svs.py.
"""

import argparse
import os
import warnings
from multiprocessing import Pool, cpu_count

import pandas as pd
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


def cut_svs_by_size(svs_path, save_path, patch_size, level=0):
    """Cut one SVS file into non-overlapping patches."""

    import openslide

    os.makedirs(save_path, exist_ok=True)

    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as exc:
        print(f"Failed to open SVS file: {svs_path}, error: {exc}")
        return None, None, None

    if level >= slide.level_count:
        print(f"Level {level} not available for {svs_path}, using lowest available level {slide.level_count - 1}")
        level = slide.level_count - 1

    original_size = slide.level_dimensions[level]
    width, height = original_size
    patch_width, patch_height = patch_size

    num_width = width // patch_width + (1 if width % patch_width != 0 else 0)
    num_height = height // patch_height + (1 if height % patch_height != 0 else 0)
    total_patches = num_width * num_height

    print("=" * 50)
    print("Original SVS file:")
    print(f"  Path: {svs_path}")
    print(f"  Level: {level}")
    print(f"  Size: {width} x {height}")
    print(f"  Patch size: {patch_width} x {patch_height}")
    print(f"  Total levels: {slide.level_count}")
    print("=" * 50)
    print("Tiling plan:")
    print(f"  Columns: {num_width}")
    print(f"  Rows: {num_height}")
    print(f"  Total patches: {total_patches}")
    print("=" * 50)
    print(f"Saving patches to: {save_path}")

    for row in range(num_height):
        for col in range(num_width):
            left = col * patch_width
            upper = row * patch_height
            right = min(left + patch_width, width)
            lower = min(upper + patch_height, height)
            actual_width = right - left
            actual_height = lower - upper

            try:
                patch = slide.read_region((left, upper), level, (actual_width, actual_height))
                patch = patch.convert("RGB")
                patch_index = row * num_width + col + 1
                patch_path = os.path.join(save_path, f"patch_{patch_index}.png")
                patch.save(patch_path)
                if patch_index % 1000 == 0 or patch_index == total_patches:
                    print(f"Saved {patch_index}/{total_patches} patches")
            except Exception as exc:
                print(f"Error while cutting patch {patch_index}: {exc}")
                continue

    slide.close()
    print("=" * 50)
    print("SVS tiling finished")
    print(f"Patch size: {patch_size}")
    print(f"Rows: {num_height}")
    print(f"Columns: {num_width}")
    print(f"Original size: {original_size}")
    print(f"Level: {level}")
    print("=" * 50)

    return num_height, num_width, original_size


def init_stitch_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header_df = pd.DataFrame(columns=[
        "patient_id",
        "svs_path",
        "patch_save_path",
        "stitch_rows",
        "stitch_cols",
        "original_image_size",
        "patch_width",
        "patch_height",
    ])
    header_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Initialized stitch CSV: {csv_path}")


def append_to_stitch_csv(csv_path, params):
    try:
        pd.DataFrame([params]).to_csv(csv_path, mode="a", index=False, header=False, encoding="utf-8")
        print(f"Wrote CSV row for: {params['patient_id']}")
    except Exception as exc:
        print(f"Failed to write CSV row for {params['patient_id']}: {exc}")


def process_single_svs(args):
    svs_filename, svs_root_dir, save_root_dir, patch_size, level = args
    patient_id = os.path.splitext(svs_filename)[0]
    svs_path = os.path.join(svs_root_dir, svs_filename)
    save_path = os.path.join(save_root_dir, patient_id)

    print(f"\n{'=' * 80}")
    print(f"Processing SVS: {svs_filename}")
    print(f"Output folder: {patient_id}")
    print(f"Save path: {save_path}")
    print(f"{'=' * 80}")

    rows, cols, original_size = None, None, None
    if not os.path.exists(svs_path):
        print(f"Warning: SVS file not found -> {svs_path}")
    else:
        try:
            rows, cols, original_size = cut_svs_by_size(
                svs_path=svs_path,
                save_path=save_path,
                patch_size=patch_size,
                level=level,
            )
        except Exception as exc:
            print(f"Error processing SVS {svs_filename}: {exc}")
            rows, cols, original_size = None, None, None

    return {
        "patient_id": patient_id,
        "svs_path": svs_path,
        "patch_save_path": save_path,
        "stitch_rows": rows if rows is not None else "failed",
        "stitch_cols": cols if cols is not None else "failed",
        "original_image_size": original_size if original_size is not None else "failed",
        "patch_width": patch_size[0],
        "patch_height": patch_size[1],
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Step 1: generate H&E WSI patch tiles from SVS files.")
    parser.add_argument("--svs_root_dir", required=True, help="Directory containing raw .svs files.")
    parser.add_argument("--save_root_dir", required=True, help="Output directory for per-slide patch folders.")
    parser.add_argument("--stitch_csv_path", required=True, help="CSV path for recording tiling metadata.")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch width/height. Paper default: 224.")
    parser.add_argument("--level", type=int, default=0, help="OpenSlide pyramid level. Paper default: 0.")
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel worker count.")
    parser.add_argument("--append", action="store_true", help="Append to an existing stitch CSV.")
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

    process_args = [(svs_filename, svs_root_dir, save_root_dir, patch_size, args.level) for svs_filename in svs_files]
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
