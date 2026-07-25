# PathFM-Cox

PathFM-Cox: Coordinate-aware Survival Prediction from H&E Whole Slide Images Using Pathology Foundation Representations

PathFM-Cox is a weakly supervised survival prediction framework for hematoxylin and eosin (H&E) whole-slide images (WSIs). It uses a frozen pathology foundation encoder to extract patch-level morphological representations, aggregates morphology and slide-local coordinates with coordinate-aware attention, and predicts patient-level survival risk with a Cox model.

## Graphic Abstract

![Graphic Abstract](HE_GE_figure_components/HE_GE_figure_components/PNG/00_component_contact_sheet.png)

## Step 1: generate H&E WSI patch tiles

First, prepare the TCGA-SKCM H&E whole-slide images in `.svs` format.

Second, edit the paths in `data_cut_svs.py` according to your local environment:

- `svs_root_dir`: root directory of raw `.svs` files.
- `save_root_dir`: output directory for tiled patch folders.
- `stitch_csv_path`: output CSV file for slide tiling metadata.
- `patch_size`: patch size; the paper uses non-overlapping `224 x 224` patches.
- `level`: OpenSlide pyramid level; the paper uses level `0`.

Run:

```bash
python data_cut_svs.py
```

or submit the provided SLURM script:

```bash
sbatch run_TCGA_cut.sh
```

This step cuts each SVS into non-overlapping H&E patches named `patch_1.png`, `patch_2.png`, ..., and records the tiling parameters for each slide.

## Step 2: extract Lunit foundation patch features

PathFM-Cox uses a frozen pathology foundation encoder for patch-level representation extraction. In the paper-level setting, we use `lunit` as the default encoder. The encoder is defined in `encoders.py`, and feature extraction is launched by `pipeline.py` with `--stage encode`.

Run:

```bash
python pipeline.py \
  --stage encode \
  --survival_mode direct \
  --tcga_slide_dir /path/to/slide-cut \
  --clinical_total_path /path/to/TCGA_primary.csv \
  --output_dir /path/to/results \
  --foundation_root /path/to/HE_foundation \
  --encoder lunit \
  --feature_dim 512 \
  --image_size 224 \
  --batch_size 256 \
  --num_workers 16 \
  --tissue_threshold 0.01 \
  --resume
```

This step removes background patches with tissue ratio below `0.01`, normalizes retained patches with ImageNet mean and standard deviation, extracts frozen Lunit embeddings, projects features to `512` dimensions when needed, and caches patch features with slide-local grid coordinates.

## Step 3: train and evaluate H&E-only PathFM-Cox

Run `pipeline.py` with `--stage downstream` and `--survival_mode direct` to train the H&E-only PathFM-Cox model. Only the coordinate-aware attention aggregation module and Cox survival head are optimized; the Lunit encoder remains frozen.

Run:

```bash
python pipeline.py \
  --stage downstream \
  --survival_mode direct \
  --tcga_slide_dir /path/to/slide-cut \
  --clinical_total_path /path/to/TCGA_primary.csv \
  --output_dir /path/to/results \
  --foundation_root /path/to/HE_foundation \
  --encoder lunit \
  --feature_dim 512 \
  --experiment_name PathFM_Cox_Lunit \
  --attention_hidden_dim 256 \
  --attention_dropout 0.1 \
  --mlp_hidden_dims 128 \
  --mlp_dropout 0.2 \
  --lambda_cox 1.0 \
  --lambda_l1 0.0 \
  --cv_mode kfold \
  --cv_folds 5 \
  --cv_repeats 5 \
  --val_size 0.15 \
  --patient_batch_size 16 \
  --epochs 200 \
  --early_stopping_patience 20
```

This step reconstructs slide-local tile-grid coordinates from patch indices, applies sinusoidal coordinate encoding, combines morphology-driven and coordinate-guided attention scores, aggregates patch features into a patient-level WSI representation, and optimizes the Cox partial-likelihood loss for survival risk prediction.

The outputs include trained model checkpoints, cross-validation summaries, patient-level risk scores, C-index, log-rank test, hazard ratio, and Kaplan-Meier curves.
