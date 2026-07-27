# PathFM-Cox

PathFM-Cox: Coordinate-aware Survival Prediction from H&E Whole Slide Images Using Pathology Foundation Representations

PathFM-Cox is a weakly supervised survival prediction framework for hematoxylin and eosin (H&E) whole-slide images (WSIs). It uses a frozen pathology foundation encoder to extract patch-level morphological representations, aggregates morphology and slide-local coordinates with coordinate-aware attention, and predicts patient-level survival risk with a Cox model.

## Graphic Abstract

<img width="1752" height="725" alt="framework" src="https://github.com/user-attachments/assets/4cf7329a-4dd4-4a02-a53a-ffb30e4341d1" />


## Step 1: generate H&E WSI patch tiles

Prepare the TCGA-SKCM H&E whole-slide images in `.svs` format, then run:

```bash
python STEP1_generate_wsi_patch_tiles.py \
  --svs_root_dir /path/to/slide-svs \
  --save_root_dir /path/to/slide-cut \
  --stitch_csv_path /path/to/svs_stitch_parameters.csv \
  --patch_size 224 \
  --level 0 \
  --num_workers 32
```

This step cuts each SVS into non-overlapping `224 x 224` H&E patches named `patch_1.png`, `patch_2.png`, ... and records the tiling parameters for each slide.

## Step 2: extract Lunit foundation patch features

Use the cached patch folders from Step 1 and run:

```bash
python STEP2_extract_lunit_patch_features.py \
  --tcga_slide_dir /path/to/slide-cut \
  --clinical_total_path /path/to/TCGA_primary.csv \
  --output_dir /path/to/results \
  --stitch_csv_path /path/to/svs_stitch_parameters.csv \
  --encoder lunit \
  --feature_dim 512 \
  --image_size 224 \
  --batch_size 256 \
  --num_workers 16 \
  --tissue_threshold 0.01
```

This step filters background patches, normalizes retained patches with ImageNet statistics, extracts frozen Lunit embeddings, projects features to `512` dimensions when needed, and caches patch features with slide-local grid coordinates.

## Step 3: train and evaluate H&E-only PathFM-Cox

Use the cached features from Step 2 and run:

```bash
python STEP3_train_pathfm_cox_lunit.py \
  --clinical_total_path /path/to/TCGA_primary.csv \
  --output_dir /path/to/results \
  --feature_cache_root /path/to/results/feature_cache/he_lunit_spot_cache \
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

This step trains the H&E-only PathFM-Cox model. The Lunit encoder remains frozen, while the coordinate-aware attention aggregation module and Cox survival head are optimized. The model reconstructs slide-local tile-grid coordinates from patch indices, applies sinusoidal coordinate encoding, combines morphology-driven and coordinate-guided attention scores, aggregates patch features into a patient-level WSI representation, and optimizes the Cox partial-likelihood loss for survival risk prediction.

The outputs include trained model checkpoints, cross-validation summaries, patient-level risk scores, C-index, log-rank test, hazard ratio, and Kaplan-Meier curves.
