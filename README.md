# PathFM-Cox

PathFM-Cox: Coordinate-aware Survival Prediction from H&E Whole Slide Images Using Pathology Foundation Representations

PathFM-Cox is a weakly supervised survival prediction framework for hematoxylin and eosin (H&E) whole-slide images (WSIs). It uses a frozen pathology foundation encoder to extract patch-level morphological representations, integrates morphological features and slide-local coordinates through coordinate-aware attention, and predicts patient-level survival risk using a Cox survival model.

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

This step divides each WSI into non-overlapping patches with a nominal size of 224 × 224 pixels, named patch_1.png, patch_2.png, and so on. Boundary patches may be smaller. The original patch indices and slide-specific grid dimensions are recorded to enable subsequent coordinate reconstruction.


## Step 2: extract Lunit foundation patch features

Use the cached patch folders from Step 1 and run:

```bash
python STEP2_extract_patch_features.py \
    --tcga_slide_dir /path/to/slide-cut \
    --clinical_total_path /path/to/TCGA_primary.csv \
    --output_dir /path/to/results \
    --stitch_csv_path /path/to/svs_stitch_parameters.csv \
    --foundation_root /path/to/foundation_models \
    --encoder lunit \
    --feature_dim 512 \
    --image_size 224 \
    --batch_size 256 \
    --num_workers 16 \
    --tissue_threshold 0.01 \
    --seed 45
```

This step reconstructs slide-local grid coordinates using the original patch indices and recorded grid dimensions, filters background patches, resizes retained patches to 224 × 224 pixels, and normalizes them using ImageNet statistics.

Patch-level morphological representations are extracted using the frozen Lunit encoder and projected to 512 dimensions through a fixed, randomly initialized projection when necessary. The original spatial indices are preserved after background filtering, and the extracted features and corresponding coordinates are cached for downstream survival modeling.

## Step 3: train and evaluate H&E-only PathFM-Cox

Use the cached features from Step 2 and run:

```bash
python STEP3_train_pathfm_cox.py \
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
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --lambda_cox 1.0 \
    --cv_mode kfold \
    --cv_folds 5 \
    --cv_repeats 5 \
    --cv_seeds 42 45 123 456 789 \
    --val_size 0.15 \
    --patient_batch_size 16 \
    --epochs 200 \
    --early_stopping_patience 20
```

This step trains the H&E-only PathFM-Cox model using frozen patch-level morphological representations. For patients with multiple WSIs, patch features and their corresponding slide-local coordinates are concatenated at the patient level.

The model applies patient-level coordinate normalization and sinusoidal positional encoding, combines morphology-driven and coordinate-guided attention scores, and aggregates patch features into a patient-level representation. The coordinate-aware attention module and survival prediction head are optimized using the mini-batch Cox partial-likelihood loss.

Model performance is evaluated using five repetitions of patient-level stratified five-fold cross-validation. Model checkpoints are selected according to validation C-index and evaluated on the corresponding held-out test sets.

For exact reproduction of the original patient-level data splits, the original split manifest can be supplied using --manifest /path/to/folds.json, together with the corresponding experimental configuration.

The outputs include trained model checkpoints, training histories, cross-validation summaries, patient-level risk scores, C-index, log-rank test results, hazard ratios, and Kaplan–Meier survival curves.

The outputs include trained model checkpoints, cross-validation summaries, patient-level risk scores, C-index, log-rank test, hazard ratio, and Kaplan-Meier curves.
