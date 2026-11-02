# Facial Keypoint Detection — Ablation Study

Group course project for the **ISAO 2025–2026 Practical Deep Learning** course, taught by **Professor Mélodie Monod**.

This project studies facial landmark regression on 96 × 96 grayscale images. It compares three neural network architectures, raw and deduplicated data splits, and five augmentation strategies. It supports both fully labeled examples and partially labeled examples through a masked mean squared error loss.

The course report is included in [report.pdf](report.pdf). The source has been reorganized from the original single-file submission into the `facial_keypoints` package. `code.py` remains a command-line entry point.

## Repository structure

```text
.
├── README.md
├── report.pdf                  # Submitted course report
├── code.py                     # Compatibility CLI entry point
├── pyproject.toml              # Package metadata and dependencies
├── configs/
│   └── ablation.yaml           # Example 45-cell augmentation matrix
└── facial_keypoints/
    ├── __main__.py             # python -m facial_keypoints
    ├── cli.py                  # Command-line commands
    ├── paths.py                # Portable paths and environment overrides
    ├── schemas.py              # Ordered landmark coordinate schemas
    ├── profile.py              # Lazy loading of calibrated EDA thresholds
    ├── cohorts.py              # Frontal, asymmetric, hard-profile flags
    ├── data.py                 # CSV loading, tensors, datasets, loaders
    ├── losses.py               # Plain and masked MSE
    ├── models.py               # MLP-2, shallow CNN, pretrained ResNet-18
    ├── augmentation.py         # Transform atoms and strategies S1–S5
    ├── training.py             # Training, validation, checkpoints, metrics
    ├── matrix.py               # YAML experiment expansion and execution
    ├── tuning.py               # Random hyperparameter search
    ├── data_prep/
    │   ├── audit.py            # Cross-split duplicate audit
    │   ├── full.py             # Full-data splits, masks, deduplication
    │   └── large.py            # Large-data deduplication
    └── analysis/
        ├── style.py            # Shared plotting style
        ├── summary.py          # Aggregate experiment results
        ├── figures.py          # Main figures
        ├── tables.py           # CSV and Word tables
        ├── deep.py             # Additional analyses
        ├── hp.py               # Legacy HP-search log visualization
        ├── augmentation.py     # Augmentation illustrations
        ├── detail.py           # Data and experiment detail figures
        └── architecture.py     # Model architecture diagram
```

## Installation

Use **Python 3.10 or newer**. Python 3.12 is a suitable choice. From a terminal:

```bash
git clone https://github.com/WuqianMa/Course_Project_Face_Keypoint.git
cd Course_Project_Face_Keypoint
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate the environment with `.venv\Scripts\activate`. PyTorch and torchvision must be installed as a compatible pair; use the appropriate PyTorch build for your CUDA environment if you train on a GPU. Training selects CUDA when available and otherwise uses the CPU.

All three entry points expose the same commands:

```bash
python code.py --help
python -m facial_keypoints --help
facial-keypoints --help
```

These diagnostics work without the dataset or experiment thresholds:

```bash
python code.py loss-test
python code.py model-info
python code.py arch-figure
```

`model-info` builds the MLP and shallow CNN. It skips constructing the pretrained ResNet to avoid downloading weights. Training ResNet-18 downloads torchvision's ImageNet weights on first use unless they are already cached. `arch-figure` writes `figures/F_arch.pdf`.

## Data and configuration

**Dataset CSV files, trained checkpoints, historical run logs, and calibrated EDA thresholds are not included.** Restore the course project's data or prepare CSVs with the schemas below before training. The raw source is the Kaggle Facial Keypoints Detection `training.csv`.

By default, the expected layout is:

```text
Data/
├── facial-keypoints-detection/
    └── training.csv
├── data_small_{train,val,test}.csv
├── data_large_{train,val,test}.csv
├── Clean Data/
    ├── data_large_clean_dedup_{train,val,test}.csv
    ├── data_full_clean_{train,val,test}.csv
    └── data_full_clean_dedup_{train,val,test}.csv
└── Raw Data/                   # Additional inputs for detail-figures
    ├── data_small.csv
    ├── data_large.csv
    ├── data.csv
    └── data_small_train.csv
results/
└── eda/
    └── profile_thresholds.json
```

The `{train,val,test}` notation represents three separate files. `Data/` contains source inputs; lowercase `data/` contains generated analysis summaries. On filesystems that ignore case, these share a directory, so keep their filenames distinct.

| CLI dataset | Location and filename prefix | Coordinates | Missing-label masks |
| --- | --- | --- | --- |
| `small_raw` | `Data/data_small` | 30 (15 landmarks) | No |
| `large_raw` | `Data/data_large` | 8 (4 landmarks) | No |
| `large_dedup` | `Data/Clean Data/data_large_clean_dedup` | 8 | No |
| `full_raw_clean` | `Data/Clean Data/data_full_clean` | 30 | Yes |
| `full_dedup` | `Data/Clean Data/data_full_clean_dedup` | 30 | Yes |

Every CSV needs an `Image` column containing 9,216 space-separated grayscale pixel values, plus landmark columns in the schema defined in `facial_keypoints/schemas.py`. The four-landmark schema contains the left and right eye centers, nose tip, and bottom-lip center, each with `_x` and `_y` coordinates.

Images are scaled by `1/255`. Coordinates are normalized as `(coordinate - 48) / 48`. Full-data CSVs also need a `mask_<coordinate_column>` for each coordinate: `1` means observed and `0` means missing. Missing target values are sanitized before loss computation. The masked loss averages squared error over observed coordinates only.

Paths can be configured before starting Python:

```bash
export FACE_KEYPOINTS_DATA_DIR="/absolute/path/to/Data"
export FACE_KEYPOINTS_RESULTS_DIR="/absolute/path/to/results"
export FACE_KEYPOINTS_PROFILE_PATH="/absolute/path/to/profile_thresholds.json"
# Optional: clean CSV directory used specifically by augmentation/detail figures.
export PSL_DL_DATA_ROOT="/absolute/path/to/Data/Clean Data"
```

The profile file must contain the original EDA-calibrated values for these keys:

| JSON section | Required numeric keys |
| --- | --- |
| `data_small` | `profile_cutoff_px` |
| `data_large` | `tau_asym_px`, `tau_hard_px`, `eps_dx_px` |
| `data_full` (optional) | `profile_cutoff_px` |

If `data_full` is omitted, its cutoff is inherited from `data_small`. Thresholds are loaded only when needed. They are not estimated by `eda-dup`; restore them from the original experiment or calibrate them on your training data. Missing thresholds produce an error explaining the required path.

## Prepare and audit splits

```bash
# Build full-data splits and mask columns from Kaggle training.csv.
python code.py build-data-full

# Deduplicate existing data_large train/validation/test CSVs.
python code.py build-data-large-dedup

# Audit existing small, large, and full clean splits for duplicate images.
python code.py eda-dup
```

The full-data builder creates deterministic 60/20/20 splits with seed 42, then writes both nondeduplicated and deduplicated versions. The large-data builder requires the existing large-data splits; this repository does not generate the original small/large source splits. Deduplication uses SHA1 fingerprints of pixel strings, retaining test examples ahead of validation examples and validation examples ahead of training examples. It removes cross-split overlap; it does not remove every duplicate within a split. Preparation commands write or replace generated CSVs and JSON reports. The duplicate audit writes under `results/eda/`.

## Models and augmentation

| Model argument | Architecture | Default learning rate |
| --- | --- | --- |
| `mlp2` | Flatten → 512-unit ReLU hidden layer → coordinates | `1e-3` |
| `cnn_shallow` | Three convolution/pooling blocks → dense regression head | `1e-3` |
| `resnet18_pretrained` | ImageNet ResNet-18 with grayscale input and a new regression head | `1e-4` |

ResNet's first convolution is initialized by averaging its pretrained RGB weights across channels. All models use Adam; defaults are 30 epochs, batch size 128, and weight decay `1e-4`.

| Strategy | Transforms |
| --- | --- |
| `none` | No augmentation |
| `S1` | Brightness, contrast, Gaussian noise |
| `S2` | Horizontal flip, rotation, translation |
| `S3` | S2 geometry + S1 photometric transforms |
| `S4` | S3 + scaling and Gaussian blur |
| `S5` | S3 + landmark-aware random erasing |

Transforms are probabilistic. Horizontal flips swap left/right landmark identities, and the dataset synchronizes label masks. Cohort gates reduce flipping and rotation for asymmetric or hard-profile examples. See `augmentation.py` and `cohorts.py` for probabilities, bounds, and retry rules.

## Run experiments

Train one experiment:

```bash
python code.py train \
  --dataset large_dedup \
  --model cnn_shallow \
  --strategy S3 \
  --seed 0 \
  --epochs 30 \
  --out_dir results/03_aug
```

Each run writes a checkpoint in `ckpt/`, per-epoch JSONL in `logs/`, a JSON summary in `results/`, and `training_log.txt` inside the chosen output directory. Filenames use `dataset__model__strategy__seedN`. Existing completed runs are skipped unless `--overwrite` is supplied. Checkpoints contain the state from the best validation epoch.

The trainer reports normalized validation MSE and pixel RMSE. It does **not** evaluate the held-out test set. Compare metrics within matching landmark schemas and evaluation splits.

Preview or execute the supplied 45-cell augmentation matrix:

```bash
python code.py matrix configs/ablation.yaml --dry-run
python code.py matrix configs/ablation.yaml
python code.py matrix configs/ablation.yaml --only cnn_shallow full_dedup
```

Matrix YAML supports a Cartesian `matrix` over `dataset`, `model`, `strategy`, and `seed`, plus explicit `cells`. Duplicate cells are removed. Set training options in `defaults`; relative matrix output paths resolve from the repository root. Single-run and HP-search `--out_dir` paths resolve from the current working directory. The matrix writes `manifest.json`, including any per-run failures; check it after execution.

Run random hyperparameter search:

```bash
python code.py hp-search \
  --dataset large_dedup \
  --model mlp2 \
  --strategy S3 \
  --n-trials 8 \
  --epochs 30 \
  --out_dir runs_hpsearch
```

The search varies learning rate, batch size, and weight decay; MLP searches also vary hidden width. It writes per-trial runs and `search_results.csv`, ordered by validation MSE. The CSV's legacy `test_mse` field remains empty because test evaluation is not implemented. Training runs the requested number of epochs without early stopping.

## Rebuild analysis outputs

The summary stage expects experiments grouped as:

```text
results/
├── 01_naive/     # Baseline runs
├── 02_dedup/     # Deduplicated-data runs
└── 03_aug/       # Augmentation runs
```

Each block should contain the `results/*.json` and `logs/*.jsonl` produced by training. Configure matrix output directories accordingly. The example YAML covers the augmentation block only.

```bash
python code.py summary          # Writes data/tier1_full_{long,epoch}.csv
python code.py figures          # Main figures F1–F5
python code.py tables           # Main CSV/Word tables T1–T5
python code.py deep-analysis    # Additional figures F6–F11 and tables T6–T12
python code.py hp-figure        # Historical master.log visualization
python code.py aug-figures      # Augmentation figures F12–F16; requires data
python code.py detail-figures   # Detail figures F17–F23; requires data
python code.py arch-figure      # Standalone architecture diagram
```

`all-analysis` runs these stages in order, including summary. Figures go to `figures/`, tables to `tables/`, and merged CSVs to lowercase `data/`.

These scripts retain the course report's experiment layout: summary count checks expect 9 baseline, 6 deduplication, and 45 augmentation cells. Some detail diagrams contain fixed sample counts from the original experiment. T1 is a placeholder for legacy motivation results. `hp-figure` expects the original `logs/master.log` format, rather than the new search CSV. Full report reproduction therefore requires the original data, thresholds, and experiment artifacts; a single new training run does not supply all inputs.

## Development and verification

```bash
ruff check .
```

The refactor passed 15 temporary regression checks using CPU computation, covering imports, CLI entry points, model shapes, masked-loss gradients, flip/mask synchronization, telemetry, deduplication, matrix expansion, and one-epoch training/search/aggregation on synthetic CSVs. The temporary tests were deleted after passing, as requested by the maintainer. Full-scale training and report reproduction require the external inputs described above.

## Contributors

- **Course project team** — original experiments, implementation, and [course report](report.pdf).
- **[WuqianMa](https://github.com/WuqianMa)** — repository owner.
- **OpenAI Codex (GPT-based assistant)** — contributor for the modular refactor, README, portability fixes, hyperparameter-search wiring, and temporary CPU regression checks. This is an attribution of AI assistance.

## License and data rights

No project license has been specified. Consult the project authors before reuse or redistribution, and follow the source dataset's terms when obtaining or sharing data.
