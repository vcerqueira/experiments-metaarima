# MetaARIMA: Automatic Configuration of ARIMA using Metalearning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Reproducibility package for:**
> MetaARIMA: Automatic Configuration of ARIMA using Metalearning.

`assets/README_example.md` is a formatting example from another paper. It is not part of this package.

**Everything except the two workflow diagrams is produced by the scripts.** `assets/` ships only `stage1.png` and `stage2.png`. Models, metadata, scores, and plots are written there when you run the commands below. TSFM inference runs on CPU.

---

## Reproducibility Package Information

- **Date assembled:** 13 October 2026
- **Authors:** TODO — replace with the package author name (`TODO — email`)
- **Affiliation:** TODO — replace with affiliation
- **Licence:** MIT (`LICENSE`)

Replace the copyright line in `LICENSE` with the author name when you fill this block in. Questions about running the package should go to the email above, or to an issue on this repository.

---

## Repository Structure

```
.
├── src/                        # MetaARIMA library
│   ├── chronos_data.py         # Hugging Face loader, horizons, time split
│   ├── config.py               # Grids, CatBoost hyperparameters, RANDOM_SEED
│   └── meta/arima/             # Model, metadata reader, optimizers
├── scripts/
│   ├── collecting_metadata/    # Feature extraction, ARIMA metadata, training
│   ├── experiments/
│   │   ├── main/               # Benchmark (main.py) and sensitivity/
│   │   └── analysis/           # Tables and figures from the CSVs the benchmark writes
│   └── tsfm/                   # Chronos-2, TimesFM, and Moirai-2 scripts
├── assets/
│   ├── stage1.png, stage2.png  # Workflow diagrams, included
│   ├── trained_metaarima_m4_{monthly,quarterly,yearly}_catboost.joblib.gz  # written by train_metaarima.py
│   ├── metadata/               # written by the collectors, gitignored
│   └── results/                # written by the benchmark, sensitivity, and TSFM scripts
│       ├── main/               # Per-dataset benchmark CSVs
│       ├── sensitivity/        # Ablation and sensitivity CSVs
│       ├── tsfm/               # Foundation-model scores and predictions
│       └── plots/              # PDFs written by the analysis scripts
├── requirements.txt            # Core environment, pinned
├── requirements-tsfm.txt       # Separate CPU environment for the TSFM scripts, pinned
└── README.md
```

Run every command below from the repository root. Scripts import `src`, so prefix them with `PYTHONPATH=.`.

---

## Computing Environment

| Component | Light check (tables and figures) | Full CPU experiments | TSFM scripts |
|---|---|---|---|
| **OS** | Any | TODO — record the machine used for the paper runs | Any. The committed scores were produced on an NVIDIA GeForce RTX 2050 |
| **CPU / RAM** | A laptop is enough | TODO — record processor and RAM | A laptop is enough. CPU inference is slower than the original GPU runs |
| **GPU** | None | None | None. The scripts load the models on CPU |
| **Python** | 3.12 | 3.12 | 3.12 |

**Language:** Python 3.12
**Package manager:** conda (or any environment that can `pip install` the pinned files)

Core packages are pinned in `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.0 | Numerical computation |
| `pandas` | 2.2.3 | Data frames |
| `scikit-learn` | 1.5.2 | PCA and multi-output wrappers |
| `statsforecast` | 1.7.8 | ARIMA and statistical baselines |
| `statsmodels` | 0.15.0 | ARIMA diagnostics |
| `utilsforecast` | 0.2.11 | MASE |
| `tsfeatures` | 0.4.5 | Time-series features |
| `datasets` | 4.4.0 | Hugging Face download |
| `catboost` | 1.2.8 | Metamodel |
| `scipy` | 1.15.3 | Ranking utilities |
| `plotnine` | 0.15.8 | Figures |

The TSFM scripts use a **separate environment** (`requirements-tsfm.txt`). `uni2ts==2.0.0` needs `torch<2.5` and `numpy~=1.26`, which conflicts with `numpy==2.4.0` in the core file. Do not install both files into one environment. The default PyTorch wheel in that file runs on CPU.

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.4.1 | Newest build that satisfies both `uni2ts` (`<2.5`) and `chronos-forecasting` (`>=2.2`) |
| `chronos-forecasting` | 2.2.2 | Chronos-2 |
| `uni2ts` | 2.0.0 | Moirai-2 |
| `transformers` | 4.57.6 | TimesFM |

---

## Installation

### Core environment (light check and CPU experiments)

```bash
conda create -n metaarima python=3.12 -y
conda activate metaarima
pip install -r requirements.txt
```

Generated metadata is read from `assets/metadata`. That directory is created by the collection scripts. No environment file is required.

### TSFM environment

```bash
conda create -n metaarima-tsfm python=3.12 -y
conda activate metaarima-tsfm
pip install -r requirements-tsfm.txt
```

---

## Data Description

All series are public configs of the Hugging Face dataset [`autogluon/chronos_datasets`](https://huggingface.co/datasets/autogluon/chronos_datasets). No registration and no local download step: `ChronosDataset` in `src/chronos_data.py` calls `datasets.load_dataset` on first use.

The loader is pinned to Hub revision `eeecad0b82a8c237e212ce6f8d1abecb513e2cec` so a later update of the dataset cannot change the series. The dataset card lists the licence as `other`. The underlying collections are the public forecasting competitions below.

| Role | Hugging Face config | Original source |
|---|---|---|
| Metamodel training | `m4_monthly`, `m4_quarterly`, `m4_yearly` | [M4 competition](https://github.com/Mcompetitions/M4-methods) (Makridakis et al.) |
| Evaluation | `monash_m1_{monthly,quarterly,yearly}` | M1 competition, via the [Monash forecasting repository](https://forecastingdata.org/) |
| Evaluation | `monash_m3_{monthly,quarterly,yearly}` | M3 competition, via the same repository |
| Evaluation | `monash_tourism_{monthly,quarterly,yearly}` | Tourism competition, via the same repository |

`ChronosDataset.load_everything` reads the `train` split. `time_wise_split` holds out the last `horizon` observations of each series as the test window. Horizons are in `src/chronos_data.py`:

| Frequency | M4 horizon | M1, M3, and Tourism horizon |
|---|---|---|
| Yearly | 6 | 6 |
| Quarterly | 8 | 8 |
| Monthly | 18 | 12 |

The metadata scripts then split that training window again, with the same horizon, into a dev set and a validation set. Features and ARIMA scores used to train MetaARIMA are computed on the dev set.

Match the saved metamodel to the target frequency:

| Target frequency | `--source` | Saved model |
|---|---|---|
| Monthly | `m4_monthly` | `assets/trained_metaarima_m4_monthly_catboost.joblib.gz` |
| Quarterly | `m4_quarterly` | `assets/trained_metaarima_m4_quarterly_catboost.joblib.gz` |
| Yearly | `m4_yearly` | `assets/trained_metaarima_m4_yearly_catboost.joblib.gz` |

### Intermediary files (not included)

`assets/metadata/` is gitignored and is not in the repository. The collectors create it. The M4 dev files are about 617 MB once built (the monthly ARIMA file alone is about 361 MB). Later steps read these files. They are not optional if you are starting from the diagrams only.

| File | Generated by | Used by |
|---|---|---|
| `assets/metadata/features-dev,{group}.csv` | `scripts/collecting_metadata/feature_extraction.py --source {group}` | Training, ablation, quantile and component-count sweeps, config EDA |
| `assets/metadata/arima-dev,{group}.csv` | `scripts/collecting_metadata/performance_estimates.py --source {group}` | Same |

`{group}` for the paper is `m4_monthly`, `m4_quarterly`, or `m4_yearly`. The collectors write only the `-dev` names, and training reads those files (`from_dev_set=True`).

`train_metaarima.py` writes `assets/trained_metaarima_{source}_catboost.joblib.gz`. `main.py` and the lambda / n-trials sweeps load that file and do not read `assets/metadata/`. Ablation, the quantile sweep, the component-count sweep, and `eda_configs.py` do read the metadata, so they come after the collectors.

---

## Reproducing the Results

### Tables and figures

Run this after the benchmark and sensitivity CSVs exist. These scripts only read those files and write PDFs under `assets/results/plots/`.

```bash
conda activate metaarima
PYTHONPATH=. python scripts/experiments/analysis/overall_analysis.py
PYTHONPATH=. python scripts/experiments/analysis/ablation_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_lambda_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_quantile_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_ntrials_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_ncomps_results_all.py
```

`plot_tsfm.py` also needs the Chronos-2 prediction CSV and a download of `monash_m3_monthly`:

```bash
PYTHONPATH=. python scripts/experiments/analysis/plot_tsfm.py
```

The TSFM score summary reads the score CSVs written by the prediction scripts:

```bash
PYTHONPATH=. python scripts/tsfm/summarize.py
```

`sens_lambda_results_all.py` ignores `assets/results/sensitivity/lambda,monash_tourism_monthly,2.csv` (an extra run). The file it uses is `lambda,monash_tourism_monthly.csv`.

### Benchmark from the trained models

This step needs the joblib files from `train_metaarima.py`. It downloads each target from Hugging Face and does not retrain.

```bash
for target in monash_m1_monthly monash_m3_monthly monash_tourism_monthly; do
  PYTHONPATH=. python scripts/experiments/main/main.py --source m4_monthly --target "$target"
done
for target in monash_m1_quarterly monash_m3_quarterly monash_tourism_quarterly; do
  PYTHONPATH=. python scripts/experiments/main/main.py --source m4_quarterly --target "$target"
done
for target in monash_m1_yearly monash_m3_yearly monash_tourism_yearly; do
  PYTHONPATH=. python scripts/experiments/main/main.py --source m4_yearly --target "$target"
done
```

The lambda and n-trials sweeps are the same kind of run (saved model, no metadata):

```bash
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_lambda.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ntrials.py --source m4_monthly --target monash_m1_monthly
```

Repeat with `--source` / `--target` pairs from the frequency table. Defaults in each script are the values last left in the file; pass both flags explicitly for a paper dataset.

### Full reproduction from scratch

1. Build dev metadata for each M4 frequency (the expensive step; 8 worker processes, one BLAS thread each):

```bash
for source in m4_monthly m4_quarterly m4_yearly; do
  PYTHONPATH=. python scripts/collecting_metadata/feature_extraction.py --source "$source"
  PYTHONPATH=. python scripts/collecting_metadata/performance_estimates.py --source "$source"
done
```

2. Train the three metamodels:

```bash
for source in m4_monthly m4_quarterly m4_yearly; do
  PYTHONPATH=. python scripts/collecting_metadata/train_metaarima.py --source "$source"
done
```

3. Run the nine-dataset benchmark in the previous section.
4. Retrain the sweeps that read metadata. Use the same frequency pairing:

```bash
PYTHONPATH=. python scripts/experiments/main/sensitivity/ablation.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_quantile.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ncomps.py --source m4_monthly --target monash_m1_monthly
```

5. Config EDA, only after the monthly metadata exists:

```bash
PYTHONPATH=. python scripts/experiments/analysis/eda_configs.py
```

6. TSFM, from the TSFM environment. The scripts run on CPU and write under `assets/results/tsfm/<model>/`:

```bash
conda activate metaarima-tsfm
for target in monash_m1_monthly monash_m3_monthly monash_tourism_monthly; do
  PYTHONPATH=. python scripts/tsfm/preds_chronos2.py --source m4_monthly --target "$target"
  PYTHONPATH=. python scripts/tsfm/preds_timesfm.py --source m4_monthly --target "$target"
  PYTHONPATH=. python scripts/tsfm/preds_moirai2.py --source m4_monthly --target "$target"
done
PYTHONPATH=. python scripts/tsfm/summarize.py
```

CatBoost is trained with `random_seed` 42 and `thread_count` 1, which is what makes that seed repeatable. PCA and the Monte Carlo ablation use `RANDOM_SEED` (1) in `src/config.py`. The Optuna search in `ht_catboost.py` uses the same seed 42; the paper runs use the hyperparameters already stored in `src/config.py`, so that script is not part of the main path. TSFM scripts seed PyTorch from `RANDOM_SEED` and load pinned Hub revisions of Chronos-2, TimesFM, and Moirai-2. ARIMA fits can still move slightly across BLAS libraries.

---

## Paper Tables and Figures — Output Mapping

Names below are descriptive. Renumber them to the manuscript before submission.

| Paper item | Script | Output |
|---|---|---|
| Mean MASE by dataset, and average rank | `scripts/experiments/analysis/overall_analysis.py` | Console LaTeX. Reads `assets/results/main/*.csv` |
| Median MASE by dataset, and average rank | same script | Console LaTeX |
| Ablation MASE by dataset and average rank | `scripts/experiments/analysis/ablation_results_all.py` | Console LaTeX and `assets/results/plots/ablation_scores.pdf` |
| Lambda sensitivity | `scripts/experiments/analysis/sens_lambda_results_all.py` | `assets/results/plots/lambda_scores.pdf` |
| Quantile sensitivity | `scripts/experiments/analysis/sens_quantile_results_all.py` | `assets/results/plots/quantile_scores.pdf` |
| Number-of-trials sensitivity | `scripts/experiments/analysis/sens_ntrials_results_all.py` | `assets/results/plots/ntrials_scores.pdf` |
| PCA component-count sensitivity | `scripts/experiments/analysis/sens_ncomps_results_all.py` | `assets/results/plots/ncomps_scores.pdf` |
| Configuration rank and complexity | `scripts/experiments/analysis/eda_configs.py` | `config_rank_dist.pdf` and `complexity_vs_rank.pdf` in `assets/results/plots/`. Needs M4 monthly metadata |
| TSFM score summary | `scripts/tsfm/summarize.py` | Console tables from `assets/results/tsfm/*/scores,*.csv` |
| TSFM forecast example | `scripts/experiments/analysis/plot_tsfm.py` | `assets/results/plots/tsfm_predictions_{uid}.pdf` |

`scripts/experiments/main/sensitivity/calc_cost_n_obs.py` prints observation counts for MetaARIMA and a naive AutoARIMA comparison. It does not write a figure.

---

## Expected Runtimes

These are orders of magnitude for planning a review, not a benchmark. Replace them if you have measured times. Analysis is last: it reads files the earlier steps write.

| Step | Script | Approximate runtime |
|---|---|---|
| Feature extraction, one M4 frequency | `feature_extraction.py` | On the order of hours |
| ARIMA performance metadata, M4 monthly | `performance_estimates.py` | On the order of days (400 configurations, 8 workers). Quarterly and yearly are shorter |
| Train one metamodel | `train_metaarima.py` | On the order of hours. `thread_count` is 1 so the seed stays fixed |
| Main benchmark, one target dataset | `main.py` | On the order of hours |
| Ablation, quantile sweep, or component-count sweep, one target | `ablation.py`, `sensitivity_quantile.py`, `sensitivity_ncomps.py` | On the order of days (they retrain) |
| Lambda or n-trials sweep, one target | `sensitivity_lambda.py`, `sensitivity_ntrials.py` | Hours to about a day |
| One TSFM script, three monthly datasets | `scripts/tsfm/preds_*.py` | On the order of hours to about a day on CPU |
| Tables and figures | `scripts/experiments/analysis/` | A few minutes, once the CSVs exist |

---

## Special Setup

- **GPU:** not required. The scripts under `scripts/tsfm/` load Chronos-2, TimesFM, and Moirai-2 on CPU.
- **Parallel metadata collection:** `performance_estimates.py` uses 8 processes and pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 1. `feature_extraction.py` uses 6 threads inside `tsfeatures`. No cluster and no extra scheduler.
- **Order:** collectors, then `train_metaarima.py`, then `main.py` and the sensitivity scripts, then the analysis scripts. TSFM scripts need the monthly joblib from training. The analysis scripts stop with a clear error if their input CSVs are missing.
- **Working directory:** repository root. Prefix the Python commands with `PYTHONPATH=.`.

---

## Citation

```bibtex
@article{todo,
  title={MetaARIMA: Automatic Configuration of ARIMA using Metalearning},
  author={TODO},
  journal={International Journal of Forecasting},
  year={TODO}
}
```

Replace this entry with the published citation, and replace the author block at the top of this file, before submission.
