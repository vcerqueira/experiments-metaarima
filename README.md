# MetaARIMA: Automatic Configuration of ARIMA using Metalearning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Reproducibility package for:**
> MetaARIMA: Automatic Configuration of ARIMA using Metalearning.


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
│   │   ├── main/               # Benchmark (main.py), CatBoost search (ht_catboost.py), sensitivity/
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
├── requirements.txt            # Single environment, pinned
└── README.md
```

Run every command below from the repository root. Scripts import `src`, so prefix them with `PYTHONPATH=.`.

---

## Computing Environment

| Component | Analysis scripts | Full CPU experiments | TSFM scripts |
|---|---|---|---|
| **OS** | Any | TODO — record the machine used for the paper runs | Any. The committed scores were produced on an NVIDIA GeForce RTX 2050 |
| **CPU / RAM** | A laptop is enough | TODO — record processor and RAM | A laptop is enough. CPU inference is slower than the original GPU runs |
| **GPU** | None | None | None. The scripts load the models on CPU |
| **Python** | 3.12 | 3.12 | 3.12 |

**Language:** Python 3.12
**Package manager:** conda (or any environment that can `pip install` the pinned file)

Everything, including the TSFM scripts, installs from `requirements.txt`. Moirai-2 (`uni2ts==2.0.0`) requires `numpy~=1.26`, `scipy~=1.11`, `datasets~=2.17`, and `torch<2.5`. Chronos-2 requires `scikit-learn>=1.6` and `torch>=2.2`. The file uses that intersection. The default PyTorch wheel runs on CPU.

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Numerical computation |
| `pandas` | 2.2.3 | Data frames |
| `scikit-learn` | 1.6.1 | PCA and multi-output wrappers |
| `scipy` | 1.11.4 | Ranking utilities |
| `statsforecast` | 1.7.8 | ARIMA and statistical baselines |
| `statsmodels` | 0.15.0 | ARIMA diagnostics |
| `utilsforecast` | 0.2.11 | MASE |
| `tsfeatures` | 0.4.5 | Time-series features |
| `datasets` | 2.17.1 | Hugging Face download |
| `pyarrow` | 25.0.1 | Arrow backend for `datasets`. Later PyArrow releases require NumPy 2 |
| `catboost` | 1.2.8 | Metamodel |
| `optuna` | 5.0.0 | CatBoost hyperparameter search |
| `plotnine` | 0.15.8 | Figures |
| `matplotlib` | 3.9.2 | Figure backend |
| `torch` | 2.4.1 | Chronos-2, TimesFM, and Moirai-2 |
| `chronos-forecasting` | 2.2.2 | Chronos-2 |
| `uni2ts` | 2.0.0 | Moirai-2 |
| `transformers` | 4.57.6 | TimesFM |

---

## Installation

```bash
conda create -n metaarima python=3.12 -y
conda activate metaarima
pip install -r requirements.txt
```

Generated metadata is read from `assets/metadata`. 
That directory is created by the collection scripts. 

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

A fresh clone contains the workflow diagrams only. Run the steps below in order from the repository root. They rebuild the metadata, models, scores, and figures.

### 1. Dev metadata

Build features and ARIMA scores for each M4 frequency. This is the expensive step: `performance_estimates.py` uses 8 worker processes and one BLAS thread each.

```bash
PYTHONPATH=. python scripts/collecting_metadata/feature_extraction.py --source m4_monthly
PYTHONPATH=. python scripts/collecting_metadata/performance_estimates.py --source m4_monthly
PYTHONPATH=. python scripts/collecting_metadata/feature_extraction.py --source m4_quarterly
PYTHONPATH=. python scripts/collecting_metadata/performance_estimates.py --source m4_quarterly
PYTHONPATH=. python scripts/collecting_metadata/feature_extraction.py --source m4_yearly
PYTHONPATH=. python scripts/collecting_metadata/performance_estimates.py --source m4_yearly
```

If you run into some cache issues you might need to delete previous files like so:

`rm -rf ~/.cache/huggingface/datasets/autogluon___chronos_datasets/m4_monthly`

Step 1 takes about 48h-72h to complete with the predefined parallel computing setup. 
You can be skip it and download the resulting metadata files directly from the following link:

https://www.dropbox.com/scl/fo/4urjrqjaxk73t4o3sdj6e/AIN9mZBiT94poDpmWGfAvFg?rlkey=6xt57hdr5y36qqyd7eqp5806g&st=m3y0h6fv&dl=0

Make sure that the files are placed in the folder `assets/metadata` for downstream scripts to work properly.

### 2. Train the metamodels

`BEST_CATBOOST_PARAMS` in `src/config.py` is the CatBoost configuration used by every later step. Those values come from `scripts/experiments/main/ht_catboost.py`. The search is fixed by seed 42: the 20% holdout uses `random_state` 42, each of the 50 trials fits CatBoost with `random_seed` 42 and `thread_count` 1, and Optuna draws trials one at a time from `TPESampler(seed=42)`. With the same dev metadata, a re-run prints the same parameters.

The script leaves `source` set to `m4_yearly`. Set `source` to `m4_monthly` and `m4_quarterly` and run it again for the other two frequencies. It prints the parameters and does not edit `src/config.py`. After a re-run, replace the matching entry in `BEST_CATBOOST_PARAMS` with that printout. Training, the benchmark, the sensitivity sweeps, and the TSFM scripts all read this dictionary, so downstream results stay on the stored values until the dictionary is updated.

```bash
PYTHONPATH=. python scripts/experiments/main/ht_catboost.py
```

Training then uses the dictionary in `src/config.py`:

```bash
PYTHONPATH=. python scripts/collecting_metadata/train_metaarima.py --source m4_monthly
PYTHONPATH=. python scripts/collecting_metadata/train_metaarima.py --source m4_quarterly
PYTHONPATH=. python scripts/collecting_metadata/train_metaarima.py --source m4_yearly
```

### 3. Benchmark

This step loads the joblib from the previous step, downloads each target from Hugging Face, and does not retrain.

```bash
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_quarterly --target monash_m1_quarterly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_quarterly --target monash_m3_quarterly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_quarterly --target monash_tourism_quarterly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_yearly --target monash_m1_yearly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_yearly --target monash_m3_yearly
PYTHONPATH=. python scripts/experiments/main/main.py --source m4_yearly --target monash_tourism_yearly
```

### 4. Sensitivity

The lambda and n-trials sweeps load the saved model and do not read metadata. They cover the three monthly and three quarterly evaluation datasets:

```bash
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_lambda.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_lambda.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_lambda.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_lambda.py --source m4_quarterly --target monash_m1_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_lambda.py --source m4_quarterly --target monash_m3_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_lambda.py --source m4_quarterly --target monash_tourism_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ntrials.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ntrials.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ntrials.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ntrials.py --source m4_quarterly --target monash_m1_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ntrials.py --source m4_quarterly --target monash_m3_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ntrials.py --source m4_quarterly --target monash_tourism_quarterly
```

Ablation, the quantile sweep, and the component-count sweep retrain and read the metadata from step 1. They use the same six datasets:

```bash
PYTHONPATH=. python scripts/experiments/main/sensitivity/ablation.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/ablation.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/ablation.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/ablation.py --source m4_quarterly --target monash_m1_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/ablation.py --source m4_quarterly --target monash_m3_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/ablation.py --source m4_quarterly --target monash_tourism_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_quantile.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_quantile.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_quantile.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_quantile.py --source m4_quarterly --target monash_m1_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_quantile.py --source m4_quarterly --target monash_m3_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_quantile.py --source m4_quarterly --target monash_tourism_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ncomps.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ncomps.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ncomps.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ncomps.py --source m4_quarterly --target monash_m1_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ncomps.py --source m4_quarterly --target monash_m3_quarterly
PYTHONPATH=. python scripts/experiments/main/sensitivity/sensitivity_ncomps.py --source m4_quarterly --target monash_tourism_quarterly
```

### 5. Config EDA

Run this after the monthly metadata from step 1 exists:

```bash
PYTHONPATH=. python scripts/experiments/analysis/eda_configs.py
```

### 6. TSFM

The scripts run on CPU, in the same environment, and write under `assets/results/tsfm/<model>/`:

```bash
PYTHONPATH=. python scripts/tsfm/preds_chronos2.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/tsfm/preds_timesfm.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/tsfm/preds_moirai2.py --source m4_monthly --target monash_m1_monthly
PYTHONPATH=. python scripts/tsfm/preds_chronos2.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/tsfm/preds_timesfm.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/tsfm/preds_moirai2.py --source m4_monthly --target monash_m3_monthly
PYTHONPATH=. python scripts/tsfm/preds_chronos2.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/tsfm/preds_timesfm.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/tsfm/preds_moirai2.py --source m4_monthly --target monash_tourism_monthly
PYTHONPATH=. python scripts/tsfm/summarize.py
```

### 7. Tables and figures

These scripts read the CSVs from the steps above and write PDFs under `assets/results/plots/`.

```bash
PYTHONPATH=. python scripts/experiments/analysis/overall_analysis.py
PYTHONPATH=. python scripts/experiments/analysis/ablation_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_lambda_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_quantile_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_ntrials_results_all.py
PYTHONPATH=. python scripts/experiments/analysis/sens_ncomps_results_all.py
```

`plot_tsfm.py` also needs the Chronos-2 prediction CSV from step 6 and a download of `monash_m3_monthly`:

```bash
PYTHONPATH=. python scripts/experiments/analysis/plot_tsfm.py
```

`sens_lambda_results_all.py` ignores `assets/results/sensitivity/lambda,monash_tourism_monthly,2.csv` (an extra run). The file it uses is `lambda,monash_tourism_monthly.csv`.

CatBoost training uses `random_seed` 42 and `thread_count` 1. The hyperparameter search is fixed by seed 42 (`train_test_split`, CatBoost, and `TPESampler`, one trial at a time). Downstream steps follow `BEST_CATBOOST_PARAMS` in `src/config.py`, so a re-run changes them only after that dictionary is updated from the script's printout. PCA and the Monte Carlo ablation use `RANDOM_SEED` (1) in `src/config.py`. TSFM scripts seed PyTorch from `RANDOM_SEED` and load pinned Hub revisions of Chronos-2, TimesFM, and Moirai-2. ARIMA fits can still move slightly across BLAS libraries.

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
| Feature extraction, one M4 frequency | `feature_extraction.py` | Up to 2 hours |
| ARIMA performance metadata, M4 monthly | `performance_estimates.py` | 48-72 hours (parallel computing) |
| CatBoost hyperparameter search, one M4 frequency | `ht_catboost.py` | Up to one hour |
| Train one metamodel | `train_metaarima.py` | Up to 1 hour (all frequencies)|
| Main benchmark, one target dataset | `main.py` | On the order of hours |
| Ablation, quantile sweep, or component-count sweep, one target | `ablation.py`, `sensitivity_quantile.py`, `sensitivity_ncomps.py` | On the order of days (they retrain) |
| Lambda or n-trials sweep, one target | `sensitivity_lambda.py`, `sensitivity_ntrials.py` | Hours to about a day |
| One TSFM script, three monthly datasets | `scripts/tsfm/preds_*.py` | On the order of hours to about a day on CPU |
| Tables and figures | `scripts/experiments/analysis/` | A few minutes, once the CSVs exist |

---

## Special Setup

- **GPU:** not required. The scripts under `scripts/tsfm/` load Chronos-2, TimesFM, and Moirai-2 on CPU.
- **Parallel metadata collection:** `performance_estimates.py` uses 8 processes and pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 1. `feature_extraction.py` uses 6 threads inside `tsfeatures`. No cluster and no extra scheduler.
- **Order:** collectors, then `ht_catboost.py` when regenerating `BEST_CATBOOST_PARAMS`, then `train_metaarima.py`, then `main.py` and the sensitivity scripts, then the analysis scripts. TSFM scripts need the monthly joblib from training. The analysis scripts stop with a clear error if their input CSVs are missing.
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
