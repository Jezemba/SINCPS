# SINCPS Training — Replication Guide

This directory contains the **actual training code** used to produce the 22 SINCPS
checkpoints, plus the setup you need for the parts that **cannot be copied** (the
Cerebras stack, the datasets, and machine-specific paths).

> **Important honesty note:** SINCPS models are **not** trained with plain PyTorch.
> The code is a [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo) package and
> was trained on a **Cerebras CS-3 (CSX)** wafer-scale engine. You cannot `pip install`
> your way to an identical run — you need access to the Cerebras software stack
> (`cerebras.pytorch`, `cerebras.modelzoo`). A CPU/GPU fallback exists for small-scale
> testing (see below) but still requires the Cerebras Python package.

---

## 1. What's in here

```
training/
├── README.md                  ← you are here
└── sincs/                     ← the Cerebras Model Zoo package (drop-in)
    ├── __init__.py            registers SINCSModel / SINCSConfig
    ├── model.py               SIREN + Fourier-encoding network (the model)
    ├── data.py                SINCSDataProcessor: reads The Well HDF5, samples coords
    ├── run.py                 training entry point (calls cerebras run_utils.run)
    ├── sincs_evaluate.py      reconstruct + score a trained checkpoint
    ├── test_all_datasets.py   sanity check across datasets
    ├── configs/               one YAML per dataset (29 files; the exact run configs)
    └── validation/            physics + universal-metric validators
        ├── physics_class_validators.py     metric definitions
        ├── run_tier1_physics_validation.py validation entry point
        └── ...
```

Everything under `sincs/` is copied verbatim from the original location
`WellImplementation/Well/modelzoo/src/cerebras/modelzoo/models/vision/sincs/`.

---

## 2. Prerequisites you must provide (cannot be copied)

### a) Cerebras software stack
The code imports `cerebras.pytorch` (`cstorch`) and `cerebras.modelzoo`. You need a
Cerebras Model Zoo checkout and the matching `cerebras-pytorch` wheel.

- Original training environment: **Cerebras PyTorch 2.9.0** on CSX
  (a CPU build, **2.4.0**, was used for the CPU benchmark).
- A virtualenv `/cra-1272/venvs/2.6.0/` was used on this cluster for validation.
- For real (wafer-scale) training you also need **CS-3 / CSX cluster access**.

### b) The Well datasets
Models are trained per-dataset on [The Well](https://github.com/PolymathicAI/the_well).
Download the datasets you want and place them under a single directory, one
subdirectory per dataset, e.g.:

```
<DATA_DIR>/
├── shear_flow/
├── mhd_64/
└── ...
```

Optionally install the `the_well` Python library (used for coordinate-system metadata;
the code falls back gracefully if it's missing).

### c) Trained checkpoints (optional)
The 22 trained checkpoints are not in this directory — they live in `../checkpoints/`
(`checkpoint_50000.mdl` per dataset). You only need those to *use* a model; to *train*
from scratch you don't need them.

---

## 3. Setup

1. **Get a Cerebras Model Zoo checkout** matching your `cerebras-pytorch` version.

2. **Place this package** inside that checkout at the canonical path so the model
   registry discovers the `sincs` model and `SINCSDataProcessor`:

   ```
   <modelzoo>/src/cerebras/modelzoo/models/vision/sincs/   ←  copy training/sincs/ here
   ```

3. **Download datasets** (§2b) and note your `<DATA_DIR>`.

4. **Fix machine-specific paths in the configs.** The YAMLs were written for this
   cluster and hardcode absolute paths. In every config you plan to run, update:

   | Field | Change to |
   |---|---|
   | `trainer.fit.train_dataloader.data_dir` | your `<DATA_DIR>` |
   | `trainer.init.backend.cluster_config.mount_dirs` | your `<DATA_DIR>` + your modelzoo `src/` |
   | `trainer.init.backend.cluster_config.python_paths` | your modelzoo `src/` |

   Quick sweep (review before running):
   ```bash
   cd training/sincs/configs
   sed -i 's#/cra-1272/PhysicsAlchemists/datasets#<DATA_DIR>#g' *.yaml
   sed -i 's#/cra-1272/PhysicsAlchemists/WellImplementation/Well/modelzoo/src#<MODELZOO>/src#g' *.yaml
   ```
   The default `data_dir` in `data.py` (`SINCSDataProcessorConfig`) is also hardcoded —
   override it via the config (above) or edit the default.

---

## 4. Train

From your modelzoo `src/` directory:

```bash
# Wafer-scale (the real runs):
python cerebras/modelzoo/models/vision/sincs/run.py CSX \
    --params cerebras/modelzoo/models/vision/sincs/configs/sincs_shear_flow.yaml

# CPU / GPU fallback for small-scale testing (still needs cerebras-pytorch):
python cerebras/modelzoo/models/vision/sincs/run.py CPU \
    --mode train \
    --params .../configs/sincs_shear_flow.yaml \
    --model_dir sincs_shear_flow_hd1024
```

Output (checkpoints + logs) lands in the `model_dir` named in the config
(e.g. `sincs_shear_flow_hd1024/`), with a `checkpoint_<step>.mdl` every 10k steps.

### Key hyperparameters (from the configs; identical across the 22 runs)

| Setting | Value |
|---|---|
| Architecture | SIREN, 4 hidden layers × 1024, Fourier encoding (10 levels) |
| `omega_0` / `omega_hidden` | 30.0 / 30.0 |
| Optimizer | Adam, lr 1e-4, betas (0.9, 0.999), weight_decay 1e-5 |
| LR schedule | Linear warmup 5k steps → cosine decay to 1e-5 over 45k |
| Steps | 50,000 (checkpoint every 10,000) |
| Batch size | 16,384 coordinates |
| Samples/epoch | 500,000 |
| Loss | MSE |
| Seed | 42 |
| Normalization | z-score per field; static/dynamic field split |

**Per-dataset differences** are mostly `input_dim`/`output_dim` (2D vs 3D, number of
fields) and **`num_timesteps`** — most datasets trained on only a *partial* timestep
rollout (as low as 15 of 100+), which is why "effective" compression ratios differ from
raw ones. Each config records the exact `num_timesteps` used.

Expected wall-clock: **~1.7 h on CSX**, **~11 h on CPU** for 50k steps
(acoustic reference; `../../WellImplementation/Well/experiments/cpu_benchmark/`).

---

## 5. Validate

```bash
python cerebras/modelzoo/models/vision/sincs/validation/run_tier1_physics_validation.py \
    --checkpoint sincs_shear_flow_hd1024/checkpoint_50000.mdl \
    --config    .../configs/sincs_shear_flow.yaml \
    --output_dir validation_results/ \
    --num_timesteps 10
```

> ⚠️ **Known metric issues — read before trusting validation output.**
> Several physics metrics are buggy on z-score-normalized data:
> - **Conservation errors** (mass/energy/concentration) divide by a near-zero mean → can
>   return values in the thousands.
> - **Wave-speed and impedance errors** are always `0.0` (reconstruction lacks the
>   derived field, so the code compares ground truth to itself).
> - **`spectral_error`** is the plain L2 norm of the FFT difference; by Parseval's theorem
>   it is mathematically proportional to the real-space L2 error and carries no extra
>   information.
>
> **Trustworthy metrics:** `psnr`, `rel_l2_error` / `l2_error_pct`, `rmse`,
> `gradient_error`, `temporal_correlation`. See
> `../../WellImplementation/Well/experiments/PHYSICS_VALIDATION_BUGS_SUMMARY.md`.

---

## 6. Use a trained model (no training needed)

To decode/query an existing checkpoint, use the lightweight inference package one level
up — it has no Cerebras dependency:

```python
from sincps import SINCPSDecompressor   # see ../sincps/ and ../README.md
```

Note: each `checkpoint_50000.mdl` is ~37.6 MB but only ~13–17 MB is model weights — the
rest is **Adam optimizer state** (`exp_avg` + `exp_avg_sq`, 2× the weights). For
inference you only need the `model.*` tensors.

---

## 7. Reproducibility checklist

- [ ] Cerebras Model Zoo checkout + matching `cerebras-pytorch` installed
- [ ] `training/sincs/` placed at `<modelzoo>/src/cerebras/modelzoo/models/vision/sincs/`
- [ ] The Well dataset(s) downloaded to `<DATA_DIR>`
- [ ] Config paths updated (`data_dir`, `mount_dirs`, `python_paths`) — §3 step 4
- [ ] (Optional) `the_well` library installed for coordinate metadata
- [ ] Run `run.py` (CSX for full scale; CPU/GPU for testing)
- [ ] Validate with the trustworthy metrics only (§5)

Seed is fixed (42), but exact bitwise reproduction depends on the Cerebras runtime
version and backend (CSX vs CPU/GPU will differ numerically).
