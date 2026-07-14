# Improved-GS integration

This fork contains an opt-in implementation of
[Improved-GS (arXiv:2508.12313)](https://arxiv.org/abs/2508.12313), the maintained
successor to arXiv:2411.10133, on top of this project's existing depth,
exposure, anti-aliasing, and radial-camera changes.

The default remains the original project behavior:

```text
--density_control 3dgs
```

Use `--density_control improvedgs` to enable the new path.

## Fresh-clone setup

The rasterizer is an upstream Git submodule. Its AbsGrad/EAS changes are stored
as a tracked patch in the parent repository so they are not lost by a fresh
recursive clone.

```bash
git submodule update --init --recursive
python scripts/apply_improved_gs_rasterizer_patch.py
pip install --no-build-isolation --no-cache-dir --force-reinstall \
  submodules/diff-gaussian-rasterization
pip install submodules/simple-knn submodules/fused-ssim plyfile
```

The patch helper is idempotent. Rebuilding the rasterizer is mandatory because
the Python/C++ extension ABI gains `pixel_weights`, `accum_weights`, and four
screen-gradient channels.

Run the CPU/static regression suite with:

```bash
python -m unittest discover -s tests -v
python scripts/apply_improved_gs_rasterizer_patch.py --check-only
```

## Kaggle command

The project wrapper can train one or more scenes:

```bash
python train_full.py \
  --subset HCM0249 \
  --iterations 30000 \
  --density_control improvedgs \
  --gaussian_budget 1500000 \
  --use_las 1 \
  --use_rap 1 \
  --use_gc 1 \
  --use_absgrad 1 \
  --use_eas 1 \
  --use_mu 1 \
  --seed 0
```

`notebooks/kaggle_run.ipynb` contains the same setup, an extension smoke check,
and the competition preprocessing/rendering flow. The 1.5M budget is a
project/Kaggle default chosen for memory control; override it explicitly when
the GPU and scene justify a larger budget. `--cap_max` remains a baseline-only
option and is ignored by Improved-GS.

## Implemented components

- **LAS:** one parent is replaced by two children along its longest rotated
  axis. With `rho=0.45`, centers move by `+/-3*rho*sigma_max`; the long scale is
  multiplied by `1-rho`, short scales by `sqrt(1-rho^2)`, and activated opacity
  by `0.6`.
- **RAP:** optional opacity prune below `0.02` at iteration 300, opacity caps at
  3000-step intervals, and exact bottom-20% recovery prunes at iterations 3300
  and 6300 by default.
- **Growth Control:** a square-root active budget reaches the final hard growth
  cap at `densify_until_iter - 500`. A model whose initial/checkpoint count is
  already above the final cap is rejected instead of silently violating it.
- **AbsGrad:** CUDA backward keeps signed gradients in channels 0-1 and sums
  absolute per-pixel contributions in channels 2-3. Geometry optimization still
  consumes only the signed channels.
- **EAS:** CPU-half Laplacian edge maps (masked by an eroded alpha mask when
  available) weight front-to-back `T * alpha` contributions per Gaussian.
  Only positive EAS candidates are sampled; EAS is never treated as a quota.
- **MU:** dense gradients are accumulated with update cadence 1/5/20 views at
  iterations 0/15000/22500 by default. Sparse Adam is rejected when MU is on.

Every component has a `0/1` switch for ablations. The active settings and seed
are written to `improvedgs_config.json` in each model output directory.

## Checkpoints and reproducibility

Improved-GS checkpoints additionally store pending MU gradients, exposure and
exposure-optimizer state, Python/NumPy/Torch/CUDA RNG states, and the remaining
camera stack. This avoids losing accumulated views or changing camera/parent
sampling after resume. Baseline checkpoints keep the original two-item outer
payload and 12-field Gaussian-model state; old 12/13-field model checkpoints
remain readable.

## GPU validation

The CUDA extension cannot be compiled on a CPU-only machine. Before a long
Kaggle run, execute the notebook smoke cell and a short run (for example 700
iterations) to verify that the rebuilt extension loads, AbsGrad backward works,
EAS returns one finite score per Gaussian, and the count does not exceed the
configured final budget.

```bash
python scripts/smoke_test_improved_gs_rasterizer.py
```
