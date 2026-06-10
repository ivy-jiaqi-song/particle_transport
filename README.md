# Particle Transport Pipeline

This repository contains a compact particle-transport workflow for static MHD
HDF5 fields. The user-facing entry point is `run_pipeline.jl`; implementation
helpers live under `src/`.

## Files

- `run_pipeline.jl`
  - Root-level Julia entry point for configured runs.

- `src/multimode_multienergy_cache_pipeline.jl`
  - Main pipeline implementation.
  - Builds campaigns from the TOML config, integrates trajectories, writes cache
    files, and optionally computes D_mumu products.

- `src/phase_space_gpu_runner.jl`
  - GPU trajectory integration helper.
  - Implements field loading, particle initialization, and the relativistic
    Boris pusher.

- `src/compute_delta_mu2_curve.jl`
  - Shared post-processing utilities for reconstructing pitch angle and plotting
    `<(Delta mu)^2>`.

- `src/compute_delta_mu2_dmumu_full.jl`
  - Shared full-pair transport post-processor used by the main runner.

- `compare_transport_products.py`
  - Follow-up plotting script for generated `delta_mu2_dmumu_full.h5` products.

- `configs/run_config.example.toml`
  - Example user configuration. Copy it to `configs/run_config.local.toml` and
    edit machine-specific paths.

- `scripts/run_pipeline.sh`
  - Linux foreground wrapper around `run_pipeline.jl`.

- `scripts/run_pipeline_background.sh`
  - Linux background wrapper using `nohup`.

## Requirements

Julia packages used by the runner:

```julia
import Pkg
Pkg.add(["CUDA", "HDF5", "CSV", "DataFrames", "PyPlot"])
```

Python packages used by the comparison script:

```bash
python -m pip install h5py matplotlib numpy
```

The trajectory stage expects a CUDA-capable GPU.

## Configuration

Copy the example config and edit it for the local machine:

```bash
cp configs/run_config.example.toml configs/run_config.local.toml
```

The current public config layout is:

- `[input]`
  - `h5_dir`: directory containing the input HDF5 files.
  - `label`: user label for this dataset, for example `0_5` or `512_a_00100`.
  - `medium`: `iso` or `mp`. This is reflected in output paths.
  - `file_stem`, `mode_file_pattern`, and optional `total_file`: file naming.

- `[run]`
  - `cache_mode`: `phase-space` records positions and momenta; `mu` writes a
    compact pitch-angle cache for D_mumu-focused runs.
  - `compute_dmumu`: set `false` to stop after trajectory/cache generation.
  - `mode_decomposition_available`: `true` means files such as
    `<stem>_alfven.h5`, `<stem>_fast.h5`, and `<stem>_slow.h5` exist. `false`
    means only the total input file is used.
  - `modes` and one particle-scale input: `energies_gev`,
    `larmor_radii_box_fraction`, or `larmor_radii_pc`.
  - `[[run.campaigns]]`: optional selective mode/particle-scale campaign
    entries. `r_L/L_box = 0.04` matches the dimensionless Larmor radius used
    in Maiti et al. for their Figure 1 setup.

- `[particles]`
  - Physical trajectory settings: field dataset names, velocity unit, box size,
    eta, integration time, boundary behavior.
  - Particle and cache burden settings: particle count, precision, field subset,
    saved time stride, output precision.

- `[dmumu]`
  - Optional transport-analysis settings.
  - `particles = "sample"` uses `n_particles_to_use` and `particle_selection`.
    `particles = "all"` uses all cached particles. Lag settings still apply in
    both cases.
  - Lag sampling, mu binning, backend, chunk size, and safety-limit controls.
  - `mu_bin_coordinate = "absolute"` bins by `|mu_0|` on `[0, 1]`; this is the
    paper-style choice for Maiti et al. Figure 1 comparisons.

Legacy `[input].layout` configs for `mp-weakb` and `mhd512` are still accepted,
but new configs should use the generic `[input]` form.

## Campaign Examples

Full repeat over all available modes and all configured energies:

```toml
[run]
mode_decomposition_available = true
available_modes = ["alfven", "fast", "slow"]
energies_gev = [1e5, 1e6, 1e7]
modes = "all"
```

Use the paper-style dimensionless Larmor radius instead of GeV inputs:

```toml
[run]
mode_decomposition_available = true
available_modes = ["alfven", "fast", "slow"]
larmor_radii_box_fraction = [0.04]
modes = "all"

[dmumu]
mu_bin_coordinate = "absolute"
mu_min = 0.0
mu_max = 1.0
```

Run only selected mode/energy pairs:

```toml
[[run.campaigns]]
mode = "alfven"
energies_gev = [1e5]

[[run.campaigns]]
mode = "fast"
energies_gev = [1e6, 1e7]
```

Use one total HDF5 file with no mode decomposition:

```toml
[input]
h5_dir = "/path/to/h5"
label = "512_a_00100"
medium = "iso"
file_stem = "512_a.00100"
total_file = "512_a.00100.h5"

[run]
mode_decomposition_available = false
energies_gev = [1e5]
```

## Running

Smoke test through the shell wrapper:

```bash
bash scripts/run_pipeline.sh smoke
```

Full configured foreground run:

```bash
bash scripts/run_pipeline.sh full
```

Background run:

```bash
bash scripts/run_pipeline_background.sh full
```

Direct Julia commands:

```bash
julia run_pipeline.jl --config=configs/run_config.local.toml --smoke
julia run_pipeline.jl --config=configs/run_config.local.toml --campaign=mp/0_5/alfven --cache-mode=mu
julia run_pipeline.jl --config=configs/run_config.local.toml --no-compute-dmumu --cache-mode=phase-space
```

Useful runtime flags:

- `--config=PATH`
- `--output-root=PATH`
- `--input-h5-dir=PATH`
- `--input-label=LABEL`
- `--input-medium=iso` or `--input-medium=mp`
- `--file-stem=STEM`
- `--total-file=FILENAME`
- `--mode-file-pattern=PATTERN`
- `--mode-decomposition-available` or `--no-mode-decomposition`
- `--modes=alfven,fast` or `--mode=alfven`
- `--campaign=mp/0_5/alfven`
- `--energy=1e5` or `--energies=1e5,1e6,1e7`
- `--larmor-radius-box-fraction=0.04` or `--larmor-radii-box-fraction=0.04,0.08`
- `--larmor-radius-pc=8.0` or `--larmor-radii-pc=8.0,16.0`
- `--mu-bin-coordinate=absolute` for paper-style `|mu_0|` D_mumu bins
- `--cache-mode=mu` or `--cache-mode=phase-space`
- `--compute-dmumu` or `--no-compute-dmumu`
- `--keep-caches`
- `--regenerate-caches`
- `--force-recompute`
- `--all-particles-dmumu`
- `--smoke`

## Outputs

For the generic config, successful D_mumu runs are written under:

```text
outputs/campaigns_cache/<cache_mode>/<medium>/<label>/<mode>/<energy>_GeV/
```

For example:

```text
outputs/campaigns_cache/mu/mp/0_5/alfven/100000_GeV/
```

Each energy folder contains:

- `delta_mu2_dmumu_full.h5`
- `delta_mu2_curve_full.png`
- `dmumu_mu_tau_full.png`
- `dmumu_tau_average_full.png`

Each campaign folder also contains `campaign_summary.tsv`. Intermediate cache
files live under each campaign's `cache/` folder. They are deleted after a
successful D_mumu run when `delete_cache_on_success = true`; cache-only runs
keep the cache because it is the final product.

## Comparison Figures

After the Julia runner creates `delta_mu2_dmumu_full.h5` files, use:

```bash
python compare_transport_products.py --list
```

Generate fixed-energy mode comparisons:

```bash
python compare_transport_products.py --figure-set=modes --turbulence=0_5 --energy=1e5
```

Generate a Figure-1-style mode comparison after an `r_L/L_box = 0.04`,
absolute-mu run:

```bash
python compare_transport_products.py --figure-set=modes --turbulence=0_5 \
  --dmumu-variant=raw --plot-kind=dmumu --mu-coordinate=absolute
```

Add `--dmumu-tau=VALUE` to plot the nearest stored `D_mumu(mu,tau)` slice, or
`--dmumu-tau-min=MIN --dmumu-tau-max=MAX` to average over a selected tau range.

Generate energy scans:

```bash
python compare_transport_products.py --figure-set=energies --turbulence=0_5 --mode=alfven,fast,slow
```

Generate cross-turbulence comparisons:

```bash
python compare_transport_products.py --figure-set=turbulence --mode=alfven --energy=1e5
```

Comparison outputs are written under:

```text
outputs/comparisons/
```

## Git Notes

`.gitignore` excludes generated outputs, logs, caches, HDF5 products,
machine-specific `configs/*.local.toml`, and local task notes.
