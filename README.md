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
  - `compute_dmumu`: set `false` to skip `D_mumu` and `delta_mu2` products;
    if `compute_dpp` is also `false`, the run stops after cache generation.
  - `compute_dpp`: set `true` to compute global momentum diffusion and energy
    snapshot histograms into a separate `dpp_full.h5`; this automatically uses
    `phase-space` cache output.
  - `mode_decomposition_available`: `true` means files such as
    `<stem>_alfven.h5`, `<stem>_fast.h5`, and `<stem>_slow.h5` exist. `false`
    means only the total input file is used.
  - `modes` and `energies_gev`: full repeat selection.
  - `[[run.campaigns]]`: optional selective mode/energy campaign entries.

- `[particles]`
  - Physical trajectory settings: field dataset names, velocity unit, box size,
    eta, integration time, boundary behavior.
  - Injection controls: `injection_position_mode = "random"` keeps the existing
    random-in-box start positions; `"fixed"` uses `injection_position`. With
    `injection_position_unit = "box-fraction"`, `[0.5, 0.5, 0.5]` is the box
    center. Units `"pc"` and `"m"` are also accepted.
  - Momentum injection defaults to `injection_mode = "isotropic"`. Set
    `injection_mode = "fixed-mu"` and `injection_mu0 = VALUE` to inject every
    particle with the requested pitch-angle cosine relative to the local
    magnetic field while keeping gyrophase random.
  - Particle and cache burden settings: particle count, precision, field subset,
    saved time stride, output precision.

- `[dmumu]`
  - D_mumu-only transport-analysis settings.
  - `particles = "sample"` uses `n_particles_to_use` and `particle_selection`.
    `particles = "all"` uses all cached particles. Lag settings still apply in
    both cases.
  - Lag sampling, particle sampling, mu binning, backend, chunk size, and
    safety-limit controls are independent from `[dpp]`.
  - `mu_bin_abs = true` bins D_mumu by `abs(mu_start)` over `mu_min = 0.0` to
    `mu_max = 1.0`; stored `mu` values and `Delta mu` remain signed.

- `[dpp]`
  - D_pp-only controls: particle selection, lag sampling, chunk size, and
    `n_energy_snapshots` are independent from `[dmumu]`.
  - D_pp is a one-dimensional function of lag, `D_pp(tau)`, so its figure is
    named `dpp_tau_curve_full.png` rather than a tau-average product.
  - When `compute_dpp = true`, the pipeline also runs
    `scripts/plot_energy_distribution.py` after each campaign to build
    campaign-level energy-distribution figures.

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

Standalone postprocessing from retained phase-space caches:

```bash
julia src/compute_delta_mu2_dmumu_full.jl --trajectory-h5=/path/to/phase_space_100000_GeV.h5 --turbulence-h5=/path/to/turbulence.h5 --output-dir=/path/to/dmumu_out
julia src/compute_dpp_full.jl --trajectory-h5=/path/to/phase_space_100000_GeV.h5 --output-dir=/path/to/dpp_out
```

The D_mumu standalone path reconstructs `mu(t)`, so it needs both the
phase-space cache and the turbulence HDF5. The D_pp standalone path reads only
the phase-space cache momenta plus time metadata. Direct D_mumu-only
postprocessing from a compact `mu_cache_*.h5` is not exposed as a standalone
script; the pipeline uses that path internally when `cache_mode = "mu"`.

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
- `--dmumu-start-mode=injection` for injection-anchored D_mumu bins using
  `Delta mu = mu(tau) - mu(0)` and binning by injected `mu(0)`.
- `--dmumu-mu-bin-abs` or `[dmumu].mu_bin_abs = true` to bin by `abs(mu_start)`.
  With `dmumu_start_mode=injection`, this gives `|mu_0|` bins from 0 to 1.
- `--dmumu-n-mu-bins=N --dmumu-mu-min=0 --dmumu-mu-max=1` to control the D_mumu bin axis.
- `--dmumu-lag-mode=stride --dmumu-min-lag-steps=MIN --dmumu-lag-step-stride=STRIDE` to use
  D_mumu lag steps `MIN, MIN+STRIDE, MIN+2*STRIDE, ...` through the maximum lag.
- `--dpp-lag-mode=stride --dpp-min-lag-steps=MIN --dpp-lag-step-stride=STRIDE` to control
  the independent D_pp lag grid.
- `--dmumu-n-particles=N` and `--dpp-n-particles=N` to choose independent
  estimator particle subsets from the same generated cache.
- `--cache-mode=mu` or `--cache-mode=phase-space`
- `--compute-dmumu` or `--no-compute-dmumu`
- `--compute-dpp` or `--no-compute-dpp`
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
- `dpp_full.h5` when `compute_dpp = true`
- `dpp_tau_curve_full.png` when `compute_dpp = true`

Each campaign folder also contains `campaign_summary.tsv`. Intermediate cache
files live under each campaign's `cache/` folder. They are deleted only after
all requested products for that energy are verified when
`delete_cache_on_success = true`; cache-only or partial-failure runs keep the
cache.

`campaign_summary.tsv` records product status separately with columns for
`dmumu_h5`, `dpp_h5`, `dmumu_status`, `dpp_status`, `dmumu_note`, and
`dpp_note`. A D_mumu failure does not prevent the requested D_pp product from
running, and a D_pp failure does not prevent the requested D_mumu product from
running. With `stop_on_error = true`, the campaign stops after both requested
products for the current energy have been attempted and the summary row has
been written.

When `compute_dpp = true`, each campaign folder also contains:

- `energy_distribution_evolution_1e5.png`
- `energy_distribution_evolution_1e6.png`
- `energy_distribution_evolution_1e7.png`
- `energy_distribution_comparison.png`
- `energy_distribution_summary.tsv`

By default, D_mumu uses the sliding start-time estimator: for each lag it bins
by `mu(t)` and accumulates `Delta mu = mu(t + tau) - mu(t)` over all valid start
times. With `dmumu_start_mode = "injection"`, each particle contributes at most
one pair per lag, binned by `mu(0)` with `Delta mu = mu(tau) - mu(0)`. The
selected D_mumu lag grid is controlled by `[dmumu].lag_mode`,
`[dmumu].min_lag_steps`, `[dmumu].max_lag_steps`, and
`[dmumu].lag_step_stride`/`[dmumu].n_lag_samples`.

When `mu_bin_abs = true`, only the bin coordinate changes: sliding mode bins by
`|mu(t)|`, injection mode bins by `|mu(0)|`, and the generated D_mumu plots use
a 0 to 1 `|mu|` axis.

When `compute_dpp = true`, the phase-space postprocessor also computes global
scalar momentum diffusion with `p = sqrt(px^2 + py^2 + pz^2)`,
`Delta p = p(t + tau) - p(t)`, and normalized output
`D_pp/(p0^2 Omega0) = Var(Delta p / p0) / (2 tau Omega0)`. It also saves
evenly spaced kinetic-energy snapshots in the `dpp_full.h5` `energy_snapshots` group.
After all configured energies in a campaign finish, the pipeline plots the
energy-distribution evolution with `scripts/plot_energy_distribution.py`.

The energy-distribution plotter uses a fixed recipe so different campaigns are
directly comparable: 200 step-histogram bins over the finite energy range padded
by 10%, a log y-axis with empty bins clipped to count 1, true GeV x-axis labels
with Matplotlib offset text disabled, and up to six snapshots selected as
`[0, 1, 10, 50, 200, last]` with evenly spaced fill-ins when needed. The
comparison figure uses `Delta E / E0`, allowing `1e5`, `1e6`, and `1e7` runs to
share one x-axis.

The plotter can also be run manually on existing campaign outputs:

```bash
python scripts/plot_energy_distribution.py outputs/campaigns_cache/phase_space/iso/0_9/total
python scripts/plot_energy_distribution.py outputs/campaigns_cache/phase_space/iso/0_9/total 1e5 1e6
```

It supports both retained phase-space caches under `cache/phase_space_*.h5` and
post-cleanup per-energy `dpp_full.h5` files containing the `energy_snapshots`
group.

To support a new energy-data layout, add a loader function in
`scripts/plot_energy_distribution.py` and append it to `LOADERS`. A loader
should return `(E, t_s, source)` where `E` is shaped `(n_snapshots, n_particles)`
in GeV and `t_s` contains snapshot times in seconds; return `None` when that
layout is absent so the next loader can try.

## Comparison Figures

After the Julia runner creates `delta_mu2_dmumu_full.h5` files, use:

```bash
python compare_transport_products.py --list
```

Generate fixed-energy mode comparisons:

```bash
python compare_transport_products.py --figure-set=modes --turbulence=0_5 --energy=1e5
```

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
