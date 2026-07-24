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
  - `total_file`: required total-field HDF5 file. This file is always used to
    define the shared reference clock and is also the trajectory field for the
    `total` campaign.
  - `label`: user label for this dataset, for example `0_5` or `512_a_00100`.
  - `medium`: `iso` or `mp`. This is reflected in output paths.
  - `file_stem`, `mode_decomposition_available`, `available_modes`, and
    `mode_file_pattern`: optional mode-file discovery. If mode decomposition is
    disabled, only `total_file` is transported.

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
    reference-gyroperiod duration, integration resolution, cache cadence, and
    boundary behavior.
  - Injection controls: `injection_position_mode = "random"` keeps the existing
    random-in-box start positions; `"fixed"` uses `injection_position`. With
    `injection_position_unit = "box-fraction"`, `[0.5, 0.5, 0.5]` is the box
    center. Units `"pc"` and `"m"` are also accepted.
  - Momentum injection defaults to `injection_mode = "isotropic"`. Set
    `injection_mode = "fixed-mu"` and `injection_mu0 = VALUE` to inject every
    particle with the requested pitch-angle cosine relative to the local
    magnetic field while keeping gyrophase random.
  - Particle and cache burden settings: particle count, precision, field subset,
    output precision.

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

Legacy `[input].layout` configs are retained only where a total-field reference
file can be resolved. New configs should use the generic `[input]` form with an
explicit `total_file`.

## Shared Reference Clock For Mode Decomposition

For decomposed campaigns, total, Alfven, fast, and slow trajectories use
different Lorentz-force fields but the same reference clock. The reference is
always derived from the total-field file:

```text
B0_reference_T = mean(|B_total|)
Omega0_reference_s_inv = q_e * B0_reference_T / (gamma0 * m_p)
T_g0 = 2π / Omega0_reference_s_inv
```

The reference depends on energy through `gamma0`, but not on trajectory mode. A
selected-mode-only run such as `modes = ["alfven", "fast"]` still loads
`total_file` to resolve `B0_reference_T`; it does not require a completed total
trajectory run. If `total_file` is missing, the run stops before trajectory
generation.

Primary cached analysis arrays are uniformly sampled. If the exact integration
end does not fall on the requested cache cadence, that off-cadence final sample
is not appended to `/positions`, `/momenta`, `/mu`, or the time axes used by
transport estimators. Requested physical lags are resolved against the positive
cached-step lag range before transport analysis starts; the loops still use
integer cached-step offsets.

## Legacy Cache Compatibility

Transport analysis requires a uniformly sampled trajectory cache because integer
lag offsets are assumed to correspond to fixed physical time separations. Older
caches that contain an appended off-cadence final sample are not repaired,
trimmed, or partially consumed. They are invalid for fixed-index D_mumu and D_pp
analysis and must be regenerated with the current pipeline. The exact final
integration state may be stored separately in future cache layouts, but it is not
part of the primary transport-analysis time series.

Low-level decomposed trajectory runs also require an explicit campaign time
reference derived from the total-field dataset. Only the total-field convenience
wrapper may derive the reference clock from its own field.

## Time Conventions

User-facing trajectory and lag controls are expressed in reference gyroperiods.
The fixed reference is computed once per energy from the loaded field arrays:

```text
B0_T = mean(sqrt(Bx^2 + By^2 + Bz^2)) after conversion to tesla
Omega0 = q_e * B0_T / (gamma0 * m_p)
T_g0 = 2π / Omega0
N_g = t / T_g0 = t Omega0 / (2π)
```

Here "reference" means this fixed campaign convention, not the instantaneous
local gyroperiod along a particle orbit. The integration timestep is still stored
and used internally in seconds. Cache axes store seconds (`t_s`), angular
normalized time (`t_norm = t_s Omega0`), and reference gyroperiods
(`t_gyroperiods = t_norm / 2π`). Transport loops still use integer cached-step
lag offsets, resolved from requested reference-gyroperiod lag grids.

Preferred time controls:

| Quantity | Preferred control |
| --- | --- |
| Trajectory duration | `[particles].trajectory_duration_gyroperiods` |
| Gyro-limited integration resolution | `[particles].integration_steps_per_gyroperiod` |
| Cache sampling interval | `[particles].trajectory_save_interval_gyroperiods` |
| D_mumu lag grid | `[dmumu].lag_min_gyroperiods`, `lag_max_gyroperiods`, `lag_stride_gyroperiods` |
| D_pp lag grid | `[dpp].lag_min_gyroperiods`, `lag_max_gyroperiods`, `lag_stride_gyroperiods` |

Legacy migration table:

| Legacy control | Preferred replacement |
| --- | --- |
| `tOmega0_max` | `trajectory_duration_gyroperiods = tOmega0_max / (2π)` |
| `eta` | `integration_steps_per_gyroperiod = 2π / eta` |
| `trajectory_time_stride` | `trajectory_save_interval_gyroperiods` |
| `min_lag_steps` | `lag_min_gyroperiods` |
| `max_lag_steps` | `lag_max_gyroperiods` |
| `lag_step_stride` | `lag_stride_gyroperiods` |

Step-stride lag and cache-cadence mappings depend on the resolved integration
and saved-cache cadence, so the code reports requested and actual values rather
than requiring users to calculate them manually. Mixing old and new keys for the
same quantity is rejected.

The actual integration timestep is the smaller of the requested gyroperiod
timestep and the CFL cell-crossing limiter when CFL is enabled:

```text
dt_actual = min(dt_gyro, dt_CFL)
dt_save_actual = round(dt_save_requested / dt_actual) * dt_actual
```

Because the save cadence is an integer number of actual integration steps, the
same requested cache interval can resolve slightly differently across particle
energies. This matters for physical lag requests near the first cached step or
near the cache duration.

Lag range controls in `[dmumu]` and `[dpp]`:

| Control | Values |
| --- | --- |
| `lag_range_policy` | `fixed`, `first-cache-step`, `common-cache-intersection` |
| `lag_boundary_policy` | `strict`, `nearest` |
| `max_lag_boundary_relative_error` | nonnegative relative tolerance for `nearest` |

`fixed` uses the configured `lag_min_gyroperiods` and `lag_max_gyroperiods`.
`first-cache-step` starts each job at its first positive cached lag and rejects a
numeric `lag_min_gyroperiods`. `common-cache-intersection` intersects the
representable cache ranges across the selected energies in the current campaign
and maps one common nominal requested grid onto each job's actual cache axis.

`strict` preserves the historical behavior: configured boundaries must already
lie inside the representable cache range. `nearest` can adjust a boundary to the
nearest positive cached lag only when the absolute error is no more than half of
the actual cache interval and the relative error is within
`max_lag_boundary_relative_error`; this is bounded cache-grid snapping, not
arbitrary clamping. Interior requested lags keep the same half-cache-interval
mapping bound, duplicate mapped offsets are removed deterministically, and the
HDF5 output records requested lags, actual lags, mapping errors, policy names,
effective boundaries, duplicate counts, and the comparison group identity.

Before any trajectory cache is generated or reused, each campaign prints a timing
preflight table for all selected energies showing the active timestep limiter,
requested and actual cache interval, and representable lag range. If a requested
`D_mumu` or `D_pp` lag policy cannot produce a valid grid, the run fails at this
preflight stage.

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
- `--trajectory-duration-gyroperiods=VALUE`
- `--integration-steps-per-gyroperiod=VALUE`
- `--trajectory-save-interval-gyroperiods=VALUE`
- `--dmumu-start-mode=injection` for injection-anchored D_mumu bins using
  `Delta mu = mu(tau) - mu(0)` and binning by injected `mu(0)`.
- `--dmumu-mu-bin-abs` or `[dmumu].mu_bin_abs = true` to bin by `abs(mu_start)`.
  With `dmumu_start_mode=injection`, this gives `|mu_0|` bins from 0 to 1.
- `--dmumu-n-mu-bins=N --dmumu-mu-min=0 --dmumu-mu-max=1` to control the D_mumu bin axis.
- `--dmumu-lag-mode=stride --dmumu-min-lag-steps=MIN --dmumu-lag-step-stride=STRIDE` to use
  D_mumu lag steps `MIN, MIN+STRIDE, MIN+2*STRIDE, ...` through the maximum lag.
- `--dmumu-lag-min-gyroperiods=VALUE --dmumu-lag-max-gyroperiods=VALUE` to request
  the D_mumu lag range in reference gyroperiods.
- `--dmumu-lag-stride-gyroperiods=VALUE` when `--dmumu-lag-mode=stride`.
- `--dpp-lag-mode=stride --dpp-min-lag-steps=MIN --dpp-lag-step-stride=STRIDE` to control
  the independent D_pp lag grid.
- `--dpp-lag-min-gyroperiods=VALUE --dpp-lag-max-gyroperiods=VALUE` to request
  the D_pp lag range in reference gyroperiods.
- `--dpp-lag-stride-gyroperiods=VALUE` when `--dpp-lag-mode=stride`.
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

Both per-energy HDF5 products preserve the cache and physical metadata needed
to interpret the result, including cache path/mode, energy, timestep,
`Omega0`, `B0_T`, particle count, trajectory stride, boundary mode, units,
output precision when present, and injection settings.

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
