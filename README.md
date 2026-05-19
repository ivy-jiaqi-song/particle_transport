# Particle Transport Pipeline

This folder contains the compact, GitHub-ready version of the multimode,
multienergy particle transport workflow.

## Files

- `multimode_multienergy_cache_pipeline.jl`
  - Main runner.
  - Loads each selected turbulence/mode HDF5 file.
  - Integrates test-particle trajectories on the GPU.
  - Builds a compact pitch-angle cache or full phase-space cache.
  - Computes `<(Delta mu)^2>(tau)` and `D_mumu(mu, tau)` products.

- `phase_space_gpu_runner.jl`
  - GPU trajectory integration helper used by the main runner.
  - Implements field loading, particle initialization, and the relativistic Boris pusher.

- `compute_delta_mu2_curve.jl`
  - Shared post-processing utilities for reconstructing pitch angle and plotting
    `<(Delta mu)^2>`.

- `compute_delta_mu2_dmumu_full.jl`
  - Shared full-pair transport post-processor used by the main runner.
  - Writes the combined HDF5 product and diagnostic PNGs.

- `compare_transport_products.py`
  - Follow-up plotting script.
  - Discovers `delta_mu2_dmumu_full.h5` products and creates mode, energy, or
    turbulence comparison figures automatically.

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

The Julia runner expects a CUDA-capable GPU for the trajectory stage.

## Input Data

The main runner reads static turbulence HDF5 files. The input paths are defined
near the top of `multimode_multienergy_cache_pipeline.jl`:

```julia
const MODE_DECOMPOSITION_ROOT = raw"/home/user0001/MHDFlows_replicate/multiphase_mode_decomposition/outputs"
const MHD512_TOTAL_H5 = raw"/home/user0001/MHDFlows_replicate/h5_outputs/512_a.00100.h5"
const MHD512_MODE_DIR = raw"/home/user0001/MHDFlows_replicate/mhdflows512_mode_decomposition/outputs/512_a_00100_cs10_L200/mode_h5"
```

Edit these constants before running on another machine.

For `--mp-weakb`, the script expects mode-decomposed files like:

```text
<MODE_DECOMPOSITION_ROOT>/MP_WeakB_0_5tcs_cs1_L200/mode_h5/MP_WeakB_0_5tcs_alfven.h5
<MODE_DECOMPOSITION_ROOT>/MP_WeakB_0_5tcs_cs1_L200/mode_h5/MP_WeakB_0_5tcs_fast.h5
<MODE_DECOMPOSITION_ROOT>/MP_WeakB_0_5tcs_cs1_L200/mode_h5/MP_WeakB_0_5tcs_slow.h5
```

For `--mhd512`, it expects one total file plus mode files named like:

```text
512_a_00100_alfven.h5
512_a_00100_fast.h5
512_a_00100_slow.h5
```

Each turbulence HDF5 file should contain magnetic field datasets named
`i_mag_field`, `j_mag_field`, `k_mag_field` and velocity datasets named
`i_velocity`, `j_velocity`, `k_velocity`, unless you edit the config.

## Main Configuration

Most runtime settings live in `CACHE_PIPELINE_CFG` inside
`multimode_multienergy_cache_pipeline.jl`.

Important top-level entries:

- `:cache_mode`
  - `:mu` writes only pitch angle versus time and is usually the practical default.
  - `:phase_space` writes positions and momenta, which is much larger.

- `:energies`
  - Energies in GeV, for example `[1e5, 1e6, 1e7]`.

- `:delete_cache_on_success`
  - `true` removes intermediate cache files after the final products are verified.
  - Use `--keep-caches` at runtime if you want to keep them.

- `:reuse_existing_cache`
  - `true` reuses an existing cache if found.
  - Use `--regenerate-caches` to force rerunning trajectory/cache generation.

- `:skip_completed_outputs`
  - `true` skips an energy if the final HDF5 and PNG outputs already verify.
  - Use `--force-recompute` to recompute final products.

Important `:trajectory_overrides` entries:

- `:n_particles`
- `:tOmega0_max`
- `:trajectory_time_stride`
- `:field_subset`
- `:boundary`
- `:precision`
- `:eta`

Important `:combined_overrides` entries:

- `:n_particles_to_use`
- `:particle_selection`
- `:particle_chunk_size`
- `:n_lag_samples`
- `:n_mu_bins`
- `:min_count_per_cell`
- `:compute_backend`
- `:compute_precision`

For a quick test, use `--smoke`; it reduces particle count, field size, lag
count, and runtime.

## Running The Main Pipeline

Run from this folder:

```bash
cd particle_transport_pipeline
julia multimode_multienergy_cache_pipeline.jl --smoke
```

Run selected multiphase modes:

```bash
julia multimode_multienergy_cache_pipeline.jl --mp-weakb --turbulence=0_5 --mode=alfven,fast,slow --cache-mode=mu
```

Run one exact campaign selector:

```bash
julia multimode_multienergy_cache_pipeline.jl --mp-weakb --campaign=0_5/alfven --cache-mode=mu
```

Run the MHD512 total plus modes:

```bash
julia multimode_multienergy_cache_pipeline.jl --mhd512 --mode=total,alfven,fast,slow --cache-mode=mu
```

Useful runtime flags:

- `--layout=mp-weakb` or `--mp-weakb`
- `--layout=mhd512` or `--mhd512`
- `--turbulence=0_5,0_9`
- `--mode=alfven,fast,slow`
- `--campaign=0_5/alfven`
- `--cache-mode=mu`
- `--cache-mode=phase-space`
- `--keep-caches`
- `--regenerate-caches`
- `--force-recompute`
- `--all-particles-dmumu`
- `--smoke`

## Main Pipeline Outputs

For `--cache-mode=mu`, outputs are written under:

```text
outputs/campaigns_cache/mu/<turbulence>/<mode>/<energy>_GeV/
```

Each energy folder contains:

- `delta_mu2_dmumu_full.h5`
- `delta_mu2_curve_full.png`
- `dmumu_mu_tau_full.png`
- `dmumu_tau_average_full.png`

Each campaign folder also contains:

- `campaign_summary.tsv`

The combined HDF5 file has two main groups:

- `delta_mu2`
  - `tau_norm`
  - `delta_mu2_particle_mean`
  - `delta_mu2_particle_sem`
  - `delta_mu2_pair_mean`
  - `n_particles_used`
  - `n_pairs_used`

- `dmumu`
  - `mu_centers`
  - `tau_norm`
  - `D_mumu_centered_norm`
  - `D_mumu_raw_norm`
  - `D_mumu_centered_tau_average_norm`
  - `D_mumu_raw_tau_average_norm`
  - count-weighted tau-average variants

Intermediate cache files live under each campaign's `cache/` folder. They are
deleted by default after successful verification.

## Comparison Figures

After the Julia runner creates `delta_mu2_dmumu_full.h5` files, use:

```bash
python compare_transport_products.py --list
```

This lists the products discovered under the default input root:

```text
outputs/campaigns_cache/mu
```

Generate fixed-energy mode comparisons:

```bash
python compare_transport_products.py --figure-set=modes --turbulence=0_5 --energy=1e5
```

Generate energy scans for each selected turbulence/mode:

```bash
python compare_transport_products.py --figure-set=energies --turbulence=0_5 --mode=alfven,fast,slow
```

Generate cross-turbulence comparisons for the same mode and energy:

```bash
python compare_transport_products.py --figure-set=turbulence --mode=alfven --energy=1e5
```

Generate all comparison families:

```bash
python compare_transport_products.py --figure-set=all
```

Useful comparison options:

- `--input-root=PATH`
- `--output-dir=PATH`
- `--turbulence=all` or `--turbulence=0_5,512_a_00100`
- `--mode=all` or `--mode=total,alfven,fast,slow`
- `--energy=all` or `--energy=1e5,1e6,1e7`
- `--dmumu-variant=centered`
- `--dmumu-variant=raw`
- `--dmumu-variant=centered-count-weighted`
- `--dmumu-variant=raw-count-weighted`
- `--delta-y-scale=linear`
- `--dmumu-y-scale=linear`
- `--no-sem`
- `--verbose`

The comparison script infers metadata from paths:

- If the parent of `<energy>_GeV` is a known mode, that parent is the mode and
  the previous folder is the turbulence tag.
- Otherwise, the parent is treated as the turbulence tag and mode is `total`.

Comparison outputs are written under:

```text
outputs/comparisons/
```

The script also writes:

```text
outputs/comparisons/comparison_manifest.tsv
```

The manifest records each figure path and the HDF5 files used to create it.

## Git Notes

The repository-level `.gitignore` ignores generated outputs, logs, caches, and
large HDF5 products by default. The intended GitHub commit is this source
folder plus project notes, not the heavy generated result tree.
