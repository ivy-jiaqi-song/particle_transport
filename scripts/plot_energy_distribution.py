#!/usr/bin/env python3
"""
Plot campaign-level particle energy-distribution evolution.

The script auto-detects either supported campaign layout:

1. Direct pipeline campaign root:
   <campaign>/cache/phase_space_<energy_dir>.h5
   <campaign>/<energy_dir>/dpp_full.h5

2. Legacy helper layout:
   <campaign>/total/cache/phase_space_<energy_dir>.h5
   <campaign>/total/<energy_dir>/dpp_full.h5

Outputs are written beside the detected cache/energy folders.
"""

from __future__ import annotations

import csv
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


C_M_S = 2.99792458e8
J_PER_GEV = 1.602176634e-10

DEFAULT_ENERGIES = [
    ("100000_GeV", "1e5", 1.0e5),
    ("1000000_GeV", "1e6", 1.0e6),
    ("10000000_GeV", "1e7", 1.0e7),
]
COLORS = {"1e5": "tab:blue", "1e6": "tab:orange", "1e7": "tab:green"}


def resolve_base_total(campaign_folder: str) -> tuple[str, str]:
    campaign_folder = os.path.abspath(campaign_folder)
    legacy_total = os.path.join(campaign_folder, "total")
    if os.path.isdir(legacy_total):
        return legacy_total, campaign_folder
    if os.path.isdir(os.path.join(campaign_folder, "cache")):
        return campaign_folder, campaign_folder
    if os.path.basename(os.path.normpath(campaign_folder)) == "total":
        return campaign_folder, os.path.dirname(campaign_folder)
    raise FileNotFoundError(
        f"No supported campaign layout found in {campaign_folder}. Expected a "
        "cache/ directory there, or a total/cache/ directory below it."
    )


def read_scalar(handle, name: str):
    if name not in handle:
        return None
    value = handle[name][()]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def resolve_time_gyroperiods(handle):
    if "t_gyroperiods" in handle:
        return np.asarray(handle["t_gyroperiods"][:], dtype=float)
    if "t_norm" in handle:
        return np.asarray(handle["t_norm"][:], dtype=float) / (2.0 * np.pi)
    omega0 = read_scalar(handle, "Omega0_reference_s_inv")
    if omega0 is None:
        omega0 = read_scalar(handle, "Omega0")
    if "t_s" in handle and omega0 is not None:
        return np.asarray(handle["t_s"][:], dtype=float) * float(omega0) / (2.0 * np.pi)
    raise ValueError(
        f"Cannot resolve reference-gyroperiod time axis in {handle.filename}; "
        "need t_gyroperiods, t_norm, or t_s plus Omega0."
    )


def resolve_snapshot_time_gyroperiods(handle):
    group = handle["energy_snapshots"]
    if "snapshot_t_gyroperiods" in group:
        return np.asarray(group["snapshot_t_gyroperiods"][:], dtype=float)
    if "snapshot_t_norm" in group:
        return np.asarray(group["snapshot_t_norm"][:], dtype=float) / (2.0 * np.pi)
    omega0 = read_scalar(handle, "Omega0_reference_s_inv")
    if omega0 is None:
        omega0 = read_scalar(handle, "Omega0")
    if "snapshot_t_s" in group and omega0 is not None:
        return np.asarray(group["snapshot_t_s"][:], dtype=float) * float(omega0) / (2.0 * np.pi)
    raise ValueError(
        f"Cannot resolve snapshot reference-gyroperiod time axis in {handle.filename}; "
        "need snapshot_t_gyroperiods, snapshot_t_norm, or snapshot_t_s plus Omega0."
    )


def load_from_cache_phase_space(base_total: str, energy_dir: str):
    path = os.path.join(base_total, "cache", f"phase_space_{energy_dir}.h5")
    if not os.path.isfile(path):
        return None
    with h5py.File(path, "r") as handle:
        momenta = handle["momenta"][:]
        t_axis = resolve_time_gyroperiods(handle)
    if momenta.ndim != 3 or momenta.shape[1] != 3:
        raise ValueError(f"Unsupported momenta layout in {path}: expected a 3D array with vector axis 1")
    if momenta.shape[0] == len(t_axis):
        p_mag = np.linalg.norm(momenta, axis=1)
    elif momenta.shape[2] == len(t_axis):
        p_mag = np.linalg.norm(momenta, axis=1).T
    else:
        raise ValueError(f"Could not align momenta snapshots with time axis in {path}: shape={momenta.shape}, len={len(t_axis)}")
    energy_gev = p_mag * C_M_S / J_PER_GEV
    return energy_gev, t_axis, f"cache/phase_space_{energy_dir}.h5 (momenta, kg*m/s -> GeV)"


def load_from_dpp_energy_snapshots(base_total: str, energy_dir: str):
    path = os.path.join(base_total, energy_dir, "dpp_full.h5")
    if not os.path.isfile(path):
        return None
    with h5py.File(path, "r") as handle:
        energy_gev = handle["energy_snapshots/energy_GeV"][:]
        t_axis = resolve_snapshot_time_gyroperiods(handle)
    return energy_gev.T, t_axis, f"{energy_dir}/dpp_full.h5 (energy_GeV, already GeV)"


LOADERS = [
    load_from_cache_phase_space,
    load_from_dpp_energy_snapshots,
]


def load_energy(base_total: str, energy_dir: str):
    for loader in LOADERS:
        result = loader(base_total, energy_dir)
        if result is not None:
            return result
    raise FileNotFoundError(
        f"No usable energy data found for {energy_dir} under {base_total}. "
        f"Tried: {[loader.__name__ for loader in LOADERS]}"
    )


def pick_snapshots(snapshot_count: int, want: int = 6):
    candidates = [0, 1, 10, 50, 200, snapshot_count - 1]
    seen = set()
    selected = []
    for index in candidates:
        if 0 <= index < snapshot_count and index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) >= want:
            break
    if len(selected) < want and snapshot_count > 1:
        extra = np.linspace(0, snapshot_count - 1, want + 2).astype(int)[1:-1]
        for index in extra:
            if index not in seen:
                selected.append(index)
                seen.add(index)
            if len(selected) >= want:
                break
    return sorted(selected)[:want]


def make_bins(values: np.ndarray, nbins: int = 200, pad_frac: float = 0.10):
    value_min = float(values.min())
    value_max = float(values.max())
    span = value_max - value_min if value_max > value_min else max(abs(value_min) * 1e-3, 1e-3)
    lower = value_min - pad_frac * span
    upper = value_max + pad_frac * span
    return np.linspace(lower, upper, nbins + 1), (upper - lower) / nbins


def plot_evolution(energy_gev: np.ndarray, t_axis: np.ndarray, tag: str, outpath: str, title_suffix: str = ""):
    snapshots = pick_snapshots(energy_gev.shape[0])
    bins, width = make_bins(energy_gev)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index in snapshots:
        counts, _ = np.histogram(energy_gev[index], bins=bins)
        ax.stairs(np.clip(counts, 1, None), bins, label=f"t/Tg0={t_axis[index]:.3g} (#{index})")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(f"counts per {width * 1000:.2f} MeV bin")
    ax.set_title(f"Energy distribution evolution {title_suffix}({tag} GeV)")
    ax.legend(title="snapshot", loc="best")
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_comparison(results: dict, outpath: str, title_suffix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for tag, (energy_gev, _t_axis, _source) in results.items():
        initial_energy = float(energy_gev[0].mean())
        relative_energy = (energy_gev - initial_energy) / initial_energy
        bins, _ = make_bins(relative_energy)
        snapshots = pick_snapshots(relative_energy.shape[0])
        for offset, index in enumerate(snapshots):
            counts, _ = np.histogram(relative_energy[index], bins=bins)
            ax.stairs(
                np.clip(counts, 1, None),
                bins,
                color=COLORS.get(tag, "gray"),
                alpha=0.35 + 0.65 * offset / max(len(snapshots) - 1, 1),
                label=f"{tag} GeV" if offset == len(snapshots) - 1 else None,
            )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta E / E_0$")
    ax.set_ylabel("counts per bin")
    ax.set_title("Relative energy shift (snapshots, darker = later)")
    ax.legend(title="injection E", loc="best")

    ax = axes[1]
    for tag, (energy_gev, t_axis, _source) in results.items():
        initial_energy = float(energy_gev[0].mean())
        relative_energy = (energy_gev - initial_energy) / initial_energy
        index = relative_energy.shape[0] - 1
        bins, _ = make_bins(relative_energy)
        counts, _ = np.histogram(relative_energy[index], bins=bins)
        ax.stairs(
            np.clip(counts, 1, None),
            bins,
            color=COLORS.get(tag, "gray"),
            label=f"{tag} GeV (t/Tg0={t_axis[index]:.3g})",
            linewidth=1.5,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta E / E_0$")
    ax.set_ylabel("counts per bin")
    ax.set_title("Final snapshot - relative energy shift comparison")
    ax.legend(title="injection E", loc="best")

    fig.suptitle(f"Energy distribution comparison {title_suffix}", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_folder(campaign_folder: str, energy_tags=None):
    base_total, display_folder = resolve_base_total(campaign_folder)
    energies = DEFAULT_ENERGIES if energy_tags is None else [energy for energy in DEFAULT_ENERGIES if energy[1] in set(energy_tags)]

    folder_name = os.path.basename(os.path.normpath(display_folder))
    print(f"=== {folder_name} ===")

    results = {}
    rows = []
    for energy_dir, tag, nominal_energy in energies:
        try:
            energy_gev, t_axis, source = load_energy(base_total, energy_dir)
        except FileNotFoundError as error:
            print(f"  {tag}: SKIP ({error})")
            continue

        snapshot_count, particle_count = energy_gev.shape
        energy_min = float(energy_gev.min())
        energy_max = float(energy_gev.max())
        initial = energy_gev[0]
        final = energy_gev[-1]
        accelerated = int(np.sum(final > initial))
        decelerated = int(np.sum(final < initial))
        unchanged = int(np.sum(final == initial))

        outpath = os.path.join(base_total, f"energy_distribution_evolution_{tag}.png")
        plot_evolution(energy_gev, t_axis, tag, outpath, title_suffix=f"- {folder_name} ")
        print(
            f"  {tag}: N={particle_count} nT={snapshot_count} span={energy_max - energy_min:.4f} GeV  "
            f"acc={accelerated} dec={decelerated} un={unchanged}  src={source.split('/')[-1]}"
        )
        print(f"        -> {outpath}")

        results[tag] = (energy_gev, t_axis, source)
        rows.append(
            dict(
                folder=folder_name,
                tag=tag,
                N=particle_count,
                nT=snapshot_count,
                E0=nominal_energy,
                Emin=energy_min,
                Emax=energy_max,
                span_GeV=energy_max - energy_min,
                acc=accelerated,
                dec=decelerated,
                un=unchanged,
                source=source,
            )
        )

    if results:
        comparison_path = os.path.join(base_total, "energy_distribution_comparison.png")
        plot_comparison(results, comparison_path, title_suffix=f"- {folder_name}")
        print(f"  -> {comparison_path}")

    if rows:
        tsv_path = os.path.join(base_total, "energy_distribution_summary.tsv")
        with open(tsv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {tsv_path}")
    return rows


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_folder(argv[1], argv[2:] if len(argv) > 2 else None)


if __name__ == "__main__":
    main(sys.argv)
