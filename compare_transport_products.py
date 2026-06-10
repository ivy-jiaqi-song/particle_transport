#!/usr/bin/env python3
"""
Build comparison figures from multimode/multienergy transport HDF5 products.

The script discovers files named delta_mu2_dmumu_full.h5, infers
    turbulence / mode / energy
from the folder structure, and writes comparison PNGs plus a manifest.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "particle_transport_matplotlib"),
)

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "outputs" / "campaigns_cache" / "mu"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "comparisons"

ENERGY_DIR_RE = re.compile(r"^(.+)_GeV$")
MODE_ORDER = ("total", "alfven", "fast", "slow")
KNOWN_MODES = set(MODE_ORDER)
MODE_LABELS = {
    "total": "Total",
    "alfven": "Alfven",
    "fast": "Fast",
    "slow": "Slow",
}

DMUMU_VARIANTS = {
    "centered": "D_mumu_centered_tau_average_norm",
    "raw": "D_mumu_raw_tau_average_norm",
    "centered-count-weighted": "D_mumu_centered_tau_average_norm_count_weighted",
    "raw-count-weighted": "D_mumu_raw_tau_average_norm_count_weighted",
}

DMUMU_2D_VARIANTS = {
    "centered": "D_mumu_centered_norm",
    "raw": "D_mumu_raw_norm",
}


@dataclass(frozen=True)
class ProductRef:
    path: Path
    turbulence: str
    mode: str
    energy_gev: float


@dataclass(frozen=True)
class Product:
    ref: ProductRef
    tau_norm: np.ndarray
    delta_mu2_mean: np.ndarray
    delta_mu2_sem: np.ndarray
    mu_centers: np.ndarray
    dmumu_tau_average: np.ndarray
    dmumu_reduction_label: str
    mu_coordinate: str
    energy_input_kind: str | None
    larmor_radius_box_fraction: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover delta_mu2_dmumu_full.h5 products and plot mode, energy, "
            "or turbulence comparisons."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root searched recursively for delta_mu2_dmumu_full.h5 files. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for comparison figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--figure-set",
        default="modes",
        help="Comma-separated set: modes, energies, turbulence, or all. Default: modes.",
    )
    parser.add_argument(
        "--turbulence",
        default="all",
        help="Comma-separated turbulence tags to include, or all. Examples: 0_5,512_a_00100",
    )
    parser.add_argument(
        "--mode",
        default="all",
        help="Comma-separated modes to include, or all. Examples: total,alfven,fast,slow",
    )
    parser.add_argument(
        "--energy",
        default="all",
        help="Comma-separated energies in GeV, or all. Examples: 1e5,100000,100000_GeV",
    )
    parser.add_argument(
        "--dmumu-variant",
        choices=tuple(DMUMU_VARIANTS),
        default="centered",
        help="D_mumu estimator to plot. Default: centered.",
    )
    parser.add_argument(
        "--dmumu-tau",
        type=float,
        default=None,
        help="Plot the nearest D_mumu(mu,tau) slice at this tau*Omega0 instead of the stored tau average.",
    )
    parser.add_argument(
        "--dmumu-tau-min",
        type=float,
        default=None,
        help="Average D_mumu(mu,tau) over tau*Omega0 >= this value instead of the stored tau average.",
    )
    parser.add_argument(
        "--dmumu-tau-max",
        type=float,
        default=None,
        help="Average D_mumu(mu,tau) over tau*Omega0 <= this value instead of the stored tau average.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=("both", "delta", "dmumu"),
        default="both",
        help="Plot both panels, only <(Delta mu)^2>, or only D_mumu. Default: both.",
    )
    parser.add_argument(
        "--mu-coordinate",
        choices=("auto", "signed", "absolute"),
        default="auto",
        help="Use stored mu bins, force signed labels, or fold signed bins to |mu|. Default: auto.",
    )
    parser.add_argument(
        "--delta-y-scale",
        choices=("linear", "log"),
        default="log",
        help="Y-axis scale for <(Delta mu)^2>. Default: log.",
    )
    parser.add_argument(
        "--dmumu-y-scale",
        choices=("linear", "log"),
        default="log",
        help="Y-axis scale for D_mumu. Default: log.",
    )
    parser.add_argument(
        "--no-sem",
        action="store_true",
        help="Do not draw SEM bands around <(Delta mu)^2> curves.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered products after filtering, then exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print files that were skipped during discovery.",
    )
    return parser.parse_args()


def parse_energy_value(value: str) -> float:
    cleaned = value.strip()
    if cleaned.lower().endswith("_gev"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.replace("p", ".")
    return float(cleaned)


def energy_tag(energy_gev: float) -> str:
    if math.isfinite(energy_gev) and abs(energy_gev - round(energy_gev)) < 1.0e-9:
        return f"{int(round(energy_gev))}_GeV"
    return f"{energy_gev:.12g}_GeV".replace("+", "").replace(".", "p")


def format_energy_label(energy_gev: float) -> str:
    if energy_gev > 0:
        exponent = round(math.log10(energy_gev))
        if math.isclose(energy_gev, 10.0**exponent, rel_tol=1.0e-9, abs_tol=0.0):
            return rf"$10^{{{exponent}}}$ GeV"
    return f"{energy_gev:g} GeV"


def normalize_mode(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def split_selector(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def split_mode_selector(value: str) -> set[str] | None:
    selected = split_selector(value)
    if selected is None:
        return None
    return {normalize_mode(item) for item in selected}


def split_energy_selector(value: str) -> list[float] | None:
    if value.strip().lower() == "all":
        return None
    return [parse_energy_value(item) for item in value.split(",") if item.strip()]


def selected_figure_sets(value: str) -> set[str]:
    choices = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not choices:
        raise ValueError("--figure-set cannot be empty.")
    if "all" in choices:
        return {"modes", "energies", "turbulence"}
    allowed = {"modes", "energies", "turbulence"}
    unknown = choices - allowed
    if unknown:
        raise ValueError(f"Unknown --figure-set value(s): {', '.join(sorted(unknown))}")
    return choices


def find_energy_dir(path: Path) -> Path:
    for parent in path.parents:
        if ENERGY_DIR_RE.match(parent.name):
            return parent
    raise ValueError(f"Cannot infer energy directory from {path}")


def infer_product_ref(path: Path) -> ProductRef:
    energy_dir = find_energy_dir(path)
    energy_gev = parse_energy_value(energy_dir.name)
    before_energy = energy_dir.parent
    inferred_mode = normalize_mode(before_energy.name)

    if inferred_mode in KNOWN_MODES:
        mode = inferred_mode
        turbulence = before_energy.parent.name
    else:
        mode = "total"
        turbulence = before_energy.name

    return ProductRef(
        path=path,
        turbulence=turbulence,
        mode=mode,
        energy_gev=energy_gev,
    )


def discover_products(input_root: Path, verbose: bool = False) -> list[ProductRef]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    refs: list[ProductRef] = []
    for path in sorted(input_root.rglob("delta_mu2_dmumu_full.h5")):
        try:
            refs.append(infer_product_ref(path))
        except ValueError as exc:
            if verbose:
                print(f"Skipping {path}: {exc}")

    refs.sort(key=lambda ref: (ref.turbulence.lower(), mode_sort_key(ref.mode), ref.energy_gev, str(ref.path)))
    return deduplicate_refs(refs)


def deduplicate_refs(refs: list[ProductRef]) -> list[ProductRef]:
    by_key: dict[tuple[str, str, float], ProductRef] = {}
    for ref in refs:
        key = (ref.turbulence.lower(), ref.mode.lower(), ref.energy_gev)
        current = by_key.get(key)
        if current is None or ref_preferred_over(ref, current):
            by_key[key] = ref
    return sorted(
        by_key.values(),
        key=lambda ref: (ref.turbulence.lower(), mode_sort_key(ref.mode), ref.energy_gev, str(ref.path)),
    )


def ref_preferred_over(candidate: ProductRef, current: ProductRef) -> bool:
    candidate_parts = [part.lower() for part in candidate.path.parts]
    current_parts = [part.lower() for part in current.path.parts]
    candidate_is_smoke = any("smoke" in part for part in candidate_parts)
    current_is_smoke = any("smoke" in part for part in current_parts)
    if candidate_is_smoke != current_is_smoke:
        return current_is_smoke
    return candidate.path.stat().st_mtime >= current.path.stat().st_mtime


def mode_sort_key(mode: str) -> tuple[int, str]:
    try:
        return (MODE_ORDER.index(mode), mode)
    except ValueError:
        return (len(MODE_ORDER), mode)


def energy_matches(energy_gev: float, selected: Iterable[float] | None) -> bool:
    if selected is None:
        return True
    return any(math.isclose(energy_gev, wanted, rel_tol=1.0e-9, abs_tol=1.0e-6) for wanted in selected)


def filter_refs(
    refs: list[ProductRef],
    turbulence_selector: set[str] | None,
    mode_selector: set[str] | None,
    energy_selector: list[float] | None,
) -> list[ProductRef]:
    filtered: list[ProductRef] = []
    for ref in refs:
        if turbulence_selector is not None and ref.turbulence.lower() not in turbulence_selector:
            continue
        if mode_selector is not None and normalize_mode(ref.mode) not in mode_selector:
            continue
        if not energy_matches(ref.energy_gev, energy_selector):
            continue
        filtered.append(ref)
    return filtered


def read_array(group: h5py.Group, dataset_name: str) -> np.ndarray:
    if dataset_name not in group:
        raise KeyError(f"Missing dataset {group.name}/{dataset_name}")
    return np.asarray(group[dataset_name][()], dtype=float)


def read_scalar(h5: h5py.File, dataset_name: str):
    if dataset_name not in h5:
        return None
    value = h5[dataset_name][()]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = np.ravel(value)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_mu_coordinate(value) -> str:
    if value is None:
        return "signed"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"absolute", "abs", "abs_mu"}:
        return "absolute"
    return "signed"


def fold_to_absolute_mu(mu_centers: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[float, list[float]] = {}
    for mu, value in zip(mu_centers, values):
        key = round(abs(float(mu)), 12)
        groups.setdefault(key, []).append(float(value))

    folded_mu: list[float] = []
    folded_values: list[float] = []
    for key in sorted(groups):
        samples = np.asarray(groups[key], dtype=float)
        finite = samples[np.isfinite(samples)]
        folded_mu.append(key)
        folded_values.append(float(np.nanmean(finite)) if finite.size else math.nan)
    return np.asarray(folded_mu), np.asarray(folded_values)


def prepare_mu_axis(
    mu_centers: np.ndarray,
    dmumu_values: np.ndarray,
    stored_coordinate: str,
    requested_coordinate: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if requested_coordinate == "absolute" and np.any(mu_centers < 0.0):
        folded_mu, folded_values = fold_to_absolute_mu(mu_centers, dmumu_values)
        return folded_mu, folded_values, "absolute"
    if requested_coordinate == "absolute":
        return np.abs(mu_centers), dmumu_values, "absolute"
    if requested_coordinate == "signed":
        return mu_centers, dmumu_values, "signed"
    return mu_centers, dmumu_values, stored_coordinate


def select_dmumu_curve(dmumu_group: h5py.Group, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    uses_tau_slice = args.dmumu_tau is not None
    uses_tau_range = args.dmumu_tau_min is not None or args.dmumu_tau_max is not None
    if uses_tau_slice and uses_tau_range:
        raise ValueError("--dmumu-tau cannot be combined with --dmumu-tau-min/--dmumu-tau-max.")

    if uses_tau_slice or uses_tau_range:
        if args.dmumu_variant not in DMUMU_2D_VARIANTS:
            raise ValueError("Tau selection only supports --dmumu-variant=centered or raw.")
        tau_norm = read_array(dmumu_group, "tau_norm")
        values = read_array(dmumu_group, DMUMU_2D_VARIANTS[args.dmumu_variant])
        if uses_tau_slice:
            index = int(np.nanargmin(np.abs(tau_norm - float(args.dmumu_tau))))
            return values[:, index], rf"$D_{{\mu\mu}}/\Omega_0$ at $\tau\Omega_0={tau_norm[index]:.4g}$"

        tau_min = -math.inf if args.dmumu_tau_min is None else float(args.dmumu_tau_min)
        tau_max = math.inf if args.dmumu_tau_max is None else float(args.dmumu_tau_max)
        if tau_min > tau_max:
            raise ValueError("--dmumu-tau-min must be <= --dmumu-tau-max.")
        mask = np.isfinite(tau_norm) & (tau_norm >= tau_min) & (tau_norm <= tau_max)
        if not np.any(mask):
            raise ValueError("No tau samples matched the requested D_mumu tau range.")
        return np.nanmean(values[:, mask], axis=1), rf"$\langle D_{{\mu\mu}}/\Omega_0\rangle_{{{tau_min:g}\leq\tau\Omega_0\leq{tau_max:g}}}$"

    dataset_name = DMUMU_VARIANTS[args.dmumu_variant]
    return read_array(dmumu_group, dataset_name), r"Tau-averaged $D_{\mu\mu}(\mu)$"


def load_product(ref: ProductRef, args: argparse.Namespace) -> Product:
    with h5py.File(ref.path, "r") as h5:
        delta_group = h5["delta_mu2"]
        dmumu_group = h5["dmumu"]
        mean = read_array(delta_group, "delta_mu2_particle_mean")
        if "delta_mu2_particle_sem" in delta_group:
            sem = read_array(delta_group, "delta_mu2_particle_sem")
        else:
            sem = np.zeros_like(mean)

        stored_mu_coordinate = normalize_mu_coordinate(read_scalar(h5, "mu_bin_coordinate"))
        mu_centers = read_array(dmumu_group, "mu_centers")
        dmumu_values, dmumu_reduction_label = select_dmumu_curve(dmumu_group, args)
        mu_centers, dmumu_values, plotted_mu_coordinate = prepare_mu_axis(
            mu_centers,
            dmumu_values,
            stored_mu_coordinate,
            args.mu_coordinate,
        )
        larmor_box_fraction = read_scalar(h5, "larmor_radius_box_fraction")
        if larmor_box_fraction is not None:
            larmor_box_fraction = float(larmor_box_fraction)

        return Product(
            ref=ref,
            tau_norm=read_array(delta_group, "tau_norm"),
            delta_mu2_mean=mean,
            delta_mu2_sem=sem,
            mu_centers=mu_centers,
            dmumu_tau_average=dmumu_values,
            dmumu_reduction_label=dmumu_reduction_label,
            mu_coordinate=plotted_mu_coordinate,
            energy_input_kind=read_scalar(h5, "energy_input_kind"),
            larmor_radius_box_fraction=larmor_box_fraction,
        )


def finite_positive_floor(arrays: Iterable[np.ndarray]) -> float:
    values = np.concatenate([np.ravel(array) for array in arrays])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0e-30
    return float(np.nanmin(values)) * 0.5


def values_for_scale(values: np.ndarray, scale: str) -> np.ndarray:
    plotted = np.asarray(values, dtype=float).copy()
    if scale == "log":
        plotted[~np.isfinite(plotted) | (plotted <= 0)] = np.nan
    return plotted


def set_scale(axis: plt.Axes, axis_name: str, scale: str) -> None:
    if scale == "log":
        getattr(axis, f"set_{axis_name}scale")("log")


def style_for_count(count: int):
    if count <= 10:
        return plt.cm.tab10(np.linspace(0.0, 1.0, max(count, 1)))
    return plt.cm.viridis(np.linspace(0.08, 0.92, count))


def mu_axis_label(products: list[Product]) -> str:
    if products and all(product.mu_coordinate == "absolute" for product in products):
        return r"$|\mu_0|$"
    return r"$\mu_0$"


def dmumu_panel_title(products: list[Product]) -> str:
    if products and all(product.dmumu_reduction_label == products[0].dmumu_reduction_label for product in products):
        return products[0].dmumu_reduction_label
    return r"$D_{\mu\mu}(\mu)$"


def plot_product_group(
    products: list[Product],
    labels: list[str],
    title: str,
    output_path: Path,
    delta_y_scale: str,
    dmumu_y_scale: str,
    draw_sem: bool,
    plot_kind: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = style_for_count(len(products))
    include_delta = plot_kind in {"both", "delta"}
    include_dmumu = plot_kind in {"both", "dmumu"}
    ncols = int(include_delta) + int(include_dmumu)
    figsize = (11.5, 4.6) if ncols == 2 else (6.2, 4.6)
    fig, raw_axes = plt.subplots(1, ncols, figsize=figsize)
    axes_list = np.atleast_1d(raw_axes).tolist()
    delta_axis = axes_list.pop(0) if include_delta else None
    dmumu_axis = axes_list.pop(0) if include_dmumu else None
    delta_floor = finite_positive_floor(product.delta_mu2_mean for product in products)

    for product, label, color in zip(products, labels, colors):
        if delta_axis is not None:
            tau = product.tau_norm
            mean = product.delta_mu2_mean
            y = values_for_scale(mean, delta_y_scale)
            delta_axis.plot(tau, y, "o-", color=color, linewidth=1.6, markersize=3.8, label=label)

            if draw_sem:
                sem = product.delta_mu2_sem
                lower = np.maximum(mean - sem, delta_floor)
                upper = mean + sem
                lower = values_for_scale(lower, delta_y_scale)
                upper = values_for_scale(upper, delta_y_scale)
                valid = np.isfinite(tau) & np.isfinite(lower) & np.isfinite(upper)
                delta_axis.fill_between(tau[valid], lower[valid], upper[valid], color=color, alpha=0.14, linewidth=0)

        if dmumu_axis is not None:
            dmumu = values_for_scale(product.dmumu_tau_average, dmumu_y_scale)
            dmumu_axis.plot(product.mu_centers, dmumu, "o-", color=color, linewidth=1.6, markersize=3.8, label=label)

    legend_source = delta_axis if delta_axis is not None else dmumu_axis

    if delta_axis is not None:
        delta_axis.set_xscale("log")
        set_scale(delta_axis, "y", delta_y_scale)
        delta_axis.set_xlabel(r"$\tau \Omega_0$")
        delta_axis.set_ylabel(r"$\langle(\Delta\mu)^2\rangle$")
        delta_axis.set_title(r"$\langle(\Delta\mu)^2\rangle$")
        delta_axis.grid(True, which="both", alpha=0.28)

    if dmumu_axis is not None:
        set_scale(dmumu_axis, "y", dmumu_y_scale)
        dmumu_axis.set_xlabel(mu_axis_label(products))
        dmumu_axis.set_ylabel(r"$D_{\mu\mu}/\Omega_0$")
        dmumu_axis.set_title(dmumu_panel_title(products))
        dmumu_axis.grid(True, which="both", alpha=0.28)

    handles, legend_labels = legend_source.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(products), 4), frameon=False)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "comparison"


def comparison_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.plot_kind != "both":
        parts.append(args.plot_kind)
    if args.dmumu_variant != "centered":
        parts.append(args.dmumu_variant.replace("-", "_"))
    if args.mu_coordinate != "auto":
        parts.append(args.mu_coordinate)
    if args.dmumu_tau is not None:
        parts.append(f"tau_{args.dmumu_tau:g}".replace(".", "p"))
    if args.dmumu_tau_min is not None or args.dmumu_tau_max is not None:
        tau_min = "min" if args.dmumu_tau_min is None else f"{args.dmumu_tau_min:g}".replace(".", "p")
        tau_max = "max" if args.dmumu_tau_max is None else f"{args.dmumu_tau_max:g}".replace(".", "p")
        parts.append(f"tau_{tau_min}_{tau_max}")
    return "" if not parts else "_" + "_".join(parts)


def format_product_scale_label(product: Product) -> str:
    kind = (product.energy_input_kind or "").lower()
    if "larmor" in kind and product.larmor_radius_box_fraction is not None and math.isfinite(product.larmor_radius_box_fraction):
        return rf"$r_L/L_{{box}}={product.larmor_radius_box_fraction:.3g}$"
    return format_energy_label(product.ref.energy_gev)


def product_scale_key(product: Product) -> tuple[str, float]:
    kind = (product.energy_input_kind or "").lower()
    if "larmor" in kind and product.larmor_radius_box_fraction is not None and math.isfinite(product.larmor_radius_box_fraction):
        return ("larmor_box_fraction", round(product.larmor_radius_box_fraction, 12))
    return ("energy_gev", round(product.ref.energy_gev, 9))


def product_scale_filename_tag(product: Product) -> str:
    kind = (product.energy_input_kind or "").lower()
    if "larmor" in kind and product.larmor_radius_box_fraction is not None and math.isfinite(product.larmor_radius_box_fraction):
        value = f"{product.larmor_radius_box_fraction:.12g}".replace(".", "p")
        return f"rLbox_{value}"
    return energy_tag(product.ref.energy_gev)


def group_products_custom(products: list[Product], key_func) -> dict[tuple, list[Product]]:
    grouped: dict[tuple, list[Product]] = {}
    for product in products:
        grouped.setdefault(key_func(product), []).append(product)
    return grouped


def group_products(products: list[Product], key_names: tuple[str, ...]) -> dict[tuple, list[Product]]:
    grouped: dict[tuple, list[Product]] = {}
    for product in products:
        key = tuple(getattr(product.ref, name) for name in key_names)
        grouped.setdefault(key, []).append(product)
    return grouped


def sort_products(products: list[Product], by: str) -> list[Product]:
    if by == "mode":
        return sorted(products, key=lambda product: mode_sort_key(product.ref.mode))
    if by == "energy":
        return sorted(products, key=lambda product: product.ref.energy_gev)
    if by == "turbulence":
        return sorted(products, key=lambda product: product.ref.turbulence.lower())
    return products


def build_mode_figures(products: list[Product], output_dir: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    grouped = group_products_custom(products, lambda product: (product.ref.turbulence, product_scale_key(product)))
    for (turbulence, _scale_key), group in sorted(grouped.items(), key=lambda item: (item[0][0].lower(), item[0][1])):
        group = sort_products(group, "mode")
        labels = [MODE_LABELS.get(product.ref.mode, product.ref.mode) for product in group]
        figure_path = output_dir / "modes" / sanitize_filename(turbulence) / f"{product_scale_filename_tag(group[0])}_modes{comparison_suffix(args)}.png"
        title = f"{turbulence}: mode comparison at {format_product_scale_label(group[0])}"
        plot_product_group(group, labels, title, figure_path, args.delta_y_scale, args.dmumu_y_scale, not args.no_sem, args.plot_kind)
        rows.extend(manifest_rows("modes", figure_path, group))
    return rows


def build_energy_figures(products: list[Product], output_dir: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (turbulence, mode), group in sorted(group_products(products, ("turbulence", "mode")).items()):
        group = sort_products(group, "energy")
        labels = [format_product_scale_label(product) for product in group]
        figure_path = output_dir / "energies" / sanitize_filename(turbulence) / f"{sanitize_filename(mode)}_energies{comparison_suffix(args)}.png"
        title = f"{turbulence} {MODE_LABELS.get(mode, mode)}: energy comparison"
        plot_product_group(group, labels, title, figure_path, args.delta_y_scale, args.dmumu_y_scale, not args.no_sem, args.plot_kind)
        rows.extend(manifest_rows("energies", figure_path, group))
    return rows


def build_turbulence_figures(products: list[Product], output_dir: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    grouped = group_products_custom(products, lambda product: (product.ref.mode, product_scale_key(product)))
    for (mode, _scale_key), group in sorted(grouped.items(), key=lambda item: (mode_sort_key(item[0][0]), item[0][1])):
        group = sort_products(group, "turbulence")
        labels = [product.ref.turbulence for product in group]
        mode_label = MODE_LABELS.get(mode, mode)
        figure_path = output_dir / "turbulence" / sanitize_filename(mode) / f"{product_scale_filename_tag(group[0])}_turbulence{comparison_suffix(args)}.png"
        title = f"{mode_label}: turbulence comparison at {format_product_scale_label(group[0])}"
        plot_product_group(group, labels, title, figure_path, args.delta_y_scale, args.dmumu_y_scale, not args.no_sem, args.plot_kind)
        rows.extend(manifest_rows("turbulence", figure_path, group))
    return rows


def manifest_rows(figure_set: str, figure_path: Path, products: list[Product]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for product in products:
        rows.append(
            {
                "figure_set": figure_set,
                "figure_path": str(figure_path),
                "turbulence": product.ref.turbulence,
                "mode": product.ref.mode,
                "energy_GeV": f"{product.ref.energy_gev:g}",
                "mu_coordinate": product.mu_coordinate,
                "energy_input_kind": product.energy_input_kind or "",
                "larmor_radius_box_fraction": ""
                if product.larmor_radius_box_fraction is None
                else f"{product.larmor_radius_box_fraction:g}",
                "input_h5": str(product.ref.path),
            }
        )
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "comparison_manifest.tsv"
    fieldnames = [
        "figure_set",
        "figure_path",
        "turbulence",
        "mode",
        "energy_GeV",
        "mu_coordinate",
        "energy_input_kind",
        "larmor_radius_box_fraction",
        "input_h5",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def print_refs(refs: list[ProductRef]) -> None:
    if not refs:
        print("No products selected.")
        return
    for ref in refs:
        print(f"{ref.turbulence}\t{ref.mode}\t{ref.energy_gev:g}\t{ref.path}")


def main() -> None:
    args = parse_args()
    figure_sets = selected_figure_sets(args.figure_set)
    all_refs = discover_products(args.input_root, args.verbose)
    selected_refs = filter_refs(
        all_refs,
        split_selector(args.turbulence),
        split_mode_selector(args.mode),
        split_energy_selector(args.energy),
    )

    print(f"Discovered {len(all_refs)} product(s) under {args.input_root}")
    print(f"Selected {len(selected_refs)} product(s)")

    if args.list:
        print_refs(selected_refs)
        return

    if not selected_refs:
        raise SystemExit("No selected HDF5 products. Use --list to inspect what was discovered.")

    products = [load_product(ref, args) for ref in selected_refs]
    rows: list[dict[str, str]] = []
    if "modes" in figure_sets:
        rows.extend(build_mode_figures(products, args.output_dir, args))
    if "energies" in figure_sets:
        rows.extend(build_energy_figures(products, args.output_dir, args))
    if "turbulence" in figure_sets:
        rows.extend(build_turbulence_figures(products, args.output_dir, args))

    manifest_path = write_manifest(args.output_dir, rows)
    figure_paths = sorted({row["figure_path"] for row in rows})

    print("Wrote figures:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
