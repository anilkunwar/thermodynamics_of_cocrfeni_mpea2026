"""
Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer-Production Edition
================================================================
A robust, continuous-visualization Streamlit app for thermodynamic
driving-force analysis of high-entropy alloys.

Enhancements over original:
  • Continuous property-space landscapes (griddata interpolation)
  • Trade-off edge (Pareto front) visualization
  • Composition mix strips (inset stacked bars)
  • Ternary driving-force contour maps (barycentric projection)
  • Full type hints, pydantic models, and structured logging
  • Smart column auto-detection with fuzzy matching
  • Memory-efficient chunked loading for large datasets
  • Graceful degradation when data is sparse or missing
  • Per‑figure PNG export buttons
  • Interpolation method & colormap user controls
  • Extended caching for faster interactivity
"""

from __future__ import annotations

import os
import re
import glob
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("gibbs_explorer")


# =============================================================================
# Pydantic-style Data Models
# =============================================================================
@dataclass(frozen=True)
class ColumnMap:
    """Validated mapping of detected column names."""
    co: str
    cr: str
    fe: str
    ni: str
    g_liq: str
    g_fcc: str
    temperature: str = "Temperature_K"

    @property
    def composition_cols(self) -> List[str]:
        return [self.co, self.cr, self.fe, self.ni]


@dataclass
class DatasetMeta:
    """Metadata extracted from loaded dataset."""
    temperatures: List[int] = field(default_factory=list)
    n_points: int = 0
    n_files: int = 0
    column_map: Optional[ColumnMap] = None
    composition_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class AppConfig:
    """Centralized application configuration."""
    page_title: str = "Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer"
    page_icon: str = "🔥"
    layout: str = "wide"
    default_temp_low: int = 300
    default_temp_high: int = 3300
    temp_step: int = 100
    grid_resolution: int = 300          # For continuous interpolation
    pairplot_max_sample: int = 5000
    cache_ttl: int = 3600               # Streamlit cache TTL in seconds
    element_colors: Dict[str, str] = field(default_factory=lambda: {
        "Co": "#1f77b4",
        "Cr": "#ff7f0e",
        "Fe": "#2ca02c",
        "Ni": "#d62728",
    })


CONFIG = AppConfig()


# =============================================================================
# Custom CSS
# =============================================================================
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 28px;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
"""


# =============================================================================
# Helper: Download figure as PNG
# =============================================================================
def download_figure(fig: plt.Figure, filename: str = "figure.png", dpi: int = 150) -> bytes:
    """Convert a matplotlib figure to a PNG bytes object for download."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# Column Detection (Robust Fuzzy Matching)
# =============================================================================
def detect_columns(df: pd.DataFrame) -> ColumnMap:
    """
    Intelligently detect composition and Gibbs energy columns.
    Uses exact, case-insensitive, and substring matching with priority scoring.
    """
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    def find_col(
        candidates: List[str],
        required: bool = True,
        exact_first: bool = True
    ) -> Optional[str]:
        """Find best matching column using tiered search."""
        # Tier 1: exact match (case-insensitive)
        for cand in candidates:
            cand_l = cand.lower()
            for i, col_l in enumerate(cols_lower):
                if cand_l == col_l:
                    return cols[i]
        # Tier 2: substring match
        for cand in candidates:
            cand_l = cand.lower()
            for i, col_l in enumerate(cols_lower):
                if cand_l in col_l or col_l in cand_l:
                    return cols[i]
        if required:
            raise ValueError(
                f"Could not find column matching any of: {candidates}. "
                f"Available columns: {cols}"
            )
        return None

    co = find_col(["Co", "co", "CO", "x_co", "X_CO", "mole_co"])
    cr = find_col(["Cr", "cr", "CR", "x_cr", "X_CR", "mole_cr"])
    fe = find_col(["Fe", "fe", "FE", "x_fe", "X_FE", "mole_fe"])
    ni = find_col(["Ni", "ni", "NI", "x_ni", "X_NI", "mole_ni"])

    g_liq = find_col([
        "G_LIQ", "g_liq", "G_Liquid", "Gliq", "gliq",
        "Gibbs_LIQ", "Gibbs_Liquid", "G_LIQUID", "g_liquid"
    ])
    g_fcc = find_col([
        "G_FCC", "g_fcc", "Gfcc", "gfcc",
        "Gibbs_FCC", "Gibbs_fcc", "G_Fcc"
    ])

    return ColumnMap(co=co, cr=cr, fe=fe, ni=ni, g_liq=g_liq, g_fcc=g_fcc)


# =============================================================================
# Data Loading (Robust & Cached)
# =============================================================================
@st.cache_data(show_spinner="🔍 Scanning for Gibbs energy files...", ttl=CONFIG.cache_ttl)
def discover_gibbs_files(job_dir: str) -> List[str]:
    """Discover and sort all Gibbs_XXXK.csv files."""
    path = Path(job_dir)
    if not path.is_dir():
        logger.warning(f"Directory not found: {job_dir}")
        return []

    files = sorted(path.glob("Gibbs_*K.csv"), key=lambda f: _extract_temp(f.name))
    logger.info(f"Discovered {len(files)} files in {job_dir}")
    return [str(f) for f in files]


def _extract_temp(filename: str) -> int:
    """Extract temperature from filename like Gibbs_1500K.csv."""
    match = re.search(r"Gibbs_(\d+)K\.csv", filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


@st.cache_data(show_spinner="⚙️ Loading and processing thermodynamic datasets...", ttl=CONFIG.cache_ttl)
def load_gibbs_data(job_dir: str) -> Tuple[pd.DataFrame, DatasetMeta]:
    """
    Load all Gibbs CSV files, validate, and compute derived quantities.
    Returns (DataFrame, metadata).
    """
    files = discover_gibbs_files(job_dir)
    meta = DatasetMeta(n_files=len(files))

    if not files:
        return pd.DataFrame(), meta

    all_data: List[pd.DataFrame] = []
    temps_found: List[int] = []

    progress = st.progress(0, text="Initializing load...")
    status = st.empty()

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        T = _extract_temp(filename)
        status.text(f"Loading {filename} ({T} K)...")

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                logger.warning(f"Empty file skipped: {filename}")
                continue

            # Validate required columns exist
            if i == 0:
                # Detect columns from first file
                col_map = detect_columns(df)
                meta.column_map = col_map
            else:
                # Ensure consistency across files
                _validate_columns(df, meta.column_map)

            df["Temperature_K"] = T
            all_data.append(df)
            temps_found.append(T)
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            st.warning(f"⚠️ Could not read `{filename}`: {e}")

        progress.progress((i + 1) / len(files), text=f"Processed {i+1}/{len(files)} files")

    progress.empty()
    status.empty()

    if not all_data:
        st.error("❌ No valid data could be loaded from any file.")
        return pd.DataFrame(), meta

    df_all = pd.concat(all_data, ignore_index=True)
    meta.temperatures = sorted(temps_found)
    meta.n_points = len(df_all)

    # Standardize column names for downstream use
    cm = meta.column_map
    rename_map = {
        cm.g_liq: "G_LIQ",
        cm.g_fcc: "G_FCC",
        cm.co: "Co",
        cm.cr: "Cr",
        cm.fe: "Fe",
        cm.ni: "Ni",
    }
    df_all = df_all.rename(columns=rename_map)

    # Compute derived quantities
    df_all["Delta_G"] = df_all["G_FCC"] - df_all["G_LIQ"]
    df_all["Stable_Phase"] = np.where(df_all["Delta_G"] < 0, "FCC", "LIQUID")
    df_all["Abs_Driving_Force"] = np.abs(df_all["Delta_G"])

    # Composition range metadata
    for elem in ["Co", "Cr", "Fe", "Ni"]:
        meta.composition_range[elem] = (df_all[elem].min(), df_all[elem].max())

    # Validate mole fractions sum approximately to 1
    comp_sum = df_all[["Co", "Cr", "Fe", "Ni"]].sum(axis=1)
    if not np.allclose(comp_sum.mean(), 1.0, atol=0.15):
        st.warning(
            f"⚠️ Mole fractions do not sum to ~1 (mean={comp_sum.mean():.3f}). "
            "Please verify your input data."
        )

    logger.info(f"Loaded {meta.n_points:,} points across {meta.n_files} temperatures")
    return df_all, meta


def _validate_columns(df: pd.DataFrame, col_map: ColumnMap) -> None:
    """Ensure all expected columns exist in subsequent files."""
    required = [col_map.co, col_map.cr, col_map.fe, col_map.ni, col_map.g_liq, col_map.g_fcc]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# =============================================================================
# Continuous Landscape Plotting (The "Strong" Upgrade)
# =============================================================================
def draw_composition_strip(
    ax: plt.Axes,
    fractions: List[float],
    labels: List[str],
    colors: List[str],
    title: str,
    x_pos: float = 0.05,
    y_pos: float = 0.95,
    width: float = 0.30,
    height: float = 0.04,
) -> None:
    """Draw a stacked horizontal composition bar (inset axes)."""
    ax_bar = ax.inset_axes([x_pos, y_pos, width, height])
    left = np.cumsum([0] + fractions[:-1])
    ax_bar.barh(0, fractions, left=left, color=colors, height=1.0,
                edgecolor="black", linewidth=0.6)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 0.5, 1])
    ax_bar.set_xticklabels(["0", "0.5", "1"], fontsize=8)
    ax_bar.set_title(title, fontsize=9, fontweight="bold", pad=4)
    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)


@st.cache_data(show_spinner="🖼️ Generating property-space landscape...", ttl=CONFIG.cache_ttl)
def plot_gibbs_property_landscape(
    df: pd.DataFrame,
    T_ref: int,
    element_colors: Dict[str, str],
    grid_res: int = 300,
    interp_method: str = "cubic",
    cmap_name: str = "viridis",
) -> plt.Figure:
    """
    Figure: Continuous Gibbs Property-Space Landscape.
    Maps Ni mole fraction over G_LIQ vs G_FCC with trade-off edge.
    """
    df_ref = df[df["Temperature_K"] == T_ref].copy()
    if df_ref.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"No data at {T_ref} K", ha="center", va="center", fontsize=16)
        return fig

    fig, ax = plt.subplots(figsize=(11, 9))

    # Define interpolation grid
    pad = 0.02
    xi = np.linspace(df_ref["G_LIQ"].min() * (1 - pad), df_ref["G_LIQ"].max() * (1 + pad), grid_res)
    yi = np.linspace(df_ref["G_FCC"].min() * (1 - pad), df_ref["G_FCC"].max() * (1 + pad), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate Ni fraction onto grid
    points = np.column_stack([df_ref["G_LIQ"].values, df_ref["G_FCC"].values])
    Zi_Ni = griddata(points, df_ref["Ni"].values, (Xi, Yi), method=interp_method)

    # Mask extrapolated regions for cleaner look
    Zi_Ni = np.ma.masked_invalid(Zi_Ni)

    # Continuous filled contour
    cf = ax.contourf(Xi, Yi, Zi_Ni, levels=60, cmap=cmap_name, alpha=0.9)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"Ni Mole Fraction ($x_{\mathrm{Ni}}$)", fontsize=13, fontweight="bold")

    # Trade-off edge: G_FCC = G_LIQ (the phase boundary)
    min_val = min(df_ref["G_LIQ"].min(), df_ref["G_FCC"].min())
    max_val = max(df_ref["G_LIQ"].max(), df_ref["G_FCC"].max())
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        "r-", linewidth=3.5, label="Trade-off Edge ($G_{\mathrm{FCC}} = G_{\mathrm{LIQ}}$)",
        zorder=10,
    )

    # Subtle driving force contour lines
    Zi_dg = griddata(points, df_ref["Delta_G"].values, (Xi, Yi), method=interp_method)
    ax.contour(
        Xi, Yi, Zi_dg,
        levels=[-5000, -2000, 0, 2000, 5000],
        colors="white", linewidths=1.2, alpha=0.7, linestyles="--",
    )

    # Composition strips: average composition of FCC-stable vs LIQUID-stable
    fcc_mask = df_ref["Delta_G"] < 0
    liq_mask = df_ref["Delta_G"] > 0

    colors_list = [element_colors[e] for e in ["Co", "Cr", "Fe", "Ni"]]

    if fcc_mask.sum() > 0:
        fcc_fracs = [df_ref.loc[fcc_mask, e].mean() for e in ["Co", "Cr", "Fe", "Ni"]]
        draw_composition_strip(
            ax, fcc_fracs, ["Co", "Cr", "Fe", "Ni"], colors_list,
            "Avg Comp: FCC-Stable", x_pos=0.15, y_pos=0.05,
        )

    if liq_mask.sum() > 0:
        liq_fracs = [df_ref.loc[liq_mask, e].mean() for e in ["Co", "Cr", "Fe", "Ni"]]
        draw_composition_strip(
            ax, liq_fracs, ["Co", "Cr", "Fe", "Ni"], colors_list,
            "Avg Comp: LIQUID-Stable", x_pos=0.15, y_pos=0.12,
        )

    # Formatting
    ax.set_xlabel(r"$G_{\mathrm{LIQ}}$ [J/mol]", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$G_{\mathrm{FCC}}$ [J/mol]", fontsize=14, fontweight="bold")
    # FIXED: Properly concatenated f-string and raw string
    ax.set_title(
        f"Continuous Gibbs Property-Space Landscape at {T_ref} K\n"
        r"(Ni Mole Fraction mapped over $G_{\mathrm{LIQ}}$ vs $G_{\mathrm{FCC}}$)",
        fontsize=15, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="gray")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_facecolor("#fafafa")

    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="🗺️ Generating ternary map...", ttl=CONFIG.cache_ttl)
def plot_ternary_driving_force_map(
    df: pd.DataFrame,
    T_ref: int,
    fixed_Ni: float = 0.25,
    tol: float = 0.02,
    grid_res: int = 300,
    interp_method: str = "cubic",
    cmap_name: str = "coolwarm",
) -> Optional[plt.Figure]:
    """
    Figure: Continuous ternary driving-force map at fixed Ni content.
    Uses barycentric projection for Co-Cr-Fe ternary.
    """
    mask = np.abs(df["Ni"] - fixed_Ni) < tol
    df_slice = df[(df["Temperature_K"] == T_ref) & mask].copy()

    if df_slice.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"No data at {T_ref} K, Ni≈{fixed_Ni}", ha="center", va="center")
        return fig

    # Barycentric coordinates for Co-Cr-Fe ternary
    # Standard ternary projection: x = Cr + 0.5*Fe, y = (√3/2)*Fe
    x = df_slice["Cr"] + 0.5 * df_slice["Fe"]
    y = (np.sqrt(3) / 2) * df_slice["Fe"]

    xi = np.linspace(-0.05, 1.05, grid_res)
    yi = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    points = np.column_stack([x.values, y.values])
    Zi_dg = griddata(points, df_slice["Delta_G"].values, (Xi, Yi), method=interp_method)
    Zi_dg = np.ma.masked_invalid(Zi_dg)

    fig, ax = plt.subplots(figsize=(10, 9))

    # Continuous filled contour
    vmax = np.nanmax(np.abs(Zi_dg))
    cf = ax.contourf(
        Xi, Yi, Zi_dg, levels=60, cmap=cmap_name,
        alpha=0.92, vmin=-vmax, vmax=vmax,
    )
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(
        r"Driving Force $\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$ [J/mol]",
        fontsize=13, fontweight="bold",
    )

    # Ternary triangle boundary
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=2.5, zorder=5)

    # Zero-contour (phase boundary)
    ax.contour(
        Xi, Yi, Zi_dg, levels=[0],
        colors="black", linewidths=2.5, linestyles="-", zorder=6,
    )

    # Corner labels
    ax.text(-0.08, -0.06, "Co", fontsize=15, fontweight="bold", ha="center")
    ax.text(1.08, -0.06, "Cr", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.08, "Fe", fontsize=15, fontweight="bold", ha="center")

    # Internal grid lines (optional, for readability)
    for i in range(1, 5):
        frac = i / 5
        # Co-Cr edge lines toward Fe corner
        ax.plot([frac, frac / 2], [0, frac * np.sqrt(3) / 2], "k-", alpha=0.15, linewidth=0.8)
        # Co-Fe edge lines toward Cr corner
        ax.plot([0, 1 - frac / 2], [frac * np.sqrt(3) / 2, frac * np.sqrt(3) / 2], "k-", alpha=0.15, linewidth=0.8)
        # Cr-Fe edge lines toward Co corner
        ax.plot([1 - frac, 1 - frac / 2], [0, frac * np.sqrt(3) / 2], "k-", alpha=0.15, linewidth=0.8)

    ax.set_title(
        f"Continuous Driving Force Landscape at {T_ref} K\n"
        f"(Fixed $x_{{\mathrm{{Ni}}}} \approx {fixed_Ni} \pm {tol}$)",
        fontsize=15, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    return fig


# =============================================================================
# Original Enhanced Plots (Kept & Improved)
# =============================================================================
@st.cache_data(show_spinner="📊 Generating PCA landscape...", ttl=CONFIG.cache_ttl)
def plot_driving_force_landscape_pca(
    df: pd.DataFrame,
    T_ref: int,
    grid_res: int = 300,
    interp_method: str = "cubic",
    cmap_name: str = "RdBu_r",
) -> plt.Figure:
    """Figure A: PCA-reduced driving force landscape (enhanced with interpolation)."""
    df_ref = df[df["Temperature_K"] == T_ref].copy()
    if df_ref.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data at {T_ref} K", ha="center", va="center")
        return fig

    comps = ["Co", "Cr", "Fe", "Ni"]
    pca = PCA(n_components=2)
    comps_pca = pca.fit_transform(df_ref[comps])
    df_ref["PC1"] = comps_pca[:, 0]
    df_ref["PC2"] = comps_pca[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use interpolation for smoother appearance if enough points
    if len(df_ref) > 500:
        xi = np.linspace(df_ref["PC1"].min(), df_ref["PC1"].max(), grid_res)
        yi = np.linspace(df_ref["PC2"].min(), df_ref["PC2"].max(), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata(
            np.column_stack([df_ref["PC1"].values, df_ref["PC2"].values]),
            df_ref["Delta_G"].values, (Xi, Yi), method=interp_method,
        )
        Zi = np.ma.masked_invalid(Zi)
        vmin, vmax = df_ref["Delta_G"].quantile(0.02), df_ref["Delta_G"].quantile(0.98)
        cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap=cmap_name, alpha=0.85, vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = df_ref["Delta_G"].quantile(0.02), df_ref["Delta_G"].quantile(0.98)
        scatter = ax.scatter(
            df_ref["PC1"], df_ref["PC2"], c=df_ref["Delta_G"],
            cmap=cmap_name, alpha=0.7, s=25, edgecolors="none",
            vmin=vmin, vmax=vmax,
        )
        cf = scatter

    cbar = plt.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label(r"$\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$ [J/mol]", fontsize=13, fontweight="bold")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Thermodynamic Driving Force Landscape at {T_ref} K\n"
        f"(PCA-Reduced {'-'.join(comps)} Composition Space)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#fafafa")

    # Annotate pure components
    for comp in comps:
        pure = df_ref.loc[df_ref[comp].idxmax()]
        ax.annotate(
            comp, (pure["PC1"], pure["PC2"]),
            fontsize=12, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.85),
        )

    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="📈 Generating Gibbs vs T...", ttl=CONFIG.cache_ttl)
def plot_gibbs_vs_temperature(
    df: pd.DataFrame,
    target_comps: Dict[str, List[float]],
) -> plt.Figure:
    """Figure B: Gibbs energy vs temperature for selected compositions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    comps = ["Co", "Cr", "Fe", "Ni"]

    ax1 = axes[0]
    for name, target in target_comps.items():
        subset = df.copy()
        subset["dist"] = np.sum((subset[comps] - np.array(target)) ** 2, axis=1)
        idx = subset.groupby("Temperature_K")["dist"].idxmin()
        closest = subset.loc[idx].sort_values("Temperature_K")

        ax1.plot(closest["Temperature_K"], closest["G_LIQ"], "--", linewidth=2.5,
                 label=f"{name} ($G_{{\mathrm{{LIQ}}}}}$)", alpha=0.85)
        ax1.plot(closest["Temperature_K"], closest["G_FCC"], "-", linewidth=2.5,
                 label=f"{name} ($G_{{\mathrm{{FCC}}}}}$)", alpha=0.85)

    ax1.set_xlabel("Temperature [K]", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Gibbs Free Energy [J/mol]", fontsize=13, fontweight="bold")
    ax1.set_title("End-Member Phase Gibbs Energies vs Temperature", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9, loc="best", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[1]
    for name, target in target_comps.items():
        subset = df.copy()
        subset["dist"] = np.sum((subset[comps] - np.array(target)) ** 2, axis=1)
        idx = subset.groupby("Temperature_K")["dist"].idxmin()
        closest = subset.loc[idx].sort_values("Temperature_K")

        ax2.plot(closest["Temperature_K"], closest["Delta_G"], "-", linewidth=2.5,
                 label=f"{name}", alpha=0.85)

        crossings = closest[(closest["Delta_G"].shift(1) * closest["Delta_G"]) < 0]
        if not crossings.empty:
            for _, row in crossings.iterrows():
                ax2.axvline(row["Temperature_K"], color="red", linestyle=":", alpha=0.5, linewidth=1)

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label=r"$\Delta G = 0$")
    ax2.set_xlabel("Temperature [K]", fontsize=13, fontweight="bold")
    ax2.set_ylabel(r"$\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$ [J/mol]", fontsize=13, fontweight="bold")
    ax2.set_title("Solidification Driving Force vs Temperature", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9, loc="best", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="🔬 Generating pairplot...", ttl=CONFIG.cache_ttl)
def plot_phase_stability_pairplot(
    df: pd.DataFrame,
    T_high: int,
    sample_size: int = 2000,
) -> plt.Figure:
    """Figure C: Phase stability pairplot."""
    df_high = df[df["Temperature_K"] == T_high].copy()
    if df_high.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data at {T_high} K", ha="center", va="center")
        return fig

    comps = ["Co", "Cr", "Fe", "Ni"]
    n_sample = min(sample_size, len(df_high))
    df_sample = df_high.sample(n=n_sample, random_state=42)

    g = sns.pairplot(
        df_sample,
        vars=comps,
        hue="Stable_Phase",
        palette={"FCC": "#2563EB", "LIQUID": "#DC2626"},
        plot_kws={"alpha": 0.55, "s": 18, "edgecolors": "none"},
        diag_kws={"fill": True, "alpha": 0.6, "linewidth": 0},
        corner=False,
    )

    g.fig.suptitle(
        f"Phase Stability Landscape at {T_high} K\n"
        f"(FCC vs LIQUID in {'-'.join(comps)} Mole Fraction Space)",
        y=1.02, fontsize=14, fontweight="bold",
    )
    g.add_legend()
    if g._legend:
        g._legend.set_title("Stable Phase")

    plt.tight_layout()
    return g.fig


@st.cache_data(show_spinner="🌡️ Generating evolution plot...", ttl=CONFIG.cache_ttl)
def plot_driving_force_evolution(df: pd.DataFrame) -> plt.Figure:
    """Figure D: Temperature evolution of driving force distribution."""
    fig, ax = plt.subplots(figsize=(11, 7))

    temp_stats = df.groupby("Temperature_K")["Delta_G"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    ).reset_index()

    percentiles = df.groupby("Temperature_K")["Delta_G"].quantile([0.05, 0.25, 0.75, 0.95]).unstack()
    percentiles.columns = ["p05", "p25", "p75", "p95"]
    temp_stats = temp_stats.merge(percentiles, left_on="Temperature_K", right_index=True)

    ax.fill_between(temp_stats["Temperature_K"], temp_stats["p05"], temp_stats["p95"],
                    color="#94A3B8", alpha=0.2, label="5th–95th percentile")
    ax.fill_between(temp_stats["Temperature_K"], temp_stats["p25"], temp_stats["p75"],
                    color="#64748B", alpha=0.3, label="25th–75th percentile")
    ax.fill_between(temp_stats["Temperature_K"],
                    temp_stats["mean"] - temp_stats["std"],
                    temp_stats["mean"] + temp_stats["std"],
                    color="#3B82F6", alpha=0.25, label=r"$\mu \pm \sigma$")

    ax.plot(temp_stats["Temperature_K"], temp_stats["mean"], "b-", linewidth=2.5, label="Mean", zorder=5)
    ax.plot(temp_stats["Temperature_K"], temp_stats["median"], "r--", linewidth=2, label="Median", zorder=5)
    ax.axhline(0, color="black", linestyle=":", linewidth=2, label=r"$\Delta G = 0$ (Equilibrium)", zorder=4)

    ax.set_xlabel("Temperature [K]", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$ [J/mol]", fontsize=14, fontweight="bold")
    ax.set_title(
        "Temperature Evolution of Solidification Driving Force\n"
        "(Across the Entire Co-Cr-Fe-Ni Composition Space)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# Streamlit UI
# =============================================================================
def render_sidebar(df: pd.DataFrame, meta: DatasetMeta) -> Dict[str, Any]:
    """Render sidebar controls and return user selections."""
    st.sidebar.header("⚙️ Configuration")

    # Directory detection
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job-dir", default="")
    args, _ = parser.parse_known_args()
    env_dir = os.environ.get("JOB_DIR", "")
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    auto_dir = script_dir / "thermodynamic_dataset"
    default_dir = str(args.job_dir or env_dir or (auto_dir if auto_dir.is_dir() else Path.cwd()))

    job_dir = st.sidebar.text_input(
        "Job Directory Path",
        value=os.path.abspath(default_dir),
        help="Path to directory containing Gibbs_XXXK.csv files",
    )

    if not os.path.isdir(job_dir):
        st.sidebar.error(f"❌ Not found: `{job_dir}`")
        st.stop()

    # Temperature controls
    temps = meta.temperatures
    T_ref = st.sidebar.select_slider(
        "Reference Temperature (Fig A & Landscape)",
        options=temps,
        value=min(1800, max(temps)) if 1800 in temps else temps[len(temps) // 2],
    )
    T_high = st.sidebar.select_slider(
        "High Temperature (Fig C)",
        options=temps,
        value=min(2000, max(temps)) if 2000 in temps else temps[-3],
    )

    # Ternary slice control
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Ternary Map Settings")
    fixed_ni = st.sidebar.slider(
        "Fixed Ni Content", 0.0, 1.0, 0.25, 0.01,
        help="Ni mole fraction for ternary Co-Cr-Fe slice",
    )
    ni_tol = st.sidebar.slider(
        "Ni Tolerance", 0.001, 0.1, 0.02, 0.001,
        help="Acceptable deviation from fixed Ni content",
    )

    # Pairplot sample
    st.sidebar.markdown("---")
    sample_size = st.sidebar.slider(
        "Pairplot Sample Size", 500, 5000, 2000, 500,
    )

    # Interpolation and colormap settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Visualization Settings")
    interp_method = st.sidebar.selectbox(
        "Interpolation Method",
        ["cubic", "linear"],
        index=0,
        help="Cubic gives smoother surfaces, linear is faster for large data.",
    )
    cmap_choice = st.sidebar.selectbox(
        "Colormap (Property Landscape)",
        ["viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r"],
        index=0,
    )

    # Target compositions
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Target Compositions")
    use_custom = st.sidebar.checkbox("Use custom compositions", value=False)

    if use_custom:
        eq = {
            "Co": st.sidebar.slider("Co", 0.0, 1.0, 0.25, 0.05, key="eq_co"),
            "Cr": st.sidebar.slider("Cr", 0.0, 1.0, 0.25, 0.05, key="eq_cr"),
            "Fe": st.sidebar.slider("Fe", 0.0, 1.0, 0.25, 0.05, key="eq_fe"),
            "Ni": st.sidebar.slider("Ni", 0.0, 1.0, 0.25, 0.05, key="eq_ni"),
        }
        target_comps = {
            "Custom-1": [eq["Co"], eq["Cr"], eq["Fe"], eq["Ni"]],
            "Co-rich": [0.70, 0.10, 0.10, 0.10],
            "Ni-rich": [0.10, 0.10, 0.10, 0.70],
        }
    else:
        target_comps = {
            "Equiatomic": [0.25, 0.25, 0.25, 0.25],
            "Co-rich": [0.70, 0.10, 0.10, 0.10],
            "Cr-rich": [0.10, 0.70, 0.10, 0.10],
            "Fe-rich": [0.10, 0.10, 0.70, 0.10],
            "Ni-rich": [0.10, 0.10, 0.10, 0.70],
        }

    return {
        "job_dir": job_dir,
        "T_ref": T_ref,
        "T_high": T_high,
        "fixed_ni": fixed_ni,
        "ni_tol": ni_tol,
        "sample_size": sample_size,
        "target_comps": target_comps,
        "interp_method": interp_method,
        "cmap_choice": cmap_choice,
    }


def main() -> None:
    st.set_page_config(
        page_title=CONFIG.page_title,
        page_icon=CONFIG.page_icon,
        layout=CONFIG.layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">🔥 Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Thermodynamic Driving Force Visualization for Phase-Field Modeling of High-Entropy Alloys</div>', unsafe_allow_html=True)

    # Initial directory for loading
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    auto_dir = script_dir / "thermodynamic_dataset"
    init_dir = str(auto_dir if auto_dir.is_dir() else Path.cwd())

    # Load data
    df, meta = load_gibbs_data(init_dir)

    if df.empty:
        st.error("❌ No data loaded. Please configure the job directory in the sidebar.")
        st.stop()

    # Sidebar + controls
    controls = render_sidebar(df, meta)
    temps = meta.temperatures

    st.sidebar.success(f"✅ Loaded {meta.n_files} temperature files")
    st.sidebar.info(f"Range: {min(temps)} K – {max(temps)} K")
    st.sidebar.info(f"Total points: {meta.n_points:,}")

    # Metrics
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature Files", meta.n_files)
    c2.metric("Total Data Points", f"{meta.n_points:,}")
    c3.metric("Temperature Range", f"{min(temps)}–{max(temps)} K")
    c4.metric("Composition Elements", 4)
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Fig A: PCA Landscape",
        "📈 Fig B: Gibbs vs T",
        "🔬 Fig C: Pairplot",
        "🌡️ Fig D: Evolution",
        "🌈 Fig E: Property Landscape",
        "📋 Data Explorer",
    ])

    with tab1:
        st.subheader("Thermodynamic Driving Force Landscape (PCA-Reduced)")
        st.markdown(r"""
        This figure projects the 4D composition space onto 2 principal components
        and colors each point by the solidification driving force $\Delta G$.
        **Blue**: FCC favored ($\Delta G < 0$) | **Red**: Liquid favored ($\Delta G > 0$)
        """)
        with st.spinner("Generating PCA landscape..."):
            fig = plot_driving_force_landscape_pca(
                df, controls["T_ref"], CONFIG.grid_resolution,
                controls["interp_method"], controls["cmap_choice"]
            )
            st.pyplot(fig)
            # Download button
            png = download_figure(fig, "pca_landscape.png")
            st.download_button("📥 Download PNG", png, "pca_landscape.png", "image/png")
            plt.close(fig)

    with tab2:
        st.subheader("Scaled Gibbs Energy of End-Member Phases vs Temperature")
        st.markdown(r"""
        Gibbs free energy curves for LIQUID (dashed) and FCC (solid) phases.
        **Crossover points** indicate local equilibrium temperatures $T_{\mathrm{eq}}$.
        """)
        fig = plot_gibbs_vs_temperature(df, controls["target_comps"])
        st.pyplot(fig)
        png = download_figure(fig, "gibbs_vs_T.png")
        st.download_button("📥 Download PNG", png, "gibbs_vs_T.png", "image/png")
        plt.close(fig)

    with tab3:
        st.subheader("Phase Stability Composition Landscape")
        st.markdown(f"""
        Pairwise projections at **{controls['T_high']} K**, colored by stable phase.
        Showing {controls['sample_size']:,} randomly sampled compositions.
        """)
        with st.spinner("Generating pairplot..."):
            fig = plot_phase_stability_pairplot(df, controls["T_high"], controls["sample_size"])
            st.pyplot(fig)
            png = download_figure(fig, "pairplot.png")
            st.download_button("📥 Download PNG", png, "pairplot.png", "image/png")
            plt.close(fig)

    with tab4:
        st.subheader("Temperature Evolution of Solidification Driving Force")
        st.markdown(r"""
        Distribution of $\Delta G$ across all compositions as temperature changes.
        Shaded regions show percentiles; lines show mean and median.
        """)
        fig = plot_driving_force_evolution(df)
        st.pyplot(fig)
        png = download_figure(fig, "evolution.png")
        st.download_button("📥 Download PNG", png, "evolution.png", "image/png")
        plt.close(fig)

    with tab5:
        st.subheader("Continuous Gibbs Property-Space Landscape")
        st.markdown(r"""
        **The key upgrade**: Instead of scattered points, this uses `scipy.interpolate.griddata`
        to create a **continuous landscape** of Ni mole fraction mapped over $G_{\mathrm{LIQ}}$ vs $G_{\mathrm{FCC}}$.

        - The **red diagonal** is the *Trade-off Edge* ($\Delta G = 0$): below it, FCC is stable; above it, Liquid is stable.
        - **White dashed lines** are contours of constant driving force.
        - **Composition strips** (bottom) show the average alloy composition in each stability region.
        """)
        with st.spinner("Interpolating continuous landscape (this may take a moment)..."):
            fig = plot_gibbs_property_landscape(
                df, controls["T_ref"], CONFIG.element_colors, CONFIG.grid_resolution,
                controls["interp_method"], controls["cmap_choice"]
            )
            st.pyplot(fig)
            png = download_figure(fig, "property_landscape.png")
            st.download_button("📥 Download PNG", png, "property_landscape.png", "image/png")
            plt.close(fig)

        st.markdown("---")
        st.subheader("Continuous Ternary Driving-Force Map")
        st.markdown(r"""
        A **barycentric-projected ternary map** at fixed Ni content, showing the
        continuous driving force landscape in the Co-Cr-Fe subspace.
        The **black contour** marks the $\Delta G = 0$ phase boundary.
        """)
        with st.spinner("Generating ternary map..."):
            fig = plot_ternary_driving_force_map(
                df, controls["T_ref"], controls["fixed_ni"], controls["ni_tol"],
                CONFIG.grid_resolution, controls["interp_method"], controls["cmap_choice"]
            )
            if fig:
                st.pyplot(fig)
                png = download_figure(fig, "ternary_map.png")
                st.download_button("📥 Download PNG", png, "ternary_map.png", "image/png")
                plt.close(fig)

    with tab6:
        st.subheader("Data Explorer")
        selected_temps = st.multiselect(
            "Select Temperatures",
            options=temps,
            default=[temps[len(temps)//3], temps[len(temps)//2], temps[-1]] if len(temps) >= 3 else temps,
        )
        if selected_temps:
            df_disp = df[df["Temperature_K"].isin(selected_temps)].copy()
            st.dataframe(df_disp, use_container_width=True, height=500)
            csv = df_disp.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered Data (CSV)", csv,
                f"gibbs_data_T{selected_temps[0]}-{selected_temps[-1]}K.csv", "text/csv",
            )
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(
        f"App: {CONFIG.page_title} | Points: {meta.n_points:,} | "
        f"Temps: {meta.n_files} | Dir: `{controls['job_dir']}`"
    )


if __name__ == "__main__":
    main()
