"""
Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer — AM Edition v2.0
===============================================================
A robust, continuous-visualization Streamlit app for thermodynamic
driving-force analysis of high-entropy alloys with Additive Manufacturing
(AM) specific visualizations.

Enhancements over original:
  • Continuous property-space landscapes (griddata interpolation)
  • Trade-off edge (Pareto front) visualization
  • Composition mix strips (inset stacked bars)
  • Ternary driving-force contour maps (barycentric projection)
  • Full type hints, pydantic models, and structured logging
  • Smart column auto-detection with fuzzy matching
  • Memory-efficient chunked loading for large datasets
  • Graceful degradation when data is sparse or missing
  • Per-figure PNG export buttons
  • Interpolation method & colormap user controls
  • Extended caching for faster interactivity
  • UNLOCKED: Full matplotlib colormap library (150+ colormaps with search)
  • AM EDITION: T0 solidification temperature landscape
  • AM EDITION: Thermodynamic freezing range (mushy zone) mapping
  • AM EDITION: Solidification driving force slope (dΔG/dT)
  • AM EDITION: Multi-temperature phase boundary envelope
  • AM EDITION: 3D interactive property space (Plotly)
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
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Plotly for 3D interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
    page_title: str = "Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer - AM Edition"
    page_icon: str = "🔥"
    layout: str = "wide"
    default_temp_low: int = 300
    default_temp_high: int = 3300
    temp_step: int = 100
    grid_resolution: int = 300
    pairplot_max_sample: int = 5000
    cache_ttl: int = 3600
    element_colors: Dict[str, str] = field(default_factory=lambda: {
        "Co": "#1f77b4",
        "Cr": "#ff7f0e",
        "Fe": "#2ca02c",
        "Ni": "#d62728",
    })
    freezing_range_threshold: float = 50.0
    dg_critical: float = -1500.0


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
    .am-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 8px;
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
# Composition Strip Helper
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


# =============================================================================
# Original Plots (Kept & Improved)
# =============================================================================

@st.cache_data(show_spinner="🖼️ Generating property-space landscape...", ttl=CONFIG.cache_ttl)
def plot_gibbs_property_landscape(
    df: pd.DataFrame,
    T_ref: int,
    element_colors: Dict[str, str],
    grid_res: int = 300,
    interp_method: str = "cubic",
    cmap_name: str = "viridis",
) -> plt.Figure:
    """Figure: Continuous Gibbs Property-Space Landscape. Maps Ni over G_LIQ vs G_FCC."""
    df_ref = df[df["Temperature_K"] == T_ref].copy()
    if df_ref.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"No data at {T_ref} K", ha="center", va="center", fontsize=16)
        return fig

    fig, ax = plt.subplots(figsize=(11, 9))

    pad = 0.02
    xi = np.linspace(df_ref["G_LIQ"].min() * (1 - pad), df_ref["G_LIQ"].max() * (1 + pad), grid_res)
    yi = np.linspace(df_ref["G_FCC"].min() * (1 - pad), df_ref["G_FCC"].max() * (1 + pad), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    points = np.column_stack([df_ref["G_LIQ"].values, df_ref["G_FCC"].values])
    Zi_Ni = griddata(points, df_ref["Ni"].values, (Xi, Yi), method=interp_method)
    Zi_Ni = np.ma.masked_invalid(Zi_Ni)

    cf = ax.contourf(Xi, Yi, Zi_Ni, levels=60, cmap=cmap_name, alpha=0.9)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"Ni Mole Fraction ($x_{\mathrm{Ni}}$)", fontsize=13, fontweight="bold")

    min_val = min(df_ref["G_LIQ"].min(), df_ref["G_FCC"].min())
    max_val = max(df_ref["G_LIQ"].max(), df_ref["G_FCC"].max())
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        "r-", linewidth=3.5, label="Trade-off Edge ($G_{\mathrm{FCC}} = G_{\mathrm{LIQ}}$)",
        zorder=10,
    )

    Zi_dg = griddata(points, df_ref["Delta_G"].values, (Xi, Yi), method=interp_method)
    ax.contour(
        Xi, Yi, Zi_dg,
        levels=[-5000, -2000, 0, 2000, 5000],
        colors="white", linewidths=1.2, alpha=0.7, linestyles="--",
    )

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

    ax.set_xlabel(r"$G_{\mathrm{LIQ}}$ [J/mol]", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$G_{\mathrm{FCC}}$ [J/mol]", fontsize=14, fontweight="bold")
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
    """Figure: Continuous ternary driving-force map at fixed Ni content."""
    mask = np.abs(df["Ni"] - fixed_Ni) < tol
    df_slice = df[(df["Temperature_K"] == T_ref) & mask].copy()

    if df_slice.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"No data at {T_ref} K, Ni≈{fixed_Ni}", ha="center", va="center")
        return fig

    x = df_slice["Cr"] + 0.5 * df_slice["Fe"]
    y = (np.sqrt(3) / 2) * df_slice["Fe"]

    xi = np.linspace(-0.05, 1.05, grid_res)
    yi = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    points = np.column_stack([x.values, y.values])
    Zi_dg = griddata(points, df_slice["Delta_G"].values, (Xi, Yi), method=interp_method)
    Zi_dg = np.ma.masked_invalid(Zi_dg)

    fig, ax = plt.subplots(figsize=(10, 9))

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

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=2.5, zorder=5)

    ax.contour(
        Xi, Yi, Zi_dg, levels=[0],
        colors="black", linewidths=2.5, linestyles="-", zorder=6,
    )

    ax.text(-0.08, -0.06, "Co", fontsize=15, fontweight="bold", ha="center")
    ax.text(1.08, -0.06, "Cr", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.08, "Fe", fontsize=15, fontweight="bold", ha="center")

    for i in range(1, 5):
        frac = i / 5
        ax.plot([frac, frac / 2], [0, frac * np.sqrt(3) / 2], "k-", alpha=0.15, linewidth=0.8)
        ax.plot([0, 1 - frac / 2], [frac * np.sqrt(3) / 2, frac * np.sqrt(3) / 2], "k-", alpha=0.15, linewidth=0.8)
        ax.plot([1 - frac, 1 - frac / 2], [0, frac * np.sqrt(3) / 2], "k-", alpha=0.15, linewidth=0.8)

    ax.set_title(
        f"Continuous Driving Force Landscape at {T_ref} K\n"
        f"(Fixed Ni fraction ≈ {fixed_Ni:.3f} ± {tol:.3f})",
        fontsize=15, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    return fig


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
                 label=f"{name} ($G_{{\mathrm{{LIQ}}}}$)", alpha=0.85)
        ax1.plot(closest["Temperature_K"], closest["G_FCC"], "-", linewidth=2.5,
                 label=f"{name} ($G_{{\mathrm{{FCC}}}}$)", alpha=0.85)

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
                    color="#94A3B8", alpha=0.2, label="5th-95th percentile")
    ax.fill_between(temp_stats["Temperature_K"], temp_stats["p25"], temp_stats["p75"],
                    color="#64748B", alpha=0.3, label="25th-75th percentile")
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
# AM-SPECIFIC VISUALIZATIONS (NEW)
# =============================================================================

@st.cache_data(show_spinner="🌡️ Calculating Solidification Temperature (T0) Landscape...", ttl=CONFIG.cache_ttl)
def plot_T0_landscape(
    df: pd.DataFrame,
    meta: DatasetMeta,
    fixed_Ni: float = 0.25,
    tol: float = 0.02,
    grid_res: int = 300,
    interp_method: str = "cubic",
    cmap_name: str = "inferno",
) -> Optional[plt.Figure]:
    """
    AM VISUALIZATION 1: T0 (Equilibrium Solidification Temperature) Landscape Map.
    Calculates the exact equilibrium solidification temperature (T0) where Delta_G = 0
    for every composition and plots it as a continuous contour map.
    AM Insight: Shows which compositions have higher/lower melting points.
    """
    if len(meta.temperatures) < 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Need >=2 temperatures for T0", ha="center", va="center", fontsize=14)
        return fig

    mask = np.abs(df["Ni"] - fixed_Ni) < tol
    df_slice = df[(df["Temperature_K"] == meta.temperatures[0]) & mask].copy()
    if df_slice.empty:
        return None

    # Setup Ternary Grid (Co-Cr-Fe)
    x = df_slice["Cr"] + 0.5 * df_slice["Fe"]
    y = (np.sqrt(3) / 2) * df_slice["Fe"]
    xi = np.linspace(-0.05, 1.05, grid_res)
    yi = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate Delta_G for ALL temperatures
    T_vals = np.array(meta.temperatures)
    Zi_dg_stack = []

    for T in T_vals:
        df_T = df[(df["Temperature_K"] == T) & mask].copy()
        if df_T.empty:
            Zi_dg_stack.append(None)
            continue

        pts = np.column_stack([
            df_T["Cr"] + 0.5 * df_T["Fe"],
            (np.sqrt(3) / 2) * df_T["Fe"]
        ])
        Zi_dg = griddata(pts, df_T["Delta_G"].values, (Xi, Yi), method=interp_method)
        Zi_dg_stack.append(Zi_dg)

    valid_temps = [T for T, Z in zip(T_vals, Zi_dg_stack) if Z is not None]
    valid_stack = [Z for Z in Zi_dg_stack if Z is not None]

    if len(valid_stack) < 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
        return fig

    Zi_dg_stack = np.array(valid_stack)
    T_vals = np.array(valid_temps)

    # Find T0 (where Delta_G == 0) for every grid point
    T0_map = np.full((grid_res, grid_res), np.nan)
    for i in range(grid_res):
        for j in range(grid_res):
            dg_curve = Zi_dg_stack[:, i, j]
            if not np.any(np.isnan(dg_curve)) and np.any(dg_curve > 0) and np.any(dg_curve < 0):
                T0_map[i, j] = np.interp(0, dg_curve, T_vals)

    T0_map = np.ma.masked_invalid(T0_map)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    vmin, vmax = np.nanmin(T0_map), np.nanmax(T0_map)
    cf = ax.contourf(Xi, Yi, T0_map, levels=30, cmap=cmap_name, alpha=0.9, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"Equilibrium Solidification Temperature $T_0$ [K]", fontsize=13, fontweight="bold")

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=2.5, zorder=5)
    ax.text(-0.08, -0.06, "Co", fontsize=15, fontweight="bold", ha="center")
    ax.text(1.08, -0.06, "Cr", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.08, "Fe", fontsize=15, fontweight="bold", ha="center")

    ax.set_title(
        f"Continuous Solidification Temperature ($T_0$) Landscape\n"
        f"(Fixed Ni fraction ~ {fixed_Ni:.3f})",
        fontsize=15, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="❄️ Calculating Thermodynamic Freezing Range...", ttl=CONFIG.cache_ttl)
def plot_freezing_range_map(
    df: pd.DataFrame,
    meta: DatasetMeta,
    fixed_Ni: float = 0.25,
    tol: float = 0.02,
    grid_res: int = 300,
    interp_method: str = "cubic",
    cmap_name: str = "YlOrRd",
    dg_critical: float = -1500.0,
) -> Optional[plt.Figure]:
    """
    AM VISUALIZATION 2: Thermodynamic Freezing Range (Mushy Zone Width) Map.
    Calculates Delta T = T(DeltaG=0) - T(DeltaG=dg_critical).
    Wide freezing range -> high solidification cracking susceptibility.
    """
    if len(meta.temperatures) < 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Need >=2 temperatures", ha="center", va="center", fontsize=14)
        return fig

    mask = np.abs(df["Ni"] - fixed_Ni) < tol
    df_slice = df[(df["Temperature_K"] == meta.temperatures[0]) & mask].copy()
    if df_slice.empty:
        return None

    x = df_slice["Cr"] + 0.5 * df_slice["Fe"]
    y = (np.sqrt(3) / 2) * df_slice["Fe"]
    xi = np.linspace(-0.05, 1.05, grid_res)
    yi = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    T_vals = np.array(meta.temperatures)
    Zi_dg_stack = []

    for T in T_vals:
        df_T = df[(df["Temperature_K"] == T) & mask].copy()
        if df_T.empty:
            Zi_dg_stack.append(None)
            continue
        pts = np.column_stack([df_T["Cr"] + 0.5 * df_T["Fe"], (np.sqrt(3) / 2) * df_T["Fe"]])
        Zi_dg = griddata(pts, df_T["Delta_G"].values, (Xi, Yi), method=interp_method)
        Zi_dg_stack.append(Zi_dg)

    valid_temps = [T for T, Z in zip(T_vals, Zi_dg_stack) if Z is not None]
    valid_stack = [Z for Z in Zi_dg_stack if Z is not None]

    if len(valid_stack) < 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
        return fig

    Zi_dg_stack = np.array(valid_stack)
    T_vals = np.array(valid_temps)

    freezing_range = np.full((grid_res, grid_res), np.nan)
    for i in range(grid_res):
        for j in range(grid_res):
            dg_curve = Zi_dg_stack[:, i, j]
            if not np.any(np.isnan(dg_curve)) and np.any(dg_curve > 0) and np.any(dg_curve < dg_critical):
                T0 = np.interp(0, dg_curve, T_vals)
                Tcrit = np.interp(dg_critical, dg_curve, T_vals)
                freezing_range[i, j] = T0 - Tcrit

    freezing_range = np.ma.masked_invalid(freezing_range)

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = np.nanpercentile(freezing_range, 98)
    cf = ax.contourf(Xi, Yi, freezing_range, levels=30, cmap=cmap_name, alpha=0.9, vmin=0, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"Freezing Range $\Delta T = T_0 - T_{crit}$ [K]", fontsize=13, fontweight="bold")

    # Cracking susceptibility threshold
    ax.contour(Xi, Yi, freezing_range, levels=[CONFIG.freezing_range_threshold],
               colors="red", linewidths=3, linestyles="--", zorder=6)
    ax.plot([], [], "r--", linewidth=3, label=f"Cracking Threshold ({CONFIG.freezing_range_threshold} K)")

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=2.5, zorder=5)
    ax.text(-0.08, -0.06, "Co", fontsize=15, fontweight="bold", ha="center")
    ax.text(1.08, -0.06, "Cr", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.08, "Fe", fontsize=15, fontweight="bold", ha="center")

    ax.set_title(
        f"Thermodynamic Freezing Range (Mushy Zone Width)\n"
        f"Fixed Ni ~ {fixed_Ni:.3f} | Critical dG = {dg_critical:.0f} J/mol",
        fontsize=15, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="📐 Calculating Solidification Driving Force Slope...", ttl=CONFIG.cache_ttl)
def plot_dg_slope_map(
    df: pd.DataFrame,
    meta: DatasetMeta,
    fixed_Ni: float = 0.25,
    tol: float = 0.02,
    grid_res: int = 300,
    interp_method: str = "cubic",
    cmap_name: str = "RdYlGn_r",
) -> Optional[plt.Figure]:
    """
    AM VISUALIZATION 3: Solidification Driving Force Slope (|dDeltaG/dT| at T0).
    Steep slope -> small undercooling generates massive driving force -> rapid dendritic growth.
    """
    if len(meta.temperatures) < 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Need >=3 temperatures", ha="center", va="center", fontsize=14)
        return fig

    mask = np.abs(df["Ni"] - fixed_Ni) < tol
    df_slice = df[(df["Temperature_K"] == meta.temperatures[0]) & mask].copy()
    if df_slice.empty:
        return None

    x = df_slice["Cr"] + 0.5 * df_slice["Fe"]
    y = (np.sqrt(3) / 2) * df_slice["Fe"]
    xi = np.linspace(-0.05, 1.05, grid_res)
    yi = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    T_vals = np.array(meta.temperatures)
    Zi_dg_stack = []

    for T in T_vals:
        df_T = df[(df["Temperature_K"] == T) & mask].copy()
        if df_T.empty:
            Zi_dg_stack.append(None)
            continue
        pts = np.column_stack([df_T["Cr"] + 0.5 * df_T["Fe"], (np.sqrt(3) / 2) * df_T["Fe"]])
        Zi_dg = griddata(pts, df_T["Delta_G"].values, (Xi, Yi), method=interp_method)
        Zi_dg_stack.append(Zi_dg)

    valid_temps = [T for T, Z in zip(T_vals, Zi_dg_stack) if Z is not None]
    valid_stack = [Z for Z in Zi_dg_stack if Z is not None]

    if len(valid_stack) < 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
        return fig

    Zi_dg_stack = np.array(valid_stack)
    T_vals = np.array(valid_temps)

    dg_slope = np.full((grid_res, grid_res), np.nan)
    for i in range(grid_res):
        for j in range(grid_res):
            dg_curve = Zi_dg_stack[:, i, j]
            if not np.any(np.isnan(dg_curve)) and np.any(dg_curve > 0) and np.any(dg_curve < 0):
                T0 = np.interp(0, dg_curve, T_vals)
                idx = np.argmin(np.abs(T_vals - T0))
                if idx > 0 and idx < len(T_vals) - 1:
                    slope = (dg_curve[idx+1] - dg_curve[idx-1]) / (T_vals[idx+1] - T_vals[idx-1])
                    dg_slope[i, j] = np.abs(slope)
                elif idx == 0 and len(T_vals) > 1:
                    slope = (dg_curve[1] - dg_curve[0]) / (T_vals[1] - T_vals[0])
                    dg_slope[i, j] = np.abs(slope)
                elif idx == len(T_vals) - 1 and len(T_vals) > 1:
                    slope = (dg_curve[-1] - dg_curve[-2]) / (T_vals[-1] - T_vals[-2])
                    dg_slope[i, j] = np.abs(slope)

    dg_slope = np.ma.masked_invalid(dg_slope)

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = np.nanpercentile(dg_slope, 98)
    cf = ax.contourf(Xi, Yi, dg_slope, levels=30, cmap=cmap_name, alpha=0.9, vmin=0, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$|d\Delta G/dT|$ at $T_0$ [J/(mol.K)]", fontsize=13, fontweight="bold")

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=2.5, zorder=5)
    ax.text(-0.08, -0.06, "Co", fontsize=15, fontweight="bold", ha="center")
    ax.text(1.08, -0.06, "Cr", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.08, "Fe", fontsize=15, fontweight="bold", ha="center")

    ax.set_title(
        f"Solidification Driving Force Slope at $T_0$\n"
        f"Fixed Ni ~ {fixed_Ni:.3f} | $|d\\Delta G/dT| \\approx \\Delta S_f$",
        fontsize=15, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="🌡️ Generating Multi-Temperature Phase Boundary...", ttl=CONFIG.cache_ttl)
def plot_multi_temp_phase_boundary(
    df: pd.DataFrame,
    meta: DatasetMeta,
    fixed_Ni: float = 0.25,
    tol: float = 0.02,
    grid_res: int = 300,
    interp_method: str = "cubic",
    temp_levels: Optional[List[int]] = None,
) -> Optional[plt.Figure]:
    """
    AM VISUALIZATION 4: Multi-Temperature Phase Boundary Envelope.
    Overlays DeltaG=0 contours for multiple temperatures on one ternary map.
    """
    if temp_levels is None:
        n_temps = len(meta.temperatures)
        if n_temps >= 5:
            indices = np.linspace(0, n_temps-1, 5, dtype=int)
            temp_levels = [meta.temperatures[i] for i in indices]
        else:
            temp_levels = meta.temperatures

    mask = np.abs(df["Ni"] - fixed_Ni) < tol
    df_slice = df[(df["Temperature_K"] == meta.temperatures[0]) & mask].copy()
    if df_slice.empty:
        return None

    xi = np.linspace(-0.05, 1.05, grid_res)
    yi = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    fig, ax = plt.subplots(figsize=(10, 9))
    temp_cmap = plt.cm.get_cmap("plasma", len(temp_levels))

    for idx, T in enumerate(temp_levels):
        df_T = df[(df["Temperature_K"] == T) & mask].copy()
        if df_T.empty:
            continue
        pts = np.column_stack([df_T["Cr"] + 0.5 * df_T["Fe"], (np.sqrt(3) / 2) * df_T["Fe"]])
        Zi_dg = griddata(pts, df_T["Delta_G"].values, (Xi, Yi), method=interp_method)
        Zi_dg = np.ma.masked_invalid(Zi_dg)
        color = temp_cmap(idx / max(len(temp_levels)-1, 1))
        cs = ax.contour(Xi, Yi, Zi_dg, levels=[0], colors=[color], linewidths=2.5, zorder=6)
        ax.clabel(cs, inline=True, fontsize=10, fmt=f'{T} K', colors=[color])

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=2.5, zorder=5)
    ax.text(-0.08, -0.06, "Co", fontsize=15, fontweight="bold", ha="center")
    ax.text(1.08, -0.06, "Cr", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.08, "Fe", fontsize=15, fontweight="bold", ha="center")

    sm = plt.cm.ScalarMappable(cmap=temp_cmap, norm=Normalize(vmin=min(temp_levels), vmax=max(temp_levels)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Temperature [K]", fontsize=13, fontweight="bold")

    ax.set_title(
        f"Multi-Temperature Phase Boundary Envelope ($\\Delta G = 0$)\n"
        f"Fixed Ni ~ {fixed_Ni:.3f} | {len(temp_levels)} Isotherms",
        fontsize=15, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="🧊 Generating 3D Interactive Property Space...", ttl=CONFIG.cache_ttl)
def plot_3d_property_space(
    df: pd.DataFrame,
    sample_frac: float = 0.1,
) -> Optional[Any]:
    """
    AM VISUALIZATION 5: 3D Interactive Property Space (Plotly).
    X: G_LIQ, Y: G_FCC, Z: Temperature, Color: Delta_G.
    """
    if not PLOTLY_AVAILABLE:
        return None

    df_sample = df.sample(frac=min(sample_frac, 1.0), random_state=42) if len(df) > 5000 else df

    fig = go.Figure(data=[go.Scatter3d(
        x=df_sample["G_LIQ"],
        y=df_sample["G_FCC"],
        z=df_sample["Temperature_K"],
        mode='markers',
        marker=dict(
            size=3,
            color=df_sample["Delta_G"],
            colorscale='RdBu_r',
            colorbar=dict(title="Delta G [J/mol]"),
            opacity=0.7,
            showscale=True,
        ),
        text=[f"Co:{r.Co:.3f} Cr:{r.Cr:.3f}<br>Fe:{r.Fe:.3f} Ni:{r.Ni:.3f}<br>T:{r.Temperature_K}K<br>dG:{r.Delta_G:.1f}"
              for r in df_sample.itertuples()],
        hovertemplate='%{text}<extra></extra>',
    )])

    # Trade-off Edge plane (G_FCC = G_LIQ)
    g_min = min(df_sample["G_LIQ"].min(), df_sample["G_FCC"].min())
    g_max = max(df_sample["G_LIQ"].max(), df_sample["G_FCC"].max())
    t_min, t_max = df_sample["Temperature_K"].min(), df_sample["Temperature_K"].max()
    g_range = np.linspace(g_min, g_max, 20)
    t_range = np.linspace(t_min, t_max, 20)
    G_mesh, T_mesh = np.meshgrid(g_range, t_range)

    fig.add_trace(go.Surface(
        x=G_mesh,
        y=G_mesh,
        z=T_mesh,
        opacity=0.15,
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        name="Trade-off Edge (G_FCC = G_LIQ)",
        hoverinfo='skip',
    ))

    fig.update_layout(
        title="3D Thermodynamic Property Space (AM Edition)",
        scene=dict(
            xaxis_title="G_LIQ [J/mol]",
            yaxis_title="G_FCC [J/mol]",
            zaxis_title="Temperature [K]",
            aspectmode='cube',
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


# =============================================================================
# Streamlit UI
# =============================================================================

def get_all_colormaps() -> List[str]:
    """Fetch all available matplotlib colormaps dynamically."""
    return sorted(list(plt.colormaps()))


def render_sidebar(df: pd.DataFrame, meta: DatasetMeta) -> Dict[str, Any]:
    """Render sidebar controls and return user selections."""
    st.sidebar.header("Configuration")

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
        st.sidebar.error(f"Not found: `{job_dir}`")
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
    st.sidebar.subheader("Ternary Map Settings")
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

    # === NEW: Full colormap library with search ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")

    interp_method = st.sidebar.selectbox(
        "Interpolation Method",
        ["cubic", "linear"],
        index=0,
        help="Cubic gives smoother surfaces, linear is faster for large data.",
    )

    all_cmaps = get_all_colormaps()
    cmap_search = st.sidebar.text_input("Search Colormap (e.g., 'turbo', 'jet')", "viridis", key="cmap_search")
    filtered_cmaps = [c for c in all_cmaps if cmap_search.lower() in c.lower()]

    cmap_choice = st.sidebar.selectbox(
        "Select Colormap",
        options=filtered_cmaps if filtered_cmaps else all_cmaps,
        index=0,
        help=f"Choose from {len(all_cmaps)} matplotlib colormaps. Scientific favorites: viridis, inferno, turbo, RdBu_r, coolwarm",
    )
    st.sidebar.caption(f"{len(all_cmaps)} total colormaps available")

    # AM-specific settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("AM Analysis Settings")
    dg_critical = st.sidebar.number_input(
        "dG Critical Threshold [J/mol]",
        value=-1500.0,
        step=100.0,
        help="Driving force threshold for freezing range calculation.",
    )
    freezing_threshold = st.sidebar.number_input(
        "Cracking Risk Threshold [K]",
        value=50.0,
        step=5.0,
        help="Freezing range above which solidification cracking risk is high.",
    )

    # Target compositions
    st.sidebar.markdown("---")
    st.sidebar.subheader("Target Compositions")
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
        "dg_critical": dg_critical,
        "freezing_threshold": freezing_threshold,
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
    st.markdown('<div class="main-header">Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Thermodynamic Driving Force Visualization for Phase-Field Modeling of High-Entropy Alloys <span class="am-badge">AM EDITION v2.0</span></div>', unsafe_allow_html=True)

    # Initial directory for loading
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    auto_dir = script_dir / "thermodynamic_dataset"
    init_dir = str(auto_dir if auto_dir.is_dir() else Path.cwd())

    # Load data
    df, meta = load_gibbs_data(init_dir)

    if df.empty:
        st.error("No data loaded. Please configure the job directory in the sidebar.")
        st.stop()

    # Sidebar + controls
    controls = render_sidebar(df, meta)
    temps = meta.temperatures

    st.sidebar.success(f"Loaded {meta.n_files} temperature files")
    st.sidebar.info(f"Range: {min(temps)} K - {max(temps)} K")
    st.sidebar.info(f"Total points: {meta.n_points:,}")

    # Metrics
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature Files", meta.n_files)
    c2.metric("Total Data Points", f"{meta.n_points:,}")
    c3.metric("Temperature Range", f"{min(temps)}-{max(temps)} K")
    c4.metric("Composition Elements", 4)
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Fig A: PCA Landscape",
        "Fig B: Gibbs vs T",
        "Fig C: Pairplot",
        "Fig D: Evolution",
        "Fig E: Property Landscape",
        "Fig F: AM Analysis",
        "Data Explorer",
    ])

    with tab1:
        st.subheader("Thermodynamic Driving Force Landscape (PCA-Reduced)")
        st.markdown(r"""
        This figure projects the 4D composition space onto 2 principal components
        and colors each point by the solidification driving force Delta G.
        **Blue**: FCC favored (Delta G < 0) | **Red**: Liquid favored (Delta G > 0)
        """)
        with st.spinner("Generating PCA landscape..."):
            fig = plot_driving_force_landscape_pca(
                df, controls["T_ref"], CONFIG.grid_resolution,
                controls["interp_method"], controls["cmap_choice"]
            )
            st.pyplot(fig)
            png = download_figure(fig, "pca_landscape.png")
            st.download_button("Download PNG", png, "pca_landscape.png", "image/png")
            plt.close(fig)

    with tab2:
        st.subheader("Scaled Gibbs Energy of End-Member Phases vs Temperature")
        st.markdown(r"""
        Gibbs free energy curves for LIQUID (dashed) and FCC (solid) phases.
        **Crossover points** indicate local equilibrium temperatures T_eq.
        """)
        fig = plot_gibbs_vs_temperature(df, controls["target_comps"])
        st.pyplot(fig)
        png = download_figure(fig, "gibbs_vs_T.png")
        st.download_button("Download PNG", png, "gibbs_vs_T.png", "image/png")
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
            st.download_button("Download PNG", png, "pairplot.png", "image/png")
            plt.close(fig)

    with tab4:
        st.subheader("Temperature Evolution of Solidification Driving Force")
        st.markdown(r"""
        Distribution of Delta G across all compositions as temperature changes.
        Shaded regions show percentiles; lines show mean and median.
        """)
        fig = plot_driving_force_evolution(df)
        st.pyplot(fig)
        png = download_figure(fig, "evolution.png")
        st.download_button("Download PNG", png, "evolution.png", "image/png")
        plt.close(fig)

    with tab5:
        st.subheader("Continuous Gibbs Property-Space Landscape")
        st.markdown(r"""
        **The key upgrade**: Instead of scattered points, this uses `scipy.interpolate.griddata`
        to create a **continuous landscape** of Ni mole fraction mapped over G_LIQ vs G_FCC.

        - The **red diagonal** is the *Trade-off Edge* (Delta G = 0): below it, FCC is stable; above it, Liquid is stable.
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
            st.download_button("Download PNG", png, "property_landscape.png", "image/png")
            plt.close(fig)

        st.markdown("---")
        st.subheader("Continuous Ternary Driving-Force Map")
        st.markdown(r"""
        A **barycentric-projected ternary map** at fixed Ni content, showing the
        continuous driving force landscape in the Co-Cr-Fe subspace.
        The **black contour** marks the Delta G = 0 phase boundary.
        """)
        with st.spinner("Generating ternary map..."):
            fig = plot_ternary_driving_force_map(
                df, controls["T_ref"], controls["fixed_ni"], controls["ni_tol"],
                CONFIG.grid_resolution, controls["interp_method"], controls["cmap_choice"]
            )
            if fig:
                st.pyplot(fig)
                png = download_figure(fig, "ternary_map.png")
                st.download_button("Download PNG", png, "ternary_map.png", "image/png")
                plt.close(fig)

    # =============================================================================
    # NEW: AM Analysis Tab
    # =============================================================================
    with tab6:
        st.subheader("Additive Manufacturing (AM) Specific Analysis")
        st.markdown("""
        <div class="info-box">
        These visualizations extract <strong>Additive Manufacturing-relevant metrics</strong> purely from 
        the temperature dependence of the driving force (Delta G). AM involves extreme cooling rates 
        (10^3-10^8 K/s), massive thermal gradients, and cyclic reheating. These maps help predict 
        solidification cracking, grain structure, and processability.
        </div>
        """, unsafe_allow_html=True)

        # --- T0 Landscape ---
        st.markdown("---")
        st.subheader("T0: Equilibrium Solidification Temperature Landscape")
        st.markdown("""
        <div class="info-box">
        <strong>Physics:</strong> T0 is the temperature where Delta G = 0 (G_FCC = G_LIQ) for each composition.
        This map shows the <strong>melting/solidification temperature</strong> across the entire composition space.
        <br><br>
        <strong>AM Insight:</strong> Higher T0 compositions require more laser power. Compositions with 
        steep T0 gradients across the composition space will experience severe thermal gradients during AM.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Calculating T0 landscape (interpolating across all temperatures)..."):
            fig = plot_T0_landscape(
                df, meta, controls["fixed_ni"], controls["ni_tol"],
                CONFIG.grid_resolution, controls["interp_method"], controls["cmap_choice"]
            )
            if fig:
                st.pyplot(fig)
                png = download_figure(fig, "T0_landscape.png")
                st.download_button("Download PNG", png, "T0_landscape.png", "image/png")
                plt.close(fig)
            else:
                st.warning("Could not generate T0 landscape. Need at least 2 temperature files.")

        # --- Freezing Range ---
        st.markdown("---")
        st.subheader("Thermodynamic Freezing Range (Mushy Zone Width)")
        st.markdown(f"""
        <div class="warning-box">
        <strong>Physics:</strong> The freezing range is Delta T = T(DeltaG=0) - T(DeltaG={controls['dg_critical']:.0f}).
        This represents the <strong>mushy zone</strong> where the alloy is partially solid/liquid.
        <br><br>
        <strong>AM Insight:</strong> Compositions with wide freezing ranges (>50 K) have large mushy zones 
        and are <strong>highly susceptible to solidification cracking</strong> (hot tearing) during AM.
        The red dashed line marks the cracking risk threshold.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Calculating freezing range map..."):
            fig = plot_freezing_range_map(
                df, meta, controls["fixed_ni"], controls["ni_tol"],
                CONFIG.grid_resolution, controls["interp_method"], controls["cmap_choice"],
                controls["dg_critical"]
            )
            if fig:
                st.pyplot(fig)
                png = download_figure(fig, "freezing_range.png")
                st.download_button("Download PNG", png, "freezing_range.png", "image/png")
                plt.close(fig)
            else:
                st.warning("Could not generate freezing range map.")

        # --- Driving Force Slope ---
        st.markdown("---")
        st.subheader("Solidification Driving Force Slope |dDeltaG/dT| at T0")
        st.markdown("""
        <div class="info-box">
        <strong>Physics:</strong> The slope dDeltaG/dT at T0 approximates the entropy of fusion (Delta S_f).
        A steep slope means small undercooling generates massive driving force.
        <br><br>
        <strong>AM Insight:</strong> 
        <ul>
        <li><strong>Steep slope</strong> -> Rapid solidification -> Fine equiaxed grains (strong, isotropic)</li>
        <li><strong>Shallow slope</strong> -> Sluggish solidification -> Coarse columnar grains (brittle, anisotropic)</li>
        </ul>
        This helps predict the <strong>Columnar-to-Equiaxed Transition (CET)</strong>.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Calculating driving force slope map..."):
            fig = plot_dg_slope_map(
                df, meta, controls["fixed_ni"], controls["ni_tol"],
                CONFIG.grid_resolution, controls["interp_method"], controls["cmap_choice"]
            )
            if fig:
                st.pyplot(fig)
                png = download_figure(fig, "dg_slope.png")
                st.download_button("Download PNG", png, "dg_slope.png", "image/png")
                plt.close(fig)
            else:
                st.warning("Could not generate slope map. Need at least 3 temperature files.")

        # --- Multi-Temp Phase Boundary ---
        st.markdown("---")
        st.subheader("Multi-Temperature Phase Boundary Envelope")
        st.markdown("""
        <div class="info-box">
        <strong>Physics:</strong> Overlays the Delta G = 0 contour for multiple temperatures on a single map.
        Creates nested "islands" showing how the phase stability region shifts with temperature.
        <br><br>
        <strong>AM Insight:</strong> The area between isotherms represents the compositional window that 
        solidifies within that temperature interval. Shows how the FCC stability region "shrinks" or 
        "expands" as the AM part cools from the melt pool.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Generating multi-temperature phase boundary envelope..."):
            fig = plot_multi_temp_phase_boundary(
                df, meta, controls["fixed_ni"], controls["ni_tol"],
                CONFIG.grid_resolution, controls["interp_method"]
            )
            if fig:
                st.pyplot(fig)
                png = download_figure(fig, "multi_temp_boundary.png")
                st.download_button("Download PNG", png, "multi_temp_boundary.png", "image/png")
                plt.close(fig)
            else:
                st.warning("Could not generate multi-temperature boundary.")

        # --- 3D Interactive ---
        if PLOTLY_AVAILABLE:
            st.markdown("---")
            st.subheader("3D Interactive Thermodynamic Property Space")
            st.markdown("""
            <div class="info-box">
            <strong>Interactive 3D scatter:</strong> Rotate, zoom, and pan to explore the full 
            3D thermodynamic landscape. The red transparent plane is the Trade-off Edge (G_FCC = G_LIQ).
            <br><br>
            <strong>AM Insight:</strong> See how the composition cloud twists through temperature space.
            Points below the red plane are FCC-stable; above are liquid-stable.
            </div>
            """, unsafe_allow_html=True)

            sample_frac = st.slider("3D Plot Sample Fraction", 0.01, 1.0, 0.1, 0.01,
                                    help="Fraction of data points to render (lower = faster)")
            with st.spinner("Building 3D interactive plot..."):
                fig_3d = plot_3d_property_space(df, sample_frac)
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("Could not generate 3D plot.")
        else:
            st.info("Install Plotly (`pip install plotly`) to enable 3D interactive visualizations.")

    with tab7:
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
                "Download Filtered Data (CSV)", csv,
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
