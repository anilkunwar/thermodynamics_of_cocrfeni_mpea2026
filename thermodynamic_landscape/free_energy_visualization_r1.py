# app.py  –  Gibbs Free Energy Landscape Explorer (Streamlit)
# Run:  streamlit run app.py -- --job-dir /path/to/Gibbs_csvs

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────
# 0.  GLOBAL STYLE  (mirrors screening_plots.py)
# ──────────────────────────────────────────────
INK = "#0F172A"
MUTED = "#94A3B8"
TOP = "#F59E0B"

COOLWARM_CUSTOM = LinearSegmentedColormap.from_list(
    "coolwarm_custom",
    ["#2563EB", "#E0F2FE", "#FEF3C7", "#DC2626"],
    N=256,
)

PHASE_PALETTE = {"FCC": "#2563EB", "LIQUID": "#DC2626"}

ELEMENT_COLORS = {
    "Co": "#E11D48",
    "Cr": "#059669",
    "Fe": "#D97706",
    "Ni": "#7C3AED",
}

COMPS = ["Co", "Cr", "Fe", "Ni"]


def configure_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 1.05,
            "axes.labelcolor": INK,
            "axes.labelweight": "bold",
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "sans-serif",
            "font.weight": "bold",
            "legend.frameon": True,
            "legend.fontsize": 9.5,
            "savefig.bbox": "tight",
            "savefig.format": "png",
        }
    )


configure_mpl()


# ──────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────
def discover_gibbs_files(job_dir: str) -> Dict[int, Path]:
    """Find all Gibbs_<T>K.csv files and return {temperature: path}."""
    pattern = os.path.join(job_dir, "Gibbs_*K.csv")
    files = glob.glob(pattern)
    mapping: Dict[int, Path] = {}
    for f in files:
        m = re.search(r"Gibbs_(\d+)K\.csv$", os.path.basename(f))
        if m:
            mapping[int(m.group(1))] = Path(f)
    return dict(sorted(mapping.items()))


@st.cache_data(show_spinner="Loading Gibbs free energy tables …")
def load_all_temperatures(job_dir: str) -> pd.DataFrame:
    """Load every Gibbs_XXXK.csv into a single DataFrame."""
    temp_map = discover_gibbs_files(job_dir)
    if not temp_map:
        st.error(f"No Gibbs_*K.csv files found in `{job_dir}`")
        st.stop()

    frames: List[pd.DataFrame] = []
    for T, path in temp_map.items():
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")
            continue
        df["Temperature"] = T
        frames.append(df)

    if not frames:
        st.error("All files failed to load.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)

    # Normalise column names (strip whitespace)
    df_all.columns = [c.strip() for c in df_all.columns]

    # Derived quantities
    if "G_FCC" in df_all.columns and "G_LIQ" in df_all.columns:
        df_all["Delta_G"] = df_all["G_FCC"] - df_all["G_LIQ"]
        df_all["Stable_Phase"] = np.where(df_all["Delta_G"] < 0, "FCC", "LIQUID")

    return df_all


# ──────────────────────────────────────────────
# 2.  HELPER  –  composition distance
# ──────────────────────────────────────────────
def nearest_composition_rows(
    df: pd.DataFrame, target: List[float], comps: List[str]
) -> pd.DataFrame:
    """For each Temperature, return the row closest to *target* composition."""
    df = df.copy()
    df["_dist"] = np.sum((df[comps] - target) ** 2, axis=1)
    idx = df.groupby("Temperature")["_dist"].idxmin()
    return df.loc[idx].sort_values("Temperature")


# ──────────────────────────────────────────────
# 3.  PLOTTING FUNCTIONS
# ──────────────────────────────────────────────
def fig_driving_force_landscape(
    df: pd.DataFrame, T_ref: int, comps: List[str]
) -> mpl.figure.Figure:
    """Figure A – PCA-reduced driving-force landscape at a single T."""
    sub = df[df["Temperature"] == T_ref].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"No data at {T_ref} K", ha="center", va="center")
        return fig

    pca = PCA(n_components=2)
    coords = pca.fit_transform(sub[comps].values)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=sub["Delta_G"].values,
        cmap=COOLWARM_CUSTOM,
        alpha=0.72,
        s=18,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(
        r"Driving Force  $\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$  (J/mol)",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xlabel(
        f"PC-1  ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12
    )
    ax.set_ylabel(
        f"PC-2  ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12
    )
    ax.set_title(
        f"Thermodynamic Driving-Force Landscape at {T_ref} K\n"
        "(PCA-reduced Co-Cr-Fe-Ni composition space)",
        fontsize=13,
    )
    return fig


def fig_gibbs_vs_temperature(
    df: pd.DataFrame, presets: Dict[str, List[float]], comps: List[str]
) -> mpl.figure.Figure:
    """Figure B – G_LIQ and G_FCC curves for selected compositions."""
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    for name, target in presets.items():
        rows = nearest_composition_rows(df, target, comps)
        color = ELEMENT_COLORS.get(name.split("-")[0], INK)
        ax.plot(
            rows["Temperature"],
            rows["G_LIQ"],
            "--",
            linewidth=2,
            color=color,
            label=f"{name}  $G_{{\\mathrm{{LIQ}}}}$",
        )
        ax.plot(
            rows["Temperature"],
            rows["G_FCC"],
            "-",
            linewidth=2,
            color=color,
            label=f"{name}  $G_{{\\mathrm{{FCC}}}}$",
        )

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Gibbs Free Energy (J/mol)", fontsize=12)
    ax.set_title(
        "Scaled Gibbs Energy of End-Member Phases vs Temperature\n"
        "(Crossover → local equilibrium $T_{\\mathrm{eq}}$)",
        fontsize=13,
    )
    ax.legend(ncol=2, fontsize=8.5, loc="best")
    ax.grid(True, linestyle="--", alpha=0.45)
    return fig


def fig_phase_stability_pairplot(
    df: pd.DataFrame, T_ref: int, comps: List[str], max_points: int = 3000
) -> mpl.figure.Figure:
    """Figure C – Phase-stability pair-plot at a single T."""
    sub = df[df["Temperature"] == T_ref].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"No data at {T_ref} K", ha="center", va="center")
        return fig

    if len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=42)

    n = len(comps)
    fig, axes = plt.subplots(n, n, figsize=(9, 9), constrained_layout=True)
    fig.suptitle(
        f"Phase-Stability Landscape at {T_ref} K  (FCC vs LIQUID)",
        fontsize=14,
        fontweight="bold",
    )

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal – histogram
                for phase, color in PHASE_PALETTE.items():
                    vals = sub.loc[sub["Stable_Phase"] == phase, comps[i]]
                    ax.hist(vals, bins=40, color=color, alpha=0.55, density=True)
            else:
                # Off-diagonal – scatter
                for phase, color in PHASE_PALETTE.items():
                    mask = sub["Stable_Phase"] == phase
                    ax.scatter(
                        sub.loc[mask, comps[j]],
                        sub.loc[mask, comps[i]],
                        s=8,
                        alpha=0.5,
                        color=color,
                        edgecolors="none",
                    )
            if i == n - 1:
                ax.set_xlabel(comps[j], fontweight="bold", fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(comps[i], fontweight="bold", fontsize=10)
            else:
                ax.set_yticklabels([])

    # Legend
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=p)
        for p, c in PHASE_PALETTE.items()
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, frameon=True)
    return fig


def fig_delta_g_evolution(df: pd.DataFrame) -> mpl.figure.Figure:
    """Figure D – Global driving-force statistics vs Temperature."""
    stats = (
        df.groupby("Temperature")["Delta_G"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.fill_between(
        stats["Temperature"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color="#94A3B8",
        alpha=0.30,
        label=r"Global $\mu \pm \sigma$",
    )
    ax.fill_between(
        stats["Temperature"],
        stats["min"],
        stats["max"],
        color="#CBD5E1",
        alpha=0.18,
        label="Full range",
    )
    ax.plot(stats["Temperature"], stats["mean"], "k-", linewidth=2.2, label="Mean")
    ax.plot(
        stats["Temperature"],
        stats["median"],
        color="#2563EB",
        linestyle="--",
        linewidth=2,
        label="Median",
    )
    ax.axhline(0, color="#DC2626", linestyle=":", linewidth=2, label=r"$\Delta G=0$")

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel(
        r"Driving Force  $\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$  (J/mol)",
        fontsize=12,
    )
    ax.set_title(
        "Temperature Evolution of Solidification Driving Force\n"
        "(Across entire Co-Cr-Fe-Ni composition space)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    return fig


# ──────────────────────────────────────────────
# 4.  STREAMLIT UI
# ──────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Gibbs Landscape Explorer", layout="wide")
    st.title("🔬 Gibbs Free Energy Landscape Explorer")
    st.caption(
        "Co-Cr-Fe-Ni HEA · Liquid ↔ FCC · Phase-field end-member visualisation"
    )

    # ── Sidebar: job directory ──
    st.sidebar.header("⚙️ Configuration")

    # Determine default directory: CLI arg > env > thermodynamic_dataset > current dir
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job-dir", default="")
    args, _ = parser.parse_known_args()

    default_dir = args.job_dir or os.environ.get("JOB_DIR", "")
    if not default_dir:
        # If thermodynamic_dataset exists next to the script, use it
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_dir = os.path.join(script_dir, "thermodynamic_dataset")
        if os.path.isdir(possible_dir):
            default_dir = possible_dir
        else:
            default_dir = "."

    job_dir = st.sidebar.text_input(
        "Job directory (containing Gibbs_*K.csv)", value=default_dir
    )

    # Normalize path
    job_dir = os.path.abspath(job_dir)

    if not os.path.isdir(job_dir):
        st.warning(f"Directory `{job_dir}` does not exist yet.")
        st.stop()

    # ── Load data ──
    df = load_all_temperatures(job_dir)
    available_T = sorted(df["Temperature"].unique())

    st.sidebar.success(f"Loaded **{len(df):,}** rows across **{len(available_T)}** temperatures  "
                       f"({available_T[0]}–{available_T[-1]} K)")

    # ── Quick data preview ──
    with st.expander("📋 Raw data preview", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)
        st.write(f"Columns: `{list(df.columns)}`")

    # ── Tabs for the four figures ──
    tab_a, tab_b, tab_c, tab_d = st.tabs(
        [
            "A · Driving-Force Landscape",
            "B · Gibbs Energy vs T",
            "C · Phase-Stability Pairplot",
            "D · ΔG Evolution",
        ]
    )

    # ────────── TAB A ──────────
    with tab_a:
        st.subheader("Figure A – PCA-Reduced Driving-Force Landscape")
        T_ref_a = st.slider(
            "Reference temperature (K)",
            min_value=int(available_T[0]),
            max_value=int(available_T[-1]),
            value=int(np.clip(1800, available_T[0], available_T[-1])),
            step=100,
            key="slider_a",
        )
        fig_a = fig_driving_force_landscape(df, T_ref_a, COMPS)
        st.pyplot(fig_a)
        plt.close(fig_a)

    # ────────── TAB B ──────────
    with tab_b:
        st.subheader("Figure B – Scaled Gibbs Energy vs Temperature")

        col1, col2, col3, col4 = st.columns(4)
        presets = {}
        with col1:
            st.markdown("**Equiatomic**")
            eq_co = st.number_input("Co", 0.0, 1.0, 0.25, 0.05, key="eq_co")
            eq_cr = st.number_input("Cr", 0.0, 1.0, 0.25, 0.05, key="eq_cr")
            eq_fe = st.number_input("Fe", 0.0, 1.0, 0.25, 0.05, key="eq_fe")
            eq_ni = st.number_input("Ni", 0.0, 1.0, 0.25, 0.05, key="eq_ni")
            presets["Equiatomic"] = [eq_co, eq_cr, eq_fe, eq_ni]

        with col2:
            st.markdown("**Co-rich**")
            co_co = st.number_input("Co", 0.0, 1.0, 0.70, 0.05, key="co_co")
            co_cr = st.number_input("Cr", 0.0, 1.0, 0.10, 0.05, key="co_cr")
            co_fe = st.number_input("Fe", 0.0, 1.0, 0.10, 0.05, key="co_fe")
            co_ni = st.number_input("Ni", 0.0, 1.0, 0.10, 0.05, key="co_ni")
            presets["Co-rich"] = [co_co, co_cr, co_fe, co_ni]

        with col3:
            st.markdown("**Ni-rich**")
            ni_co = st.number_input("Co", 0.0, 1.0, 0.10, 0.05, key="ni_co")
            ni_cr = st.number_input("Cr", 0.0, 1.0, 0.10, 0.05, key="ni_cr")
            ni_fe = st.number_input("Fe", 0.0, 1.0, 0.10, 0.05, key="ni_fe")
            ni_ni = st.number_input("Ni", 0.0, 1.0, 0.70, 0.05, key="ni_ni")
            presets["Ni-rich"] = [ni_co, ni_cr, ni_fe, ni_ni]

        with col4:
            st.markdown("**Cr-rich**")
            cr_co = st.number_input("Co", 0.0, 1.0, 0.10, 0.05, key="cr_co2")
            cr_cr = st.number_input("Cr", 0.0, 1.0, 0.70, 0.05, key="cr_cr")
            cr_fe = st.number_input("Fe", 0.0, 1.0, 0.10, 0.05, key="cr_fe")
            cr_ni = st.number_input("Ni", 0.0, 1.0, 0.10, 0.05, key="cr_ni")
            presets["Cr-rich"] = [cr_co, cr_cr, cr_fe, cr_ni]

        fig_b = fig_gibbs_vs_temperature(df, presets, COMPS)
        st.pyplot(fig_b)
        plt.close(fig_b)

    # ────────── TAB C ──────────
    with tab_c:
        st.subheader("Figure C – Phase-Stability Composition Pairplot")
        T_ref_c = st.slider(
            "Reference temperature (K)",
            min_value=int(available_T[0]),
            max_value=int(available_T[-1]),
            value=int(np.clip(2000, available_T[0], available_T[-1])),
            step=100,
            key="slider_c",
        )
        max_pts = st.slider("Max points to render", 500, 10000, 3000, 500, key="pts_c")
        fig_c = fig_phase_stability_pairplot(df, T_ref_c, COMPS, max_points=max_pts)
        st.pyplot(fig_c)
        plt.close(fig_c)

    # ────────── TAB D ──────────
    with tab_d:
        st.subheader("Figure D – Temperature Evolution of ΔG Distribution")
        fig_d = fig_delta_g_evolution(df)
        st.pyplot(fig_d)
        plt.close(fig_d)

    # ── Download ──
    st.divider()
    st.markdown("### 💾 Export")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download merged CSV", csv_bytes, "gibbs_merged.csv", "text/csv"
        )
    with col_dl2:
        st.info(
            "Use the ⋮ menu on each figure to download the PNG directly from Streamlit."
        )


if __name__ == "__main__":
    main()
