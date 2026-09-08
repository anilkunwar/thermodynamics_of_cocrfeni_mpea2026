import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from io import BytesIO
import base64
import shutil

# Optional cubic-spline smoothing
try:
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Real LaTeX almost never exists on Streamlit Cloud -> detect it
LATEX_AVAILABLE = shutil.which("latex") is not None

# =====================================================================
# RESEARCH QUESTION — Q1AL2
# =====================================================================
RESEARCH_QUESTION = (
    "Q1AL2 — Role of the phase-conditioned composition tensor in driving KKS "
    "phase equilibrium constraints and elemental partitioning within the "
    "CoCrFeNi multicomponent diffusion process"
)

# =====================================================================
# QDWA QUANTIFICATION FRAMEWORK
#     W_k = (alpha + raw_k) / (6*alpha + sum_j raw_j),   alpha = 0.25
# The 6-domain schema is fixed across questions; only the raw evidence
# scores are re-quantified per research question.
# =====================================================================
domain_categories_full = [
    "Thermodynamic state space & phase stability",
    "Multicomponent alloy chemistry & composition",
    "Laser processing parameters & thermal cycles",
    "Melt pool hydrodynamics & transport phenomena",
    "Phase-field kinetics & microstructural evolution",
    "Physics-informed AI surrogate & digital twin",
]
N_DOMAINS = len(domain_categories_full)

# --- CHANGED: raw evidence re-scored for Q1AL2 -----------------------
# D1  8.5   KKS mu_i^alpha = mu_i^beta constraints; CALPHAD Co-Cr-Fe-Ni manifold
# D2  9.5   phase-conditioned composition tensor; CoCrFeNi elemental partitioning
# D3  1.5   laser thermal history only -> peripheral
# D4  4.0   multicomponent diffusion / Onsager transport (no melt-pool convection)
# D5  9.0   KKS embedded in phase-field; tensor in the free-energy functional
# D6  1.0   non-mechanistic surrogate layer
PRESETS = {
    "Q1AL2 — KKS / partitioning focus (default)": [8.5, 9.5, 1.5, 4.0, 9.0, 1.0],
    "Q1TD1 — Gibbs tensor spectral (original)": [9.0, 6.0, 4.5, 3.0, 2.0, 1.5],
}

EVIDENCE_RATIONALE = {
    "Thermodynamic state space & phase stability": (
        r"KKS phase equilibrium is an equality-of-chemical-potential condition, "
        r"$\mu_i^{\alpha}=\mu_i^{\beta}$ for $i\in\{\mathrm{Co,Cr,Fe,Ni}\}$, anchored "
        r"in CALPHAD Gibbs energies; phase stability (FCC vs $\sigma$/B2) fixes the "
        r"endpoints the phase-conditioned tensor must reach. Strong, but one step "
        r"removed from the tensor itself."
    ),
    "Multicomponent alloy chemistry & composition": (
        r"The phase-conditioned composition tensor $c^p_{ij}$ lives directly in the "
        r"quaternary CoCrFeNi composition space, and elemental partitioning "
        r"(Co/Cr/Fe/Ni redistribution between phases) is the primary observable of "
        r"the KKS constraints — highest relevance."
    ),
    "Laser processing parameters & thermal cycles": (
        r"Laser parameters supply thermal boundary conditions and thermal history "
        r"only; they neither define the KKS equilibrium constraints nor intrinsic "
        r"partitioning behaviour — peripheral to Q1AL2."
    ),
    "Melt pool hydrodynamics & transport phenomena": (
        r"The multicomponent diffusion process (interdiffusion fluxes, Onsager / "
        r"diffusion matrix $D_{ik}$) carries the composition tensor toward KKS "
        r"equilibrium, but melt-pool convection is irrelevant in the solid-state "
        r"diffusion regime — moderate relevance."
    ),
    "Phase-field kinetics & microstructural evolution": (
        r"The KKS model is embedded in the phase-field formulation: the composition "
        r"tensor enters the free-energy functional, and its relaxation drives "
        r"interface motion and partitioning kinetics — core mechanism."
    ),
    "Physics-informed AI surrogate & digital twin": (
        r"Surrogate / digital-twin layers only emulate or post-process the physics; "
        r"no mechanistic role in enforcing phase-equilibrium constraints or "
        r"elemental partitioning — lowest relevance."
    ),
}

# ------------------- PLOT FUNCTION -------------------
def plot_dual_axis(df, cfg):
    mpl.rcParams.update({
        "font.family": cfg["font_family"],
        "font.size": cfg["font_size"],
        "mathtext.fontset": cfg["mathtext_fontset"],
        "text.usetex": cfg["use_usetex"],
        "axes.labelweight": cfg["label_weight"],
    })
    if cfg["use_usetex"]:
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}"
        )

    fig, ax1 = plt.subplots(figsize=(cfg["fig_width"], cfg["fig_height"]))
    xs = np.arange(len(df))

    full_names = df["Category"].tolist()
    symbolic_names = [f"D{i+1}" for i in range(len(full_names))]

    # ---------- Bars ----------
    ax1.bar(xs, df["raw_k"], width=cfg["bar_width"], color=cfg["bar_color"],
            alpha=cfg["bar_alpha"], edgecolor=cfg["bar_edge_color"],
            linewidth=cfg["bar_edge_width"], label="Raw Evidence (left)")
    ax1.set_xlabel("Domain", fontsize=cfg["font_size"])
    ax1.set_ylabel(cfg["ylabel_left"], fontsize=cfg["font_size"],
                   color=cfg["bar_color"], fontweight=cfg["label_weight"])

    ax1.set_xticks(xs)
    ax1.set_xticklabels(symbolic_names, fontsize=cfg["font_size"],
                        rotation=cfg["x_rotation"])
    if cfg["x_rotation"] != 0:
        plt.setp(ax1.get_xticklabels(), ha="right", rotation_mode="anchor")

    # ---------- Line ----------
    ax2 = ax1.twinx()
    if cfg["smooth_line"] and SCIPY_AVAILABLE and len(xs) > 3:
        x_s = np.linspace(xs.min(), xs.max(), 300)
        w_s = make_interp_spline(xs, df["w_k"].to_numpy(), k=3)(x_s)
        ax2.plot(x_s, w_s, color=cfg["line_color"],
                 linewidth=cfg["line_width"],
                 linestyle=cfg["line_style"], label="Smoothed Weight (right)")
        ax2.plot(xs, df["w_k"], linestyle="none", marker=cfg["marker_style"],
                 markersize=cfg["marker_size"], color=cfg["line_color"],
                 markeredgewidth=cfg["marker_edge_width"],
                 markeredgecolor=cfg["marker_edge_color"])
    else:
        ax2.plot(xs, df["w_k"], color=cfg["line_color"],
                 linewidth=cfg["line_width"],
                 linestyle=cfg["line_style"], marker=cfg["marker_style"],
                 markersize=cfg["marker_size"],
                 markeredgewidth=cfg["marker_edge_width"],
                 markeredgecolor=cfg["marker_edge_color"],
                 label="Smoothed Weight (right)")

    ax2.set_ylabel(cfg["ylabel_right"], fontsize=cfg["font_size"],
                   color=cfg["line_color"], fontweight=cfg["label_weight"])

    # ---------- Ticks ----------
    tkw = dict(length=cfg["tick_length"], width=cfg["tick_width"],
               direction=cfg["tick_direction"], pad=cfg["tick_pad"])
    ax1.tick_params(axis="y", labelcolor=cfg["bar_color"],
                    labelsize=cfg["font_size"], **tkw)
    ax1.tick_params(axis="x", **tkw)
    ax2.tick_params(axis="y", labelcolor=cfg["line_color"],
                    labelsize=cfg["font_size"], **tkw)

    if cfg["minor_ticks"]:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        for ax in (ax1, ax2):
            ax.tick_params(which="minor", length=cfg["tick_length"] * 0.5,
                           width=cfg["tick_width"] * 0.75,
                           direction=cfg["tick_direction"])

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(cfg["spine_width"])
        ax.spines["top"].set_visible(cfg["top_spines"])
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.spines["left"].set_color(cfg["bar_color"])
    ax2.spines["right"].set_color(cfg["line_color"])

    ax1.grid(cfg["show_grid"], axis="y", linestyle=cfg["grid_style"],
             linewidth=cfg["grid_width"], alpha=cfg["grid_alpha"])
    ax2.grid(False)
    ax1.set_axisbelow(True)

    version = "Rounded" if cfg["use_rounded"] else "Exact"
    # CHANGED: chart title now references Q1AL2
    ax1.set_title(f"Q1AL2 QDWA — Dual-Axis Chart ({version} Data)",
                  fontsize=cfg["font_size"] + 2, pad=12,
                  fontweight="bold" if cfg["bold_title"] else "normal")

    # ---------- Legend Logic (Matplotlib) ----------
    if cfg["legend_mode"] == "inside":
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        domain_key_handles = [plt.Line2D([0], [0], color='gray', lw=1.5) for _ in full_names]
        domain_key_labels = [f"{s_name} = {f_name}" for s_name, f_name in zip(symbolic_names, full_names)]

        handles = h1 + h2 + domain_key_handles
        labels = l1 + l2 + domain_key_labels

        ax1.legend(handles, labels, loc=cfg["legend_loc"],
                   frameon=cfg["legend_frame"], fontsize=cfg["legend_fontsize"],
                   framealpha=0.9, edgecolor="black")

    fig.tight_layout()
    return fig

# ------------------- HTML LEGEND GENERATOR -------------------
def render_html_legend(cfg, df):
    st.markdown("---")
    st.markdown("### Chart Legend")

    marker_map = {
        "o": "&#9679;", "s": "&#9632;", "^": "&#9650;", "D": "&#9670;",
        "v": "&#9660;", "P": "&#9673;", "*": "&#10042;", "X": "&#10006;",
    }
    marker_sym = marker_map.get(cfg["marker_style"], "&#9679;")

    bar_color = cfg["bar_color"]
    line_color = cfg["line_color"]

    full_names = df["Category"].tolist()
    symbolic_names = [f"D{i+1}" for i in range(len(full_names))]

    st.markdown("**Data Series:**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            f'<span style="color:{bar_color}; font-size:1.5em; font-weight:bold;">&#9632;</span> '
            f'<span style="font-size:1.1em;">Raw Evidence (left)</span>',
            unsafe_allow_html=True
        )
    with col_s2:
        st.markdown(
            f'<span style="color:{line_color}; font-size:1.5em; font-weight:bold;">{marker_sym}</span> '
            f'<span style="font-size:1.1em;">Domain Weight (right)</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Domain Key:**")
    col_d1, col_d2 = st.columns(2)
    for i, (s_name, f_name) in enumerate(zip(symbolic_names, full_names)):
        with col_d1 if i % 2 == 0 else col_d2:
            st.markdown(
                f'<span style="color:gray; font-weight:bold;">{s_name}</span> = '
                f'<span style="color:black;">{f_name}</span>',
                unsafe_allow_html=True
            )

# ------------------- DOWNLOAD FUNCTION -------------------
MIME_TYPES = {"png": "image/png", "pdf": "application/pdf",
              "svg": "image/svg+xml", "eps": "application/postscript",
              "tiff": "image/tiff"}

def get_download_link(fig, dpi, fmt, transparent):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                transparent=transparent)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    # CHANGED: export filename for Q1AL2
    href = (f'<a href="data:{MIME_TYPES[fmt]};base64,{b64}" '
            f'download="q1al2_qdwa_chart.{fmt}">Download {fmt.upper()}</a>')
    return href

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Q1AL2: QDWA Domain Weight Quantification", layout="wide")

st.title("📊 Q1AL2 — KKS Phase Equilibrium & CoCrFeNi Partitioning")
st.markdown(f"> **Research question:** {RESEARCH_QUESTION}")
st.markdown(
    "**Quantitative Domain Weight Analysis (QDWA)** — domain evidence re-scored for the "
    "phase-conditioned composition tensor / KKS question and normalized with a uniform prior."
)

st.sidebar.header("QDWA & Chart Customization")

# --- NEW: QDWA scoring panel (replaces the hard-coded raw_data) -------
with st.sidebar.expander("🔢 QDWA Evidence Scoring (raw_k)", expanded=True):
    preset_label = st.selectbox("Scoring preset", list(PRESETS.keys()))
    preset_vals = PRESETS[preset_label]

    # Re-initialize the per-domain inputs whenever the preset changes
    if st.session_state.get("_qdwa_preset") != preset_label:
        st.session_state["_qdwa_preset"] = preset_label
        for i in range(N_DOMAINS):
            st.session_state.pop(f"qdwa_raw_{i}", None)
        st.session_state.pop("qdwa_custom", None)

    alpha = st.number_input(
        "Prior α (Laplace smoothing)", 0.0, 1.0, 0.25, 0.05,
        key="qdwa_alpha",
        help="Added to every raw_k in the numerator (α = 0.25 in the reference table).")

    raw_data = []
    for i, (cat, val) in enumerate(zip(domain_categories_full, preset_vals)):
        r = st.number_input(f"D{i+1} · raw_k", 0.0, 20.0, float(val),
                            0.25, key=f"qdwa_raw_{i}", help=cat)
        raw_data.append(float(r))

    custom_vec = st.text_input(
        "Custom raw_k vector (overrides the inputs above)",
        key="qdwa_custom",
        placeholder="e.g. 8.5, 9.5, 1.5, 4.0, 9.0, 1.0",
        help="Comma-separated, exactly 6 values — use this to reproduce an "
             "external scoring table exactly.")
    if custom_vec.strip():
        try:
            parsed = [float(x) for x in custom_vec.replace(";", ",").split(",") if x.strip()]
            if len(parsed) == N_DOMAINS:
                raw_data = parsed
            else:
                st.warning(f"Custom vector needs {N_DOMAINS} values (got {len(parsed)}); preset kept.")
        except ValueError:
            st.warning("Could not parse the custom vector; preset kept.")

with st.sidebar.expander("🎨 Data & Colors", expanded=True):
    use_rounded = st.checkbox("Use Rounded Data", value=False)
    bar_color = st.color_picker("Bar Color", "#1f77b4")
    line_color = st.color_picker("Line Color", "#d62728")
    bar_alpha = st.slider("Bar transparency", 0.1, 1.0, 0.7, 0.05)
    bar_width = st.slider("Bar width", 0.2, 1.0, 0.8, 0.05)
    bar_edge_color = st.color_picker("Bar edge color", "#000000")
    bar_edge_width = st.slider("Bar edge width", 0.0, 2.0, 0.5, 0.1)

with st.sidebar.expander("✏️ Line / Spline", expanded=True):
    line_width = st.slider("Spline (line) width", 0.5, 6.0, 2.0, 0.1)
    line_style = st.selectbox("Line style", ["solid", "dashed", "dashdot", "dotted"])
    marker_style = st.selectbox("Marker", ["o", "s", "^", "D", "P", "*", "v", "X", "None"])
    marker_size = st.slider("Marker size", 3, 16, 8)
    marker_edge_width = st.slider("Marker edge width", 0.0, 3.0, 1.5, 0.1)
    marker_edge_color = st.color_picker("Marker edge color", "#ffffff")
    if SCIPY_AVAILABLE:
        smooth_line = st.checkbox("Smooth curve (cubic spline fit)", value=False)
    else:
        smooth_line = False

with st.sidebar.expander("🔤 Fonts & Math", expanded=True):
    font_family = st.selectbox("Font family", ["sans-serif", "serif", "monospace"])
    mathtext_fontset = st.selectbox(
        "Math font set (mathtext)",
        ["dejavusans", "dejavuserif", "cm", "stix", "stixsans"], index=2)
    ylabel_left = st.text_input("Left y-label", value=r"Raw Evidence $k_{\mathrm{raw}}$")
    ylabel_right = st.text_input("Right y-label", value=r"Domain Weight $W_k$")
    label_weight = st.selectbox("Axis label weight", ["normal", "bold"])
    bold_title = st.checkbox("Bold title", value=True)
    use_usetex = st.checkbox("Use real LaTeX (text.usetex)", value=False)

with st.sidebar.expander("📏 Ticks & Spines"):
    tick_length = st.slider("Major tick length", 0, 15, 5)
    tick_width = st.slider("Major tick width", 0.1, 3.0, 1.0, 0.1)
    tick_direction = st.selectbox("Tick direction", ["out", "in", "inout"])
    tick_pad = st.slider("Tick label padding", 1, 15, 4)
    minor_ticks = st.checkbox("Show minor ticks (y axes)", value=False)
    spine_width = st.slider("Spine (frame) width", 0.5, 3.0, 1.0, 0.1)
    top_spines = st.checkbox("Show top spine", value=False)
    x_rotation = st.select_slider("X-label rotation", options=[0, 15, 30, 45, 60, 90], value=45)

with st.sidebar.expander("🔀 Grid & Legend"):
    show_grid = st.checkbox("Show grid", value=True)
    grid_style = st.selectbox("Grid line style", ["--", "-", ":", "-."])
    grid_width = st.slider("Grid line width", 0.3, 2.0, 0.6, 0.1)
    grid_alpha = st.slider("Grid transparency", 0.05, 1.0, 0.6, 0.05)
    legend_mode = st.selectbox("Legend Location",
                               ["Below Plot (Separate Space)", "Inside Plot"])
    legend_loc = st.selectbox("Legend location (Inside Plot Only)",
                              ["upper left", "upper right", "lower left", "lower right", "best"])
    legend_frame = st.checkbox("Legend frame", value=True)
    legend_fontsize = st.slider("Legend font size", 6, 20, 10)

with st.sidebar.expander("🖼 Figure & Export"):
    font_size = st.slider("Font size", 8, 24, 12)
    fig_width = st.slider("Figure width (inches)", 4, 12, 8)
    fig_height = st.slider("Figure height (inches)", 3, 9, 5)
    dpi = st.selectbox("Export DPI (raster formats)", [100, 200, 300, 600], index=2)
    export_format = st.selectbox("Export format", ["png", "pdf", "svg", "eps", "tiff"])
    transparent_bg = st.checkbox("Transparent background", value=False)

# ------------------- QDWA COMPUTATION (CHANGED) -----------------------
sum_raw = float(sum(raw_data))
denom = N_DOMAINS * alpha + sum_raw
weights_exact = ([(alpha + r) / denom for r in raw_data] if denom > 0
                 else [0.0] * N_DOMAINS)

df_exact = pd.DataFrame({"Category": domain_categories_full,
                         "raw_k": raw_data, "w_k": weights_exact})
df_rounded = pd.DataFrame({"Category": domain_categories_full,
                           "raw_k": raw_data,
                           "w_k": [round(w, 3) for w in weights_exact]})
df = df_rounded if use_rounded else df_exact

# Quantification table in the reference-image format
# (alpha prior | raw_k | Numerator | Denominator | W_k)
df_quant = pd.DataFrame({
    "Category": domain_categories_full,
    "α (prior)": [f"{alpha:.2f}"] * N_DOMAINS,
    "raw_k": [f"{r:.2f}" for r in raw_data],
    "Numerator (α + raw_k)": [f"{alpha + r:.4f}" for r in raw_data],
    "Denominator (6α + Σ raw_j)": [f"{denom:.4f}"] * N_DOMAINS,
    "W_k": [f"{w:.4f}" for w in weights_exact],
})

cfg = dict(use_rounded=use_rounded, bar_color=bar_color, line_color=line_color,
           bar_alpha=bar_alpha, bar_width=bar_width, bar_edge_color=bar_edge_color,
           bar_edge_width=bar_edge_width, line_width=line_width, line_style=line_style,
           marker_style=marker_style, marker_size=marker_size,
           marker_edge_width=marker_edge_width, marker_edge_color=marker_edge_color,
           smooth_line=smooth_line, font_family=font_family,
           mathtext_fontset=mathtext_fontset, use_usetex=use_usetex,
           ylabel_left=ylabel_left, ylabel_right=ylabel_right,
           bold_title=bold_title, label_weight=label_weight,
           tick_length=tick_length, tick_width=tick_width,
           tick_direction=tick_direction, tick_pad=tick_pad,
           minor_ticks=minor_ticks, spine_width=spine_width,
           top_spines=top_spines, x_rotation=x_rotation, show_grid=show_grid,
           grid_style=grid_style, grid_width=grid_width, grid_alpha=grid_alpha,
           legend_loc=legend_loc, legend_frame=legend_frame,
           legend_fontsize=legend_fontsize, font_size=font_size,
           fig_width=fig_width, fig_height=fig_height, legend_mode=legend_mode)

# ------------------- MAIN PANEL -------------------
st.subheader("🧮 QDWA Quantification")
st.latex(rf"W_k \;=\; \frac{{\alpha + raw_k}}{{\,6\alpha + \sum_{{j=1}}^{{6}} raw_j\,}}"
         rf"\,,\qquad \alpha = {alpha:g}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Σ raw_j", f"{sum_raw:.2f}")
m2.metric("Denominator (6α + Σ raw_j)", f"{denom:.4f}")
m3.metric("Σ W_k (normalization check)", f"{sum(weights_exact):.4f}")
m4.metric("Top-3 weight concentration",
          f"{sum(sorted(weights_exact, reverse=True)[:3]):.1%}")

st.dataframe(df_quant, use_container_width=True)
top3_ids = sorted(range(N_DOMAINS), key=lambda i: weights_exact[i], reverse=True)[:3]
st.caption(f"Σ W_k = 1 by construction · Top-3 domains: "
           f"{', '.join(f'D{i+1}' for i in top3_ids)} "
           f"(default Q1AL2 core: D2 composition tensor, D5 phase-field/KKS, D1 thermodynamics).")

with st.expander("🔍 Scoring rationale — why each raw_k (Q1AL2)"):
    for i, cat in enumerate(domain_categories_full):
        st.markdown(f"**D{i+1} · {cat}** — $raw_k = {raw_data[i]:g}$  \n"
                    f"{EVIDENCE_RATIONALE[cat]}")

with st.expander("✍️ Notation"):
    st.markdown(r"""
    - $W_k$ — normalized domain weight, $\sum_k W_k = 1$     - $raw_k$ — raw evidence score of domain $k$ (re-scored for Q1AL2)
    - $\alpha$ — uniform prior (Laplace smoothing), default 0.25
    - Numerator $=\alpha + raw_k$ · Denominator $= 6\alpha + \sum_j raw_j$     - Left axis: raw evidence $k_{\mathrm{raw}}$ · Right axis: domain weight $W_k$     """)

st.subheader("Plot Data (raw_k vs W_k)")
st.dataframe(df)

fig = plot_dual_axis(df, cfg)
st.pyplot(fig)

if legend_mode == "Below Plot (Separate Space)":
    render_html_legend(cfg, df)

st.markdown(get_download_link(fig, dpi, export_format, transparent_bg),
            unsafe_allow_html=True)
