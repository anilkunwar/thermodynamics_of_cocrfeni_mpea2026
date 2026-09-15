import io
import re

import matplotlib
matplotlib.use("Agg")          # headless-safe backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from matplotlib.transforms import blended_transform_factory

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Concept Growth Rate | Q1LR3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Default Q1LR3 data
# =============================================================================
DEFAULT_DATA = pd.DataFrame(
    {
        "Concept": [
            "beam_diameter",
            "cocrfeni",
            "gaussian_heat_source",
            "grain_size",
            "hea",
            "laser_power",
            "melt_pool",
            "porosity",
            "scan_speed",
            "thermal_cycle",
            "thermal_gradient",
        ],
        "2020 & After": [
            9,
            502,
            4,
            269,
            1724,
            403,
            93,
            221,
            120,
            13,
            184,
        ],
        "Before 2020": [
            4,
            54,
            0,
            18,
            167,
            42,
            4,
            19,
            17,
            6,
            17,
        ],
    }
)


# =============================================================================
# Matplotlib colormap registry (70+)
# =============================================================================
MPL_COLORMAPS = [
    # Perceptual / sequential
    "viridis", "plasma", "magma", "inferno", "cividis", "turbo",
    "jet", "rainbow", "hsv", "twilight", "twilight_shifted",
    "hot", "cool", "coolwarm", "bwr", "seismic", "RdBu", "RdGy",
    "RdYlBu", "RdYlGn", "Spectral", "BrBG", "PiYG", "PRGn", "PuOr",
    "Blues", "Reds", "Greens", "Purples", "Oranges", "Greys",
    "BuPu", "PuBu", "YlGn", "YlOrRd", "YlOrBr", "YlGnBu", "PuBuGn",
    "BuGn", "GnBu", "OrRd", "PuRd", "RdPu", "pink", "spring",
    "summer", "autumn", "winter", "bone", "copper", "gray", "flag",
    "prism", "gist_earth", "terrain", "ocean", "gist_stern",
    "gist_ncar", "gist_rainbow", "gist_heat", "gist_gray", "gist_yarg",
    "CMRmap", "cubehelix", "gnuplot", "gnuplot2", "nipy_spectral",
    "tab10", "tab20", "tab20b", "tab20c", "Accent", "Dark2",
    "Set1", "Set2", "Set3", "Pastel1", "Pastel2", "Paired",
]


# =============================================================================
# Palette discovery
# Avoids AttributeError across Plotly versions.
# =============================================================================
def get_color_lists(module):
    """
    Return Plotly palettes that are stored as lists or tuples of colors.

    The function inspects the installed Plotly module dynamically, rather
    than assuming a given palette attribute exists.
    """
    color_lists = {}

    for name in dir(module):
        if name.startswith("_"):
            continue

        value = getattr(module, name)

        if isinstance(value, (list, tuple)) and len(value) > 0:
            if all(isinstance(color, str) for color in value):
                color_lists[name] = list(value)

    return color_lists


def get_available_palettes():
    """
    Build a safe palette dictionary from the installed Plotly version.

    Qualitative palettes are suitable for concept categories.
    Sequential/diverging/cyclical palettes are also exposed as color lists
    for custom concept coloring.
    """
    palettes = {}

    qualitative = get_color_lists(px.colors.qualitative)
    sequential = get_color_lists(px.colors.sequential)
    diverging = get_color_lists(px.colors.diverging)
    cyclical = get_color_lists(px.colors.cyclical)

    for name, colors in qualitative.items():
        palettes[f"Qualitative — {name}"] = colors

    for name, colors in sequential.items():
        palettes[f"Sequential — {name}"] = colors

    for name, colors in diverging.items():
        palettes[f"Diverging — {name}"] = colors

    for name, colors in cyclical.items():
        palettes[f"Cyclical — {name}"] = colors

    if not palettes:
        palettes = {
            "Fallback — Plotly": [
                "#636EFA",
                "#EF553B",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
                "#FF6692",
                "#B6E880",
                "#FF97FF",
                "#FECB52",
            ]
        }

    return dict(sorted(palettes.items()))


def to_hex_color(color, fallback="#636EFA"):
    """
    Convert Plotly/CSS colors to a #RRGGBB string accepted by st.color_picker.

    Streamlit's st.color_picker requires a hex string. Plotly palettes may
    include hex, rgb(...), rgba(...), hsl(...), named CSS colors, or
    colorscale-specific values.
    """
    if not isinstance(color, str):
        return fallback

    color = color.strip()

    # Already standard hex: #RRGGBB
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        return color.upper()

    # Short hex: #RGB -> #RRGGBB
    if re.fullmatch(r"#[0-9A-Fa-f]{3}", color):
        return (
            "#"
            + color[1] * 2
            + color[2] * 2
            + color[3] * 2
        ).upper()

    # rgb(12, 34, 56) or rgba(12, 34, 56, 0.5)
    rgb_match = re.match(
        r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)",
        color,
    )

    if rgb_match:
        red = int(float(rgb_match.group(1)))
        green = int(float(rgb_match.group(2)))
        blue = int(float(rgb_match.group(3)))

        red = max(0, min(255, red))
        green = max(0, min(255, green))
        blue = max(0, min(255, blue))

        return f"#{red:02X}{green:02X}{blue:02X}"

    # Named colors, hsl(), hsv(), or uncommon formats:
    # Streamlit needs hex, so fall back to a safe default.
    return fallback


PALETTES = get_available_palettes()


# =============================================================================
# Data preparation
# =============================================================================
def validate_and_prepare_data(input_df):
    """
    Validate input fields and calculate numerical growth metrics safely.

    Growth rate is undefined for zero pre-2020 baseline values and is stored
    as NumPy NaN rather than pandas pd.NA, avoiding Series.round errors.
    """
    required_columns = {
        "Concept",
        "2020 & After",
        "Before 2020",
    }

    missing_columns = required_columns.difference(input_df.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(
            "The uploaded CSV is missing required column(s): "
            f"{missing_names}"
        )

    df = input_df[
        [
            "Concept",
            "2020 & After",
            "Before 2020",
        ]
    ].copy()

    df["Concept"] = df["Concept"].astype(str).str.strip()

    df["2020 & After"] = pd.to_numeric(
        df["2020 & After"],
        errors="coerce",
    ).fillna(0.0)

    df["Before 2020"] = pd.to_numeric(
        df["Before 2020"],
        errors="coerce",
    ).fillna(0.0)

    df["Absolute increase"] = (
        df["2020 & After"] - df["Before 2020"]
    )

    before = df["Before 2020"].to_numpy(dtype=float)
    after = df["2020 & After"].to_numpy(dtype=float)

    df["Growth rate (%)"] = np.where(
        before > 0,
        np.round(
            ((after - before) / before) * 100,
            1,
        ),
        np.nan,
    )

    df["Growth label"] = df["Growth rate (%)"].apply(
        lambda value: (
            "N/A (zero baseline)"
            if pd.isna(value)
            else f"{value:.1f}%"
        )
    )

    df["Trend"] = np.select(
        [
            df["Absolute increase"] > 0,
            df["Absolute increase"] < 0,
        ],
        [
            "Increase",
            "Decrease",
        ],
        default="No change",
    )

    return df


def get_continuous_colorscale(selected_palette_name):
    """
    Use a valid Plotly colorscale for heatmaps.

    Plotly can accept a list of valid CSS colors directly as a colorscale.
    This is more version-safe than referring to a named scale attribute.
    """
    selected_colors = PALETTES[selected_palette_name]

    if len(selected_colors) < 2:
        return "Viridis"

    return selected_colors


def format_number(value):
    """Format numeric values for labels and summary cards."""
    if pd.isna(value):
        return "N/A"

    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{float(value):,.1f}"


@st.cache_data
def dataframe_to_csv(dataframe):
    """Convert a DataFrame to UTF-8 CSV bytes for downloading."""
    return dataframe.to_csv(index=False).encode("utf-8")


# =============================================================================
# Matplotlib rendering helpers
# =============================================================================
def _apply_fonts(ax, title_fs, label_fs, tick_fs):
    """Apply user-selected font sizes to an axis."""
    ax.set_title(ax.get_title(), fontsize=title_fs, fontweight="bold", pad=14)
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_fs)


def render_matplotlib_chart(
    df,
    chart_type,
    orientation,
    show_values,
    use_log_scale,
    # ---- Figure-level sizing ----
    fig_width,
    fig_height,
    # ---- Fonts ----
    title_fs,
    label_fs,
    tick_fs,
    # ---- Tick marks ----
    tick_length,
    tick_width,
    # ---- Figure box / spines ----
    spine_lw,
    # ---- Bars ----
    bar_width,
    group_gap,
    # ---- Colormap / axes ----
    cmap_name,
    y_axis_position,
    edge_color,
    show_grid,
    concept_colors,
    # ---- Value labels ----
    value_label_fs,
    value_label_offset,
    value_label_color,
    value_label_box_enabled,
    value_label_box_fc,
    value_label_box_ec,
    value_label_box_lw,
    value_label_box_pad,
    value_label_box_alpha,
):
    """Render the selected chart with Matplotlib and return the figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#444444",
        "axes.linewidth": spine_lw,
        "xtick.major.size": tick_length,
        "xtick.major.width": tick_width,
        "ytick.major.size": tick_length,
        "ytick.major.width": tick_width,
    })

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=140)
    cmap = plt.get_cmap(cmap_name)
    concepts = df["Concept"].tolist()
    n = len(concepts)
    norm = Normalize(vmin=0, vmax=max(1, n - 1))
    concept_index = {c: i for i, c in enumerate(concepts)}

    def palette_for_concept(concept):
        existing = concept_colors.get(concept)
        if existing:
            return existing
        idx = concept_index.get(concept, 0)
        return to_hex(cmap(norm(idx)))

    # ---- Global tick styling ------------------------------------------------
    def _apply_tick_style():
        """Push user tick length/width to both axes after all artists added."""
        ax.tick_params(
            axis="both", which="major",
            length=tick_length,
            width=tick_width,
            labelsize=tick_fs,
        )
        # Apply spine linewidth uniformly (outer box thickness)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_lw)

    # ---- Value-label helpers ------------------------------------------------
    def _bbox():
        """Build the bbox dict from user settings (or None if disabled)."""
        if not value_label_box_enabled:
            return None
        return dict(
            boxstyle=f"round,pad={value_label_box_pad}",
            fc=value_label_box_fc,
            ec=value_label_box_ec,
            lw=value_label_box_lw,
            alpha=value_label_box_alpha,
        )

    def _label(x, y, text, dx=0, dy=0, ha="center", va="bottom",
               color=None, fontsize=None):
        """Place a value label with a fixed pixel offset from (x, y)."""
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize if fontsize is not None else value_label_fs,
            color=color or value_label_color,
            bbox=_bbox(),
            zorder=5,
        )

    # ---- Tick-label relocation helpers (for center-axis mode) ---------------
    def _relocate_x_tick_labels(labels, fontsize):
        """Hide default x tick labels and place them at the bottom edge."""
        ax.tick_params(
            axis="x", which="both",
            bottom=False, top=False, labelbottom=False,
        )
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for i, lab in enumerate(labels):
            ax.annotate(
                lab,
                xy=(i, 0.0),
                xytext=(0, -(tick_length + 4)),
                textcoords="offset points",
                ha="right", va="top",
                rotation=35, rotation_mode="anchor",
                xycoords=trans,
                fontsize=fontsize,
            )

    def _relocate_y_tick_labels(labels, fontsize):
        """Hide default y tick labels and place them at the left edge."""
        ax.tick_params(
            axis="y", which="both",
            left=False, right=False, labelleft=False,
        )
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for i, lab in enumerate(labels):
            ax.annotate(
                lab,
                xy=(0.0, i),
                xytext=(-(tick_length + 4), 0),
                textcoords="offset points",
                ha="right", va="center",
                xycoords=trans,
                fontsize=fontsize,
            )

    # ---- Grouped-bar geometry helper ---------------------------------------
    def _group_offsets():
        """
        Compute the two bar offsets for grouped charts.

        ``bar_width`` controls each individual bar's thickness,
        ``group_gap`` controls the whitespace between the two bars
        of the same concept.
        """
        half = bar_width / 2.0
        offset = half + group_gap / 2.0
        return offset

    # --------------------------------------------------------------
    # Grouped bar chart
    # --------------------------------------------------------------
    if chart_type == "Grouped bar chart":
        before = df["Before 2020"].to_numpy(dtype=float)
        after = df["2020 & After"].to_numpy(dtype=float)

        if y_axis_position == "Center" and orientation == "Horizontal":
            # Tornado / back-to-back bar chart
            y = np.arange(n)
            ax.barh(y, after, height=bar_width, color=to_hex(cmap(0.20)),
                    edgecolor=edge_color, label="2020 & After")
            ax.barh(y, -before, height=bar_width, color=to_hex(cmap(0.85)),
                    edgecolor=edge_color, label="Before 2020")
            ax.set_yticks(y)
            ax.set_yticklabels([])                 # hide default labels
            _relocate_y_tick_labels(concepts, tick_fs)
            ax.invert_yaxis()
            ax.axvline(0, color="black", linewidth=1.0)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{abs(v):,.0f}")
            )
            ax.spines["left"].set_position("zero")   # Y-axis at center
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlabel("Mentions  (← Before 2020   |   2020 & After →)")
            ax.set_title("Concept Momentum (Back-to-Back / Centered Y-Axis)")
            ax.legend(loc="lower right", fontsize=tick_fs)
            if show_values:
                for yi, b, a in zip(y, before, after):
                    _label(-b, yi, f"{int(b):,}",
                           dx=-value_label_offset, dy=0,
                           ha="right", va="center")
                    _label(a, yi, f"{int(a):,}",
                           dx=value_label_offset, dy=0,
                           ha="left", va="center")

        elif y_axis_position == "Center" and orientation == "Vertical":
            # Diverging: Before goes down, After goes up
            x = np.arange(n)
            ax.bar(x, after, width=bar_width, color=to_hex(cmap(0.20)),
                   edgecolor=edge_color, label="2020 & After")
            ax.bar(x, -before, width=bar_width, color=to_hex(cmap(0.85)),
                   edgecolor=edge_color, label="Before 2020")
            ax.set_xticks(x)
            ax.set_xticklabels([])                 # hide default labels
            _relocate_x_tick_labels(concepts, tick_fs)
            ax.axhline(0, color="black", linewidth=1.0)
            ax.spines["bottom"].set_position("zero")   # X-axis at center
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{abs(v):,.0f}")
            )
            ax.set_ylabel("Mentions  (↓ Before 2020   |   2020 & After ↑)")
            ax.set_title("Concept Momentum (Diverging / Centered X-Axis)")
            ax.legend(loc="upper right", fontsize=tick_fs)
            # Push axis label to the outer edge so it doesn't collide
            ax.yaxis.set_label_coords(-0.08, 0.5)
            if show_values:
                for xi, b, a in zip(x, before, after):
                    _label(xi, -b, f"{int(b):,}",
                           dx=0, dy=-value_label_offset,
                           ha="center", va="top")
                    _label(xi, a, f"{int(a):,}",
                           dx=0, dy=value_label_offset,
                           ha="center", va="bottom")

        else:
            # Standard grouped bars, Y-axis at edge
            if orientation == "Vertical":
                x = np.arange(n)
                offset = _group_offsets()
                ax.bar(x - offset, before, width=bar_width,
                       color=to_hex(cmap(0.20)), edgecolor=edge_color,
                       label="Before 2020")
                ax.bar(x + offset, after, width=bar_width,
                       color=to_hex(cmap(0.85)), edgecolor=edge_color,
                       label="2020 & After")
                ax.set_xticks(x)
                ax.set_xticklabels(concepts, rotation=35, ha="right")
                ax.set_xlabel("Concept")
                ax.set_ylabel("Total mentions")
                if show_values:
                    for xi, b, a in zip(x, before, after):
                        _label(xi - offset, b, f"{int(b):,}",
                               dx=0, dy=value_label_offset,
                               ha="center", va="bottom")
                        _label(xi + offset, a, f"{int(a):,}",
                               dx=0, dy=value_label_offset,
                               ha="center", va="bottom")
            else:
                y = np.arange(n)
                offset = _group_offsets()
                ax.barh(y - offset, before, height=bar_width,
                        color=to_hex(cmap(0.20)), edgecolor=edge_color,
                        label="Before 2020")
                ax.barh(y + offset, after, height=bar_width,
                        color=to_hex(cmap(0.85)), edgecolor=edge_color,
                        label="2020 & After")
                ax.set_yticks(y)
                ax.set_yticklabels(concepts)
                ax.invert_yaxis()
                ax.set_xlabel("Total mentions")
                ax.set_ylabel("Concept")
                if show_values:
                    for yi, b, a in zip(y, before, after):
                        _label(b, yi - offset, f"{int(b):,}",
                               dx=value_label_offset, dy=0,
                               ha="left", va="center")
                        _label(a, yi + offset, f"{int(a):,}",
                               dx=value_label_offset, dy=0,
                               ha="left", va="center")
            ax.set_title("Concept Momentum (Grouped Bars)")
            ax.legend(fontsize=tick_fs)

    # --------------------------------------------------------------
    # Growth-rate bar chart
    # --------------------------------------------------------------
    elif chart_type == "Growth-rate bar chart":
        growth_df = df.dropna(subset=["Growth rate (%)"]).copy()
        concepts_g = growth_df["Concept"].tolist()
        values = growth_df["Growth rate (%)"].to_numpy(dtype=float)
        colors = [to_hex(cmap(norm(i))) for i in range(len(concepts_g))]

        if orientation == "Vertical":
            x = np.arange(len(concepts_g))
            ax.bar(x, values, width=bar_width, color=colors,
                   edgecolor=edge_color)
            if y_axis_position == "Center":
                ax.set_xticks(x)
                ax.set_xticklabels([])
                _relocate_x_tick_labels(concepts_g, tick_fs)
                ax.xaxis.set_label_coords(0.5, -0.10)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(concepts_g, rotation=35, ha="right")
            ax.set_xlabel("Concept")
            ax.set_ylabel("Growth rate (%)")

            if y_axis_position == "Center":
                ax.spines["bottom"].set_position("zero")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.yaxis.set_label_coords(-0.08, 0.5)

            if show_values:
                for i, v in enumerate(values):
                    if v >= 0:
                        _label(i, v, f"{v:.1f}%",
                               dx=0, dy=value_label_offset,
                               ha="center", va="bottom")
                    else:
                        _label(i, v, f"{v:.1f}%",
                               dx=0, dy=-value_label_offset,
                               ha="center", va="top")
        else:
            y = np.arange(len(concepts_g))
            ax.barh(y, values, height=bar_width, color=colors,
                    edgecolor=edge_color)
            if y_axis_position == "Center":
                ax.set_yticks(y)
                ax.set_yticklabels([])
                _relocate_y_tick_labels(concepts_g, tick_fs)
                ax.yaxis.set_label_coords(-0.08, 0.5)
                ax.spines["left"].set_position("zero")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            else:
                ax.set_yticks(y)
                ax.set_yticklabels(concepts_g)
            ax.invert_yaxis()
            ax.set_xlabel("Growth rate (%)")
            ax.set_ylabel("Concept")

            if show_values:
                for i, v in enumerate(values):
                    if v >= 0:
                        _label(v, i, f"{v:.1f}%",
                               dx=value_label_offset, dy=0,
                               ha="left", va="center")
                    else:
                        _label(v, i, f"{v:.1f}%",
                               dx=-value_label_offset, dy=0,
                               ha="right", va="center")

        ax.set_title("Concept Growth Rate (%)")

    # --------------------------------------------------------------
    # Absolute-increase bar chart
    # --------------------------------------------------------------
    elif chart_type == "Absolute-increase bar chart":
        values = df["Absolute increase"].to_numpy(dtype=float)
        colors = ["#2CA02C" if v > 0 else "#D62728" if v < 0 else "#7F7F7F"
                  for v in values]

        if orientation == "Vertical":
            x = np.arange(n)
            ax.bar(x, values, width=bar_width, color=colors,
                   edgecolor=edge_color)
            if y_axis_position == "Center":
                ax.set_xticks(x)
                ax.set_xticklabels([])
                _relocate_x_tick_labels(concepts, tick_fs)
                ax.xaxis.set_label_coords(0.5, -0.10)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(concepts, rotation=35, ha="right")
            ax.set_xlabel("Concept")
            ax.set_ylabel("Absolute increase in mentions")

            if y_axis_position == "Center":
                ax.spines["bottom"].set_position("zero")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.yaxis.set_label_coords(-0.08, 0.5)

            ax.axhline(0, color="black", linewidth=0.8)

            if show_values:
                for i, v in enumerate(values):
                    if v >= 0:
                        _label(i, v, f"{v:+,.0f}",
                               dx=0, dy=value_label_offset,
                               ha="center", va="bottom")
                    else:
                        _label(i, v, f"{v:+,.0f}",
                               dx=0, dy=-value_label_offset,
                               ha="center", va="top")
        else:
            y = np.arange(n)
            ax.barh(y, values, height=bar_width, color=colors,
                    edgecolor=edge_color)
            if y_axis_position == "Center":
                ax.set_yticks(y)
                ax.set_yticklabels([])
                _relocate_y_tick_labels(concepts, tick_fs)
                ax.yaxis.set_label_coords(-0.08, 0.5)
                ax.spines["left"].set_position("zero")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            else:
                ax.set_yticks(y)
                ax.set_yticklabels(concepts)
            ax.invert_yaxis()
            ax.set_xlabel("Absolute increase in mentions")
            ax.set_ylabel("Concept")

            ax.axvline(0, color="black", linewidth=0.8)

            if show_values:
                for i, v in enumerate(values):
                    if v >= 0:
                        _label(v, i, f"{v:+,.0f}",
                               dx=value_label_offset, dy=0,
                               ha="left", va="center")
                    else:
                        _label(v, i, f"{v:+,.0f}",
                               dx=-value_label_offset, dy=0,
                               ha="right", va="center")

        ax.set_title("Absolute Increase in Mentions")

    # --------------------------------------------------------------
    # Dumbbell chart
    # --------------------------------------------------------------
    elif chart_type == "Dumbbell chart":
        y = np.arange(n)
        for i, (_, row) in enumerate(df.iterrows()):
            color = palette_for_concept(row["Concept"])
            ax.plot([row["Before 2020"], row["2020 & After"]],
                    [i, i], color=color, lw=3, solid_capstyle="round")
            ax.scatter(row["Before 2020"], i, s=140, color=color,
                       edgecolor=edge_color, zorder=3)
            ax.scatter(row["2020 & After"], i, s=140, color=color,
                       edgecolor=edge_color, marker="s", zorder=3)
            if show_values:
                _label(row["Before 2020"], i,
                       f"{int(row['Before 2020']):,}",
                       dx=-value_label_offset, dy=0,
                       ha="right", va="center")
                _label(row["2020 & After"], i,
                       f"{int(row['2020 & After']):,}",
                       dx=value_label_offset, dy=0,
                       ha="left", va="center")
        ax.set_yticks(y)
        ax.set_yticklabels(concepts)
        ax.invert_yaxis()
        ax.set_xlabel("Total mentions")
        ax.set_ylabel("Concept")
        ax.set_title("Concept Mentions Dumbbell Chart")
        if use_log_scale:
            ax.set_xscale("symlog")

    # --------------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------------
    elif chart_type == "Heatmap":
        z = df[["Before 2020", "2020 & After"]].to_numpy(dtype=float)
        im = ax.imshow(z, aspect="auto", cmap=cmap_name)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before 2020", "2020 & After"])
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(concepts)
        ax.set_xlabel("Publication period")
        ax.set_ylabel("Concept")
        if show_values:
            zmax = float(z.max()) if z.size else 1.0
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    ax.annotate(
                        f"{int(z[i, j]):,}",
                        xy=(j, i),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha="center", va="center",
                        fontsize=value_label_fs,
                        color=("white" if z[i, j] < zmax * 0.5
                               else value_label_color),
                        bbox=_bbox(),
                        zorder=5,
                    )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mentions", fontsize=label_fs)
        cbar.ax.tick_params(labelsize=tick_fs,
                            length=tick_length, width=tick_width)
        ax.set_title("Concept Mentions Heatmap")

    # --------------------------------------------------------------
    # Treemap (fallback: proportional horizontal bar)
    # --------------------------------------------------------------
    elif chart_type == "Treemap":
        sizes = df["2020 & After"].to_numpy(dtype=float)
        total = sizes.sum() if sizes.sum() > 0 else 1.0
        cumulative = np.cumsum(sizes) / total
        starts = np.concatenate([[0], cumulative[:-1]])
        for i, concept in enumerate(concepts):
            ax.barh(0, cumulative[i] - starts[i], left=starts[i],
                    height=0.6, color=to_hex(cmap(norm(i))),
                    edgecolor=edge_color)
            label = (f"{concept}\n{int(sizes[i]):,} "
                     f"({sizes[i] / total * 100:.1f}%)")
            ax.annotate(
                label,
                xy=((starts[i] + cumulative[i]) / 2, 0),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center", va="center",
                fontsize=value_label_fs,
                color=("white"
                       if sum(cmap(norm(i))[:3]) < 1.5
                       else "black"),
                zorder=5,
            )
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of post-2020 mentions")
        ax.set_title("Concept Share Treemap (Proportional Bar View)")

    # ---- common finalization ----
    if show_grid and chart_type not in ("Heatmap", "Treemap"):
        ax.grid(True, linestyle="--", alpha=0.35)

    if use_log_scale and chart_type not in (
        "Heatmap",
        "Treemap",
        "Dumbbell chart",
    ):
        numeric_df = df.select_dtypes("number")
        has_negative = bool((numeric_df < 0).any().any())
        scale_name = "symlog" if has_negative else "log"
        if orientation == "Vertical":
            ax.set_yscale(scale_name)
        else:
            ax.set_xscale(scale_name)

    _apply_fonts(ax, title_fs, label_fs, tick_fs)
    _apply_tick_style()
    fig.tight_layout()
    return fig


# =============================================================================
# Header
# =============================================================================
st.title("Concept Growth Rate: Q1LR3")

st.markdown(
    """
**Research question:** Analyze the sensitivity of the Gaussian heat-source
thermal cycle and subsequent melt-pool penetration depth to variations in
laser power and scan speed.

The dashboard compares concept mentions **before 2020** with mentions in
**2020 and after**. It provides absolute increases, relative growth rates,
interactive visuals, palette selection, and exports.
"""
)


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Visualization controls")

    uploaded_file = st.file_uploader(
        "Upload concept-count CSV",
        type=["csv"],
        help=(
            "Required columns: Concept, 2020 & After, Before 2020"
        ),
    )

    st.caption(
        "No upload is required. The Q1LR3 data embedded in the script "
        "will be used by default."
    )

    st.divider()

    chart_type = st.selectbox(
        "Chart type",
        options=[
            "Grouped bar chart",
            "Growth-rate bar chart",
            "Absolute-increase bar chart",
            "Dumbbell chart",
            "Heatmap",
            "Treemap",
        ],
    )

    palette_name = st.selectbox(
        "Color palette",
        options=list(PALETTES.keys()),
        index=0,
        help=(
            "The palette list is detected dynamically from the Plotly "
            "version installed in the deployment environment."
        ),
    )

    sort_by = st.selectbox(
        "Sort concepts by",
        options=[
            "Original order",
            "2020 & After",
            "Before 2020",
            "Absolute increase",
            "Growth rate (%)",
            "Alphabetical",
        ],
    )

    orientation = st.radio(
        "Bar-chart orientation",
        options=[
            "Vertical",
            "Horizontal",
        ],
        horizontal=True,
    )

    show_values = st.checkbox(
        "Show data labels",
        value=True,
    )

    use_log_scale = st.checkbox(
        "Use logarithmic mention axis",
        value=False,
        help=(
            "A logarithmic axis cannot display zero-valued bars. "
            "Use a linear axis to retain zero mention counts."
        ),
    )

    use_custom_colors = st.checkbox(
        "Customize concept colors",
        value=False,
    )

    rendering_engine = st.radio(
        "Rendering engine",
        options=["Plotly", "Matplotlib"],
        horizontal=True,
        help=(
            "Matplotlib exposes font sizes, figure size, bar thickness, "
            "70+ colormaps, and Y-axis center/edge placement."
        ),
    )

    use_matplotlib = rendering_engine == "Matplotlib"

    # -------------------------------------------------------------------
    # Defaults so these names always exist downstream (even in Plotly mode)
    # -------------------------------------------------------------------
    fig_width = 12.0
    fig_height = 7.0
    title_fontsize = 16
    axis_label_fontsize = 13
    tick_fontsize = 11
    tick_length = 4.0
    tick_width = 1.0
    spine_lw = 0.9
    bar_width = 0.6
    group_gap = 0.08
    mpl_cmap = "turbo"
    y_axis_position = "Edge"
    bar_edge_color = "#222222"
    grid_toggle = True

    # Value-label styling defaults
    value_label_fs = 11
    value_label_offset = 5
    value_label_color = "#000000"
    value_label_box_enabled = True
    value_label_box_fc = "#FFFFFF"
    value_label_box_ec = "#222222"
    value_label_box_lw = 0.8
    value_label_box_pad = 0.25
    value_label_box_alpha = 0.9

    if use_matplotlib:
        st.divider()
        st.subheader("Matplotlib figure controls")

        with st.expander("Figure & fonts", expanded=True):
            fig_width = st.slider(
                "Figure width (inches)",
                min_value=4.0, max_value=30.0, value=12.0, step=0.5,
            )
            fig_height = st.slider(
                "Figure height (inches)",
                min_value=3.0, max_value=24.0, value=7.0, step=0.5,
            )
            title_fontsize = st.slider(
                "Title font size", 8, 40, 16,
            )
            axis_label_fontsize = st.slider(
                "Axis-label font size", 8, 32, 13,
            )
            tick_fontsize = st.slider(
                "Tick-label font size", 6, 28, 11,
            )

        with st.expander("Ticks & figure box", expanded=False):
            tick_length = st.slider(
                "Tick mark length (points)",
                min_value=0.0, max_value=15.0, value=4.0, step=0.5,
            )
            tick_width = st.slider(
                "Tick mark thickness (points)",
                min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            )
            spine_lw = st.slider(
                "Figure box line thickness",
                min_value=0.0, max_value=5.0, value=0.9, step=0.1,
            )

        with st.expander("Bars & spacing", expanded=False):
            bar_width = st.slider(
                "Bar thickness (0–1)",
                min_value=0.05, max_value=1.0, value=0.6, step=0.05,
            )
            group_gap = st.slider(
                "Gap between paired bars",
                min_value=0.0, max_value=0.5, value=0.08, step=0.01,
                help=(
                    "Whitespace between the Before-2020 and 2020-&-After "
                    "bars of the same concept in grouped charts."
                ),
            )

        with st.expander("Palette & axes", expanded=False):
            mpl_cmap = st.selectbox(
                "Colormap (70+)",
                options=MPL_COLORMAPS,
                index=MPL_COLORMAPS.index("turbo"),
            )
            y_axis_position = st.radio(
                "Y-axis position",
                options=["Edge", "Center"],
                horizontal=True,
                help=(
                    "Center places the category axis at x = 0 and renders "
                    "Before 2020 / 2020 & After as a back-to-back "
                    "(tornado) bar chart. Tick labels are pushed to the "
                    "outer edges so they don't overlap the centered spine."
                ),
            )
            bar_edge_color = st.color_picker(
                "Bar edge color", "#222222",
            )
            grid_toggle = st.checkbox(
                "Show grid lines", value=True,
            )

        with st.expander("Data label styling", expanded=False):
            value_label_fs = st.slider(
                "Label font size", 6, 28, 11,
            )
            value_label_offset = st.slider(
                "Label offset (points)", 0, 30, 5,
            )
            value_label_color = st.color_picker(
                "Label text color", "#000000",
            )
            value_label_box_enabled = st.checkbox(
                "Draw box behind labels", value=True,
            )
            if value_label_box_enabled:
                value_label_box_fc = st.color_picker(
                    "Box fill color", "#FFFFFF",
                )
                value_label_box_ec = st.color_picker(
                    "Box edge color", "#222222",
                )
                value_label_box_lw = st.slider(
                    "Box edge thickness",
                    0.0, 3.0, 0.8, 0.1,
                )
                value_label_box_pad = st.slider(
                    "Box padding",
                    0.0, 1.0, 0.25, 0.05,
                )
                value_label_box_alpha = st.slider(
                    "Box opacity",
                    0.0, 1.0, 0.9, 0.05,
                )

        st.caption(
            "Colormaps include jet, turbo, rainbow, inferno, viridis, "
            "plasma, magma, cividis, seismic, Spectral, RdBu, tab10, "
            "tab20, Set1–3, Pastel1–2, Paired, Accent, Dark2, gist_*, "
            "CMRmap, cubehelix, gnuplot, nipy_spectral, twilight, hsv, "
            "and many more."
        )


# =============================================================================
# Load data
# =============================================================================
if uploaded_file is not None:
    try:
        source_df = pd.read_csv(uploaded_file)
        data_source = "Uploaded CSV"
    except Exception as error:
        st.error(f"Unable to read the uploaded CSV: {error}")
        st.stop()
else:
    source_df = DEFAULT_DATA.copy()
    data_source = "Embedded Q1LR3 dataset"

try:
    df = validate_and_prepare_data(source_df)
except ValueError as error:
    st.error(str(error))
    st.info(
        "Required CSV fields: `Concept`, `2020 & After`, "
        "and `Before 2020`."
    )
    st.stop()


# =============================================================================
# Filter concepts
# =============================================================================
all_concepts = df["Concept"].tolist()

with st.sidebar:
    selected_concepts = st.multiselect(
        "Concepts to display",
        options=all_concepts,
        default=all_concepts,
    )

if not selected_concepts:
    st.warning("Select at least one concept to render a visualization.")
    st.stop()

df = df[
    df["Concept"].isin(selected_concepts)
].copy()


# =============================================================================
# Sorting
# =============================================================================
if sort_by == "2020 & After":
    df = df.sort_values(
        "2020 & After",
        ascending=False,
    )

elif sort_by == "Before 2020":
    df = df.sort_values(
        "Before 2020",
        ascending=False,
    )

elif sort_by == "Absolute increase":
    df = df.sort_values(
        "Absolute increase",
        ascending=False,
    )

elif sort_by == "Growth rate (%)":
    df = df.sort_values(
        "Growth rate (%)",
        ascending=False,
        na_position="last",
    )

elif sort_by == "Alphabetical":
    df = df.sort_values(
        "Concept",
        ascending=True,
    )


# =============================================================================
# Colors
# =============================================================================
selected_palette = PALETTES[palette_name]

# Convert every Plotly palette value to #RRGGBB before passing it to
# Streamlit's color picker.
safe_palette = [
    to_hex_color(color)
    for color in selected_palette
]

# Emergency fallback: guarantees at least one valid color.
if not safe_palette:
    safe_palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

if use_custom_colors:
    concept_colors = {}

    with st.sidebar:
        st.divider()
        st.subheader("Concept colors")

        for index, concept in enumerate(df["Concept"].tolist()):
            default_color = safe_palette[
                index % len(safe_palette)
            ]

            concept_colors[concept] = st.color_picker(
                label=concept,
                value=default_color,
                key=f"concept_color_{concept}",
            )

else:
    concept_colors = {
        concept: safe_palette[
            index % len(safe_palette)
        ]
        for index, concept in enumerate(df["Concept"].tolist())
    }


# =============================================================================
# Summary metrics
# =============================================================================
total_before = df["Before 2020"].sum()
total_after = df["2020 & After"].sum()
total_change = df["Absolute increase"].sum()

if total_before > 0:
    total_growth_rate = (
        (total_after - total_before) / total_before
    ) * 100
    growth_metric = f"{total_growth_rate:.1f}%"
else:
    growth_metric = "N/A"

st.caption(f"Data source: {data_source}")

col_1, col_2, col_3, col_4 = st.columns(4)

col_1.metric(
    "Displayed concepts",
    f"{len(df):,}",
)

col_2.metric(
    "Mentions before 2020",
    format_number(total_before),
)

col_3.metric(
    "Mentions in 2020 & after",
    format_number(total_after),
)

col_4.metric(
    "Overall growth",
    growth_metric,
    delta=f"+{format_number(total_change)} mentions",
)


# =============================================================================
# Plotly figure (always built; only shown when the Plotly engine is active)
# =============================================================================
fig = go.Figure()

chart_title = "Concept Momentum (User Defined Split: 2020)"
mention_axis_type = "log" if use_log_scale else "linear"


# -----------------------------------------------------------------------------
# Grouped bars
# -----------------------------------------------------------------------------
if chart_type == "Grouped bar chart":

    before_color = "#636EFA"
    after_color = "#EF553B"

    if orientation == "Vertical":
        fig.add_trace(
            go.Bar(
                name="Before 2020",
                x=df["Concept"],
                y=df["Before 2020"],
                marker_color=before_color,
                text=(
                    df["Before 2020"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Before 2020: %{y:,} mentions"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Bar(
                name="2020 & After",
                x=df["Concept"],
                y=df["2020 & After"],
                marker_color=after_color,
                text=(
                    df["2020 & After"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "2020 & After: %{y:,} mentions"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            title="Concept",
            tickangle=-35,
            categoryorder="array",
            categoryarray=df["Concept"].tolist(),
        )

        fig.update_yaxes(
            title="Total mentions",
            type=mention_axis_type,
        )

    else:
        fig.add_trace(
            go.Bar(
                name="Before 2020",
                y=df["Concept"],
                x=df["Before 2020"],
                orientation="h",
                marker_color=before_color,
                text=(
                    df["Before 2020"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Before 2020: %{x:,} mentions"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Bar(
                name="2020 & After",
                y=df["Concept"],
                x=df["2020 & After"],
                orientation="h",
                marker_color=after_color,
                text=(
                    df["2020 & After"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "2020 & After: %{x:,} mentions"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            title="Total mentions",
            type=mention_axis_type,
        )

        fig.update_yaxes(
            title="Concept",
            categoryorder="array",
            categoryarray=df["Concept"].tolist()[::-1],
        )

    fig.update_layout(
        barmode="group",
    )


# -----------------------------------------------------------------------------
# Growth rate
# -----------------------------------------------------------------------------
elif chart_type == "Growth-rate bar chart":

    growth_df = df.dropna(
        subset=["Growth rate (%)"]
    ).copy()

    zero_baseline_df = df[
        df["Growth rate (%)"].isna()
    ].copy()

    if growth_df.empty:
        st.warning(
            "None of the selected concepts has a nonzero pre-2020 baseline. "
            "Relative percentage growth cannot be calculated."
        )
        st.stop()

    growth_labels = [
        f"{value:.1f}%"
        for value in growth_df["Growth rate (%)"]
    ]

    growth_colors = [
        concept_colors[concept]
        for concept in growth_df["Concept"]
    ]

    if orientation == "Vertical":
        fig.add_trace(
            go.Bar(
                x=growth_df["Concept"],
                y=growth_df["Growth rate (%)"],
                marker_color=growth_colors,
                text=growth_labels if show_values else None,
                textposition="outside",
                name="Growth rate",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Growth rate: %{y:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            title="Concept",
            tickangle=-35,
        )

        fig.update_yaxes(
            title="Growth rate (%)",
        )

    else:
        fig.add_trace(
            go.Bar(
                y=growth_df["Concept"],
                x=growth_df["Growth rate (%)"],
                orientation="h",
                marker_color=growth_colors,
                text=growth_labels if show_values else None,
                textposition="outside",
                name="Growth rate",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Growth rate: %{x:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            title="Growth rate (%)",
        )

        fig.update_yaxes(
            title="Concept",
            categoryorder="array",
            categoryarray=growth_df["Concept"].tolist()[::-1],
        )

    if not zero_baseline_df.empty:
        concepts_with_zero_baseline = ", ".join(
            zero_baseline_df["Concept"].tolist()
        )

        st.info(
            "Relative growth is undefined for "
            f"`{concepts_with_zero_baseline}` because its pre-2020 count "
            "is zero. Review its absolute increase and post-2020 count "
            "instead."
        )


# -----------------------------------------------------------------------------
# Absolute increase
# -----------------------------------------------------------------------------
elif chart_type == "Absolute-increase bar chart":

    increase_colors = [
        "#2CA02C"
        if value > 0
        else "#D62728"
        if value < 0
        else "#7F7F7F"
        for value in df["Absolute increase"]
    ]

    increase_labels = [
        f"{value:+,.0f}"
        for value in df["Absolute increase"]
    ]

    if orientation == "Vertical":
        fig.add_trace(
            go.Bar(
                x=df["Concept"],
                y=df["Absolute increase"],
                marker_color=increase_colors,
                text=increase_labels if show_values else None,
                textposition="outside",
                name="Absolute increase",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Absolute increase: %{y:+,} mentions"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            title="Concept",
            tickangle=-35,
        )

        fig.update_yaxes(
            title="Absolute increase in mentions",
        )

    else:
        fig.add_trace(
            go.Bar(
                y=df["Concept"],
                x=df["Absolute increase"],
                orientation="h",
                marker_color=increase_colors,
                text=increase_labels if show_values else None,
                textposition="outside",
                name="Absolute increase",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Absolute increase: %{x:+,} mentions"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            title="Absolute increase in mentions",
        )

        fig.update_yaxes(
            title="Concept",
            categoryorder="array",
            categoryarray=df["Concept"].tolist()[::-1],
        )


# -----------------------------------------------------------------------------
# Dumbbell chart
# -----------------------------------------------------------------------------
elif chart_type == "Dumbbell chart":

    for _, row in df.iterrows():
        concept = row["Concept"]
        before_value = float(row["Before 2020"])
        after_value = float(row["2020 & After"])
        color = concept_colors[concept]

        fig.add_trace(
            go.Scatter(
                x=[before_value, after_value],
                y=[concept, concept],
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=4,
                ),
                marker=dict(
                    color=color,
                    size=11,
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{concept}</b><br>"
                    "Mentions: %{x:,}"
                    "<extra></extra>"
                ),
            )
        )

        if show_values:
            fig.add_annotation(
                x=before_value,
                y=concept,
                text=format_number(before_value),
                showarrow=False,
                yshift=17,
            )

            fig.add_annotation(
                x=after_value,
                y=concept,
                text=format_number(after_value),
                showarrow=False,
                yshift=17,
            )

    fig.update_xaxes(
        title="Total mentions",
        type=mention_axis_type,
    )

    fig.update_yaxes(
        title="Concept",
        categoryorder="array",
        categoryarray=df["Concept"].tolist()[::-1],
    )


# -----------------------------------------------------------------------------
# Heatmap
# -----------------------------------------------------------------------------
elif chart_type == "Heatmap":

    heatmap_values = df[
        [
            "Before 2020",
            "2020 & After",
        ]
    ].to_numpy(dtype=float)

    heatmap_text = None

    if show_values:
        heatmap_text = np.array(
            [
                [
                    format_number(value)
                    for value in row
                ]
                for row in heatmap_values
            ]
        )

    fig.add_trace(
        go.Heatmap(
            z=heatmap_values,
            x=[
                "Before 2020",
                "2020 & After",
            ],
            y=df["Concept"],
            colorscale=get_continuous_colorscale(palette_name),
            text=heatmap_text,
            texttemplate="%{text}" if show_values else None,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}: %{z:,} mentions"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="Mentions",
            ),
        )
    )

    fig.update_xaxes(
        title="Publication period",
    )

    fig.update_yaxes(
        title="Concept",
        categoryorder="array",
        categoryarray=df["Concept"].tolist()[::-1],
    )


# -----------------------------------------------------------------------------
# Treemap
# -----------------------------------------------------------------------------
elif chart_type == "Treemap":

    fig.add_trace(
        go.Treemap(
            labels=df["Concept"],
            parents=[""] * len(df),
            values=df["2020 & After"],
            marker=dict(
                colors=[
                    concept_colors[concept]
                    for concept in df["Concept"]
                ]
            ),
            textinfo="label+value+percent root",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "2020 & After mentions: %{value:,}<br>"
                "Share of selected total: %{percentRoot}"
                "<extra></extra>"
            ),
        )
    )


# =============================================================================
# Shared Plotly layout
# =============================================================================
fig.update_layout(
    title=dict(
        text=chart_title,
        x=0.01,
        xanchor="left",
    ),
    template="plotly_white",
    height=700,
    margin=dict(
        l=30,
        r=30,
        t=80,
        b=170,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.0,
    ),
)


# =============================================================================
# Render: dispatch between Matplotlib and Plotly
# =============================================================================
png_bytes = None

if use_matplotlib:
    mpl_fig = render_matplotlib_chart(
        df=df,
        chart_type=chart_type,
        orientation=orientation,
        show_values=show_values,
        use_log_scale=use_log_scale,
        fig_width=fig_width,
        fig_height=fig_height,
        title_fs=title_fontsize,
        label_fs=axis_label_fontsize,
        tick_fs=tick_fontsize,
        tick_length=tick_length,
        tick_width=tick_width,
        spine_lw=spine_lw,
        bar_width=bar_width,
        group_gap=group_gap,
        cmap_name=mpl_cmap,
        y_axis_position=y_axis_position,
        edge_color=bar_edge_color,
        show_grid=grid_toggle,
        concept_colors=concept_colors,
        value_label_fs=value_label_fs,
        value_label_offset=value_label_offset,
        value_label_color=value_label_color,
        value_label_box_enabled=value_label_box_enabled,
        value_label_box_fc=value_label_box_fc,
        value_label_box_ec=value_label_box_ec,
        value_label_box_lw=value_label_box_lw,
        value_label_box_pad=value_label_box_pad,
        value_label_box_alpha=value_label_box_alpha,
    )
    st.pyplot(mpl_fig, use_container_width=True)

    # Cache PNG bytes for the download button
    buf = io.BytesIO()
    mpl_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    png_bytes = buf.getvalue()
    plt.close(mpl_fig)
else:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "q1lr3_concept_growth",
                "height": 800,
                "width": 1600,
                "scale": 2,
            },
        },
    )


# =============================================================================
# Data table
# =============================================================================
st.subheader("Concept-growth statistics")

display_df = df[
    [
        "Concept",
        "Before 2020",
        "2020 & After",
        "Absolute increase",
        "Growth rate (%)",
        "Growth label",
        "Trend",
    ]
].copy()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# Downloads
# =============================================================================
st.subheader("Export")

csv_data = dataframe_to_csv(display_df)

if use_matplotlib:
    download_col_1, download_col_2 = st.columns(2)

    with download_col_1:
        st.download_button(
            label="Download statistics as CSV",
            data=csv_data,
            file_name="q1lr3_concept_growth.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col_2:
        st.download_button(
            label="Download chart as PNG",
            data=png_bytes,
            file_name="q1lr3_concept_growth.png",
            mime="image/png",
            use_container_width=True,
        )
else:
    download_col_1, download_col_2 = st.columns(2)

    with download_col_1:
        st.download_button(
            label="Download statistics as CSV",
            data=csv_data,
            file_name="q1lr3_concept_growth.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col_2:
        html_data = fig.to_html(
            include_plotlyjs="cdn",
            full_html=True,
        )

        st.download_button(
            label="Download interactive chart as HTML",
            data=html_data,
            file_name="q1lr3_concept_growth.html",
            mime="text/html",
            use_container_width=True,
        )


# =============================================================================
# Methodology
# =============================================================================
with st.expander("Growth-rate definition and interpretation"):
    st.markdown(
        r"""
The absolute change is:

\[
\text{Absolute increase}
=
\text{Mentions}_{2020+}
-
\text{Mentions}_{<2020}
\]

The relative growth rate is calculated only for concepts with a nonzero
pre-2020 mention count:

\[
\text{Growth rate (\%)}
=
\frac{
\text{Mentions}_{2020+}
-
\text{Mentions}_{<2020}
}{
\text{Mentions}_{<2020}
}
\times 100
\]

For concepts such as `gaussian_heat_source`, where the pre-2020 count is zero,
a percentage rate is not defined. The dashboard reports
`N/A (zero baseline)` and preserves the post-2020 count and absolute increase.
"""
    )
