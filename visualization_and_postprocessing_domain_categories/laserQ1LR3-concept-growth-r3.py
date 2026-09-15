import re

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
# Chart
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
# Shared layout
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
