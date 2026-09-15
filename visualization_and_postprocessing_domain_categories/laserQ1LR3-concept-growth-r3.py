import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Concept Growth Rate | Q1LR3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Default data: Q1LR3
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Plotly palettes
# The menu contains more than 50 named palette options through Plotly colors.
# -----------------------------------------------------------------------------
PALETTES = {
    # Qualitative palettes
    "Alphabet": px.colors.qualitative.Alphabet,
    "Antique": px.colors.qualitative.Antique,
    "Bold": px.colors.qualitative.Bold,
    "D3": px.colors.qualitative.D3,
    "Dark2": px.colors.qualitative.Dark2,
    "Dark24": px.colors.qualitative.Dark24,
    "G10": px.colors.qualitative.G10,
    "Light24": px.colors.qualitative.Light24,
    "Pastel": px.colors.qualitative.Pastel,
    "Pastel1": px.colors.qualitative.Pastel1,
    "Pastel2": px.colors.qualitative.Pastel2,
    "Plotly": px.colors.qualitative.Plotly,
    "Plotly3": px.colors.qualitative.Plotly3,
    "Prism": px.colors.qualitative.Prism,
    "Safe": px.colors.qualitative.Safe,
    "Set1": px.colors.qualitative.Set1,
    "Set2": px.colors.qualitative.Set2,
    "Set3": px.colors.qualitative.Set3,
    "T10": px.colors.qualitative.T10,
    "Vivid": px.colors.qualitative.Vivid,

    # Sequential palettes
    "Aggrnyl": px.colors.sequential.Aggrnyl,
    "Agsunset": px.colors.sequential.Agsunset,
    "Blackbody": px.colors.sequential.Blackbody,
    "Bluered": px.colors.sequential.Bluered,
    "Blues": px.colors.sequential.Blues,
    "Blugrn": px.colors.sequential.Blugrn,
    "Bluyl": px.colors.sequential.Bluyl,
    "Brwnyl": px.colors.sequential.Brwnyl,
    "Bugn": px.colors.sequential.Bugn,
    "Bupu": px.colors.sequential.Bupu,
    "Burg": px.colors.sequential.Burg,
    "Burgyl": px.colors.sequential.Burgyl,
    "Cividis": px.colors.sequential.Cividis,
    "Darkmint": px.colors.sequential.Darkmint,
    "Electric": px.colors.sequential.Electric,
    "Emrld": px.colors.sequential.Emrld,
    "GnBu": px.colors.sequential.GnBu,
    "Greens": px.colors.sequential.Greens,
    "Greys": px.colors.sequential.Greys,
    "Hot": px.colors.sequential.Hot,
    "Inferno": px.colors.sequential.Inferno,
    "Jet": px.colors.sequential.Jet,
    "Magenta": px.colors.sequential.Magenta,
    "Magma": px.colors.sequential.Magma,
    "Mint": px.colors.sequential.Mint,
    "Oranges": px.colors.sequential.Oranges,
    "OrRd": px.colors.sequential.OrRd,
    "Oryel": px.colors.sequential.Oryel,
    "Peach": px.colors.sequential.Peach,
    "Pinkyl": px.colors.sequential.Pinkyl,
    "Plasma": px.colors.sequential.Plasma,
    "Plotly3": px.colors.qualitative.Plotly3,
    "PuBu": px.colors.sequential.PuBu,
    "PuBuGn": px.colors.sequential.PuBuGn,
    "PuRd": px.colors.sequential.PuRd,
    "Purples": px.colors.sequential.Purples,
    "Purp": px.colors.sequential.Purp,
    "RdBu": px.colors.diverging.RdBu,
    "RdPu": px.colors.sequential.RdPu,
    "Reds": px.colors.sequential.Reds,
    "Sunset": px.colors.sequential.Sunset,
    "Sunsetdark": px.colors.sequential.Sunsetdark,
    "Teal": px.colors.sequential.Teal,
    "Tealgrn": px.colors.sequential.Tealgrn,
    "Turbo": px.colors.sequential.Turbo,
    "Viridis": px.colors.sequential.Viridis,
    "YlGn": px.colors.sequential.YlGn,
    "YlGnBu": px.colors.sequential.YlGnBu,
    "YlOrBr": px.colors.sequential.YlOrBr,
    "YlOrRd": px.colors.sequential.YlOrRd,

    # Diverging palettes
    "BrBG": px.colors.diverging.BrBG,
    "Earth": px.colors.diverging.Earth,
    "Fall": px.colors.diverging.Fall,
    "Geyser": px.colors.diverging.Geyser,
    "IceFire": px.colors.diverging.IceFire,
    "Picnic": px.colors.diverging.Picnic,
    "PiYG": px.colors.diverging.PiYG,
    "Portland": px.colors.diverging.Portland,
    "PRGn": px.colors.diverging.PRGn,
    "PuOr": px.colors.diverging.PuOr,
    "RdBu Diverging": px.colors.diverging.RdBu,
    "RdGy": px.colors.diverging.RdGy,
    "RdYlBu": px.colors.diverging.RdYlBu,
    "RdYlGn": px.colors.diverging.RdYlGn,
    "Spectral": px.colors.diverging.Spectral,
    "Tealrose": px.colors.diverging.Tealrose,
    "Temps": px.colors.diverging.Temps,
    "Tropic": px.colors.diverging.Tropic,
    "Twilight": px.colors.diverging.Twilight,
}


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def validate_and_prepare_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """Validate input columns and calculate concept-growth metrics safely."""

    required_columns = {
        "Concept",
        "2020 & After",
        "Before 2020",
    }

    missing_columns = required_columns.difference(input_df.columns)

    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required CSV column(s): {missing_list}"
        )

    df = input_df[
        ["Concept", "2020 & After", "Before 2020"]
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
        np.round(((after - before) / before) * 100, 1),
        np.nan,
    )

    df["Growth label"] = np.where(
        np.isnan(df["Growth rate (%)"]),
        "N/A (zero baseline)",
        df["Growth rate (%)"].map(lambda value: f"{value:.1f}%"),
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


def get_continuous_colorscale(selected_palette: str):
    """Return a valid Plotly continuous colorscale for heatmap use."""

    valid_continuous_scales = [
        "Aggrnyl",
        "Agsunset",
        "Blackbody",
        "Bluered",
        "Blues",
        "Blugrn",
        "Bluyl",
        "Brwnyl",
        "Bugn",
        "Bupu",
        "Burg",
        "Burgyl",
        "Cividis",
        "Darkmint",
        "Electric",
        "Emrld",
        "GnBu",
        "Greens",
        "Greys",
        "Hot",
        "Inferno",
        "Jet",
        "Magma",
        "Mint",
        "Oranges",
        "OrRd",
        "Oryel",
        "Peach",
        "Pinkyl",
        "Plasma",
        "PuBu",
        "PuBuGn",
        "PuRd",
        "Purples",
        "RdBu",
        "RdPu",
        "Reds",
        "Sunset",
        "Sunsetdark",
        "Teal",
        "Tealgrn",
        "Turbo",
        "Viridis",
        "YlGn",
        "YlGnBu",
        "YlOrBr",
        "YlOrRd",
        "BrBG",
        "Earth",
        "Fall",
        "Geyser",
        "IceFire",
        "Picnic",
        "PiYG",
        "Portland",
        "PRGn",
        "PuOr",
        "RdGy",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "Tealrose",
        "Temps",
        "Tropic",
        "Twilight",
    ]

    if selected_palette in valid_continuous_scales:
        return selected_palette

    return "Viridis"


@st.cache_data
def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    """Convert a DataFrame to a downloadable UTF-8 CSV."""

    return dataframe.to_csv(index=False).encode("utf-8")


def format_number(value):
    """Format numeric labels without unnecessary decimal places."""

    if pd.isna(value):
        return "N/A"

    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{value:,.1f}"


# -----------------------------------------------------------------------------
# App title
# -----------------------------------------------------------------------------
st.title("Concept Growth Rate: Q1LR3")

st.markdown(
    """
**Research question:** Analyze the sensitivity of the Gaussian heat source
thermal cycle and subsequent melt-pool penetration depth to variations in
laser power and scan speed.

The dashboard compares concept mentions **before 2020** against
**2020 and after**, quantifying the absolute and relative change in each
concept's occurrence.
"""
)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data and visualization")

    uploaded_file = st.file_uploader(
        "Upload concept-count CSV",
        type=["csv"],
        help=(
            "Required columns: Concept, 2020 & After, Before 2020."
        ),
    )

    st.caption(
        "If no CSV is uploaded, the embedded Q1LR3 concept-count data "
        "are used."
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
        index=0,
    )

    palette_name = st.selectbox(
        "Color palette",
        options=list(PALETTES.keys()),
        index=list(PALETTES.keys()).index("Plotly"),
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
        index=0,
    )

    chart_orientation = st.radio(
        "Bar-chart orientation",
        options=["Vertical", "Horizontal"],
        horizontal=True,
    )

    show_values = st.checkbox(
        "Show data labels",
        value=True,
    )

    use_log_scale = st.checkbox(
        "Use logarithmic scale for mentions",
        value=False,
        help=(
            "Zero-valued bars cannot be represented on a logarithmic axis. "
            "Use a linear axis when zero values must remain visible."
        ),
    )

    custom_colors_enabled = st.checkbox(
        "Customize individual concept colors",
        value=False,
    )


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        source_df = pd.read_csv(uploaded_file)
        data_origin = "Uploaded CSV"
    except Exception as error:
        st.error(f"Could not read the uploaded CSV file: {error}")
        st.stop()
else:
    source_df = DEFAULT_DATA.copy()
    data_origin = "Embedded Q1LR3 dataset"

try:
    df = validate_and_prepare_data(source_df)
except ValueError as error:
    st.error(str(error))
    st.info(
        "Use a CSV with exactly these required fields: "
        "`Concept`, `2020 & After`, and `Before 2020`."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Concept selection
# -----------------------------------------------------------------------------
all_concepts = df["Concept"].tolist()

with st.sidebar:
    selected_concepts = st.multiselect(
        "Concepts to display",
        options=all_concepts,
        default=all_concepts,
    )

if not selected_concepts:
    st.warning("Select at least one concept from the sidebar.")
    st.stop()

df = df[df["Concept"].isin(selected_concepts)].copy()


# -----------------------------------------------------------------------------
# Sorting
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Color configuration
# -----------------------------------------------------------------------------
palette_colors = PALETTES[palette_name]

if custom_colors_enabled:
    with st.sidebar:
        st.divider()
        st.subheader("Individual concept colors")

        concept_colors = {}

        for index, concept in enumerate(df["Concept"].tolist()):
            default_color = palette_colors[index % len(palette_colors)]

            concept_colors[concept] = st.color_picker(
                label=concept,
                value=default_color,
                key=f"color_picker_{concept}",
            )
else:
    concept_colors = {
        concept: palette_colors[index % len(palette_colors)]
        for index, concept in enumerate(df["Concept"].tolist())
    }


# -----------------------------------------------------------------------------
# Summary metrics
# -----------------------------------------------------------------------------
total_before = df["Before 2020"].sum()
total_after = df["2020 & After"].sum()
total_increase = df["Absolute increase"].sum()

if total_before > 0:
    total_growth = (
        (total_after - total_before) / total_before
    ) * 100
    total_growth_label = f"{total_growth:.1f}%"
else:
    total_growth_label = "N/A"

st.caption(f"Data source: {data_origin}")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Displayed concepts",
    f"{len(df):,}",
)

metric_2.metric(
    "Mentions before 2020",
    format_number(total_before),
)

metric_3.metric(
    "Mentions in 2020 & after",
    format_number(total_after),
)

metric_4.metric(
    "Overall concept growth",
    total_growth_label,
    delta=f"+{format_number(total_increase)} mentions",
)


# -----------------------------------------------------------------------------
# Chart construction
# -----------------------------------------------------------------------------
fig = go.Figure()

title_text = (
    "Concept Momentum "
    "(User Defined Split: 2020)"
)

mention_axis_type = "log" if use_log_scale else "linear"


# --- Grouped bar chart -------------------------------------------------------
if chart_type == "Grouped bar chart":

    if chart_orientation == "Vertical":
        fig.add_trace(
            go.Bar(
                name="Before 2020",
                x=df["Concept"],
                y=df["Before 2020"],
                marker_color="#636EFA",
                text=(
                    df["Before 2020"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Before 2020: %{y:,}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Bar(
                name="2020 & After",
                x=df["Concept"],
                y=df["2020 & After"],
                marker_color="#EF553B",
                text=(
                    df["2020 & After"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "2020 & After: %{y:,}"
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
                marker_color="#636EFA",
                text=(
                    df["Before 2020"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Before 2020: %{x:,}"
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
                marker_color="#EF553B",
                text=(
                    df["2020 & After"].map(format_number)
                    if show_values
                    else None
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "2020 & After: %{x:,}"
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


# --- Growth-rate bar chart ---------------------------------------------------
elif chart_type == "Growth-rate bar chart":

    plot_df = df.dropna(
        subset=["Growth rate (%)"]
    ).copy()

    undefined_growth_df = df[
        df["Growth rate (%)"].isna()
    ].copy()

    if plot_df.empty:
        st.warning(
            "Growth rate cannot be calculated because all selected concepts "
            "have a zero pre-2020 baseline."
        )
        st.stop()

    growth_labels = [
        f"{value:.1f}%"
        for value in plot_df["Growth rate (%)"]
    ]

    if chart_orientation == "Vertical":
        fig.add_trace(
            go.Bar(
                x=plot_df["Concept"],
                y=plot_df["Growth rate (%)"],
                marker_color=[
                    concept_colors[concept]
                    for concept in plot_df["Concept"]
                ],
                text=growth_labels if show_values else None,
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Growth rate: %{y:.1f}%"
                    "<extra></extra>"
                ),
                name="Growth rate",
            )
        )

        fig.update_xaxes(
            title="Concept",
            tickangle=-35,
            categoryorder="array",
            categoryarray=plot_df["Concept"].tolist(),
        )

        fig.update_yaxes(
            title="Growth rate (%)",
        )

    else:
        fig.add_trace(
            go.Bar(
                y=plot_df["Concept"],
                x=plot_df["Growth rate (%)"],
                orientation="h",
                marker_color=[
                    concept_colors[concept]
                    for concept in plot_df["Concept"]
                ],
                text=growth_labels if show_values else None,
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Growth rate: %{x:.1f}%"
                    "<extra></extra>"
                ),
                name="Growth rate",
            )
        )

        fig.update_xaxes(
            title="Growth rate (%)",
        )

        fig.update_yaxes(
            title="Concept",
            categoryorder="array",
            categoryarray=plot_df["Concept"].tolist()[::-1],
        )

    if not undefined_growth_df.empty:
        undefined_names = ", ".join(
            undefined_growth_df["Concept"].tolist()
        )

        st.info(
            "Percentage growth is undefined for: "
            f"{undefined_names}. Their pre-2020 mention count is zero; "
            "use absolute increase to interpret their emergence."
        )


# --- Absolute-increase bar chart --------------------------------------------
elif chart_type == "Absolute-increase bar chart":

    increase_labels = [
        f"{value:+,.0f}"
        for value in df["Absolute increase"]
    ]

    increase_colors = [
        "#2CA02C" if value > 0
        else "#D62728" if value < 0
        else "#7F7F7F"
        for value in df["Absolute increase"]
    ]

    if chart_orientation == "Vertical":
        fig.add_trace(
            go.Bar(
                x=df["Concept"],
                y=df["Absolute increase"],
                marker_color=increase_colors,
                text=increase_labels if show_values else None,
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Absolute increase: %{y:+,}"
                    "<extra></extra>"
                ),
                name="Absolute increase",
            )
        )

        fig.update_xaxes(
            title="Concept",
            tickangle=-35,
            categoryorder="array",
            categoryarray=df["Concept"].tolist(),
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
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Absolute increase: %{x:+,}"
                    "<extra></extra>"
                ),
                name="Absolute increase",
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


# --- Dumbbell chart ----------------------------------------------------------
elif chart_type == "Dumbbell chart":

    for _, row in df.iterrows():
        concept = row["Concept"]
        before_value = row["Before 2020"]
        after_value = row["2020 & After"]
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
                yshift=15,
                font=dict(size=11),
            )

            fig.add_annotation(
                x=after_value,
                y=concept,
                text=format_number(after_value),
                showarrow=False,
                yshift=15,
                font=dict(size=11),
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

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color="#636EFA",
                size=10,
            ),
            name="Before 2020",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color="#EF553B",
                size=10,
            ),
            name="2020 & After",
        )
    )


# --- Heatmap -----------------------------------------------------------------
elif chart_type == "Heatmap":

    heatmap_values = df[
        ["Before 2020", "2020 & After"]
    ].to_numpy(dtype=float)

    heatmap_text = (
        np.vectorize(format_number)(heatmap_values)
        if show_values
        else None
    )

    fig.add_trace(
        go.Heatmap(
            z=heatmap_values,
            x=["Before 2020", "2020 & After"],
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


# --- Treemap -----------------------------------------------------------------
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
                ],
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


# -----------------------------------------------------------------------------
# Shared chart layout
# -----------------------------------------------------------------------------
fig.update_layout(
    title=dict(
        text=title_text,
        x=0.01,
        xanchor="left",
    ),
    template="plotly_white",
    height=680,
    margin=dict(
        l=30,
        r=30,
        t=80,
        b=150,
    ),
    legend=dict(
        title="Period",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.0,
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=13,
    ),
)

if chart_type in [
    "Grouped bar chart",
    "Growth-rate bar chart",
    "Absolute-increase bar chart",
]:
    fig.update_layout(
        uniformtext_minsize=9,
        uniformtext_mode="hide",
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
            "width": 1500,
            "scale": 2,
        },
    },
)


# -----------------------------------------------------------------------------
# Data table
# -----------------------------------------------------------------------------
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
    column_config={
        "Concept": st.column_config.TextColumn(
            "Concept",
        ),
        "Before 2020": st.column_config.NumberColumn(
            "Before 2020",
            format="%d",
        ),
        "2020 & After": st.column_config.NumberColumn(
            "2020 & After",
            format="%d",
        ),
        "Absolute increase": st.column_config.NumberColumn(
            "Absolute increase",
            format="%+d",
        ),
        "Growth rate (%)": st.column_config.NumberColumn(
            "Growth rate (%)",
            format="%.1f%%",
        ),
        "Growth label": st.column_config.TextColumn(
            "Growth interpretation",
        ),
        "Trend": st.column_config.TextColumn(
            "Trend",
        ),
    },
)


# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------
st.subheader("Export")

download_col_1, download_col_2 = st.columns(2)

csv_data = convert_df_to_csv(display_df)

with download_col_1:
    st.download_button(
        label="Download concept-growth data as CSV",
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


# -----------------------------------------------------------------------------
# Method note
# -----------------------------------------------------------------------------
with st.expander("Methodological note"):
    st.markdown(
        """
- **Absolute increase** is calculated as:

  \[
  \\text{Absolute increase}
  =
  \\text{Mentions}_{2020+}
  -
  \\text{Mentions}_{<2020}
  \]

- **Growth rate** is calculated only where pre-2020 mentions are greater
  than zero:

  \[
  \\text{Growth rate (\\%)}
  =
  \\frac{
      \\text{Mentions}_{2020+}
      -
      \\text{Mentions}_{<2020}
  }{
      \\text{Mentions}_{<2020}
  }
  \\times 100
  \]

- Where the pre-2020 count equals zero, percentage growth is mathematically
  undefined. The dashboard displays `N/A (zero baseline)` rather than
  reporting an artificial infinite percentage. Use the absolute increase and
  post-2020 mention count to assess such emerging concepts.
"""
    )
