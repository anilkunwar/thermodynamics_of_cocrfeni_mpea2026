import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Concept Growth Rate", page_icon="📈", layout="wide")

DEFAULT = pd.DataFrame({
    "Concept": ["beam_diameter", "cocrfeni", "gaussian_heat_source", "grain_size", "hea", "laser_power", "melt_pool", "porosity", "scan_speed", "thermal_cycle", "thermal_gradient"],
    "2020 & After": [9, 502, 4, 269, 1724, 403, 93, 221, 120, 13, 184],
    "Before 2020": [4, 54, 0, 18, 167, 42, 4, 19, 17, 6, 17],
})

PALETTES = {
    "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "G10": px.colors.qualitative.G10,
    "T10": px.colors.qualitative.T10,
    "Set1": px.colors.qualitative.Set1,
    "Set2": px.colors.qualitative.Set2,
    "Set3": px.colors.qualitative.Set3,
    "Pastel": px.colors.qualitative.Pastel,
    "Dark24": px.colors.qualitative.Dark24,
    "Light24": px.colors.qualitative.Light24,
    "Alphabet": px.colors.qualitative.Alphabet,
    "Vivid": px.colors.qualitative.Vivid,
    "Prism": px.colors.qualitative.Prism,
    "Safe": px.colors.qualitative.Safe,
    "Bold": px.colors.qualitative.Bold,
    "Turbo": px.colors.sequential.Turbo,
    "Viridis": px.colors.sequential.Viridis,
    "Plasma": px.colors.sequential.Plasma,
    "Inferno": px.colors.sequential.Inferno,
    "Magma": px.colors.sequential.Magma,
    "Cividis": px.colors.sequential.Cividis,
    "Rainbow": px.colors.sequential.Rainbow,
    "Jet": px.colors.sequential.Jet,
    "Blues": px.colors.sequential.Blues,
    "Greens": px.colors.sequential.Greens,
    "Greys": px.colors.sequential.Greys,
    "Oranges": px.colors.sequential.Oranges,
    "Purples": px.colors.sequential.Purples,
    "Reds": px.colors.sequential.Reds,
    "YlGnBu": px.colors.sequential.YlGnBu,
    "YlOrRd": px.colors.sequential.YlOrRd,
    "Spectral": px.colors.diverging.Spectral,
    "RdBu": px.colors.diverging.RdBu,
    "RdYlBu": px.colors.diverging.RdYlBu,
    "RdYlGn": px.colors.diverging.RdYlGn,
    "BrBG": px.colors.diverging.BrBG,
    "PiYG": px.colors.diverging.PiYG,
    "PRGn": px.colors.diverging.PRGn,
    "PuOr": px.colors.diverging.PuOr,
    "RdGy": px.colors.diverging.RdGy,
    "Earth": px.colors.cyclical.Edge,
}

st.title("Concept Growth Rate")
st.caption('Query Q1LR3 — “Analyze the sensitivity of the Gaussian heat source thermal cycle and subsequent melt pool penetration depth to variations in laser power and scan speed.”')

with st.sidebar:
    st.header("Visualization controls")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        required = {"Concept", "2020 & After", "Before 2020"}
        if not required.issubset(raw.columns):
            st.error("CSV must contain: Concept, 2020 & After, Before 2020")
            df = DEFAULT.copy()
        else:
            df = raw[["Concept", "2020 & After", "Before 2020"]].copy()
    else:
        df = DEFAULT.copy()

    chart_type = st.selectbox("Chart type", ["Grouped bars", "Growth-rate bars", "Dumbbell", "Heatmap", "Treemap"])
    palette_name = st.selectbox("Color palette (40+ choices)", list(PALETTES), index=0)
    orientation = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
    sort_by = st.selectbox("Sort concepts by", ["Original order", "2020 & After", "Before 2020", "Growth rate", "Absolute increase"])
    selected = st.multiselect("Concepts to display", df["Concept"].tolist(), default=df["Concept"].tolist())
    show_values = st.checkbox("Show values", True)
    log_axis = st.checkbox("Logarithmic metric axis", False)
    custom_colors = st.checkbox("Customize concept colors", False)

if not selected:
    st.warning("Select at least one concept in the sidebar.")
    st.stop()

df = df[df["Concept"].isin(selected)].copy()
df["2020 & After"] = pd.to_numeric(df["2020 & After"], errors="coerce").fillna(0)
df["Before 2020"] = pd.to_numeric(df["Before 2020"], errors="coerce").fillna(0)
df["Absolute increase"] = df["2020 & After"] - df["Before 2020"]
df["Growth rate (%)"] = ((df["2020 & After"] - df["Before 2020"]) / df["Before 2020"].replace(0, pd.NA) * 100).round(1)
df["Growth rate (%)"] = df["Growth rate (%)"].fillna(pd.NA)

if sort_by != "Original order":
    key = {"2020 & After": "2020 & After", "Before 2020": "Before 2020", "Growth rate": "Growth rate (%)", "Absolute increase": "Absolute increase"}[sort_by]
    df = df.sort_values(key, ascending=False, na_position="last")

colors = PALETTES[palette_name]
if custom_colors:
    concept_colors = {c: st.sidebar.color_picker(c, value=colors[i % len(colors)], key=f"color_{c}") for i, c in enumerate(df["Concept"])}
else:
    concept_colors = {c: colors[i % len(colors)] for i, c in enumerate(df["Concept"])}

fig = go.Figure()
if chart_type == "Grouped bars":
    for period in ["Before 2020", "2020 & After"]:
        fig.add_trace(go.Bar(name=period, x=df["Concept"], y=df[period], text=df[period] if show_values else None, textposition="outside", marker_color="#636EFA" if period == "Before 2020" else "#EF553B"))
    fig.update_layout(barmode="group")
elif chart_type == "Growth-rate bars":
    values = df["Growth rate (%)"].astype("Float64").tolist()
    fig.add_trace(go.Bar(x=df["Concept"], y=values, text=["N/A" if pd.isna(v) else f"{v:.1f}%" for v in values] if show_values else None, textposition="outside", marker_color=[concept_colors[c] for c in df["Concept"]], name="Growth rate"))
    fig.update_yaxes(title="Growth rate (%)")
elif chart_type == "Dumbbell":
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(x=[r["Before 2020"], r["2020 & After"]], y=[r["Concept"], r["Concept"]], mode="lines+markers+text", text=[r["Before 2020"], r["2020 & After"]] if show_values else None, textposition="top center", line=dict(color=concept_colors[r["Concept"]], width=4), marker=dict(size=11), showlegend=False))
    fig.update_yaxes(title="Concept")
elif chart_type == "Heatmap":
    z = df[["Before 2020", "2020 & After"]].values
    fig.add_trace(go.Heatmap(z=z, x=["Before 2020", "2020 & After"], y=df["Concept"], colorscale=palette_name if palette_name in ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Greens", "Oranges", "Purples", "Reds", "Spectral", "RdBu"] else "Viridis", text=z if show_values else None, texttemplate="%{text}" if show_values else None, colorbar_title="Mentions"))
else:
    fig.add_trace(go.Treemap(labels=df["Concept"], parents=[""] * len(df), values=df["2020 & After"], marker_colors=[concept_colors[c] for c in df["Concept"]], textinfo="label+value"))

fig.update_layout(template="plotly_white", height=650, margin=dict(l=20, r=20, t=70, b=180), legend_title_text="Period", hovermode="closest")
if chart_type not in ["Treemap", "Heatmap"]:
    fig.update_xaxes(title="Concept", tickangle=-45 if orientation == "Vertical" else 0)
    fig.update_yaxes(title="Total mentions", type="log" if log_axis else "linear")
    if orientation == "Horizontal" and chart_type in ["Grouped bars", "Growth-rate bars"]:
        fig.update_layout(xaxis_title="Total mentions", yaxis_title="Concept")
        for trace in fig.data:
            trace.x, trace.y = trace.y, trace.x
        fig.update_yaxes(type="category")

st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Concepts", len(df))
c2.metric("Mentions before 2020", f"{df['Before 2020'].sum():,.0f}")
c3.metric("Mentions 2020 & after", f"{df['2020 & After'].sum():,.0f}")
c4.metric("Absolute increase", f"{df['Absolute increase'].sum():,.0f}")

st.subheader("Concept growth table")
st.dataframe(df[["Concept", "Before 2020", "2020 & After", "Absolute increase", "Growth rate (%)"]], use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8")
html = fig.to_html(include_plotlyjs="cdn")
st.download_button("Download processed CSV", csv, "concept_growth_q1lr3.csv", "text/csv")
st.download_button("Download interactive HTML", html, "concept_growth_q1lr3.html", "text/html")
