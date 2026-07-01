import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from io import BytesIO

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Data Loading Functions
# ==========================================
@st.cache_data(show_spinner="Scanning job directory for Gibbs energy files...")
def discover_gibbs_files(job_dir: str) -> list:
    """Discover all Gibbs_XXXK.csv files in the job directory."""
    if not os.path.isdir(job_dir):
        return []
    
    pattern = os.path.join(job_dir, "Gibbs_*K.csv")
    files = glob.glob(pattern)
    
    # Sort by temperature
    def extract_temp(f):
        match = re.search(r'Gibbs_(\d+)K\.csv', os.path.basename(f))
        return int(match.group(1)) if match else 0
    
    return sorted(files, key=extract_temp)


@st.cache_data(show_spinner="Loading and processing thermodynamic datasets...")
def load_gibbs_data(job_dir: str) -> pd.DataFrame:
    """Load all Gibbs CSV files and compute derived quantities."""
    files = discover_gibbs_files(job_dir)
    if not files:
        return pd.DataFrame()
    
    all_data = []
    temps_found = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        match = re.search(r'Gibbs_(\d+)K\.csv', filename)
        if not match:
            continue
        
        T = int(match.group(1))
        status_text.text(f"Loading {filename} ({T} K)...")
        
        try:
            df = pd.read_csv(filepath)
            df['Temperature_K'] = T
            all_data.append(df)
            temps_found.append(T)
        except Exception as e:
            st.warning(f"Could not read {filename}: {e}")
        
        progress_bar.progress((i + 1) / len(files))
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        return pd.DataFrame()
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Identify composition columns (mole fractions)
    comp_cols = [c for c in ['Co', 'Cr', 'Fe', 'Ni'] if c in df_all.columns]
    
    # Identify Gibbs energy columns
    g_liq_col = None
    g_fcc_col = None
    for c in df_all.columns:
        cl = c.lower()
        if 'liq' in cl and ('gibbs' in cl or 'g_' in cl or cl == 'g_liq'):
            g_liq_col = c
        if 'fcc' in cl and ('gibbs' in cl or 'g_' in cl or cl == 'g_fcc'):
            g_fcc_col = c
    
    # Fallback: try exact names
    if g_liq_col is None and 'G_LIQ' in df_all.columns:
        g_liq_col = 'G_LIQ'
    if g_fcc_col is None and 'G_FCC' in df_all.columns:
        g_fcc_col = 'G_FCC'
    
    if g_liq_col is None or g_fcc_col is None:
        st.error(f"Could not identify Gibbs energy columns. Found columns: {list(df_all.columns)}")
        return pd.DataFrame()
    
    # Standardize column names
    df_all = df_all.rename(columns={g_liq_col: 'G_LIQ', g_fcc_col: 'G_FCC'})
    
    # Compute derived quantities
    df_all['Delta_G'] = df_all['G_FCC'] - df_all['G_LIQ']
    df_all['Stable_Phase'] = np.where(df_all['Delta_G'] < 0, 'FCC', 'LIQUID')
    df_all['Abs_Driving_Force'] = np.abs(df_all['Delta_G'])
    
    # Store metadata
    df_all.attrs['comp_cols'] = comp_cols
    df_all.attrs['temperatures'] = sorted(temps_found)
    df_all.attrs['g_liq_col'] = 'G_LIQ'
    df_all.attrs['g_fcc_col'] = 'G_FCC'
    
    return df_all


# ==========================================
# Plotting Functions
# ==========================================
def plot_driving_force_landscape(df, T_ref, comps, n_components=2):
    """Figure A: PCA-reduced driving force landscape at a specific temperature."""
    df_ref = df[df['Temperature_K'] == T_ref].copy()
    
    if df_ref.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data at {T_ref} K", ha='center', va='center')
        return fig
    
    # PCA on composition space
    pca = PCA(n_components=min(n_components, len(comps)))
    comps_pca = pca.fit_transform(df_ref[comps])
    df_ref['PC1'] = comps_pca[:, 0]
    df_ref['PC2'] = comps_pca[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        df_ref['PC1'], df_ref['PC2'],
        c=df_ref['Delta_G'],
        cmap='RdBu_r',
        alpha=0.7,
        s=25,
        edgecolors='none',
        vmin=df_ref['Delta_G'].quantile(0.02),
        vmax=df_ref['Delta_G'].quantile(0.98)
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label(r'$\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$ [J/mol]', fontsize=13, fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=13, fontweight='bold')
    ax.set_title(f'Thermodynamic Driving Force Landscape at {T_ref} K\n(PCA-Reduced {"-".join(comps)} Composition Space)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add composition annotations
    for i, comp in enumerate(comps):
        pure_comp = df_ref[df_ref[comp] == df_ref[comp].max()]
        if not pure_comp.empty:
            ax.annotate(comp, (pure_comp['PC1'].iloc[0], pure_comp['PC2'].iloc[0]),
                       fontsize=12, fontweight='bold', ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_gibbs_vs_temperature(df, target_comps, comps):
    """Figure B: Scaled Gibbs energy vs temperature for selected compositions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left panel: G_LIQ and G_FCC curves
    ax1 = axes[0]
    for name, target in target_comps.items():
        subset = df.copy()
        subset['dist'] = np.sum((subset[comps] - np.array(target))**2, axis=1)
        idx = subset.groupby('Temperature_K')['dist'].idxmin()
        closest = subset.loc[idx].sort_values('Temperature_K')
        
        ax1.plot(closest['Temperature_K'], closest['G_LIQ'], '--', linewidth=2.5,
                label=f'{name} ($G_{{\\mathrm{{LIQ}}}}$)', alpha=0.85)
        ax1.plot(closest['Temperature_K'], closest['G_FCC'], '-', linewidth=2.5,
                label=f'{name} ($G_{{\\mathrm{{FCC}}}}$)', alpha=0.85)
    
    ax1.set_xlabel('Temperature [K]', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Gibbs Free Energy [J/mol]', fontsize=13, fontweight='bold')
    ax1.set_title('End-Member Phase Gibbs Energies vs Temperature', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=1, loc='best', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right panel: Driving force Delta_G
    ax2 = axes[1]
    for name, target in target_comps.items():
        subset = df.copy()
        subset['dist'] = np.sum((subset[comps] - np.array(target))**2, axis=1)
        idx = subset.groupby('Temperature_K')['dist'].idxmin()
        closest = subset.loc[idx].sort_values('Temperature_K')
        
        ax2.plot(closest['Temperature_K'], closest['Delta_G'], '-', linewidth=2.5,
                label=f'{name}', alpha=0.85)
        
        # Mark equilibrium temperature (Delta_G = 0 crossing)
        crossings = closest[(closest['Delta_G'].shift(1) * closest['Delta_G']) < 0]
        if not crossings.empty:
            for _, row in crossings.iterrows():
                ax2.axvline(row['Temperature_K'], color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\Delta G = 0$')
    ax2.set_xlabel('Temperature [K]', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'$\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$ [J/mol]', fontsize=13, fontweight='bold')
    ax2.set_title('Solidification Driving Force vs Temperature', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_phase_stability_pairplot(df, T_high, comps, sample_size=2000):
    """Figure C: Phase stability composition landscape (pairplot)."""
    df_high = df[df['Temperature_K'] == T_high].copy()
    
    if df_high.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data at {T_high} K", ha='center', va='center')
        return fig
    
    # Sample for performance
    n_sample = min(sample_size, len(df_high))
    df_sample = df_high.sample(n=n_sample, random_state=42)
    
    g = sns.pairplot(
        df_sample,
        vars=comps,
        hue='Stable_Phase',
        palette={'FCC': '#2563EB', 'LIQUID': '#DC2626'},
        plot_kws={'alpha': 0.55, 's': 18, 'edgecolors': 'none'},
        diag_kws={'fill': True, 'alpha': 0.6, 'linewidth': 0},
        corner=False
    )
    
    g.fig.suptitle(
        f'Phase Stability Landscape at {T_high} K\n(FCC vs LIQUID in {"-".join(comps)} Mole Fraction Space)',
        y=1.02, fontsize=14, fontweight='bold'
    )
    
    # Adjust legend
    g.add_legend()
    if g._legend:
        g._legend.set_title('Stable Phase')
    
    plt.tight_layout()
    return g.fig


def plot_driving_force_evolution(df, comps):
    """Figure D: Temperature evolution of driving force distribution."""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Compute statistics per temperature
    temp_stats = df.groupby('Temperature_K')['Delta_G'].agg(
        ['mean', 'std', 'median', 'min', 'max', 'count']
    ).reset_index()
    
    # Percentiles
    percentiles = df.groupby('Temperature_K')['Delta_G'].quantile([0.05, 0.25, 0.75, 0.95]).unstack()
    percentiles.columns = ['p05', 'p25', 'p75', 'p95']
    temp_stats = temp_stats.merge(percentiles, left_on='Temperature_K', right_index=True)
    
    # Plot shaded regions
    ax.fill_between(temp_stats['Temperature_K'],
                    temp_stats['p05'], temp_stats['p95'],
                    color='#94A3B8', alpha=0.2, label='5th–95th percentile')
    ax.fill_between(temp_stats['Temperature_K'],
                    temp_stats['p25'], temp_stats['p75'],
                    color='#64748B', alpha=0.3, label='25th–75th percentile')
    ax.fill_between(temp_stats['Temperature_K'],
                    temp_stats['mean'] - temp_stats['std'],
                    temp_stats['mean'] + temp_stats['std'],
                    color='#3B82F6', alpha=0.25, label=r'$\mu \pm \sigma$')
    
    ax.plot(temp_stats['Temperature_K'], temp_stats['mean'],
            'b-', linewidth=2.5, label='Mean', zorder=5)
    ax.plot(temp_stats['Temperature_K'], temp_stats['median'],
            'r--', linewidth=2, label='Median', zorder=5)
    
    ax.axhline(0, color='black', linestyle=':', linewidth=2, label=r'$\Delta G = 0$ (Equilibrium)', zorder=4)
    
    ax.set_xlabel('Temperature [K]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\Delta G = G_{\mathrm{FCC}} - G_{\mathrm{LIQ}}$ [J/mol]', fontsize=14, fontweight='bold')
    ax.set_title('Temperature Evolution of Solidification Driving Force\n(Across the Entire Co-Cr-Fe-Ni Composition Space)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_3d_composition_triangle(df, T_ref, comps):
    """Bonus: Ternary/quaternary composition visualization."""
    df_ref = df[df['Temperature_K'] == T_ref].copy()
    if df_ref.empty or len(comps) < 3:
        return None
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use first 3 components for 3D scatter
    c1, c2, c3 = comps[0], comps[1], comps[2]
    
    scatter = ax.scatter(
        df_ref[c1], df_ref[c2], df_ref[c3],
        c=df_ref['Delta_G'],
        cmap='RdBu_r',
        alpha=0.6,
        s=20,
        edgecolors='none'
    )
    
    ax.set_xlabel(f'{c1} mole fraction', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{c2} mole fraction', fontsize=12, fontweight='bold')
    ax.set_zlabel(f'{c3} mole fraction', fontsize=12, fontweight='bold')
    ax.set_title(f'3D Composition Space at {T_ref} K\n(Color = Driving Force)', fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, shrink=0.6, label=r'$\Delta G$ [J/mol]')
    plt.tight_layout()
    return fig


# ==========================================
# Main App
# ==========================================
def main():
    # Header
    st.markdown('<div class="main-header">🔥 Co-Cr-Fe-Ni Gibbs Energy Landscape Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Thermodynamic Driving Force Visualization for Phase-Field Modeling of High-Entropy Alloys</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # ──────────────────────────────────────────────
    # IMPROVED FOLDER DETECTION
    # ──────────────────────────────────────────────
    # 1. Try CLI argument first (if any)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job-dir", default="")
    args, _ = parser.parse_known_args()
    
    # 2. Environment variable
    env_dir = os.environ.get('JOB_DIR', '')
    
    # 3. Automatic detection: thermodynamic_dataset next to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    auto_dir = os.path.join(script_dir, "thermodynamic_dataset")
    if os.path.isdir(auto_dir):
        auto_dir = auto_dir  # use it
    else:
        auto_dir = os.getcwd()  # fallback to current working directory
    
    # Priority: CLI arg > env var > auto_dir
    default_dir = args.job_dir or env_dir or auto_dir
    
    # Ensure it's an absolute path
    default_dir = os.path.abspath(default_dir)
    
    job_dir = st.sidebar.text_input(
        "Job Directory Path",
        value=default_dir,
        help="Path to the directory containing Gibbs_XXXK.csv files (300 K to 3300 K, 100 K intervals)"
    )
    
    # Show detected folder if auto-detected
    if job_dir == auto_dir and os.path.isdir(auto_dir):
        st.sidebar.info(f"📁 Auto-detected: `{auto_dir}`")
    
    # Check directory
    if not os.path.isdir(job_dir):
        st.error(f"❌ Directory not found: `{job_dir}`")
        st.info("Please enter a valid path containing `Gibbs_XXXK.csv` files.")
        st.stop()
    
    # Discover files
    gibbs_files = discover_gibbs_files(job_dir)
    if not gibbs_files:
        st.error(f"❌ No `Gibbs_XXXK.csv` files found in `{job_dir}`")
        st.info("Expected files: `Gibbs_300K.csv`, `Gibbs_400K.csv`, ..., `Gibbs_3300K.csv`")
        st.stop()
    
    # Load data
    df = load_gibbs_data(job_dir)
    if df.empty:
        st.error("❌ Could not load any data from the CSV files.")
        st.stop()
    
    comps = df.attrs.get('comp_cols', ['Co', 'Cr', 'Fe', 'Ni'])
    temperatures = df.attrs.get('temperatures', [])
    
    # Sidebar controls
    st.sidebar.success(f"✅ Loaded {len(gibbs_files)} temperature files")
    st.sidebar.info(f"Temperature range: {min(temperatures)} K – {max(temperatures)} K")
    st.sidebar.info(f"Total data points: {len(df):,}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Plot Parameters")
    
    # Temperature selectors
    T_ref = st.sidebar.select_slider(
        "Reference Temperature (Fig A)",
        options=temperatures,
        value=min(1800, max(temperatures)) if 1800 in temperatures else temperatures[len(temperatures)//2],
        help="Temperature for the driving force landscape"
    )
    
    T_high = st.sidebar.select_slider(
        "High Temperature (Fig C)",
        options=temperatures,
        value=min(2000, max(temperatures)) if 2000 in temperatures else temperatures[-3],
        help="Temperature for phase stability pairplot"
    )
    
    sample_size = st.sidebar.slider(
        "Pairplot Sample Size",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500,
        help="Number of points to display in the pairplot"
    )
    
    # Composition targets
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Target Compositions")
    
    use_custom = st.sidebar.checkbox("Use custom compositions", value=False)
    
    if use_custom:
        st.sidebar.markdown("**Equiatomic:**")
        eq_co = st.sidebar.slider("Co", 0.0, 1.0, 0.25, 0.05, key='eq_co')
        eq_cr = st.sidebar.slider("Cr", 0.0, 1.0, 0.25, 0.05, key='eq_cr')
        eq_fe = st.sidebar.slider("Fe", 0.0, 1.0, 0.25, 0.05, key='eq_fe')
        eq_ni = st.sidebar.slider("Ni", 0.0, 1.0, 0.25, 0.05, key='eq_ni')
        
        target_comps = {
            'Custom-1': [eq_co, eq_cr, eq_fe, eq_ni],
            'Co-rich': [0.70, 0.10, 0.10, 0.10],
            'Ni-rich': [0.10, 0.10, 0.10, 0.70],
        }
    else:
        target_comps = {
            'Equiatomic': [0.25, 0.25, 0.25, 0.25],
            'Co-rich': [0.70, 0.10, 0.10, 0.10],
            'Cr-rich': [0.10, 0.70, 0.10, 0.10],
            'Fe-rich': [0.10, 0.10, 0.70, 0.10],
            'Ni-rich': [0.10, 0.10, 0.10, 0.70],
        }
    
    # Main content
    st.markdown("---")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temperature Files", len(gibbs_files))
    with col2:
        st.metric("Total Data Points", f"{len(df):,}")
    with col3:
        st.metric("Temperature Range", f"{min(temperatures)}–{max(temperatures)} K")
    with col4:
        st.metric("Composition Elements", len(comps))
    
    st.markdown("---")
    
    # Tabs for figures
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Fig A: Driving Force Landscape",
        "📈 Fig B: Gibbs Energy vs T",
        "🔬 Fig C: Phase Stability Pairplot",
        "🌡️ Fig D: Driving Force Evolution",
        "📋 Data Explorer"
    ])
    
    with tab1:
        st.subheader("Thermodynamic Driving Force Landscape (PCA-Reduced)")
        st.markdown("""
        This figure projects the 4D composition space (Co-Cr-Fe-Ni) onto 2 principal components 
        and colors each point by the solidification driving force $\\Delta G = G_{\\mathrm{FCC}} - G_{\\mathrm{LIQ}}$.
        
        - **Blue regions**: FCC is thermodynamically favored ($\\Delta G < 0$)
        - **Red regions**: Liquid is thermodynamically favored ($\\Delta G > 0$)
        """)
        
        fig_a = plot_driving_force_landscape(df, T_ref, comps)
        st.pyplot(fig_a)
        plt.close(fig_a)
    
    with tab2:
        st.subheader("Scaled Gibbs Energy of End-Member Phases vs Temperature")
        st.markdown("""
        This figure shows the Gibbs free energy curves for the LIQUID and FCC phases 
        as a function of temperature for selected alloy compositions.
        
        - **Dashed lines**: $G_{\\mathrm{LIQ}}$ (liquid phase)
        - **Solid lines**: $G_{\\mathrm{FCC}}$ (solid FCC phase)
        - **Crossover points**: Local equilibrium temperatures $T_{eq}$
        """)
        
        fig_b = plot_gibbs_vs_temperature(df, target_comps, comps)
        st.pyplot(fig_b)
        plt.close(fig_b)
    
    with tab3:
        st.subheader("Phase Stability Composition Landscape")
        st.markdown(f"""
        This pairplot shows pairwise projections of the composition space at **{T_high} K**, 
        colored by which phase is thermodynamically stable.
        
        - 🔵 **Blue**: FCC stable ($G_{\\mathrm{{FCC}}} < G_{\\mathrm{{LIQ}}}$)
        - 🔴 **Red**: LIQUID stable ($G_{\\mathrm{{LIQ}}} < G_{\\mathrm{{FCC}}}$)
        
        *Showing {sample_size:,} randomly sampled compositions.*
        """)
        
        with st.spinner("Generating pairplot (this may take a moment)..."):
            fig_c = plot_phase_stability_pairplot(df, T_high, comps, sample_size)
            st.pyplot(fig_c)
            plt.close(fig_c)
    
    with tab4:
        st.subheader("Temperature Evolution of Solidification Driving Force")
        st.markdown("""
        This figure shows how the driving force $\\Delta G$ evolves across the entire 
        composition space as temperature changes.
        
        - **Shaded regions**: Distribution of $\\Delta G$ across all compositions
        - **Blue line**: Mean driving force
        - **Red dashed**: Median driving force
        - **Black dotted**: $\\Delta G = 0$ (equilibrium)
        """)
        
        fig_d = plot_driving_force_evolution(df, comps)
        st.pyplot(fig_d)
        plt.close(fig_d)
    
    with tab5:
        st.subheader("Data Explorer")
        
        # Temperature filter
        selected_temps = st.multiselect(
            "Select Temperatures to Display",
            options=temperatures,
            default=[temperatures[len(temperatures)//3], temperatures[len(temperatures)//2], temperatures[-1]] if len(temperatures) >= 3 else temperatures,
            help="Choose which temperature slices to show in the table"
        )
        
        if selected_temps:
            df_display = df[df['Temperature_K'].isin(selected_temps)].copy()
            st.dataframe(df_display, use_container_width=True, height=500)
            
            # Download button
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"gibbs_data_filtered_T{selected_temps[0]}-{selected_temps[-1]}K.csv",
                mime="text/csv"
            )
        
        # Raw data statistics
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"App running on job directory: `{job_dir}` | Data loaded: {len(df):,} points across {len(temperatures)} temperatures")


if __name__ == "__main__":
    main()
