#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoCrFeNi MPEA Quantitative Descriptor Graph v6.1 (AgNPs Port + Batch Mode + OOM Hotfix)
=======================================================================================
Multi-level reasoning concept graph for numerical/quantitative description of CoCrFeNi MPEAs.
Focus: Thermodynamic, Compositional, and Mechanical Descriptors.

This is a TRUE architectural port of the AgNP-Sustainability-ConceptGraph codebase,
preserving every memory-safe pattern, visualization pattern, and session-state management
pattern from the working AgNPs code. The domain ontology and extraction patterns have been
replaced with those for CoCrFeNi MPEA quantitative descriptors.

NEW in v6.0 — BATCH PROCESSING MODE (Streamlit Cloud ≤ 1 GB RAM):
- Sidebar toggle "Enable batch processing" switches the analysis pipeline
  into a memory-efficient incremental mode.
- Documents are processed in small batches (default 1000 docs), the concept
  graph is merged incrementally, and memory is released after every batch.
- GNN training runs once on the final merged graph (configurable epochs).
- Works with the full 4707-document dataset on the Streamlit Cloud free tier.

NEW in v6.1 — MEMORY-CRASH HOTFIX (OOM at batch 2/5, > 1 GB RSS):
Five unbounded accumulators used to compound across batches and cross the
Streamlit Cloud 1 GB limit by batch ~2. v6.1 caps every one of them:
- Patch 1: batch state no longer stores EVERY abstract. `all_texts` is now
  a dict {doc_idx: text} that only keeps documents containing at least one
  concept at/above MIN_CONCEPT_FREQ (plus a per-text character cap).
  A `docs_processed` counter keeps the UI honest.
- Patch 2: `EnhancedConceptExtractor.concept_contexts` (dead code that
  stored a 200-char snippet per concept per doc) is removed; per-doc
  `document_concepts` accumulation can be disabled and IS disabled in
  batch mode (it was leak #6 hiding behind the same pattern).
- Patch 3: `AdvancedConceptResolver.embedding_cache` and
  `resolution_cache` are now bounded LRU-style caches (default 2000
  entries; oldest 30% evicted on overflow).
- Patch 4: `compute_concept_distillation` is rewritten memory-safe:
  ≤30 docs joined per concept, TF-IDF max_features reduced 5000→2000,
  coherence computed on the first 20 words only, dict-or-list `all_texts`
  compatible, explicit del/gc cleanup.
Expected peak RSS after the patches: ~400 MB total (vs. > 1 GB before).

DOMAIN: CoCrFeNi MPEA Quantitative Descriptors
- Compositional: atomic size difference (δ), valence electron concentration (VEC),
  electronegativity difference (Δχ), nominal composition
- Thermodynamic: enthalpy of mixing (ΔH_mix), entropy of mixing (ΔS_mix),
  Ω parameter, Gibbs free energy
- Mechanical: hardness (HV), elongation (%), Pugh's ratio (B/G), Cauchy pressure,
  yield strength, tensile strength
- Asymmetry factors: melting temperature, shear modulus, enthalpy asymmetries
- Phase constituents: FCC, BCC, intermetallic (IM), solid solution (SS), Laves phase
- Processing routes: casting, wrought, sintering, annealing

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run:
    streamlit run mpea_concept_graph.py

Place JSON/BibTeX/CSV files in ./json_metadatabase/ folder next to this script.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import math
import os
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
import functools
import time
import io
import base64
import copy
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Iterator
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score, r2_score, mean_absolute_error,
    mean_squared_error, davies_bouldin_score, pairwise_distances
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches as mpatches
import seaborn as sns

from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')


# ============================================================================
# PERFORMANCE MONITORING DECORATOR
# ============================================================================
class PerformanceMonitor:
    _timings: Dict[str, float] = {}
    _call_counts: Dict[str, int] = {}

    @classmethod
    def reset(cls) -> None:
        cls._timings.clear()
        cls._call_counts.clear()

    @classmethod
    def get_report(cls) -> str:
        report = []
        for func_name, total_time in sorted(
            cls._timings.items(), key=lambda x: x[1], reverse=True
        ):
            count = cls._call_counts.get(func_name, 1)
            avg_time = total_time / count
            report.append(
                f"  {func_name}: {total_time:.3f}s total "
                f"({count} calls, {avg_time:.4f}s avg)"
            )
        return "\n".join(report)


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        func_name = func.__qualname__
        PerformanceMonitor._timings[func_name] = (
            PerformanceMonitor._timings.get(func_name, 0) + elapsed
        )
        PerformanceMonitor._call_counts[func_name] = (
            PerformanceMonitor._call_counts.get(func_name, 0) + 1
        )
        return result
    return wrapper


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CoCrFeNi MPEA Quantitative Descriptor Graph v6.1",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)


# ============================================================================
# COLORMAP REGISTRY (50+)
# ============================================================================
SUPPORTED_COLORMAPS = {
    "viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno", "magma": "Magma",
    "cividis": "Cividis", "turbo": "Turbo", "jet": "Jet", "rainbow": "Rainbow",
    "hsv": "Hsv", "nipy_spectral": "NipySpectral", "gist_rainbow": "GistRainbow",
    "coolwarm": "Coolwarm", "RdBu": "RdBu", "seismic": "Seismic", "Spectral": "Spectral",
    "tab10": "Set1", "tab20": "Set2", "tab20b": "Set3", "Accent": "Accent",
    "Dark2": "Dark2", "Paired": "Paired", "Pastel1": "Pastel1", "Pastel2": "Pastel2",
    "cubehelix": "Cubehelix", "bone": "Bone", "gray": "Gray", "pink": "Pink",
    "spring": "Spring", "summer": "Summer", "autumn": "Autumn", "winter": "Winter",
    "cool": "Cool", "hot": "Hot", "twilight": "Twilight", "copper": "Copper",
    "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu",
    "BuPu": "BuPu", "GnBu": "GnBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn",
    "BuGn": "BuGn", "YlGn": "YlGn", "Greys": "Greys", "afmhot": "Afmhot",
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean",
}


def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        try:
            cmap = cm.get_cmap(cmap_name, n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
        except Exception:
            try:
                cmap = matplotlib.colormaps.get_cmap("viridis").resampled(n)
            except Exception:
                cmap = cm.get_cmap("viridis", n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


# ============================================================================
# ROBUST FILE LOADER (JSON / JSONL / CSV / BibTeX)
# ============================================================================
def robust_load_file(filepath: Path):
    suffix = filepath.suffix.lower()
    if suffix == '.bib':
        return parse_bibtex_file(filepath)

    text = filepath.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"File is empty (0 bytes or only whitespace).")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    sanitized = re.sub(r'NaN', 'null', text)
    sanitized = re.sub(r'Infinity', 'null', sanitized)
    sanitized = re.sub(r'-Infinity', 'null', sanitized)
    sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if records:
        return records

    try:
        df = pd.read_csv(filepath)
        return df.to_dict(orient="records")
    except Exception:
        pass

    preview = text[:300]
    raise ValueError(
        f"Could not parse {filepath.name}. First 200 chars: {preview[:200]}..."
    )


def parse_bibtex_file(filepath: Path) -> List[Dict]:
    try:
        import bibtexparser
        from bibtexparser.bparser import BibTexParser
        from bibtexparser.customization import convert_to_unicode
        with open(filepath, 'r', encoding='utf-8') as bibfile:
            parser = BibTexParser()
            parser.customization = convert_to_unicode
            bib_database = bibtexparser.load(bibfile, parser=parser)
            records = []
            for entry in bib_database.entries:
                record = {
                    'title': entry.get('title', ''),
                    'abstract': entry.get('abstract', ''),
                    'author': entry.get('author', ''),
                    'year': entry.get('year', ''),
                    'journal': entry.get('journal', entry.get('booktitle', '')),
                    'doi': entry.get('doi', ''),
                    'keywords': entry.get('keywords', ''),
                    'entry_type': entry.get('ENTRYTYPE', ''),
                    'id': entry.get('ID', ''),
                    '_source_file': filepath.name,
                }
                records.append(record)
            return records
    except ImportError:
        st.warning(
            "bibtexparser not installed. Install with: pip install bibtexparser"
        )
        return []
    except Exception as e:
        st.error(f"BibTeX parse error for {filepath.name}: {e}")
        return []


@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    files = (
        sorted(Path(directory).glob("*.json"))
        + sorted(Path(directory).glob("*.bib"))
        + sorted(Path(directory).glob("*.csv"))
    )
    if not files:
        return []
    loaded = []
    for fp in files:
        try:
            data = robust_load_file(fp)
            if isinstance(data, list):
                loaded.append((str(fp.name), data))
            elif isinstance(data, dict):
                loaded.append((str(fp.name), [data]))
            else:
                loaded.append((str(fp.name), []))
        except Exception as e:
            st.error(f"Error loading `{fp.name}`: {e}")
            try:
                raw_bytes = fp.read_bytes()[:300]
                hex_str = raw_bytes.hex()
                formatted = ' '.join(
                    hex_str[i:i + 2] for i in range(0, len(hex_str), 2)
                )
                st.code(
                    f"Hex preview (first {len(raw_bytes)} bytes):\n{formatted}",
                    language="text",
                )
            except Exception:
                pass
    return loaded


@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    rows = []
    for fname, records in file_records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec["_source_file"] = fname
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df = df.replace({
        float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA
    })
    year_cols = [c for c in df.columns if 'year' in c.lower()]
    if year_cols:
        df["Year"] = pd.to_numeric(df[year_cols[0]], errors="coerce")
    elif "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


# ============================================================================
# ENHANCED ONTOLOGY & NLP REASONING SYSTEM (MPEA QUANTITATIVE DESCRIPTORS)
# ============================================================================
class ConceptType(Enum):
    MATERIAL = "material"
    PROCESS = "process"
    PROPERTY = "property"
    PHENOMENON = "phenomenon"
    METHOD = "method"
    PARAMETER = "parameter"
    MICROSTRUCTURE = "microstructure"
    MODEL = "model"
    GENERAL = "general"


class RelationshipType(Enum):
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    CAUSES = "causes"
    RESULTS_IN = "results_in"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    CO_OCCURS = "co_occurs"
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    BRIDGE = "bridge"
    CONSTRAINS = "constrains"
    MODIFIES = "modifies"
    CORRECTS = "corrects"
    SELECTS = "selects"
    INITIATES = "initiates"
    DRIVES = "drives"
    TRANSITIONS_TO = "transitions_to"
    REPLACES = "replaces"
    TRAINS = "trains"
    OUTPUTS = "outputs"
    LEARNS = "learns"
    CAPTURES = "captures"
    PARALLELIZES = "parallelizes"
    POSITIONS = "positions"
    IDENTIFIES = "identifies"
    FORMS = "forms"
    PROCESSES = "processes"
    STABILIZES = "stabilizes"
    PRESERVES = "preserves"
    GENERATES = "generates"
    COMPOSES = "composes"
    QUALIFIES = "qualifies"
    ENABLES = "enables"
    DISCOVERS = "disovers"
    PRE_TRAINS = "pre_trains"
    GENERALIZES = "generalizes"
    QUERIES = "queries"
    OPTIMIZES = "optimizes"
    VALIDATES = "validates"
    BOUNDS = "bounds"
    QUANTIFIES = "quantifies"
    EVALUATES = "evaluates"
    COMPARES = "compares"
    COMPUTES = "computes"
    MODELS = "models"
    AVERAGES = "averages"
    MAPS = "maps"
    SIMULATES = "simulates"
    DETECTS = "detects"
    INTEGRATES = "integrates"
    COUPLES = "couples"
    UPSCALES = "upscales"
    RESOLVES = "resolves"
    SYNCHRONIZES = "synchronizes"
    CHARACTERIZES = "characterizes"
    DECOMPOSES = "decomposes"
    DESIGNS = "designs"
    APPROXIMATES = "approximates"
    STRENGTHENS = "strengthens"
    EXPLAINS = "explains"
    INTERPRETS = "interprets"
    GROUPS = "groups"
    VISUALIZES = "visualizes"
    CONSTRUCTS = "constructs"
    FRAMES = "frames"
    ACCELERATES = "accelerates"
    ENFORCES = "enforces"
    CORRELATES = "correlates"

# ============================================================================
# EDGE COLOR REGISTRY — one distinct color per RelationshipType category
# ============================================================================
EDGE_COLOR_REGISTRY: Dict[RelationshipType, str] = {
    # --- Semantic / structural ---
    RelationshipType.SYNONYM:           "#AAAAAA",   # light grey
    RelationshipType.HYPERNYM:          "#5B9BD5",   # steel blue
    RelationshipType.HYPONYM:           "#5B9BD5",   # steel blue (same family)
    RelationshipType.PART_OF:           "#70AD47",   # green
    RelationshipType.HAS_PART:          "#70AD47",   # green
    RelationshipType.CO_OCCURS:         "#BFBFBF",   # silver

    # --- Causal / directional ---
    RelationshipType.CAUSES:            "#FF4444",   # red
    RelationshipType.RESULTS_IN:        "#E06040",   # red-orange
    RelationshipType.INFLUENCES:        "#FF8C00",   # dark orange
    RelationshipType.DEPENDS_ON:        "#DAA520",   # goldenrod
    RelationshipType.CONSTRAINS:        "#CC5500",   # burnt orange
    RelationshipType.MODIFIES:          "#FF6347",   # tomato
    RelationshipType.CORRECTS:          "#CD5C5C",   # indian red
    RelationshipType.DRIVES:            "#DC143C",   # crimson
    RelationshipType.ENABLES:           "#FF7F50",   # coral

    # --- Phase / thermodynamic transitions ---
    RelationshipType.TRANSITIONS_TO:    "#8A2BE2",   # blue-violet
    RelationshipType.REPLACES:          "#9932CC",   # dark orchid
    RelationshipType.FORMS:             "#9370DB",   # medium purple
    RelationshipType.STABILIZES:        "#7B68EE",   # medium slate blue
    RelationshipType.PRESERVES:         "#6A5ACD",   # slate blue

    # --- Computation / modeling ---
    RelationshipType.TRAINS:            "#00CED1",   # dark turquoise
    RelationshipType.OUTPUTS:           "#20B2AA",   # light sea green
    RelationshipType.LEARNS:            "#48D1CC",   # medium turquoise
    RelationshipType.CAPTURES:          "#40E0D0",   # turquoise
    RelationshipType.COMPUTES:          "#008B8B",   # dark cyan
    RelationshipType.SIMULATES:         "#5F9EA0",   # cadet blue
    RelationshipType.MODELS:            "#4682B4",   # steel blue variant
    RelationshipType.APPROXIMATES:      "#87CEEB",   # sky blue
    RelationshipType.MAPS:              "#00BFFF",   # deep sky blue

    # --- Analysis / evaluation ---
    RelationshipType.QUANTIFIES:        "#32CD32",   # lime green
    RelationshipType.EVALUATES:         "#228B22",   # forest green
    RelationshipType.COMPARES:          "#3CB371",   # medium sea green
    RelationshipType.VALIDATES:         "#2E8B57",   # sea green
    RelationshipType.AVERAGES:          "#66CDAA",   # medium aquamarine
    RelationshipType.CORRELATES:        "#00FA9A",   # medium spring green

    # --- Structural / architectural ---
    RelationshipType.PARALLELIZES:      "#FFD700",   # gold
    RelationshipType.POSITIONS:         "#FFC125",   # golden rod 2
    RelationshipType.IDENTIFIES:        "#F0E68C",   # khaki
    RelationshipType.PROCESSES:         "#EEE8AA",   # pale golden rod
    RelationshipType.GROUPS:            "#DAA520",   # goldenrod variant
    RelationshipType.INTEGRATES:        "#B8860B",   # dark goldenrod
    RelationshipType.COUPLES:           "#CD950C",   # dark goldenrod 2

    # --- Discovery / optimization ---
    RelationshipType.DISCOVERS:         "#FF69B4",   # hot pink
    RelationshipType.PRE_TRAINS:        "#FF1493",   # deep pink
    RelationshipType.GENERALIZES:       "#DB7093",   # pale violet red
    RelationshipType.QUERIES:           "#C71585",   # medium violet red
    RelationshipType.OPTIMIZES:         "#FF00FF",   # magenta
    RelationshipType.DESIGNS:           "#BA55D3",   # medium orchid
    RelationshipType.CONSTRUCTS:        "#DA70D6",   # orchid

    # --- Advanced modeling ---
    RelationshipType.UPSCALES:          "#8B4513",   # saddle brown
    RelationshipType.RESOLVES:          "#A0522D",   # sienna
    RelationshipType.SYNCHRONIZES:      "#D2691E",   # chocolate
    RelationshipType.CHARACTERIZES:     "#CD853F",   # peru
    RelationshipType.DECOMPOSES:        "#DEB887",   # burlywood
    RelationshipType.FRAMES:            "#D2B48C",   # tan
    RelationshipType.COMPOSES:          "#BC8F8F",   # rosy brown
    RelationshipType.QUALIFIES:         "#F4A460",   # sandy brown

    # --- Explanation / visualization ---
    RelationshipType.STRENGTHENS:       "#7FFF00",   # chartreuse
    RelationshipType.EXPLAINS:          "#ADFF2F",   # green yellow
    RelationshipType.INTERPRETS:        "#7CFC00",   # lawn green
    RelationshipType.VISUALIZES:        "#00FF7F",   # spring green
    RelationshipType.ACCELERATES:       "#98FB98",   # pale green
    RelationshipType.ENFORCES:          "#90EE90",   # light green

    # --- Generic fallback ---
    RelationshipType.SEMANTIC:          "#808080",   # grey
    RelationshipType.INFERRED:          "#A9A9A9",   # dark grey
    RelationshipType.BRIDGE:            "#C0C0C0",   # silver
    RelationshipType.SELECTS:           "#D3D3D3",   # light grey
    RelationshipType.INITIATES:         "#696969",   # dim grey
    RelationshipType.DETECTS:           "#556B2F",   # dark olive green
    RelationshipType.GENERATES:         "#6B8E23",   # olive drab
}

# Color for edges whose RelationshipType is not in the registry
EDGE_COLOR_FALLBACK = "#888888"


def get_edge_color(rel_type: RelationshipType) -> str:
    """Return the hex color associated with a relationship type."""
    return EDGE_COLOR_REGISTRY.get(rel_type, EDGE_COLOR_FALLBACK)


def get_edge_width(rel_type: RelationshipType) -> float:
    """Return an edge width proportional to relationship 'strength' category."""
    STRONG = {RelationshipType.CAUSES, RelationshipType.DRIVES,
              RelationshipType.FORMS, RelationshipType.STABILIZES,
              RelationshipType.DEPENDS_ON, RelationshipType.CONSTRAINS}
    MEDIUM = {RelationshipType.INFLUENCES, RelationshipType.RESULTS_IN,
              RelationshipType.MODIFIES, RelationshipType.ENABLES,
              RelationshipType.TRANSITIONS_TO, RelationshipType.COMPUTES}
    if rel_type in STRONG:
        return 3.0
    elif rel_type in MEDIUM:
        return 2.0
    return 1.0


def get_edge_style(rel_type: RelationshipType) -> str:
    """Dashed lines for inferred / weak relationships, solid otherwise."""
    DASHED = {RelationshipType.INFERRED, RelationshipType.CO_OCCURS,
              RelationshipType.SEMANTIC, RelationshipType.BRIDGE}
    return "dashed" if rel_type in DASHED else "solid"


# -----------------------------------------------------------------------------
# NEW: Helper to lighten a hex color
# -----------------------------------------------------------------------------
def lighten_hex_color(hex_color: str, factor: float) -> str:
    """
    Lighten a hex color by mixing with white.
    factor: 0.0 = original, 1.0 = white.
    """
    if not hex_color.startswith('#'):
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class ConceptNode:
    canonical_name: str
    concept_type: ConceptType
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    related_processes: Set[str] = field(default_factory=set)
    related_properties: Set[str] = field(default_factory=set)
    definition: str = ""
    embedding: Optional[np.ndarray] = None

    def add_synonym(self, synonym: str) -> None:
        self.synonyms.add(synonym.lower().strip())

    def is_match(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if text_lower == self.canonical_name.lower():
            return True
        return text_lower in self.synonyms


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: RelationshipType
    confidence: float = 1.0
    evidence: str = ""
    inferred: bool = False


class DomainOntology:
    """Comprehensive ontology for MPEA Quantitative Descriptors."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self._build_ontology()

    def _build_ontology(self) -> None:
        # === COMPOSITIONAL DESCRIPTORS ===
        self._add_concept("atomic_size_difference", ConceptType.PARAMETER,
            synonyms={"atomic radius difference", "delta", "atomic mismatch", "mean atomic radius difference"},
            definition=r"Mean atomic radius difference ($\delta$) among constituent elements, capturing relative atomic sizes")
        self._add_concept("electronegativity_difference", ConceptType.PARAMETER,
            synonyms={"delta chi", "electronegativity mismatch", "chemical compatibility"},
            definition="Electronegativity difference among constituent elements")
        self._add_concept("valence_electron_concentration", ConceptType.PARAMETER,
            synonyms={"vec", "average vec", "electron concentration"},
            definition="Valence Electron Concentration (VEC), a unified indicator for phase stability (FCC vs BCC) and mechanical properties")
        self._add_concept("nominal_composition", ConceptType.PARAMETER,
            synonyms={"atomic fraction", "mole fraction", "equiatomic", "non-equiatomic", "composition vector"},
            definition="Nominal composition or atomic fractions of constituent elements")

        # === THERMODYNAMIC PARAMETERS ===
        self._add_concept("enthalpy_of_mixing", ConceptType.PARAMETER,
            synonyms={"delta h mix", "mixing enthalpy", "chemical compatibility"},
            definition=r"Enthalpy of mixing ($\Delta H_{mix}$), indicating probability of solid solution vs intermetallic formation")
        self._add_concept("entropy_of_mixing", ConceptType.PARAMETER,
            synonyms={"delta s mix", "mixing entropy", "configurational entropy"},
            definition=r"Entropy of mixing ($\Delta S_{mix}$), driving force for single-phase solid solution stabilization")
        self._add_concept("omega_parameter", ConceptType.PARAMETER,
            synonyms={"dimensionless omega", "omega", "phase prediction parameter"},
            definition=r"Dimensionless parameter $\Omega = T_m \Delta S_{mix} / |\Delta H_{mix}|$ for predicting solid solution stability")
        self._add_concept("gibbs_free_energy", ConceptType.PROPERTY,
            synonyms={"gibbs energy", "free energy", "thermodynamic potential"},
            definition="Gibbs free energy and related thermodynamic potentials governing phase stability")

        # === MECHANICAL PROPERTIES ===
        self._add_concept("hardness", ConceptType.PROPERTY,
            synonyms={"hv", "vickers hardness", "microhardness", "mechanical hardness"},
            definition="Resistance to localized plastic deformation (Hardness in HV)")
        self._add_concept("elongation", ConceptType.PROPERTY,
            synonyms={"ductility", "percentage elongation", "el %", "tensile ductility"},
            definition="Percentage elongation, a primary measure of material ductility")
        self._add_concept("pughs_ratio", ConceptType.PROPERTY,
            synonyms={"b/g ratio", "bulk to shear modulus ratio", "pugh criterion"},
            definition="Ratio of bulk modulus to shear modulus (B/G), a de-facto indicator of hardness and ductility")
        self._add_concept("cauchy_pressure", ConceptType.PROPERTY,
            synonyms={"cauchy criterion", "pettifor criterion"},
            definition="Cauchy pressure, an indicator of material ductility and metallic bonding character")

        # === ASYMMETRY FACTORS ===
        self._add_concept("asymmetry_factor", ConceptType.PARAMETER,
            synonyms={"elemental asymmetry", "property asymmetry", "asymmetry"},
            definition="Asymmetry in physical properties among constituent elements, highly predictive of mechanical properties")
        self._add_concept("melting_temp_asymmetry", ConceptType.PARAMETER,
            synonyms={"melting temperature asymmetry"},
            definition="Asymmetry in melting temperatures of constituent elements")
        self._add_concept("shear_modulus_asymmetry", ConceptType.PARAMETER,
            synonyms={"shear modulus asymmetry"},
            definition="Asymmetry in shear moduli of constituent elements")

        # === PHASE CONSTITUENTS ===
        self._add_concept("fcc_phase", ConceptType.MICROSTRUCTURE,
            synonyms={"fcc solid solution", "face centered cubic", "gamma phase"},
            definition="Face-centered cubic crystal structure, typically associated with high ductility and VEC >= 8")
        self._add_concept("bcc_phase", ConceptType.MICROSTRUCTURE,
            synonyms={"bcc solid solution", "body centered cubic", "alpha phase"},
            definition="Body-centered cubic crystal structure, typically associated with higher hardness and VEC < 6.87")
        self._add_concept("intermetallic_phase", ConceptType.MICROSTRUCTURE,
            synonyms={"im phase", "laves phase", "intermetallic compound", "im"},
            definition="Intermetallic phase, often forming at high negative enthalpy of mixing, increasing hardness but reducing ductility")
        self._add_concept("solid_solution", ConceptType.MICROSTRUCTURE,
            synonyms={"ss phase", "solid solution phase", "single phase"},
            definition="Disordered solid solution phase (FCC, BCC, or HCP)")

        # === PROCESSING ROUTES ===
        self._add_concept("casting", ConceptType.PROCESS,
            synonyms={"cast", "as-cast", "casting process"},
            definition="Casting manufacturing route")
        self._add_concept("wrought", ConceptType.PROCESS,
            synonyms={"wrought process", "thermomechanical processing"},
            definition="Wrought manufacturing route involving smelting, mixing, and forming")
        self._add_concept("sintering", ConceptType.PROCESS,
            synonyms={"powder metallurgy", "pm", "sintered"},
            definition="Powder metallurgy and sintering route")
        self._add_concept("annealing", ConceptType.PROCESS,
            synonyms={"annealed", "heat treatment"},
            definition="Annealing or heat treatment route")

        # === MATERIALS ===
        self._add_concept("cocrfeni", ConceptType.MATERIAL,
            synonyms={"co-cr-fe-ni", "co cr fe ni", "cocofeni", "cocrfeni hea", "cocrfeni mpea"},
            definition="Quaternary CoCrFeNi multi-principal element alloy system")
        self._add_concept("mpea", ConceptType.MATERIAL,
            synonyms={"multi-principal element alloy", "high entropy alloy", "hea", "medium entropy alloy", "mea"},
            definition="Multi-principal element alloy class")

        # === THERMODYNAMIC DATA TENSOR (advanced) ===
        self._add_concept("tensor_rank", ConceptType.PARAMETER,
            synonyms={"cp rank", "canonical polyadic rank", "tensor decomposition rank", "rank"},
            definition="Rank of thermodynamic property tensors describing multi-component interactions")
        self._add_concept("tucker_decomposition", ConceptType.MODEL,
            synonyms={"higher-order svd", "hosvd", "tensor decomposition", "tucker"},
            definition="Decomposition of multi-dimensional thermodynamic data into core tensor and factor matrices")
        self._add_concept("tensor_contraction", ConceptType.METHOD,
            synonyms={"mode-n product", "tensor product", "tensor multiplication"},
            definition="Contraction of composition-dependent property tensors with crystal orientation tensors")
        self._add_concept("kronecker_product", ConceptType.METHOD,
            synonyms={"kronecker", "tensor outer product", "direct product"},
            definition="Build higher-order interaction tensors from binary sub-systems")
        self._add_concept("core_tensor", ConceptType.PARAMETER,
            synonyms={"core", "tensor core", "multibody interaction tensor"},
            definition="Capture multi-body interactions beyond pairwise in CoCrFeNi")
        self._add_concept("factor_matrix", ConceptType.PARAMETER,
            synonyms={"factor matrices", "factor loading", "loading matrix"},
            definition="Element-specific contributions to thermodynamic properties")
        self._add_concept("tensor_completion", ConceptType.METHOD,
            synonyms={"tensor imputation", "missing data recovery", "low-rank completion"},
            definition="Fill missing CALPHAD data points via low-rank approximation")
        self._add_concept("alternating_least_squares", ConceptType.METHOD,
            synonyms={"als", "als convergence", "tensor optimization"},
            definition="Iterative optimization algorithm for tensor decomposition")
        self._add_concept("low_rank_approximation", ConceptType.METHOD,
            synonyms={"low-rank", "truncated decomposition", "tensor compression"},
            definition="Approximate high-dimensional tensors with reduced rank for efficiency")

        # === CALPHAD-SPECIFIC TENSOR TERMS ===
        self._add_concept("excess_gibbs_energy", ConceptType.PARAMETER,
            synonyms={"g_xs", "excess gibbs energy", "redlich-kister", "non-ideal mixing"},
            definition="Non-ideal mixing contribution to Gibbs energy, parameterized via Redlich-Kister polynomials")
        self._add_concept("redlich_kister_polynomials", ConceptType.METHOD,
            synonyms={"redlich-kister", "interaction polynomial", "binary expansion"},
            definition="Polynomial expansion for binary interaction energies in multi-component systems")
        self._add_concept("sublattice_model", ConceptType.MODEL,
            synonyms={"compound energy formalism", "cem", "cel model", "site occupancy model"},
            definition="CALPHAD model describing site occupancy on crystallographic sublattices")
        self._add_concept("interaction_parameter", ConceptType.PARAMETER,
            synonyms={"l_ij", "binary interaction", "redlich-kister coefficient", "interaction energy"},
            definition="Composition-dependent interaction energy between element pairs")
        self._add_concept("activity_coefficient", ConceptType.PARAMETER,
            synonyms={"gamma_i", "thermodynamic activity", "raoultian activity", "activity"},
            definition="Deviation from ideal solution behavior for component i")
        self._add_concept("chemical_potential", ConceptType.PARAMETER,
            synonyms={"mu_i", "partial molar gibbs energy", "diffusion potential"},
            definition="Thermodynamic potential driving diffusion and phase equilibrium")
        self._add_concept("scheil_gulliver", ConceptType.METHOD,
            synonyms={"scheil", "scheil solidification", "non-equilibrium freezing", "lever rule inverse"},
            definition="Path-dependent tensor evolution during non-equilibrium solidification")
        self._add_concept("muggianu_extrapolation", ConceptType.METHOD,
            synonyms={"kohler model", "toop model", "geometric extrapolation", "geometric model"},
            definition="Geometric model for estimating ternary/quaternary properties from binary data")
        self._add_concept("gibbs_duhem_equation", ConceptType.METHOD,
            synonyms={"gibbs-duhem", "thermodynamic consistency", "phase rule"},
            definition="Tensor consistency constraint across composition space")

        # === MPEA PHENOMENA (Advanced) ===
        self._add_concept("sluggish_diffusion", ConceptType.PHENOMENON,
            synonyms={"slow diffusion", "diffusion retardation", "tracer diffusion slowdown"},
            definition="Reduced interdiffusion kinetics due to fluctuating local bonding in multi-component alloys")
        self._add_concept("severe_lattice_distortion", ConceptType.PHENOMENON,
            synonyms={"lattice strain", "atomic size mismatch effect", "local strain field"},
            definition="Local strain fields from atomic size differences in solid solution")
        self._add_concept("cocktail_effect", ConceptType.PHENOMENON,
            synonyms={"synergistic effect", "unexpected properties", "non-linear properties"},
            definition="Properties exceeding rule-of-mixtures predictions from multi-component synergy")
        self._add_concept("high_entropy_stabilization", ConceptType.PHENOMENON,
            synonyms={"entropy stabilization", "configurational entropy effect", "hea stabilization"},
            definition="Delta S_mix suppresses intermetallic formation favoring solid solutions")
        self._add_concept("entropy_enthalpy_compensation", ConceptType.PHENOMENON,
            synonyms={"entropy-enthalpy balance", "compensation effect", "phase competition"},
            definition="Competing effects of entropy and enthalpy determine phase selection")
        self._add_concept("short_range_order", ConceptType.MICROSTRUCTURE,
            synonyms={"sro", "chemical short range order", "csro", "local ordering"},
            definition="Local preferential bonding despite overall random structure")
        self._add_concept("medium_range_order", ConceptType.MICROSTRUCTURE,
            synonyms={"mro", "extended chemical order", "medium-range correlations"},
            definition="Chemical correlations extending beyond nearest-neighbor shells")
        self._add_concept("chemical_complexity", ConceptType.PARAMETER,
            synonyms={"compositional complexity", "multi-component complexity", "alloy complexity"},
            definition="Multi-component thermodynamic tensor dimensionality and interaction multiplicity")
        self._add_concept("compositional_space", ConceptType.PARAMETER,
            synonyms={"composition space", "alloy space", "phase space", "simplex"},
            definition="Multi-dimensional space of possible compositions (3D simplex for quaternary)")
        self._add_concept("equiatomic", ConceptType.PARAMETER,
            synonyms={"equimolar", "equal atomic fraction", "equiatomic composition"},
            definition="Composition with equal atomic fractions of all constituent elements")
        self._add_concept("non_equiatomic", ConceptType.PARAMETER,
            synonyms={"non-equimolar", "off-equiatomic", "deviated composition"},
            definition="Composition with unequal atomic fractions, often optimized for properties")
        self._add_concept("solid_solution_strengthening", ConceptType.PHENOMENON,
            synonyms={"solution hardening", "lattice distortion strengthening", "random solid solution strengthening"},
            definition="Lattice distortion contribution to hardness and strength in random solid solutions")

        # === PHASE-FIELD MODELING (Advanced) ===
        self._add_concept("phase_field_model", ConceptType.MODEL,
            synonyms={"phase-field", "pf model", "diffuse interface model", "phase field"},
            definition="Computational model using order parameters to describe phase transitions and microstructure evolution")
        self._add_concept("allen_cahn_equation", ConceptType.MODEL,
            synonyms={"allen-cahn", "non-conserved dynamics", "order parameter evolution", "ginzburg-landau"},
            definition="Governing equation for non-conserved order parameters (grain orientation, ordering)")
        self._add_concept("cahn_hilliard_equation", ConceptType.MODEL,
            synonyms={"cahn-hilliard", "conserved dynamics", "composition evolution", "spinodal decomposition"},
            definition="Governing equation for conserved composition fields with chemical diffusion")
        self._add_concept("kks_model", ConceptType.MODEL,
            synonyms={"kim-kim-suzuki", "partitioning phase-field", "quantitative pf", "kks"},
            definition="Phase-field model with explicit solute partitioning for quantitative solidification")
        self._add_concept("grand_potential_formulation", ConceptType.MODEL,
            synonyms={"grand potential", "grand canonical", "omega potential", "chemical potential formulation"},
            definition="Phase-field formulation using chemical potentials as primary variables")
        self._add_concept("order_parameter", ConceptType.PARAMETER,
            synonyms={"eta", "phase field variable", "structural order parameter", "phase indicator"},
            definition="Field variable distinguishing different phases or orientations")
        self._add_concept("gradient_energy_coefficient", ConceptType.PARAMETER,
            synonyms={"kappa", "interfacial gradient term", "gradient penalty", "surface energy coefficient"},
            definition="Energy penalty for composition/order parameter gradients at interfaces")
        self._add_concept("double_well_potential", ConceptType.PARAMETER,
            synonyms={"double well", "phase separation barrier", "landau potential", "bulk free energy"},
            definition="Potential function with minima at distinct phases, driving phase separation")
        self._add_concept("interface_mobility", ConceptType.PARAMETER,
            synonyms={"m", "kinetic coefficient", "boundary mobility", "interface velocity coefficient"},
            definition="Proportionality constant relating thermodynamic driving force to interface velocity")
        self._add_concept("anti_trapping_current", ConceptType.MODEL,
            synonyms={"atc", "solute trapping correction", "anti-trapping", "solute redistribution correction"},
            definition="Correction term in phase-field to suppress spurious solute trapping at interface")
        self._add_concept("thin_interface_limit", ConceptType.METHOD,
            synonyms={"sharp interface limit", "asymptotic analysis", "quantitative matching"},
            definition="Asymptotic limit linking diffuse interface model to sharp interface physics")
        self._add_concept("quantitative_phase_field", ConceptType.METHOD,
            synonyms={"quantitative pf", "matched asymptotics", "converged phase-field"},
            definition="Phase-field model parameters calibrated to reproduce sharp-interface kinetics")

        # === MICROSTRUCTURE EVOLUTION (Phase-Field Outputs) ===
        self._add_concept("dendritic_growth", ConceptType.PHENOMENON,
            synonyms={"dendrite", "tree-like solidification", "primary arm growth", "branched growth"},
            definition="Instability-driven branched solidification morphology")
        self._add_concept("mullins_sekerka_instability", ConceptType.PHENOMENON,
            synonyms={"morphological instability", "interface instability", "constitutional undercooling"},
            definition="Wavelength selection mechanism for dendritic/cellular solidification")
        self._add_concept("sidebranching", ConceptType.PHENOMENON,
            synonyms={"secondary dendrite arms", "tertiary arms", "dendrite branching"},
            definition="Formation of secondary and tertiary arms on primary dendrite stalks")
        self._add_concept("tip_radius", ConceptType.PARAMETER,
            synonyms={"dendrite tip radius", "tip curvature", "radius of curvature"},
            definition="Radius of curvature at dendrite tip controlling growth kinetics")
        self._add_concept("growth_velocity", ConceptType.PARAMETER,
            synonyms={"dendrite velocity", "tip velocity", "solidification speed"},
            definition="Velocity of advancing solidification front")
        self._add_concept("ostwald_ripening", ConceptType.PHENOMENON,
            synonyms={"coarsening", "particle coarsening", "aging", "lsw theory"},
            definition="Thermally-driven growth of larger precipitates at expense of smaller ones")
        self._add_concept("grain_boundary_migration", ConceptType.PHENOMENON,
            synonyms={"gb migration", "boundary motion", "interface migration"},
            definition="Movement of grain boundaries driven by curvature or stored energy")
        self._add_concept("nucleation_rate", ConceptType.PARAMETER,
            synonyms={"i", "homogeneous nucleation", "heterogeneous nucleation", "nucleation frequency"},
            definition="Frequency of critical nucleus formation per unit volume and time")
        self._add_concept("critical_nucleus_size", ConceptType.PARAMETER,
            synonyms={"r_star", "critical radius", "nucleation barrier size"},
            definition="Minimum radius for stable nucleus formation against surface energy penalty")
        self._add_concept("columnar_equiaxed_transition", ConceptType.PHENOMENON,
            synonyms={"cet", "equiaxed transition", "grain morphology transition"},
            definition="Transition from directional columnar to randomly oriented equiaxed grains")
        self._add_concept("texture_development", ConceptType.PHENOMENON,
            synonyms={"crystallographic texture", "preferred orientation", "grain orientation"},
            definition="Evolution of preferred crystallographic orientations during solidification")
        self._add_concept("interfacial_anisotropy", ConceptType.PARAMETER,
            synonyms={"surface energy anisotropy", "kinetic anisotropy", "growth anisotropy"},
            definition="Directional dependence of interfacial energy or growth kinetics")

        # === AI SURROGATE MODELS ===
        self._add_concept("physics_informed_neural_network", ConceptType.MODEL,
            synonyms={"pinn", "physics-informed nn", "physics-constrained ml", "physics-guided neural network"},
            definition="Neural network with physical law constraints embedded as loss terms")
        self._add_concept("fourier_neural_operator", ConceptType.MODEL,
            synonyms={"fno", "neural operator", "fourier operator", "solution operator"},
            definition="Neural architecture learning solution operators of PDEs in Fourier space")
        self._add_concept("deeponet", ConceptType.MODEL,
            synonyms={"deep operator network", "branch-trunk network", "deep operator", "operator learning"},
            definition="Neural operator architecture with branch and trunk sub-networks for parametric PDEs")
        self._add_concept("gaussian_process_regression", ConceptType.MODEL,
            synonyms={"gpr", "kriging", "bayesian surrogate", "gaussian process"},
            definition="Non-parametric probabilistic regression with uncertainty quantification")
        self._add_concept("proper_orthogonal_decomposition", ConceptType.MODEL,
            synonyms={"pod", "karhunen-loeve", "modal decomposition", "principal component analysis"},
            definition="Dimensionality reduction via dominant eigenmodes of snapshot matrix")
        self._add_concept("variational_autoencoder", ConceptType.MODEL,
            synonyms={"vae", "generative autoencoder", "latent variable model", "probabilistic autoencoder"},
            definition="Probabilistic generative model for microstructure synthesis and latent representation")
        self._add_concept("autoencoder", ConceptType.MODEL,
            synonyms={"ae", "encoder-decoder", "bottleneck network", "compression network"},
            definition="Neural network learning compressed representations through encoder-decoder architecture")
        self._add_concept("generative_adversarial_network", ConceptType.MODEL,
            synonyms={"gan", "adversarial network", "generator-discriminator"},
            definition="Generative model using competing generator and discriminator networks")

        # === TRANSFORMER & ATTENTION ===
        self._add_concept("self_attention", ConceptType.MODEL,
            synonyms={"attention mechanism", "intra-attention", "scaled dot-product attention"},
            definition="Learned weighted aggregation capturing element-element interactions in composition space")
        self._add_concept("multi_head_attention", ConceptType.MODEL,
            synonyms={"multi-head", "parallel attention", "attention head ensemble"},
            definition="Parallel attention mechanisms operating on different representation subspaces")
        self._add_concept("positional_encoding", ConceptType.MODEL,
            synonyms={"pos enc", "position embedding", "spatial encoding"},
            definition="Encoding of crystal structure sites or composition dimensions in sequence models")
        self._add_concept("attention_visualization", ConceptType.METHOD,
            synonyms={"attention map", "attention weights", "saliency map", "attention heatmap"},
            definition="Visualization of attention weights to identify driving element interactions")
        self._add_concept("query_key_value", ConceptType.PARAMETER,
            synonyms={"qkv", "query key value", "attention triplet"},
            definition="Three linear projections forming the basis of attention computation")
        self._add_concept("feed_forward_network", ConceptType.MODEL,
            synonyms={"ffn", "mlp", "position-wise feed-forward", "dense layer"},
            definition="Fully-connected sublayer processing attention outputs in transformer blocks")
        self._add_concept("layer_normalization", ConceptType.MODEL,
            synonyms={"layer norm", "batch normalization", "instance norm", "group norm"},
            definition="Normalization technique stabilizing training of deep neural networks")
        self._add_concept("residual_connection", ConceptType.MODEL,
            synonyms={"skip connection", "residual link", "highway connection"},
            definition="Direct path preserving gradient flow in deep architectures")
        self._add_concept("transformer_encoder", ConceptType.MODEL,
            synonyms={"encoder", "transformer block", "self-attention encoder"},
            definition="Stack of self-attention and feed-forward layers processing input sequences")
        self._add_concept("transformer_decoder", ConceptType.MODEL,
            synonyms={"decoder", "cross-attention decoder", "autoregressive decoder"},
            definition="Stack with cross-attention generating outputs conditioned on encoder representations")

        # === LEARNING STRATEGIES ===
        self._add_concept("transfer_learning", ConceptType.METHOD,
            synonyms={"pre-training", "fine-tuning", "domain adaptation", "knowledge transfer"},
            definition="Leveraging knowledge from binary/ternary systems for quaternary prediction")
        self._add_concept("meta_learning", ConceptType.METHOD,
            synonyms={"learning to learn", "few-shot learning", "model-agnostic meta-learning", "maml"},
            definition="Learn to learn new CoCrFeNi variants quickly with minimal data")
        self._add_concept("active_learning", ConceptType.METHOD,
            synonyms={"query strategy", "optimal experimental design", "uncertainty sampling"},
            definition="Iterative selection of most informative compositions for DFT/experiment labeling")
        self._add_concept("bayesian_optimization", ConceptType.METHOD,
            synonyms={"bo", "sequential experimental design", "surrogate optimization", "gp-ucb"},
            definition="Global optimization of expensive black-box functions with uncertainty-guided exploration")
        self._add_concept("cross_validation", ConceptType.METHOD,
            synonyms={"cv", "k-fold", "leave-one-out", "loo", "validation strategy"},
            definition="Statistical validation across compositional subspaces to prevent overfitting")

        # === UNCERTAINTY QUANTIFICATION ===
        self._add_concept("epistemic_uncertainty", ConceptType.PARAMETER,
            synonyms={"model uncertainty", "knowledge uncertainty", "reducible uncertainty"},
            definition="Uncertainty from model inadequacy or limited training data")
        self._add_concept("aleatoric_uncertainty", ConceptType.PARAMETER,
            synonyms={"data uncertainty", "inherent noise", "irreducible uncertainty"},
            definition="Uncertainty from intrinsic stochasticity in measurements and processes")
        self._add_concept("confidence_interval", ConceptType.PARAMETER,
            synonyms={"ci", "prediction interval", "credible interval", "uncertainty bounds"},
            definition="Statistical range quantifying prediction reliability")
        self._add_concept("prediction_uncertainty", ConceptType.PARAMETER,
            synonyms={"total uncertainty", "combined uncertainty", "predictive variance"},
            definition="Combined epistemic and aleatoric uncertainty in model predictions")

        # === MODEL EVALUATION ===
        self._add_concept("mean_absolute_error", ConceptType.PARAMETER,
            synonyms={"mae", "l1 loss", "mean absolute deviation"},
            definition="Average absolute difference between predicted and true values")
        self._add_concept("root_mean_square_error", ConceptType.PARAMETER,
            synonyms={"rmse", "l2 loss", "root mean squared error"},
            definition="Square root of average squared prediction errors")
        self._add_concept("r2_score", ConceptType.PARAMETER,
            synonyms={"r squared", "coefficient of determination", "explained variance"},
            definition="Proportion of variance in dependent variable predictable from independent variables")
        self._add_concept("normalized_rmse", ConceptType.PARAMETER,
            synonyms={"nrmse", "relative rmse", "percentage rmse"},
            definition="RMSE normalized by data range for cross-property comparison")
        self._add_concept("mean_absolute_percentage_error", ConceptType.PARAMETER,
            synonyms={"mape", "percentage error", "relative error"},
            definition="Average percentage difference between predictions and observations")
        self._add_concept("dice_coefficient", ConceptType.PARAMETER,
            synonyms={"dice score", "sorensen-dice", "f1 segmentation"},
            definition="Overlap metric for microstructure phase segmentation accuracy")
        self._add_concept("intersection_over_union", ConceptType.PARAMETER,
            synonyms={"iou", "jaccard index", "segmentation overlap"},
            definition="Ratio of intersection to union for predicted vs. true microstructure regions")

        # === MULTI-SCALE METHODS ===
        self._add_concept("density_functional_theory", ConceptType.METHOD,
            synonyms={"dft", "ab initio", "first-principles", "kohn-sham"},
            definition="Quantum mechanical method for electronic structure and thermodynamic properties")
        self._add_concept("special_quasirandom_structures", ConceptType.METHOD,
            synonyms={"sqs", "quasirandom", "disordered supercell", "random structure model"},
            definition="Periodic supercells mimicking random solid solution statistics")
        self._add_concept("coherent_potential_approximation", ConceptType.METHOD,
            synonyms={"cpa", "effective medium", "single-site approximation", "virtual crystal"},
            definition="Effective medium theory for electronic structure of disordered alloys")
        self._add_concept("cluster_expansion", ConceptType.METHOD,
            synonyms={"ce", "lattice hamiltonian", "configuration interaction", "ising expansion"},
            definition="Mapping of DFT energies to Ising-like Hamiltonian for configurational space")
        self._add_concept("molecular_dynamics", ConceptType.METHOD,
            synonyms={"md", "atomistic simulation", "classical potential", "force field simulation"},
            definition="Newtonian dynamics simulation for diffusion and defect kinetics")
        self._add_concept("kinetic_monte_carlo", ConceptType.METHOD,
            synonyms={"kmc", "stochastic simulation", "rare event kinetics", "event-driven simulation"},
            definition="Event-driven simulation for diffusion-limited precipitation and phase evolution")
        self._add_concept("cellular_automaton", ConceptType.METHOD,
            synonyms={"ca", "grain growth model", "microstructure ca", "probabilistic ca"},
            definition="Discrete grid-based model for grain structure evolution")
        self._add_concept("finite_element_method", ConceptType.METHOD,
            synonyms={"fem", "finite element analysis", "fea", "galerkin method"},
            definition="Numerical method for solving partial differential equations on meshed domains")
        self._add_concept("finite_difference_method", ConceptType.METHOD,
            synonyms={"fdm", "finite difference", "discrete derivative"},
            definition="Numerical differentiation on regular grids for differential equations")
        self._add_concept("finite_volume_method", ConceptType.METHOD,
            synonyms={"fvm", "finite volume", "conservative scheme", "control volume"},
            definition="Conservative numerical method based on flux balance over control volumes")
        self._add_concept("spectral_method", ConceptType.METHOD,
            synonyms={"fourier method", "chebyshev method", "pseudo-spectral"},
            definition="High-accuracy method using global basis functions for smooth problems")

        # === SCALE BRIDGING ===
        self._add_concept("hierarchical_modeling", ConceptType.METHOD,
            synonyms={"hierarchical multiscale", "sequential multiscale", "information passing"},
            definition="Sequential coupling from electronic to continuum scales")
        self._add_concept("concurrent_multiscale", ConceptType.METHOD,
            synonyms={"concurrent coupling", "handshake method", "domain decomposition multiscale"},
            definition="Simultaneous simulation of multiple scales with interface coupling")
        self._add_concept("representative_volume_element", ConceptType.PARAMETER,
            synonyms={"rve", "statistical volume element", "sve", "unit cell"},
            definition="Minimum volume capturing effective material response with periodic boundary conditions")
        self._add_concept("homogenization", ConceptType.METHOD,
            synonyms={"effective property", "averaging", "upscaling", "coarse-graining"},
            definition="Derive macroscopic properties from microscopic structure and behavior")
        self._add_concept("crystal_plasticity", ConceptType.MODEL,
            synonyms={"cp", "crystal plasticity finite element", "cpfem", "dislocation-based model"},
            definition="Mesoscale model resolving anisotropic plastic deformation by crystal slip systems")
        self._add_concept("dislocation_dynamics", ConceptType.METHOD,
            synonyms={"dd", "discrete dislocation", "line defect dynamics"},
            definition="Simulation of dislocation motion and interaction in crystal lattices")

        # === DIGITAL TWIN & OPTIMIZATION ===
        self._add_concept("digital_twin", ConceptType.MODEL,
            synonyms={"virtual prototype", "digital shadow", "virtual replica", "real-time model"},
            definition="Virtual representation synchronized with physical system for prediction and control")
        self._add_concept("uncertainty_quantification", ConceptType.METHOD,
            synonyms={"uq", "sensitivity analysis", "propagation of uncertainty", "monte carlo sampling"},
            definition="Systematic characterization and propagation of uncertainties through models")
        self._add_concept("sobol_indices", ConceptType.PARAMETER,
            synonyms={"sobol sensitivity", "global sensitivity", "variance decomposition"},
            definition="Variance-based sensitivity metrics decomposing output variance by input factors")
        self._add_concept("pareto_front", ConceptType.PARAMETER,
            synonyms={"pareto optimality", "trade-off surface", "non-dominated set"},
            definition="Set of optimal solutions where improving one objective worsens another")
        self._add_concept("design_of_experiments", ConceptType.METHOD,
            synonyms={"doe", "experimental design", "factorial design", "latin hypercube"},
            definition="Systematic selection of simulation/experiment conditions for maximum information")
        self._add_concept("response_surface_methodology", ConceptType.METHOD,
            synonyms={"rsm", "surrogate model", "meta-model", "response surface"},
            definition="Statistical approximation of input-output relationships for optimization")

        # === MATERIALS PROPERTIES (Thermophysical) ===
        self._add_concept("thermal_conductivity", ConceptType.PROPERTY,
            synonyms={"k", "heat conductivity", "thermal diffusivity related"},
            definition="Ability to conduct heat, critical for thermal gradient calculations")
        self._add_concept("specific_heat_capacity", ConceptType.PROPERTY,
            synonyms={"c_p", "heat capacity", "thermal capacity", "isobaric heat capacity"},
            definition="Heat required to raise unit mass temperature by one degree")
        self._add_concept("thermal_expansion_coefficient", ConceptType.PROPERTY,
            synonyms={"alpha", "cte", "coefficient of thermal expansion"},
            definition="Fractional change in length per degree temperature increase")
        self._add_concept("latent_heat_of_fusion", ConceptType.PROPERTY,
            synonyms={"l_f", "heat of fusion", "enthalpy of fusion", "melting enthalpy"},
            definition="Energy absorbed during solid-to-liquid phase transition")
        self._add_concept("density", ConceptType.PROPERTY,
            synonyms={"rho", "mass density", "specific weight", "volumetric density"},
            definition="Mass per unit volume, composition-dependent in MPEAs")
        self._add_concept("viscosity", ConceptType.PROPERTY,
            synonyms={"mu", "dynamic viscosity", "melt viscosity", "fluidity"},
            definition="Resistance to flow in liquid state, affecting melt pool dynamics")
        self._add_concept("surface_tension", ConceptType.PROPERTY,
            synonyms={"gamma", "interfacial tension", "capillary force"},
            definition="Energy per unit area of interface, driving Marangoni convection")
        self._add_concept("emissivity", ConceptType.PROPERTY,
            synonyms={"epsilon", "radiative emissivity", "thermal radiation coefficient"},
            definition="Efficiency of thermal radiation emission from material surface")

        # === ELASTIC PROPERTIES ===
        self._add_concept("youngs_modulus", ConceptType.PROPERTY,
            synonyms={"e", "elastic modulus", "stiffness", "modulus of elasticity"},
            definition="Ratio of stress to strain in elastic deformation regime")
        self._add_concept("shear_modulus", ConceptType.PROPERTY,
            synonyms={"g", "rigidity modulus", "torsion modulus"},
            definition="Ratio of shear stress to shear strain")
        self._add_concept("bulk_modulus", ConceptType.PROPERTY,
            synonyms={"k", "compression modulus", "incompressibility"},
            definition="Resistance to uniform compression")
        self._add_concept("poissons_ratio", ConceptType.PROPERTY,
            synonyms={"nu", "poisson ratio", "transverse contraction ratio"},
            definition="Ratio of transverse strain to axial strain under uniaxial stress")
        self._add_concept("elastic_stiffness_tensor", ConceptType.PROPERTY,
            synonyms={"c_ijkl", "elastic constants", "stiffness matrix", "fourth-rank tensor"},
            definition="Fourth-rank tensor relating stress and strain in anisotropic elasticity")
        self._add_concept("elastic_compliance_tensor", ConceptType.PROPERTY,
            synonyms={"s_ijkl", "compliance constants", "compliance matrix"},
            definition="Inverse of stiffness tensor, relating strain to stress")
        self._add_concept("zener_anisotropy", ConceptType.PROPERTY,
            synonyms={"a", "anisotropy ratio", "elastic anisotropy"},
            definition="Ratio characterizing degree of elastic anisotropy in cubic crystals")

        # === MECHANICAL PROPERTIES (Advanced) ===
        self._add_concept("yield_strength", ConceptType.PROPERTY,
            synonyms={"sigma_y", "proof strength", "0.2% offset yield"},
            definition="Stress at onset of permanent plastic deformation")
        self._add_concept("ultimate_tensile_strength", ConceptType.PROPERTY,
            synonyms={"uts", "tensile strength", "ultimate strength", "ts"},
            definition="Maximum engineering stress before necking and fracture")
        self._add_concept("elongation_to_failure", ConceptType.PROPERTY,
            synonyms={"total elongation", "fracture elongation", "ductility"},
            definition="Total strain at fracture, measure of ductility")
        self._add_concept("uniform_elongation", ConceptType.PROPERTY,
            synonyms={"uniform strain", "necking onset strain", "considere criterion"},
            definition="Strain at onset of necking instability")
        self._add_concept("work_hardening_rate", ConceptType.PROPERTY,
            synonyms={"theta", "strain hardening rate", "hardening capacity"},
            definition="Rate of stress increase with plastic strain")
        self._add_concept("strain_hardening_exponent", ConceptType.PROPERTY,
            synonyms={"n", "hollomon exponent", "power-law exponent"},
            definition="Exponent in power-law relationship between true stress and true strain")
        self._add_concept("strength_coefficient", ConceptType.PROPERTY,
            synonyms={"k", "hollomon coefficient", "flow stress constant"},
            definition="Prefactor in Hollomon power-law hardening equation")
        self._add_concept("hall_petch", ConceptType.MODEL,
            synonyms={"hall-petch relationship", "grain size strengthening", "boundary strengthening"},
            definition="Inverse square-root dependence of yield strength on grain size")
        self._add_concept("precipitation_strengthening", ConceptType.PHENOMENON,
            synonyms={"age hardening", "particle strengthening", "orowan mechanism"},
            definition="Strength increase from obstacles posed by fine precipitates to dislocation motion")
        self._add_concept("critical_resolved_shear_stress", ConceptType.PROPERTY,
            synonyms={"crss", "tau_crss", "critical shear stress"},
            definition="Minimum resolved shear stress required to initiate slip on a crystal system")
        self._add_concept("taylor_hardening", ConceptType.MODEL,
            synonyms={"taylor law", "dislocation density strengthening", "forest hardening"},
            definition="Linear relationship between flow stress and square root of dislocation density")

        # === EXPERIMENTAL VALIDATION ===
        self._add_concept("scanning_electron_microscopy", ConceptType.METHOD,
            synonyms={"sem", "electron microscopy", "secondary electron imaging"},
            definition="Electron beam imaging for surface morphology and microstructure")
        self._add_concept("transmission_electron_microscopy", ConceptType.METHOD,
            synonyms={"tem", "electron diffraction", "high-resolution tem"},
            definition="Electron transmission imaging for atomic-scale structure and defects")
        self._add_concept("electron_backscatter_diffraction", ConceptType.METHOD,
            synonyms={"ebsd", "orientation imaging microscopy", "oim", "kikuchi diffraction"},
            definition="Crystallographic orientation mapping from backscattered electron diffraction patterns")
        self._add_concept("energy_dispersive_spectroscopy", ConceptType.METHOD,
            synonyms={"eds", "edx", "x-ray microanalysis", "elemental mapping"},
            definition="Elemental composition analysis via characteristic X-ray emission")
        self._add_concept("x_ray_diffraction", ConceptType.METHOD,
            synonyms={"xrd", "powder diffraction", "bragg diffraction", "rietveld refinement"},
            definition="Phase identification and lattice parameter determination from diffraction patterns")
        self._add_concept("atom_probe_tomography", ConceptType.METHOD,
            synonyms={"apt", "3d atom probe", "field ion microscopy", "atom probe"},
            definition="Three-dimensional elemental mapping at atomic resolution")
        self._add_concept("differential_scanning_calorimetry", ConceptType.METHOD,
            synonyms={"dsc", "thermal analysis", "calorimetry", "phase transition detection"},
            definition="Measurement of heat flow associated with phase transitions and reactions")
        self._add_concept("differential_thermal_analysis", ConceptType.METHOD,
            synonyms={"dta", "thermal differential analysis", "temperature difference method"},
            definition="Comparison of sample and reference temperatures during heating/cooling")

        # === DATA-DRIVEN MATERIALS SCIENCE ===
        self._add_concept("materials_informatics", ConceptType.METHOD,
            synonyms={"materials data science", "computational materials discovery", "materials ai"},
            definition="Application of data science and machine learning to materials discovery and design")
        self._add_concept("materials_genome_initiative", ConceptType.METHOD,
            synonyms={"mgi", "materials genome", "accelerated materials discovery"},
            definition="Systematic framework for integrating computation, data, and experiment")
        self._add_concept("high_throughput_computing", ConceptType.METHOD,
            synonyms={"htc", "high-throughput dft", "computational screening"},
            definition="Automated large-scale simulation campaigns for materials screening")
        self._add_concept("materials_database", ConceptType.METHOD,
            synonyms={"materials data repository", "curated dataset", "materials project"},
            definition="Structured repositories of computed and experimental materials properties")
        self._add_concept("feature_engineering", ConceptType.METHOD,
            synonyms={"descriptor design", "fingerprint construction", "representation learning"},
            definition="Systematic creation of input features capturing materials physics")
        self._add_concept("dimensionality_reduction", ConceptType.METHOD,
            synonyms={"pca", "t-sne", "umap", "manifold learning", "embedding"},
            definition="Projection of high-dimensional data to lower-dimensional representations")
        self._add_concept("clustering", ConceptType.METHOD,
            synonyms={"k-means", "hierarchical clustering", "dbscan", "unsupervised classification"},
            definition="Grouping of materials by similarity in feature space without labels")
        self._add_concept("explainable_ai", ConceptType.METHOD,
            synonyms={"xai", "interpretable ml", "model interpretability", "transparent ai"},
            definition="Methods making AI model decisions understandable to human experts")
        self._add_concept("shap_values", ConceptType.METHOD,
            synonyms={"shap", "shapley additive explanations", "game theory explanation"},
            definition="Game-theoretic attribution of predictions to individual features")
        self._add_concept("symbolic_regression", ConceptType.METHOD,
            synonyms={"equation discovery", "genetic programming", "analytical model discovery"},
            definition="Discovery of closed-form analytical expressions from data")
        self._add_concept("sparse_identification_nonlinear_dynamics", ConceptType.METHOD,
            synonyms={"sindy", "sparse regression", "equation-free modeling", "data-driven dynamics"},
            definition="Sparse regression for discovering governing equations from time-series data")

        # Build indices and causal chains
        self._build_synonym_index()
        self._build_causal_chains()

    def _add_concept(
        self,
        canonical_name: str,
        concept_type: ConceptType,
        synonyms: Set[str] = None,
        hypernyms: Set[str] = None,
        hyponyms: Set[str] = None,
        definition: str = "",
        related_processes: Set[str] = None,
        related_properties: Set[str] = None,
    ) -> None:
        node = ConceptNode(
            canonical_name=canonical_name,
            concept_type=concept_type,
            synonyms=synonyms or set(),
            hypernyms=hypernyms or set(),
            hyponyms=hyponyms or set(),
            related_processes=related_processes or set(),
            related_properties=related_properties or set(),
            definition=definition,
        )
        self.concepts[canonical_name] = node

    def _build_synonym_index(self) -> None:
        self.synonym_to_canonical: Dict[str, str] = {}
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def _build_causal_chains(self) -> None:
        # === TENSOR → CALPHAD → DESCRIPTOR CAUSAL CHAINS ===
        causal_chains = [
            ("tucker_decomposition", RelationshipType.INFLUENCES, "excess_gibbs_energy", 0.95),
            ("tensor_rank", RelationshipType.INFLUENCES, "core_tensor", 0.90),
            ("kronecker_product", RelationshipType.INFLUENCES, "factor_matrix", 0.85),
            ("tensor_completion", RelationshipType.INFLUENCES, "excess_gibbs_energy", 0.80),
            ("alternating_least_squares", RelationshipType.DEPENDS_ON, "tensor_completion", 0.90),
            ("low_rank_approximation", RelationshipType.INFLUENCES, "excess_gibbs_energy", 0.85),
            # CALPHAD → Descriptors
            ("excess_gibbs_energy", RelationshipType.CAUSES, "enthalpy_of_mixing", 0.95),
            ("excess_gibbs_energy", RelationshipType.CAUSES, "entropy_of_mixing", 0.95),
            ("redlich_kister_polynomials", RelationshipType.INFLUENCES, "excess_gibbs_energy", 0.90),
            ("sublattice_model", RelationshipType.INFLUENCES, "excess_gibbs_energy", 0.90),
            ("interaction_parameter", RelationshipType.INFLUENCES, "excess_gibbs_energy", 0.85),
            ("chemical_potential", RelationshipType.INFLUENCES, "enthalpy_of_mixing", 0.90),
            ("activity_coefficient", RelationshipType.INFLUENCES, "enthalpy_of_mixing", 0.85),
            ("muggianu_extrapolation", RelationshipType.INFLUENCES, "excess_gibbs_energy", 0.80),
            ("gibbs_duhem_equation", RelationshipType.DEPENDS_ON, "excess_gibbs_energy", 0.90),
            # Descriptors → MPEA Phenomena
            ("enthalpy_of_mixing", RelationshipType.CAUSES, "intermetallic_phase", 0.85),
            ("entropy_of_mixing", RelationshipType.CAUSES, "high_entropy_stabilization", 0.90),
            ("entropy_of_mixing", RelationshipType.CAUSES, "cocktail_effect", 0.80),
            ("atomic_size_difference", RelationshipType.CAUSES, "severe_lattice_distortion", 0.90),
            ("atomic_size_difference", RelationshipType.CAUSES, "sluggish_diffusion", 0.85),
            ("electronegativity_difference", RelationshipType.CAUSES, "short_range_order", 0.85),
            ("valence_electron_concentration", RelationshipType.INFLUENCES, "phase_stability_parameter", 0.90),
            ("omega_parameter", RelationshipType.INFLUENCES, "high_entropy_stabilization", 0.85),
            ("lambda_parameter", RelationshipType.INFLUENCES, "high_entropy_stabilization", 0.80),
            ("phase_stability_parameter", RelationshipType.INFLUENCES, "fcc_phase", 0.85),
            ("phase_stability_parameter", RelationshipType.INFLUENCES, "bcc_phase", 0.85),
            ("undercooling", RelationshipType.CAUSES, "nucleation_rate", 0.90),
            ("partition_coefficient", RelationshipType.INFLUENCES, "severe_lattice_distortion", 0.85),
            ("severe_lattice_distortion", RelationshipType.CAUSES, "short_range_order", 0.80),
            ("high_entropy_stabilization", RelationshipType.CAUSES, "intermetallic_phase", -0.75),
            ("short_range_order", RelationshipType.CAUSES, "sluggish_diffusion", 0.85),
            ("medium_range_order", RelationshipType.CAUSES, "cocktail_effect", 0.80),
            ("entropy_enthalpy_compensation", RelationshipType.INFLUENCES, "phase_stability_parameter", 0.85),
            ("chemical_complexity", RelationshipType.INFLUENCES, "cocktail_effect", 0.80),
            ("compositional_space", RelationshipType.INFLUENCES, "phase_stability_parameter", 0.85),
            ("equiatomic", RelationshipType.INFLUENCES, "entropy_of_mixing", 0.90),
            ("non_equiatomic", RelationshipType.INFLUENCES, "enthalpy_of_mixing", 0.85),
            ("solid_solution_strengthening", RelationshipType.CAUSES, "hardness", 0.80),
            # Descriptors → Phase-Field
            ("enthalpy_of_mixing", RelationshipType.INFLUENCES, "phase_field_model", 0.90),
            ("entropy_of_mixing", RelationshipType.INFLUENCES, "phase_field_model", 0.90),
            ("atomic_size_difference", RelationshipType.INFLUENCES, "phase_field_model", 0.85),
            ("electronegativity_difference", RelationshipType.INFLUENCES, "phase_field_model", 0.80),
            ("valence_electron_concentration", RelationshipType.INFLUENCES, "phase_field_model", 0.85),
            ("omega_parameter", RelationshipType.INFLUENCES, "phase_field_model", 0.85),
            ("chemical_driving_pressure", RelationshipType.CAUSES, "phase_field_model", 0.85),
            ("undercooling", RelationshipType.CAUSES, "nucleation_rate", 0.90),
            ("capillary_length", RelationshipType.INFLUENCES, "dendritic_growth", 0.80),
            # Phase-Field Internal
            ("phase_field_model", RelationshipType.RESULTS_IN, "allen_cahn_equation", 0.95),
            ("phase_field_model", RelationshipType.RESULTS_IN, "cahn_hilliard_equation", 0.95),
            ("phase_field_model", RelationshipType.RESULTS_IN, "kks_model", 0.90),
            ("phase_field_model", RelationshipType.RESULTS_IN, "grand_potential_formulation", 0.90),
            ("order_parameter", RelationshipType.DEPENDS_ON, "allen_cahn_equation", 0.90),
            ("gradient_energy_coefficient", RelationshipType.INFLUENCES, "allen_cahn_equation", 0.85),
            ("gradient_energy_coefficient", RelationshipType.INFLUENCES, "cahn_hilliard_equation", 0.85),
            ("double_well_potential", RelationshipType.CAUSES, "allen_cahn_equation", 0.85),
            ("interface_mobility", RelationshipType.INFLUENCES, "allen_cahn_equation", 0.90),
            ("anti_trapping_current", RelationshipType.CORRECTS, "kks_model", 0.85),
            ("thin_interface_limit", RelationshipType.DEPENDS_ON, "quantitative_phase_field", 0.90),
            ("allen_cahn_equation", RelationshipType.CAUSES, "dendritic_growth", 0.85),
            ("cahn_hilliard_equation", RelationshipType.CAUSES, "ostwald_ripening", 0.85),
            ("mullins_sekerka_instability", RelationshipType.SELECTS, "dendritic_growth", 0.90),
            ("nucleation_rate", RelationshipType.INITIATES, "dendritic_growth", 0.85),
            ("grain_boundary_migration", RelationshipType.DRIVES, "ostwald_ripening", 0.80),
            ("columnar_equiaxed_transition", RelationshipType.TRANSITIONS_TO, "dendritic_growth", 0.85),
            ("interfacial_anisotropy", RelationshipType.INFLUENCES, "dendritic_growth", 0.85),
            ("sidebranching", RelationshipType.RESULTS_IN, "dendritic_growth", 0.80),
            ("tip_radius", RelationshipType.INFLUENCES, "growth_velocity", 0.85),
            ("texture_development", RelationshipType.RESULTS_IN, "columnar_equiaxed_transition", 0.80),
            ("critical_nucleus_size", RelationshipType.CONSTRAINS, "nucleation_rate", 0.90),
            # AI Internal Architectures
            ("physics_informed_neural_network", RelationshipType.ENFORCES, "enthalpy_of_mixing", 0.90),
            ("physics_informed_neural_network", RelationshipType.ENFORCES, "entropy_of_mixing", 0.90),
            ("fourier_neural_operator", RelationshipType.LEARNS, "phase_field_model", 0.90),
            ("deeponet", RelationshipType.LEARNS, "phase_field_model", 0.90),
            ("self_attention", RelationshipType.CAPTURES, "fourier_neural_operator", 0.85),
            ("multi_head_attention", RelationshipType.PARALLELIZES, "self_attention", 0.90),
            ("positional_encoding", RelationshipType.POSITIONS, "self_attention", 0.85),
            ("attention_visualization", RelationshipType.IDENTIFIES, "short_range_order", 0.80),
            ("query_key_value", RelationshipType.FORMS, "self_attention", 0.90),
            ("feed_forward_network", RelationshipType.PROCESSES, "self_attention", 0.85),
            ("layer_normalization", RelationshipType.STABILIZES, "transformer_encoder", 0.85),
            ("residual_connection", RelationshipType.PRESERVES, "transformer_encoder", 0.85),
            ("transformer_encoder", RelationshipType.PROCESSES, "fourier_neural_operator", 0.80),
            # AI Learning Strategies
            ("transfer_learning", RelationshipType.PRE_TRAINS, "physics_informed_neural_network", 0.90),
            ("meta_learning", RelationshipType.GENERALIZES, "transfer_learning", 0.85),
            ("active_learning", RelationshipType.QUERIES, "gaussian_process_regression", 0.85),
            ("cross_validation", RelationshipType.VALIDATES, "physics_informed_neural_network", 0.85),
            # Uncertainty
            ("epistemic_uncertainty", RelationshipType.BOUNDS, "uncertainty_quantification", 0.90),
            ("aleatoric_uncertainty", RelationshipType.BOUNDS, "uncertainty_quantification", 0.90),
            ("confidence_interval", RelationshipType.QUANTIFIES, "prediction_uncertainty", 0.85),
            # Evaluation Metrics
            ("mean_absolute_error", RelationshipType.EVALUATES, "physics_informed_neural_network", 0.80),
            ("root_mean_square_error", RelationshipType.EVALUATES, "physics_informed_neural_network", 0.80),
            ("r2_score", RelationshipType.EVALUATES, "physics_informed_neural_network", 0.85),
            ("normalized_rmse", RelationshipType.COMPARES, "physics_informed_neural_network", 0.75),
            ("dice_coefficient", RelationshipType.EVALUATES, "phase_field_model", 0.80),
            ("intersection_over_union", RelationshipType.EVALUATES, "phase_field_model", 0.80),
            # Multi-Scale Bridge
            ("density_functional_theory", RelationshipType.COMPUTES, "enthalpy_of_mixing", 0.90),
            ("density_functional_theory", RelationshipType.COMPUTES, "atomic_size_difference", 0.85),
            ("special_quasirandom_structures", RelationshipType.MODELS, "density_functional_theory", 0.90),
            ("coherent_potential_approximation", RelationshipType.AVERAGES, "density_functional_theory", 0.85),
            ("cluster_expansion", RelationshipType.MAPS, "density_functional_theory", 0.85),
            ("molecular_dynamics", RelationshipType.SIMULATES, "sluggish_diffusion", 0.85),
            ("molecular_dynamics", RelationshipType.DETECTS, "short_range_order", 0.80),
            ("kinetic_monte_carlo", RelationshipType.SIMULATES, "ostwald_ripening", 0.85),
            ("kinetic_monte_carlo", RelationshipType.SIMULATES, "nucleation_rate", 0.80),
            ("finite_element_method", RelationshipType.COMPUTES, "yield_strength", 0.85),
            ("cellular_automaton", RelationshipType.SIMULATES, "grain_boundary_migration", 0.80),
            ("hierarchical_modeling", RelationshipType.INTEGRATES, "density_functional_theory", 0.90),
            ("hierarchical_modeling", RelationshipType.INTEGRATES, "molecular_dynamics", 0.90),
            ("hierarchical_modeling", RelationshipType.INTEGRATES, "kinetic_monte_carlo", 0.85),
            ("hierarchical_modeling", RelationshipType.INTEGRATES, "finite_element_method", 0.85),
            ("concurrent_multiscale", RelationshipType.COUPLES, "molecular_dynamics", 0.80),
            ("representative_volume_element", RelationshipType.CAPTURES, "finite_element_method", 0.85),
            ("homogenization", RelationshipType.UPSCALES, "crystal_plasticity", 0.85),
            ("crystal_plasticity", RelationshipType.RESOLVES, "yield_strength", 0.80),
            ("dislocation_dynamics", RelationshipType.SIMULATES, "work_hardening_rate", 0.80),
            # Multi-Scale → AI
            ("hierarchical_modeling", RelationshipType.ACCELERATES, "physics_informed_neural_network", 0.85),
            ("density_functional_theory", RelationshipType.TRAINS, "physics_informed_neural_network", 0.90),
            ("molecular_dynamics", RelationshipType.TRAINS, "fourier_neural_operator", 0.85),
            # Digital Twin & Optimization
            ("digital_twin", RelationshipType.SYNCHRONIZES, "physics_informed_neural_network", 0.90),
            ("uncertainty_quantification", RelationshipType.CHARACTERIZES, "physics_informed_neural_network", 0.85),
            ("sobol_indices", RelationshipType.DECOMPOSES, "uncertainty_quantification", 0.80),
            ("pareto_front", RelationshipType.OPTIMIZES, "nominal_composition", 0.85),
            ("design_of_experiments", RelationshipType.DESIGNS, "active_learning", 0.80),
            ("response_surface_methodology", RelationshipType.APPROXIMATES, "physics_informed_neural_network", 0.75),
            # Properties → Outputs
            ("hardness", RelationshipType.RESULTS_IN, "yield_strength", 0.85),
            ("elongation", RelationshipType.RESULTS_IN, "elongation_to_failure", 0.80),
            ("yield_strength", RelationshipType.RESULTS_IN, "ultimate_tensile_strength", 0.85),
            # Thermophysical Properties → Phase-Field
            ("thermal_conductivity", RelationshipType.INFLUENCES, "phase_field_model", 0.80),
            ("specific_heat_capacity", RelationshipType.INFLUENCES, "phase_field_model", 0.80),
            ("latent_heat_of_fusion", RelationshipType.INFLUENCES, "phase_field_model", 0.85),
            ("density", RelationshipType.INFLUENCES, "phase_field_model", 0.75),
            ("surface_tension", RelationshipType.INFLUENCES, "phase_field_model", 0.80),
            ("viscosity", RelationshipType.INFLUENCES, "phase_field_model", 0.75),
            # Elastic Properties → Mechanical
            ("youngs_modulus", RelationshipType.INFLUENCES, "yield_strength", 0.80),
            ("shear_modulus", RelationshipType.INFLUENCES, "hardness", 0.80),
            ("bulk_modulus", RelationshipType.INFLUENCES, "pughs_ratio", 0.85),
            ("poissons_ratio", RelationshipType.INFLUENCES, "elongation", 0.75),
            ("elastic_stiffness_tensor", RelationshipType.COMPUTES, "youngs_modulus", 0.90),
            ("zener_anisotropy", RelationshipType.INFLUENCES, "texture_development", 0.80),
            # Mechanical Properties
            ("yield_strength", RelationshipType.CAUSES, "elongation", -0.70),
            ("ultimate_tensile_strength", RelationshipType.CORRELATES, "yield_strength", 0.90),
            ("work_hardening_rate", RelationshipType.INFLUENCES, "ultimate_tensile_strength", 0.85),
            ("strain_hardening_exponent", RelationshipType.INFLUENCES, "elongation", 0.75),
            ("hall_petch", RelationshipType.STRENGTHENS, "yield_strength", 0.85),
            ("solid_solution_strengthening", RelationshipType.STRENGTHENS, "yield_strength", 0.80),
            ("precipitation_strengthening", RelationshipType.STRENGTHENS, "yield_strength", 0.85),
            ("critical_resolved_shear_stress", RelationshipType.CONSTRAINS, "yield_strength", 0.80),
            ("taylor_hardening", RelationshipType.MODELS, "work_hardening_rate", 0.85),
            # Experimental → Validation
            ("scanning_electron_microscopy", RelationshipType.VALIDATES, "phase_field_model", 0.80),
            ("transmission_electron_microscopy", RelationshipType.VALIDATES, "short_range_order", 0.85),
            ("electron_backscatter_diffraction", RelationshipType.VALIDATES, "texture_development", 0.85),
            ("energy_dispersive_spectroscopy", RelationshipType.VALIDATES, "compositional_space", 0.80),
            ("x_ray_diffraction", RelationshipType.VALIDATES, "fcc_phase", 0.90),
            ("atom_probe_tomography", RelationshipType.VALIDATES, "short_range_order", 0.90),
            ("differential_scanning_calorimetry", RelationshipType.VALIDATES, "enthalpy_of_mixing", 0.85),
            ("differential_thermal_analysis", RelationshipType.VALIDATES, "entropy_of_mixing", 0.80),
            # Data-Driven → AI
            ("materials_informatics", RelationshipType.DRIVES, "physics_informed_neural_network", 0.90),
            ("materials_genome_initiative", RelationshipType.FRAMES, "physics_informed_neural_network", 0.85),
            ("high_throughput_computing", RelationshipType.GENERATES, "density_functional_theory", 0.85),
            ("materials_database", RelationshipType.TRAINS, "physics_informed_neural_network", 0.85),
            ("feature_engineering", RelationshipType.CONSTRUCTS, "physics_informed_neural_network", 0.80),
            ("dimensionality_reduction", RelationshipType.VISUALIZES, "compositional_space", 0.75),
            ("clustering", RelationshipType.GROUPS, "fcc_phase", 0.75),
            ("explainable_ai", RelationshipType.INTERPRETS, "physics_informed_neural_network", 0.85),
            ("shap_values", RelationshipType.EXPLAINS, "self_attention", 0.80),
            ("symbolic_regression", RelationshipType.DISCOVERS, "phase_stability_parameter", 0.75),
            ("sparse_identification_nonlinear_dynamics", RelationshipType.DISCOVERS, "phase_field_model", 0.70),
            # Material Hierarchy
            ("cocrfeni", RelationshipType.HYPONYM, "mpea", 1.0),
        ]
        for source, rel_type, target, confidence in causal_chains:
            self.relationships.append(
                Relationship(source, target, rel_type, abs(confidence))
            )

    def resolve_concept(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in self.synonym_to_canonical:
            return self.synonym_to_canonical[text_lower]
        normalized = self._normalize_text(text_lower)
        if normalized in self.synonym_to_canonical:
            return self.synonym_to_canonical[normalized]
        variants = [
            text_lower.replace("-", " "),
            text_lower.replace(" ", "-"),
            text_lower.replace(" of ", " "),
            text_lower.replace(" for ", " "),
            text_lower.replace(" in ", " "),
            re.sub(r'\bs\b', '', text_lower),
            re.sub(r'\bes\b', '', text_lower),
        ]
        for variant in variants:
            if variant in self.synonym_to_canonical:
                return self.synonym_to_canonical[variant]
        return None

    def _normalize_text(self, text: str) -> str:
        text = re.sub(
            r'\b(the|a|an|of|for|in|with|by|to|and|or|on|at)\b', ' ', text
        )
        text = ' '.join(text.split())
        return text.strip()

    def get_concept_type(self, canonical_name: str) -> ConceptType:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].concept_type
        return ConceptType.GENERAL

    def get_hypernyms(self, canonical_name: str) -> Set[str]:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].hypernyms
        return set()

    def get_hyponyms(self, canonical_name: str) -> Set[str]:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].hyponyms
        return set()

    def get_definition(self, canonical_name: str) -> str:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].definition
        return ""

    def infer_path(
        self, source: str, target: str, max_depth: int = 3
    ) -> List[List[str]]:
        paths: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(current: str, target: str, path: List[str], depth: int) -> None:
            if depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            if current in visited:
                return
            visited.add(current)
            for rel in self.relationships:
                if rel.source == current and rel.confidence > 0.5:
                    path.append(rel.target)
                    dfs(rel.target, target, path, depth + 1)
                    path.pop()
            if current in self.concepts:
                for hyp in self.concepts[current].hypernyms:
                    path.append(hyp)
                    dfs(hyp, target, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(source, target, [source], 0)
        return paths

    def get_related_concepts(
        self, canonical_name: str, rel_type: RelationshipType = None
    ) -> List[Tuple[str, RelationshipType, float]]:
        related: List[Tuple[str, RelationshipType, float]] = []
        for rel in self.relationships:
            if rel.source == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.target, rel.rel_type, rel.confidence))
            elif rel.target == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.source, rel.rel_type, rel.confidence))
        return related


# ============================================================================
# ADVANCED CONCEPT RESOLVER (AgNPs Pattern — Eager Precomputation)
# ============================================================================


# ============================================================================
# HIERARCHY LABEL BUILDER — enriches flat concept names with ancestor path
# ============================================================================

# Hand-curated "primary parent" map for the ontology concepts.
# Each entry says:  child → (parent_label, hierarchy_tier)
# Tiers: 0 = root domain, 1 = major category, 2 = sub-category
_HIERARCHY_PARENTS: Dict[str, Tuple[str, int]] = {
    # --- Root domains (tier 0) ---
    "cocrfeni":  (None, 0),
    "mpea":      (None, 0),

    # --- Tier 1: Major categories ---
    "atomic_size_difference":         ("Compositional Descriptors", 1),
    "electronegativity_difference":   ("Compositional Descriptors", 1),
    "valence_electron_concentration": ("Compositional Descriptors", 1),
    "nominal_composition":            ("Compositional Descriptors", 1),
    "enthalpy_of_mixing":             ("Thermodynamic Parameters", 1),
    "entropy_of_mixing":              ("Thermodynamic Parameters", 1),
    "omega_parameter":                ("Thermodynamic Parameters", 1),
    "gibbs_free_energy":              ("Thermodynamic Parameters", 1),
    "hardness":                       ("Mechanical Properties", 1),
    "elongation":                     ("Mechanical Properties", 1),
    "pughs_ratio":                    ("Mechanical Properties", 1),
    "cauchy_pressure":                ("Mechanical Properties", 1),
    "asymmetry_factor":               ("Asymmetry Factors", 1),
    "melting_temp_asymmetry":         ("Asymmetry Factors", 1),
    "shear_modulus_asymmetry":        ("Asymmetry Factors", 1),
    "fcc_phase":                      ("Phase Constituents", 1),
    "bcc_phase":                      ("Phase Constituents", 1),
    "intermetallic_phase":            ("Phase Constituents", 1),
    "solid_solution":                 ("Phase Constituents", 1),
    "casting":                        ("Processing Routes", 1),
    "wrought":                        ("Processing Routes", 1),
    "sintering":                      ("Processing Routes", 1),
    "annealing":                      ("Processing Routes", 1),
    "tensor_rank":                    ("Tensor Decomposition", 1),
    "tucker_decomposition":           ("Tensor Decomposition", 1),
    "tensor_contraction":             ("Tensor Decomposition", 1),
    "kronecker_product":              ("Tensor Decomposition", 1),
    "core_tensor":                    ("Tensor Decomposition", 1),
    "factor_matrix":                  ("Tensor Decomposition", 1),
    "tensor_completion":              ("Tensor Decomposition", 1),
    "alternating_least_squares":      ("Tensor Decomposition", 1),
    "low_rank_approximation":         ("Tensor Decomposition", 1),
    "excess_gibbs_energy":            ("CALPHAD Methods", 1),
    "redlich_kister_polynomials":     ("CALPHAD Methods", 1),
    "sublattice_model":               ("CALPHAD Methods", 1),
    "interaction_parameter":          ("CALPHAD Methods", 1),
    "activity_coefficient":           ("CALPHAD Methods", 1),
    "chemical_potential":             ("CALPHAD Methods", 1),
    "scheil_gulliver":                ("CALPHAD Methods", 1),
    "muggianu_extrapolation":         ("CALPHAD Methods", 1),
    "gibbs_duhem_equation":           ("CALPHAD Methods", 1),
    "sluggish_diffusion":             ("MPEA Core Phenomena", 1),
    "severe_lattice_distortion":      ("MPEA Core Phenomena", 1),
    "cocktail_effect":                ("MPEA Core Phenomena", 1),
    "high_entropy_stabilization":     ("MPEA Core Phenomena", 1),
    "entropy_enthalpy_compensation":  ("MPEA Core Phenomena", 1),
    "short_range_order":              ("Microstructure Order", 1),
    "medium_range_order":             ("Microstructure Order", 1),
    "chemical_complexity":            ("Compositional Space", 1),
    "compositional_space":            ("Compositional Space", 1),
    "equiatomic":                     ("Compositional Space", 1),
    "non_equiatomic":                 ("Compositional Space", 1),
    "solid_solution_strengthening":   ("Strengthening Mechanisms", 1),
    "phase_field_model":              ("Phase-Field Modeling", 1),
    "allen_cahn_equation":            ("Phase-Field Modeling", 1),
    "cahn_hilliard_equation":         ("Phase-Field Modeling", 1),
    "kks_model":                      ("Phase-Field Modeling", 1),
    "grand_potential_formulation":    ("Phase-Field Modeling", 1),
    "order_parameter":                ("Phase-Field Modeling", 1),
    "gradient_energy_coefficient":    ("Phase-Field Modeling", 1),
    "double_well_potential":          ("Phase-Field Modeling", 1),
    "interface_mobility":             ("Phase-Field Modeling", 1),
    "anti_trapping_current":          ("Phase-Field Modeling", 1),
    "thin_interface_limit":           ("Phase-Field Modeling", 1),
    "quantitative_phase_field":       ("Phase-Field Modeling", 1),
    "dendritic_growth":               ("Microstructure Evolution", 1),
    "mullins_sekerka_instability":    ("Microstructure Evolution", 1),
    "sidebranching":                  ("Microstructure Evolution", 1),
    "tip_radius":                     ("Microstructure Evolution", 1),
    "growth_velocity":                ("Microstructure Evolution", 1),
    "ostwald_ripening":               ("Microstructure Evolution", 1),
    "grain_boundary_migration":       ("Microstructure Evolution", 1),
    "nucleation_rate":                ("Microstructure Evolution", 1),
    "critical_nucleus_size":          ("Microstructure Evolution", 1),
    "columnar_equiaxed_transition":   ("Microstructure Evolution", 1),
    "texture_development":            ("Microstructure Evolution", 1),
    "interfacial_anisotropy":         ("Microstructure Evolution", 1),
    "physics_informed_neural_network":("AI Surrogate Models", 1),
    "fourier_neural_operator":        ("AI Surrogate Models", 1),
    "deeponet":                       ("AI Surrogate Models", 1),
    "gaussian_process_regression":    ("AI Surrogate Models", 1),
    "proper_orthogonal_decomposition":("AI Surrogate Models", 1),
    "variational_autoencoder":        ("AI Surrogate Models", 1),
    "autoencoder":                    ("AI Surrogate Models", 1),
    "generative_adversarial_network": ("AI Surrogate Models", 1),
    "self_attention":                 ("Transformer Architecture", 1),
    "multi_head_attention":           ("Transformer Architecture", 1),
    "positional_encoding":            ("Transformer Architecture", 1),
    "attention_visualization":        ("Transformer Architecture", 1),
    "query_key_value":                ("Transformer Architecture", 1),
    "feed_forward_network":           ("Transformer Architecture", 1),
    "layer_normalization":            ("Transformer Architecture", 1),
    "residual_connection":            ("Transformer Architecture", 1),
    "transformer_encoder":            ("Transformer Architecture", 1),
    "transformer_decoder":            ("Transformer Architecture", 1),
    "transfer_learning":              ("Learning Strategies", 1),
    "meta_learning":                  ("Learning Strategies", 1),
    "active_learning":                ("Learning Strategies", 1),
    "bayesian_optimization":          ("Learning Strategies", 1),
    "cross_validation":               ("Learning Strategies", 1),
    "epistemic_uncertainty":          ("Uncertainty Quantification", 1),
    "aleatoric_uncertainty":          ("Uncertainty Quantification", 1),
    "confidence_interval":            ("Uncertainty Quantification", 1),
    "prediction_uncertainty":         ("Uncertainty Quantification", 1),
    "mean_absolute_error":            ("Model Evaluation Metrics", 1),
    "root_mean_square_error":         ("Model Evaluation Metrics", 1),
    "r2_score":                       ("Model Evaluation Metrics", 1),
    "normalized_rmse":                ("Model Evaluation Metrics", 1),
    "mean_absolute_percentage_error": ("Model Evaluation Metrics", 1),
    "dice_coefficient":               ("Model Evaluation Metrics", 1),
    "intersection_over_union":        ("Model Evaluation Metrics", 1),
    "density_functional_theory":      ("Multi-Scale Methods", 1),
    "special_quasirandom_structures": ("Multi-Scale Methods", 1),
    "coherent_potential_approximation":("Multi-Scale Methods", 1),
    "cluster_expansion":              ("Multi-Scale Methods", 1),
    "molecular_dynamics":             ("Multi-Scale Methods", 1),
    "kinetic_monte_carlo":            ("Multi-Scale Methods", 1),
    "cellular_automaton":             ("Multi-Scale Methods", 1),
    "finite_element_method":          ("Multi-Scale Methods", 1),
    "finite_difference_method":       ("Multi-Scale Methods", 1),
}


def get_hierarchy_label(concept_key: str,
                        style: str = "arrow") -> str:
    """
    Build a human-readable hierarchy label for a concept.

    Parameters
    ----------
    concept_key : str
        The canonical name key used in the ontology (e.g. "enthalpy_of_mixing").
    style : str
        "arrow"   → "Thermodynamic Parameters → Enthalpy of Mixing"
        "bracket" → "Thermodynamic Parameters [Enthalpy of Mixing]"
        "dot"     → "Thermodynamic Parameters · Enthalpy of Mixing"
        "leaf"    → just the leaf name, but Title-Cased

    Returns
    -------
    str
    """
    SEPARATOR = {
        "arrow": " → ",
        "bracket": " [",
        "dot": " · ",
        "leaf": "",
    }

    leaf = concept_key.replace("_", " ").title()

    entry = _HIERARCHY_PARENTS.get(concept_key)
    if entry is None or entry[0] is None or style == "leaf":
        return leaf

    parent_label = entry[0]
    sep = SEPARATOR.get(style, " → ")

    if style == "bracket":
        return f"{parent_label}{sep}{leaf}]"
    return f"{parent_label}{sep}{leaf}"


def get_hierarchy_path(concept_key: str) -> List[str]:
    """
    Return the full hierarchy path as a list, e.g.
    ["CoCrFeNi MPEA", "Thermodynamic Parameters", "Enthalpy of Mixing"].

    This is directly usable as the `ids` / `labels` / `parents` arrays
    for a Plotly sunburst chart.
    """
    leaf = concept_key.replace("_", " ").title()
    entry = _HIERARCHY_PARENTS.get(concept_key)

    if entry is None or entry[0] is None:
        return ["CoCrFeNi MPEA", leaf]

    parent_label = entry[0]
    return ["CoCrFeNi MPEA", parent_label, leaf]


def build_sunburst_data(
    graph: nx.Graph,
    node_weights: Optional[Dict[str, float]] = None,
    min_weight: float = 0.0,
) -> Tuple[List[str], List[str], List[float], List[str]]:
    """
    Build the four arrays needed by ``plotly.sunburst``:
    ids, labels, values, parents.

    Parameters
    ----------
    graph : nx.Graph
        The concept graph.
    node_weights : dict or None
        Mapping concept_key → numeric weight (e.g. frequency, importance).
        If None, all nodes get weight 1.
    min_weight : float
        Skip leaf nodes below this weight.

    Returns
    -------
    ids, labels, values, parents : lists
    """
    ids: List[str] = []
    labels: List[str] = []
    values: List[float] = []
    parents: List[str] = []

    # --- Root node ---
    root_id = "CoCrFeNi MPEA"
    ids.append(root_id)
    labels.append("CoCrFeNi MPEA")
    values.append(0)  # root has no intrinsic value in sunburst
    parents.append("")

    # --- Aggregate children per category ---
    category_children: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for node in graph.nodes:
        if node not in _HIERARCHY_PARENTS:
            continue
        parent_label = _HIERARCHY_PARENTS[node][0]
        if parent_label is None:
            continue
        w = (node_weights or {}).get(node, 1.0)
        if w < min_weight:
            continue
        category_children[parent_label].append((node, w))

    # --- Category (tier-1) nodes ---
    for cat_label, children in sorted(category_children.items()):
        cat_id = cat_label  # unique enough
        cat_value = sum(w for _, w in children)
        ids.append(cat_id)
        labels.append(cat_label)
        values.append(cat_value)
        parents.append(root_id)

        # --- Leaf (tier-2) nodes ---
        for child_key, child_w in sorted(children, key=lambda x: -x[1]):
            child_label = child_key.replace("_", " ").title()
            child_id = child_key  # unique
            ids.append(child_id)
            labels.append(child_label)
            values.append(child_w)
            parents.append(cat_id)

    return ids, labels, values, parents

class AdvancedConceptResolver:
    """
    Multi-level concept resolution using ontology, embeddings, and context.
    Faithful port of AgNPs pattern:
    - EAGER single-batch precomputation of ontology embeddings
    - Batch matrix resolution
    """

    def __init__(
        self,
        ontology: DomainOntology,
        embed_model,
        cache_max: int = 2000,
    ) -> None:
        self.ontology = ontology
        self.embed_model = embed_model
        self.resolution_cache: Dict[str, str] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        # v6.1 (Patch 3): both caches are now BOUNDED. On Streamlit Cloud
        # an unbounded embedding_cache alone leaked ~200 MB by batch 2
        # (one 384-d float vector per unique query phrase).
        self._cache_max = max(100, int(cache_max))
        self.similarity_threshold = 0.85
        self.ontology_concepts_list: Optional[List[str]] = None
        self.ontology_embedding_matrix: Optional[np.ndarray] = None
        self._precompute_ontology_embeddings()

    def _trim_embedding_cache(self) -> None:
        """Evict the oldest 30% of entries once the cache overflows.

        Python dicts preserve insertion order, so the first keys are the
        oldest (LRU-ish eviction without an OrderedDict). Called after
        every embedding-match lookup; cheap because it only does real
        work past the cap.
        """
        if len(self.embedding_cache) > self._cache_max:
            keys = list(self.embedding_cache.keys())
            for k in keys[:int(len(keys) * 0.3)]:
                del self.embedding_cache[k]
            gc.collect()

    def _trim_resolution_cache(self) -> None:
        """Same bounded-cache discipline for the str→str resolution cache."""
        if len(self.resolution_cache) > self._cache_max * 4:
            keys = list(self.resolution_cache.keys())
            for k in keys[:int(len(keys) * 0.3)]:
                del self.resolution_cache[k]

    def _precompute_ontology_embeddings(self) -> None:
        concepts: List[str] = []
        all_texts: List[str] = []
        text_counts: List[int] = []

        for canonical, node in self.ontology.concepts.items():
            concepts.append(canonical)
            texts = [canonical] + list(node.synonyms)
            all_texts.extend(texts)
            text_counts.append(len(texts))

        if not all_texts:
            self.ontology_concepts_list = []
            self.ontology_embedding_matrix = np.empty((0, 0))
            return

        with torch.no_grad():
            all_embeddings = self.embed_model.encode(
                all_texts,
                show_progress_bar=False,
                batch_size=64,
                convert_to_numpy=True,
            )

        embeddings: List[np.ndarray] = []
        idx = 0
        for count in text_counts:
            concept_embs = all_embeddings[idx:idx + count]
            embeddings.append(np.mean(concept_embs, axis=0))
            idx += count

        del all_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.ontology_concepts_list = concepts
        self.ontology_embedding_matrix = (
            np.array(embeddings) if embeddings else np.empty((0, 0))
        )

    @timed
    def resolve(
        self, text: str, context: str = "", use_embedding: bool = True
    ) -> Optional[str]:
        self._trim_resolution_cache()  # v6.1: cheap no-op until cap hit
        text_lower = text.lower().strip()
        if text_lower in self.resolution_cache:
            return self.resolution_cache[text_lower]

        canonical = self.ontology.resolve_concept(text)
        if canonical:
            self.resolution_cache[text_lower] = canonical
            return canonical

        canonical = self._substring_match(text_lower)
        if canonical:
            self.resolution_cache[text_lower] = canonical
            return canonical

        if use_embedding and self.ontology_embedding_matrix.size > 0:
            canonical = self._embedding_match(text, context)
            if canonical:
                self.resolution_cache[text_lower] = canonical
                return canonical

        if context:
            canonical = self._context_disambiguation(text_lower, context)
            if canonical:
                self.resolution_cache[text_lower] = canonical
                return canonical

        return None

    @timed
    def resolve_batch(
        self, phrases: List[str], context: str = ""
    ) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        need_embedding: List[str] = []

        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower in self.resolution_cache:
                results[phrase] = self.resolution_cache[phrase_lower]
                continue
            canonical = self.ontology.resolve_concept(phrase)
            if canonical:
                self.resolution_cache[phrase_lower] = canonical
                results[phrase] = canonical
                continue
            sub_match = self._substring_match(phrase_lower)
            if sub_match:
                self.resolution_cache[phrase_lower] = sub_match
                results[phrase] = sub_match
                continue
            need_embedding.append(phrase)

        if need_embedding and self.ontology_embedding_matrix.size > 0:
            query_texts = [
                p if not context else f"{p} in context of {context}"
                for p in need_embedding
            ]
            with torch.no_grad():
                query_embs = self.embed_model.encode(
                    query_texts,
                    show_progress_bar=False,
                    batch_size=64,
                    convert_to_numpy=True,
                )
            sims = cosine_similarity(query_embs, self.ontology_embedding_matrix)
            best_indices = np.argmax(sims, axis=1)
            best_scores = np.max(sims, axis=1)
            for idx, phrase in enumerate(need_embedding):
                if best_scores[idx] > self.similarity_threshold:
                    canonical = self.ontology_concepts_list[best_indices[idx]]
                    self.resolution_cache[phrase.lower().strip()] = canonical
                    results[phrase] = canonical
                else:
                    results[phrase] = None
            del query_embs, sims, best_indices, best_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            for phrase in need_embedding:
                results[phrase] = None

        self._trim_resolution_cache()  # v6.1: bounded cache (Patch 3)
        return results

    def _substring_match(self, text: str) -> Optional[str]:
        for canonical, node in self.ontology.concepts.items():
            all_forms = {canonical.lower()} | node.synonyms
            for form in all_forms:
                if form in text or text in form:
                    if len(form) > 4 and len(text) > 4:
                        return canonical
        return None

    def _embedding_match(self, text: str, context: str = "") -> Optional[str]:
        try:
            query_text = (
                text if not context else f"{text} in context of {context}"
            )
            if query_text not in self.embedding_cache:
                with torch.no_grad():
                    self.embedding_cache[query_text] = self.embed_model.encode(
                        query_text,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
            query_emb = self.embedding_cache[query_text]
            sims = cosine_similarity(
                [query_emb], self.ontology_embedding_matrix
            )[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] > self.similarity_threshold:
                return self.ontology_concepts_list[best_idx]
            return None
        except Exception:
            return None
        finally:
            # v6.1 (Patch 3): keep the cache bounded even on the error path
            self._trim_embedding_cache()

    def _context_disambiguation(
        self, text: str, context: str
    ) -> Optional[str]:
        context_lower = context.lower()
        thermo_indicators = [
            'gibbs', 'thermodynamic', 'energy', 'enthalpy', 'entropy',
            'free energy',
        ]
        microstructure_indicators = [
            'grain', 'dendrite', 'solidification', 'microstructure', 'phase',
        ]
        fluid_indicators = [
            'flow', 'convection', 'navier', 'melt pool', 'fluid',
        ]
        is_thermo = any(ind in context_lower for ind in thermo_indicators)
        is_microstructure = any(
            ind in context_lower for ind in microstructure_indicators
        )
        is_fluid = any(ind in context_lower for ind in fluid_indicators)

        if 'phase' in text:
            if is_thermo and not is_microstructure:
                return "gibbs_free_energy"
            elif is_microstructure and not is_thermo:
                return "fcc_phase"
            elif is_fluid:
                return "liquid_phase"
        if 'interface' in text:
            if is_thermo:
                return "kks_model"
            elif is_microstructure:
                return "grain_boundary"
        return None

    def find_equivalent_concepts(
        self, concepts: List[str]
    ) -> Dict[str, str]:
        equivalence_map: Dict[str, str] = {}
        for concept in concepts:
            canonical = self.resolve(concept)
            if canonical:
                equivalence_map[concept] = canonical
            else:
                equivalence_map[concept] = concept
        return equivalence_map

    def compute_semantic_similarity(
        self, concept1: str, concept2: str
    ) -> float:
        c1 = self.resolve(concept1) or concept1
        c2 = self.resolve(concept2) or concept2
        if c1 == c2:
            return 1.0
        if (
            c2 in self.ontology.get_hypernyms(c1)
            or c1 in self.ontology.get_hypernyms(c2)
        ):
            return 0.9
        if (
            c2 in self.ontology.get_hyponyms(c1)
            or c1 in self.ontology.get_hyponyms(c2)
        ):
            return 0.9
        try:
            with torch.no_grad():
                emb1 = self.embed_model.encode(
                    c1, show_progress_bar=False, convert_to_numpy=True
                )
                emb2 = self.embed_model.encode(
                    c2, show_progress_bar=False, convert_to_numpy=True
                )
            return float(cosine_similarity([emb1], [emb2])[0][0])
        except Exception:
            return 0.0


# ============================================================================
# ENHANCED CONCEPT EXTRACTOR (MPEA-focused)
# ============================================================================
class EnhancedConceptExtractor:
    def __init__(
        self,
        ontology: DomainOntology,
        resolver: AdvancedConceptResolver,
        store_contexts: bool = False,
        store_documents: bool = True,
    ) -> None:
        self.ontology = ontology
        self.resolver = resolver
        self.concept_frequencies: Dict[str, int] = defaultdict(int)
        # v6.1 (Patch 2): `concept_contexts` stored a 200-char snippet per
        # concept per document (~100 MB by batch 2) and was never read by
        # any downstream function — dead-code leak, now disabled by default.
        self.store_contexts = store_contexts
        # v6.1: `document_concepts` kept one concept list per doc id —
        # the same unbounded pattern (leak #6). Batch mode disables it.
        self.store_documents = store_documents
        self.concept_contexts: Dict[str, List[str]] = defaultdict(list)
        self.document_concepts: Dict[int, List[str]] = defaultdict(list)
        self._build_extraction_patterns()
        # Limit keyword regex to top 500 longest keywords to prevent regex engine crash
        all_keywords = self._get_all_keywords()
        if all_keywords:
            sorted_keywords = sorted(all_keywords, key=len, reverse=True)[:500]
            pattern = r'\b(' + '|'.join(
                re.escape(k) for k in sorted_keywords
            ) + r')\b'
            self._keyword_regex = re.compile(pattern, re.IGNORECASE)
        else:
            self._keyword_regex = None

    def _build_extraction_patterns(self) -> None:
        # MPEA quantitative descriptor patterns
        self.alloy_patterns = [r'\bcocrfeni\b', r'\bco-cr-fe-ni\b', r'\bmpea\b', r'\bhea\b']
        self.process_patterns = [r'\bcasting\b', r'\bwrought\b', r'\bsintering\b', r'\bannealing\b']
        self.thermo_patterns = [
            r'\benthalpy\s+of\s+mixing\b',
            r'\bentropy\s+of\s+mixing\b',
            r'\bomega\s+parameter\b',
            r'\bgibbs\s+free\s+energy\b'
        ]
        self.pf_patterns = []
        self.fluid_patterns = []
        self.ai_patterns = []
        self.micro_patterns = [
            r'\bfcc\s+phase\b',
            r'\bbcc\s+phase\b',
            r'\bintermetallic\b',
            r'\bsolid\s+solution\b'
        ]
        self.comp_patterns = []
        self.param_patterns = [
            r'\b(hardness|elongation|yield\s+strength|tensile\s+strength)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:hv|%|gpa|mpa)\b',
            r'\b(vec|omega|delta)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\b'
        ]
        self.cause_effect_patterns = [
            r'\b(increase|decrease|enhance|reduce)\w*\s+(?:in|of)\s+([\w\s-]+?)\s+(?:lead[s]?|result[s]?|cause[s]?)\s+(?:to|in)?\s+([\w\s-]+?)\b',
        ]
        self.all_patterns = (
            self.alloy_patterns + self.process_patterns + self.thermo_patterns
            + self.pf_patterns + self.fluid_patterns + self.ai_patterns
            + self.micro_patterns + self.comp_patterns
        )
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.all_patterns
        ]
        self.compiled_param_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.param_patterns
        ]
        self.compiled_cause_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.cause_effect_patterns
        ]

    @timed
    def extract_from_text(self, text: str, doc_id: int = 0) -> List[str]:
        concepts: Set[str] = set()
        text_lower = text.lower()

        # 1. Pattern matching (domain-specific)
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = (
                        match[0] if match[0]
                        else (match[1] if len(match) > 1 else match[0])
                    )
                concept = match.lower().strip()
                if len(concept) > 3:
                    canonical = self.resolver.resolve(concept, context=text[:200])
                    if canonical:
                        concepts.add(canonical)
                    else:
                        concepts.add(concept)

        # 2. Parameter extraction (truncated context to 200 chars)
        for pattern in self.compiled_param_patterns:
            matches = pattern.findall(text)
            for param_name, value in matches:
                param_concept = f"{param_name.lower().strip()}_{value}"
                canonical = self.resolver.resolve(param_name, context=text[:200])
                if canonical:
                    concepts.add(f"{canonical}_{value}")
                else:
                    concepts.add(param_concept)

        # 3. Localized Context Window Extraction (Prevents Memory issue)
        context_concepts = self._extract_from_context_windows(text)
        concepts.update(context_concepts)

        # 4. Batch resolve remaining raw concepts (limit to 50 to prevent OOM)
        raw_concepts = set()
        for c in concepts:
            if c not in self.ontology.concepts and not self.resolver.resolve(c):
                raw_concepts.add(c)
        if raw_concepts:
            raw_list = list(raw_concepts)[:50]
            resolved_map = self.resolver.resolve_batch(raw_list, context="")
            for raw, canonical in resolved_map.items():
                if canonical:
                    concepts.add(canonical)
                else:
                    concepts.add(raw)

        # Update tracking
        for concept in concepts:
            self.concept_frequencies[concept] += 1
            if self.store_contexts:
                # Opt-in only (v6.1): unbounded per-doc snippet storage
                # was a dead-code memory leak and is off by default.
                self.concept_contexts[concept].append(text[:200])
        if self.store_documents:
            self.document_concepts[doc_id] = list(concepts)
        return list(concepts)

    def _extract_from_context_windows(
        self, text: str, window_size: int = 100
    ) -> Set[str]:
        """Optimized: Resolves locally using a 200-char window instead of full text."""
        if not self._keyword_regex:
            return set()
        candidate_phrases: Set[str] = set()
        text_lower = text.lower()
        match_count = 0
        for match in self._keyword_regex.finditer(text_lower):
            if match_count > 20:  # Cap iterations per abstract
                break
            match_count += 1
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            local_context = text_lower[start:end]
            phrases = re.findall(
                r'\b([a-z]+(?:[-\s][a-z]+){1,3})\b', local_context
            )
            for phrase in phrases:
                if 5 <= len(phrase) <= 40:
                    # Resolve using the small local window
                    canonical = self.resolver.resolve(phrase, context=local_context)
                    if canonical:
                        candidate_phrases.add(canonical)
        return candidate_phrases

    def _get_all_keywords(self) -> Set[str]:
        keywords: Set[str] = set()
        for canonical, node in self.ontology.concepts.items():
            keywords.add(canonical)
            keywords.update(node.synonyms)
        return keywords

    def extract_relationships(self, text: str) -> List[Relationship]:
        relationships: List[Relationship] = []
        for pattern in self.compiled_cause_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) >= 2:
                    source = (
                        match[0] if isinstance(match[0], str) else match[1]
                    )
                    target = (
                        match[-1] if isinstance(match[-1], str) else match[0]
                    )
                    # Truncate context
                    source_canon = self.resolver.resolve(source, context=text[:200])
                    target_canon = self.resolver.resolve(target, context=text[:200])
                    if (
                        source_canon and target_canon
                        and source_canon != target_canon
                    ):
                        rel = Relationship(
                            source=source_canon,
                            target=target_canon,
                            rel_type=RelationshipType.CAUSES,
                            confidence=0.7,
                            evidence=text[:150],
                        )
                        relationships.append(rel)
        return relationships

    def get_concept_frequencies(self) -> Dict[str, int]:
        return dict(self.concept_frequencies)

    def get_concept_contexts(self, concept: str) -> List[str]:
        return self.concept_contexts.get(concept, [])

    def get_document_concepts(self, doc_id: int) -> List[str]:
        return self.document_concepts.get(doc_id, [])


# ============================================================================
# REASONING-ENHANCED GRAPH BUILDER
# ============================================================================
class ReasoningEnhancedGraphBuilder:
    def __init__(
        self, ontology: DomainOntology, extractor: EnhancedConceptExtractor
    ) -> None:
        self.ontology = ontology
        self.extractor = extractor
        self.reasoning_paths: List[List[str]] = []
        self.inferred_edges: Set[Tuple[str, str]] = set()

    @timed
    def build_graph(
        self,
        all_concepts: List[List[str]],
        valid_concepts: List[str],
        concept_to_id: Dict[str, int],
        embed_model=None,
        config: Dict = None,
    ) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(3000)
        nx_graph = nx.Graph()

        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            freq = self.extractor.concept_frequencies.get(c, 0)
            definition = self.ontology.get_definition(c)
            nx_graph.add_node(
                c,
                frequency=freq,
                concept_type=concept_type.value,
                definition=definition,
                degree=0,
            )

        cooccurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for concepts in all_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if u != v:
                        key = tuple(sorted([u, v]))
                        cooccurrence_map[key] += 1

        for (u, v), count in cooccurrence_map.items():
            nx_graph.add_edge(
                u, v,
                weight=count,
                cooccurrence=count,
                semantic=0,
                edge_type='cooccurrence',
                inferred=False,
            )

        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(nx_graph, valid_concepts, embed_model, config)

        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)
            self._add_cause_effect_edges(nx_graph)
            self._add_hierarchical_edges(nx_graph, valid_concepts)

        self._compute_final_weights(nx_graph, config)
        return nx_graph

    def _add_semantic_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str],
        embed_model, config: Dict,
    ) -> None:
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts,
                    show_progress_bar=False,
                    batch_size=64,
                    convert_to_numpy=True,
                )
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i + 1:], start=i + 1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh:
                        if nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3:
                            nx_graph.add_edge(
                                c1, c2,
                                weight=sim * 2,
                                cooccurrence=0,
                                semantic=sim,
                                edge_type='semantic',
                                inferred=False,
                            )
            del embeddings, sim_matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")

    def _add_inferred_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        for rel in self.ontology.relationships:
            if rel.source in valid_concepts and rel.target in valid_concepts:
                if not nx_graph.has_edge(rel.source, rel.target):
                    nx_graph.add_edge(
                        rel.source, rel.target,
                        weight=rel.confidence * 2,
                        cooccurrence=0,
                        semantic=rel.confidence,
                        edge_type=rel.rel_type.value,
                        inferred=True,
                        confidence=rel.confidence,
                    )
                    self.inferred_edges.add((rel.source, rel.target))
        self._infer_cross_domain_bridges(nx_graph, valid_concepts)

    def _infer_cross_domain_bridges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        process_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.PROCESS
        ]
        property_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.PROPERTY
        ]
        for proc in process_nodes:
            for prop in property_nodes:
                if not nx_graph.has_edge(proc, prop):
                    paths = self.ontology.infer_path(proc, prop, max_depth=2)
                    if paths:
                        avg_confidence = 0.6
                        nx_graph.add_edge(
                            proc, prop,
                            weight=avg_confidence,
                            cooccurrence=0,
                            semantic=avg_confidence,
                            edge_type='bridge',
                            inferred=True,
                            path=" -> ".join(paths[0]),
                        )
                        self.inferred_edges.add((proc, prop))
                        self.reasoning_paths.append(paths[0])

    def _add_cause_effect_edges(self, nx_graph: nx.Graph) -> None:
        pass

    def _add_hierarchical_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        for concept in valid_concepts:
            if concept not in self.ontology.concepts:
                continue
            node = self.ontology.concepts[concept]
            for hypernym in node.hypernyms:
                if (
                    hypernym in valid_concepts
                    and not nx_graph.has_edge(concept, hypernym)
                ):
                    nx_graph.add_edge(
                        concept, hypernym,
                        weight=1.0, cooccurrence=0, semantic=0.95,
                        edge_type='hypernym', inferred=True,
                    )
            for hyponym in node.hyponyms:
                if (
                    hyponym in valid_concepts
                    and not nx_graph.has_edge(concept, hyponym)
                ):
                    nx_graph.add_edge(
                        concept, hyponym,
                        weight=1.0, cooccurrence=0, semantic=0.95,
                        edge_type='hyponym', inferred=True,
                    )

    def _compute_final_weights(
        self, nx_graph: nx.Graph, config: Dict
    ) -> None:
        cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.7)
        sem_weight = config.get("SEMANTIC_WEIGHT", 0.2)
        inf_weight = config.get("INFERENCE_WEIGHT", 0.1)
        for u, v, data in nx_graph.edges(data=True):
            cooc = data.get('cooccurrence', 0)
            sem = data.get('semantic', 0)
            inf = 1.0 if data.get('inferred', False) else 0
            conf = data.get('confidence', 0.5)
            data['weight'] = (
                cooc_weight * cooc
                + sem_weight * sem
                + inf_weight * inf * conf
            )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    if num_abstracts <= 50:
        return {
            "MIN_CONCEPT_FREQ": 2, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 1, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.72, "COOCCURRENCE_WEIGHT": 0.5,
            "SEMANTIC_WEIGHT": 0.5, "CLUSTER_SIMILARITY": 0.75,
            "TOP_N_CONCEPTS": 200, "MAX_CONCEPT_LENGTH": 6,
            "INFERENCE_WEIGHT": 0.1,
        }
    elif num_abstracts <= 500:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.78, "COOCCURRENCE_WEIGHT": 0.6,
            "SEMANTIC_WEIGHT": 0.3, "CLUSTER_SIMILARITY": 0.72,
            "TOP_N_CONCEPTS": 500, "MAX_CONCEPT_LENGTH": 8,
            "INFERENCE_WEIGHT": 0.1,
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 5, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 3, "USE_SEMANTIC_CLUSTERING": False,
            "SIMILARITY_THRESHOLD": 0.85, "COOCCURRENCE_WEIGHT": 0.7,
            "SEMANTIC_WEIGHT": 0.2, "CLUSTER_SIMILARITY": 0.68,
            "TOP_N_CONCEPTS": 1000, "MAX_CONCEPT_LENGTH": 10,
            "INFERENCE_WEIGHT": 0.1,
        }


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )


# ============================================================================
# MPEA QUANTITATIVE DESCRIPTOR KEYWORDS
# ============================================================================
COMPOSITIONAL_DESCRIPTORS = [
    "atomic fraction", "nominal composition", "equiatomic", "non-equiatomic",
    "atomic size difference", "atomic radius", "atomic radius difference",
    "electronegativity difference", "valence electron concentration", "vec",
    "average atomic number", "atomic mismatch", "lattice distortion",
    "mean atomic radius", "concentration", "mole fraction"
]
THERMODYNAMIC_PARAMETERS = [
    "enthalpy of mixing", "entropy of mixing", "dimensionless omega parameter",
    "omega parameter", "gibbs free energy", "free energy", "calphad",
    "average melting temperature", "melting temperature", "mixing enthalpy",
    "mixing entropy", "chemical potential", "thermodynamic parameter",
    "configurational entropy", "delta h mix", "delta s mix"
]
MECHANICAL_PROPERTIES = [
    "hardness", "hv", "vickers hardness", "elongation", "ductility",
    "yield strength", "tensile strength", "ultimate tensile strength",
    "pugh's ratio", "cauchy pressure", "bulk modulus", "shear modulus",
    "young's modulus", "elastic modulus", "wear resistance", "fracture toughness",
    "microhardness", "percentage elongation"
]
ASYMMETRY_FACTORS = [
    "asymmetry factor", "melting temperature asymmetry", "shear modulus asymmetry",
    "bulk modulus asymmetry", "enthalpy of mixing asymmetry", "electronegativity asymmetry",
    "atomic size asymmetry", "elemental asymmetry", "property asymmetry"
]
PHASE_CONSTITUENTS = [
    "fcc phase", "bcc phase", "hcp phase", "solid solution", "ss phase",
    "intermetallic", "im phase", "amorphous", "am phase", "laves phase",
    "single phase", "multiphase", "duplex phase", "phase fraction",
    "phase stability", "phase diagram", "phase boundary", "fcc/bcc",
    "solid solution phase", "crystal structure"
]
PROCESSING_ROUTES = [
    "casting", "wrought", "sintering", "powder metallurgy", "annealing",
    "manufacturing route", "thermomechanical processing", "heat treatment",
    "fabrication method", "processing parameter"
]
ALL_DOMAIN_KEYWORDS = (
    COMPOSITIONAL_DESCRIPTORS + THERMODYNAMIC_PARAMETERS + MECHANICAL_PROPERTIES +
    ASYMMETRY_FACTORS + PHASE_CONSTITUENTS + PROCESSING_ROUTES
)
MPEA_QUANTITATIVE_PATTERNS = [
    r'\b(?:atomic\s+size\s+difference|atomic\s+radius\s+difference|delta)\b',
    r'\b(?:valence\s+electron\s+concentration|vec)\b',
    r'\b(?:electronegativity\s+difference|delta\s+chi)\b',
    r'\b(?:enthalpy\s+of\s+mixing|delta\s+h\s*_{0,1}mix)\b',
    r'\b(?:entropy\s+of\s+mixing|delta\s+s\s*_{0,1}mix|configurational\s+entropy)\b',
    r'\b(?:dimensionless\s+omega|omega\s+parameter)\b',
    r"\bpugh'?s\s+ratio|b/g\s+ratio|cauchy\s+pressure\b",
    r'\b(?:asymmetry\s+factor|melting\s+temperature\s+asymmetry|shear\s+modulus\s+asymmetry)\b',
    r'\b(?:hardness|hv|vickers|elongation|ductility|yield\s+strength|tensile\s+strength)\b',
    r'\b(?:fcc|bcc|hcp|solid\s+solution|intermetallic|laves\s+phase)\b',
    r'\b(?:cocrfeni|co-cr-fe-ni|co\s+cr\s+fe\s+ni|cocofeni)\b',
    r'\b(?:casting|wrought|sintering|annealing|powder\s+metallurgy)\b'
]
MPEA_DESCRIPTOR_MAPPING = {
    r'atomic\s+(?:size|radius|fraction|number| mismatch)|electronegativity|valence\s+electron|vec|composition|mole\s+fraction': 'compositional_descriptor',
    r'enthalpy|entropy|omega|gibbs|calphad|melting\s+temperature|thermodynamic|delta\s+[hs]': 'thermodynamic_parameter',
    r"hardness|hv|elongation|ductility|yield|tensile|pugh'?s\s+ratio|b/g\s+ratio|cauchy\s+pressure|bulk\s+modulus|shear\s+modulus|young": 'mechanical_property',
    r'asymmetry|asymmetric': 'asymmetry_factor',
    r'fcc|bcc|hcp|solid\s+solution|intermetallic|amorphous|laves|phase\s+(?:fraction|stability|diagram|boundary)|single\s+phase|multiphase|duplex|crystal\s+structure': 'phase_constituent',
    r'casting|wrought|sintering|powder\s+metallurgy|annealing|manufacturing\s+route|fabrication': 'processing_route'
}


def is_valid_mpea_descriptor_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw.lower() in concept_lower for kw in ALL_DOMAIN_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in MPEA_QUANTITATIVE_PATTERNS)
    generic = {
        'study', 'analysis', 'effect', 'role', 'investigation', 'research',
        'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
        'new', 'recent', 'various', 'different', 'significant', 'important',
        'report', 'demonstrate', 'show', 'result', 'data', 'find', 'present',
        'propose', 'develop', 'investigate', 'discuss', 'conclude', 'alloy',
        'material', 'system', 'sample', 'specimen'
    }
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic


def normalize_mpea_descriptor_term(concept: str) -> str:
    concept = concept.lower().strip()
    concept = re.sub(r'\bco(?:-|\s)?cr(?:-|\s)?fe(?:-|\s)?ni\b', 'cocrfeni', concept)
    concept = re.sub(r'\bcocofeni\b', 'cocrfeni', concept)
    concept = re.sub(r'\bcobalt\s+chromium\s+iron\s+nickel\b', 'cocrfeni', concept)
    concept = re.sub(r'\batomic\s+size\s+difference\b', 'atomic size difference', concept)
    concept = re.sub(r'\bvalence\s+electron\s+concentration\b', 'valence electron concentration', concept)
    concept = re.sub(r'\benthalpy\s+of\s+mixing\b', 'enthalpy of mixing', concept)
    concept = re.sub(r'\bentropy\s+of\s+mixing\b', 'entropy of mixing', concept)
    concept = re.sub(r'\bdimensionless\s+omega\s+parameter\b', 'omega parameter', concept)
    concept = re.sub(r"\bpugh'?s\s+ratio\b", "pugh's ratio", concept)
    concept = re.sub(r'\basymmetry\s+factor\b', 'asymmetry factor', concept)
    concept = re.sub(r'\bvickers\s+hardness\b', 'hardness', concept)
    concept = re.sub(r'\bpercentage\s+elongation\b', 'elongation', concept)
    concept = re.sub(r'\bsolid\s+solution\s+phase\b', 'solid solution', concept)
    concept = re.sub(r'\bintermetallic\s+compound\b', 'intermetallic', concept)
    concept = re.sub(r'\bpowder\s+metallurgy\b', 'sintering', concept)
    return concept


def extract_concepts_from_text(text: str) -> List[str]:
    concepts: Set[str] = set()
    text_lower = text.lower()
    for pattern in MPEA_QUANTITATIVE_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    noun_pattern = (
        r'\b(?:[a-z]+(?:[-\s]?[a-z]+){0,2}[-\s]?)?'
        r'(?:composition|fraction|radius|size|electronegativity|vec|enthalpy|entropy|omega|gibbs|hardness|elongation|ductility|modulus|pugh|cauchy|asymmetry|phase|fcc|bcc|intermetallic|laves|casting|wrought|sintering|annealing)\b'
    )
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_mpea_descriptor_concept(concept):
            concepts.add(concept)
    for keyword in ALL_DOMAIN_KEYWORDS:
        for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text_lower[start:end]
            context_phrases = re.findall(
                r'\b([a-z]+(?:\s+[a-z]+){1,3})\s+'
                r'(?:of|for|in|with|using|via|through|by|to|and|or)\s+'
                + re.escape(keyword) + r'\b',
                context,
            )
            for phrase in context_phrases:
                concept = f"{phrase.strip()} {keyword}"
                if is_valid_mpea_descriptor_concept(concept):
                    concepts.add(concept)
    param_pattern = (
        r'\b([a-z\s]+(?:hardness|elongation|modulus|strength|temperature|vec|omega|delta|entropy|enthalpy))\s+'
        r'(?:of|is|=|:)?\s*(\d+(?:\.\d+)?\s*(?:hv|%|gpa|mpa|k|j/mol|j/(mol\s*k)|dimensionless)?)\b'
    )
    matches = re.findall(param_pattern, text, re.I)
    for param, value in matches:
        concept = f"{param.lower().strip()} {value.lower().strip()}"
        if is_valid_mpea_descriptor_concept(concept):
            concepts.add(concept)
    return list(concepts)


def extract_concepts_from_abstracts(
    df: pd.DataFrame, text_columns: List[str]
) -> Tuple[List[List[str]], List[Dict]]:
    all_concepts: List[List[str]] = []
    all_metrics: List[Dict] = []
    for idx, row in df.iterrows():
        combined_text = ""
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])
        metrics: Dict[str, Any] = {}
        power_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:w|watt)', combined_text, re.I)
        if power_matches:
            metrics['laser_power_w'] = [float(m) for m in power_matches]
        velocity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mm/s|m/s)', combined_text, re.I)
        if velocity_matches:
            metrics['scan_velocity'] = [float(m) for m in velocity_matches]
        temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:k|°c|celsius)', combined_text, re.I)
        if temp_matches:
            metrics['temperature'] = [float(m) for m in temp_matches]
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:j|kj|mj)', combined_text, re.I)
        if energy_matches:
            metrics['energy'] = [float(m) for m in energy_matches]
        pressure_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mpa|gpa|pa)', combined_text, re.I)
        if pressure_matches:
            metrics['pressure'] = [float(m) for m in pressure_matches]
        all_metrics.append(metrics)
        concepts = extract_concepts_from_text(combined_text)
        normalized = [normalize_mpea_descriptor_term(c) for c in concepts]
        all_concepts.append(normalized)
    return all_concepts, all_metrics


def cluster_similar_concepts(
    valid_concepts: List[str], embed_model, similarity_threshold: float = 0.75
) -> Tuple[List[str], Dict[str, str]]:
    if len(valid_concepts) < 5:
        return valid_concepts, {c: c for c in valid_concepts}
    try:
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts,
                show_progress_bar=False,
                batch_size=64,
                convert_to_numpy=True,
            )
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            linkage='average',
            metric='cosine',
        ).fit(embeddings)
        cluster_members: Dict[int, List[str]] = defaultdict(list)
        concept_to_cluster: Dict[str, int] = {}
        for idx, label in enumerate(clustering.labels_):
            concept = valid_concepts[idx]
            cluster_members[label].append(concept)
            concept_to_cluster[concept] = label
        cluster_representatives: Dict[int, str] = {}
        for label, members in cluster_members.items():
            def score(m):
                domain_hits = sum(
                    1 for kw in ALL_DOMAIN_KEYWORDS if kw.lower() in m.lower()
                )
                return (domain_hits, -len(m))
            representative = max(members, key=score)
            cluster_representatives[label] = representative
        final_mapping = {
            c: cluster_representatives[label]
            for c, label in concept_to_cluster.items()
        }
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return list(cluster_representatives.values()), final_mapping
    except Exception as e:
        st.warning(f"Semantic clustering skipped: {e}")
        return valid_concepts, {c: c for c in valid_concepts}


def normalize_and_filter_concepts(
    all_concepts: List[List[str]], config: Dict
) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
    concept_counts: Dict[str, int] = defaultdict(int)
    concept_abstract_map: Dict[str, List[int]] = defaultdict(list)
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc: Set[str] = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_mpea_descriptor_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    min_freq = config.get("MIN_CONCEPT_FREQ", 5)
    min_words = config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    max_words = config.get("MAX_CONCEPT_LENGTH", 10)
    valid_concepts = [
        c for c, cnt in concept_counts.items()
        if cnt >= min_freq and min_words <= len(c.split()) <= max_words
    ]
    if config.get("USE_SEMANTIC_CLUSTERING", False) and len(valid_concepts) > 50:
        try:
            embed_model = load_embedding_model()
            valid_concepts, concept_to_cluster = cluster_similar_concepts(
                valid_concepts, embed_model,
                similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.72),
            )
            new_abstract_map: Dict[str, List[int]] = defaultdict(list)
            for orig_concept, docs in concept_abstract_map.items():
                clustered = concept_to_cluster.get(orig_concept, orig_concept)
                if clustered in valid_concepts:
                    new_abstract_map[clustered].extend(docs)
            concept_abstract_map = new_abstract_map
        except Exception as e:
            st.warning(f"Semantic clustering skipped: {e}")
    valid_concepts = sorted(
        valid_concepts, key=lambda c: concept_counts[c], reverse=True
    )
    top_n = config.get("TOP_N_CONCEPTS", 1000)
    if len(valid_concepts) > top_n:
        valid_concepts = valid_concepts[:top_n]
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map


def abstract_concepts_to_categories(concepts: List[str]) -> Dict[str, str]:
    concept_to_abstract: Dict[str, str] = {}
    for concept in concepts:
        matched = False
        for pattern, category in MPEA_DESCRIPTOR_MAPPING.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            if any(re.search(p, concept, re.I) for p in [r'\bcocrfeni', r'\bco-cr-fe-ni']):
                concept_to_abstract[concept] = 'compositional_descriptor'
            elif any(re.search(p, concept, re.I) for p in [r'\bvec', r'\batomic', r'\belectronegativity']):
                concept_to_abstract[concept] = 'compositional_descriptor'
            elif any(re.search(p, concept, re.I) for p in [r'\benthalpy', r'\bentropy', r'\bgibbs', r'\bomega']):
                concept_to_abstract[concept] = 'thermodynamic_parameter'
            elif any(re.search(p, concept, re.I) for p in [r'\bhardness', r'\belongation', r'\bmodulus', r'\bpugh']):
                concept_to_abstract[concept] = 'mechanical_property'
            elif any(re.search(p, concept, re.I) for p in [r'\basymmetry']):
                concept_to_abstract[concept] = 'asymmetry_factor'
            elif any(re.search(p, concept, re.I) for p in [r'\bfcc', r'\bbcc', r'\bphase', r'\bintermetallic']):
                concept_to_abstract[concept] = 'phase_constituent'
            elif any(re.search(p, concept, re.I) for p in [r'\bcasting', r'\bwrought', r'\bsintering', r'\bannealing']):
                concept_to_abstract[concept] = 'processing_route'
            else:
                concept_to_abstract[concept] = 'general'
    return concept_to_abstract


# ============================================================================
# CONCEPT DISTILLATION (Memory-safe)
# ============================================================================
def compute_concept_distillation(
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    all_texts: Union[List[str], Dict[int, str]],
    max_docs_per_concept: int = 30,
) -> pd.DataFrame:
    """Memory-safe concept distillation (v6.1 rewrite — Patch 4)."""
    distill_data: List[Dict[str, Any]] = []
    doc_corpus: List[str] = []

    texts_is_dict = isinstance(all_texts, dict)
    n_texts = len(all_texts)

    for c in valid_concepts:
        doc_indices = concept_abstract_map.get(c, [])
        if max_docs_per_concept and len(doc_indices) > max_docs_per_concept:
            doc_indices = doc_indices[:max_docs_per_concept]
        if texts_is_dict:
            doc_text = " ".join([
                all_texts[i] for i in doc_indices
                if i in all_texts
            ])
        else:
            doc_text = " ".join([
                all_texts[i] for i in doc_indices
                if isinstance(i, int) and 0 <= i < n_texts
            ])
        doc_corpus.append(doc_text)

    tfidf = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        stop_words='english', max_features=2000,
    )
    try:
        if any(doc_corpus) and any(t.strip() for t in doc_corpus):
            tfidf_matrix = tfidf.fit_transform(doc_corpus)
            tfidf_scores = tfidf_matrix.max(axis=1).A1
            del tfidf_matrix
        else:
            tfidf_scores = np.ones(len(valid_concepts))
    except Exception:
        tfidf_scores = np.ones(len(valid_concepts))
    gc.collect()

    embed_model = load_embedding_model()

    for i, c in enumerate(valid_concepts):
        freq = len(concept_abstract_map.get(c, []))
        semantic_density = float(tfidf_scores[i])
        coherence = 0.0
        if freq > 1 and doc_corpus[i].strip():
            try:
                words = doc_corpus[i].split()[:20]
                with torch.no_grad():
                    concept_embeddings = embed_model.encode(
                        words, show_progress_bar=False,
                        batch_size=16, convert_to_numpy=True,
                    )
                if len(concept_embeddings) > 1:
                    sim_matrix = cosine_similarity(concept_embeddings)
                    coherence = float(np.mean(
                        sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                    ))
                    del sim_matrix
                del concept_embeddings, words
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                coherence = 0.0
        distill_data.append({
            "concept": c,
            "frequency": freq,
            "tfidf_weight": semantic_density,
            "semantic_density": semantic_density,
            "coherence_score": float(coherence),
            "distillation_efficiency": float(
                semantic_density * np.log1p(freq) * (0.5 + 0.5 * coherence)
            ),
        })

    del doc_corpus
    gc.collect()
    return pd.DataFrame(distill_data).sort_values(
        "distillation_efficiency", ascending=False
    )


# ============================================================================
# LEGACY GRAPH CONSTRUCTION (FALLBACK)
# ============================================================================
def build_hybrid_graph(
    all_concepts: List[List[str]],
    valid_concepts: List[str],
    concept_to_id: Dict[str, int],
    embed_model=None,
    config: Dict = None,
    ontology: DomainOntology = None,
) -> nx.Graph:
    if config is None:
        config = get_adaptive_config(3000)
    nx_graph = nx.Graph()
    for c in valid_concepts:
        concept_type = ontology.get_concept_type(c).value if ontology else 'general'
        definition = ontology.get_definition(c) if ontology else ''
        nx_graph.add_node(
            c, frequency=0, concept_type=concept_type, definition=definition,
        )
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(
                        u, v, weight=1, cooccurrence=1, semantic=0,
                        edge_type='cooccurrence',
                    )
                nx_graph.nodes[u]['frequency'] = (
                    nx_graph.nodes[u].get('frequency', 0) + 1
                )
                nx_graph.nodes[v]['frequency'] = (
                    nx_graph.nodes[v].get('frequency', 0) + 1
                )
    if embed_model and len(valid_concepts) >= 10:
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts, show_progress_bar=False,
                    batch_size=64, convert_to_numpy=True,
                )
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i + 1:], start=i + 1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh and (
                        nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3
                    ):
                        nx_graph.add_edge(
                            c1, c2, weight=sim * 2, cooccurrence=0,
                            semantic=sim, edge_type='semantic',
                        )
            del embeddings, sim_matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.9)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.1)
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    return nx_graph


def sample_edges_for_training(
    nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_to_id: Dict[str, int],
    config: Dict = None,
    memory_safe: bool = False,
) -> Tuple[List[Tuple], List[Tuple]]:
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs: List[Tuple[int, int]] = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    if memory_safe:
        target_negs = min(len(pos_pairs) * 2 if pos_pairs else 30, 2000)
    else:
        target_negs = min(len(pos_pairs) * 3 if pos_pairs else 30, 5000)
    attempts = 0
    max_attempts = 50000
    if memory_safe:
        path_lengths = {}
    else:
        try:
            path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=3))
        except Exception:
            path_lengths = {}
    while len(neg_pairs) < target_negs and attempts < max_attempts:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            attempts += 1
            continue
        dist = path_lengths.get(u_c, {}).get(v_c, 999)
        if dist == 2 or dist == 3:
            neg_pairs.append((int(u_idx), int(v_idx)))
        elif dist == 999 and np.random.rand() < 0.1:
            neg_pairs.append((int(u_idx), int(v_idx)))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append((int(u_idx), int(v_idx)))
    return pos_pairs, neg_pairs


# ============================================================================
# GNN MODEL
# ============================================================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, adj_indices, adj_values, num_nodes, h,
        pos_u, pos_v, neg_u, neg_v,
    ):
        A = sparse.FloatTensor(
            adj_indices, adj_values, torch.Size([num_nodes, num_nodes])
        ).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(
            self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1))
        )
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(
            torch.cat([h2[pos_u], h2[pos_v]], dim=1)
        ).squeeze(1)
        neg_scores = self.decoder(
            torch.cat([h2[neg_u], h2[neg_v]], dim=1)
        ).squeeze(1)
        return pos_scores, neg_scores, h2


def train_gnn(
    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs,
    progress_callback=None, epochs: int = 50, lr: float = 1e-3,
):
    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384
    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2:
            pos_pairs = [(nodes[0], nodes[1])]
        else:
            raise ValueError("Cannot train GNN with fewer than 2 concepts")
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    target_device = (
        node_features.device if node_features.numel() > 0
        else torch.device('cpu')
    )
    pos_u = torch.tensor(
        [p[0] for p in pos_pairs], dtype=torch.long, device=target_device
    )
    pos_v = torch.tensor(
        [p[1] for p in pos_pairs], dtype=torch.long, device=target_device
    )
    neg_u = (
        torch.tensor(
            [n[0] for n in neg_pairs], dtype=torch.long, device=target_device
        )
        if neg_pairs
        else torch.tensor([], dtype=torch.long, device=target_device)
    )
    neg_v = (
        torch.tensor(
            [n[1] for n in neg_pairs], dtype=torch.long, device=target_device
        )
        if neg_pairs
        else torch.tensor([], dtype=torch.long, device=target_device)
    )
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=128).to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if len(neg_pairs) == 0:
            pos_out, _, _ = model(
                adj_indices, adj_values, num_nodes, node_features,
                pos_u, pos_v, pos_u[:1], pos_v[:1],
            )
            loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
        else:
            pos_out, neg_out, _ = model(
                adj_indices, adj_values, num_nodes, node_features,
                pos_u, pos_v, neg_u, neg_v,
            )
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            loss = 0.5 * (pos_loss + neg_loss)
        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(
            adj_indices, adj_values, num_nodes, node_features,
            pos_u[:1], pos_v[:1],
            neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1],
            neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1],
        )
    return model, final_embeddings.cpu(), adj_indices.cpu(), adj_values.cpu()


# ============================================================================
# RESEARCH DIRECTION SCORING
# ============================================================================
def compute_research_direction_scores(
    model, node_features, final_emb, nx_graph,
    valid_concepts, concept_properties, ridge,
    embed_model, n_samples: int = 5000,
) -> pd.DataFrame:
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(
        n_concepts, size=min(n_samples, n_concepts * 5)
    )
    v_ids = np.random.randint(
        n_concepts, size=min(n_samples, n_concepts * 5)
    )
    candidate_pairs: List[Tuple[int, int, str, str]] = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx:
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            continue
        candidate_pairs.append((int(u_idx), int(v_idx), u_c, v_c))
    if not candidate_pairs:
        return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pair_features = torch.cat(
            [final_emb[u_tensor], final_emb[v_tensor]], dim=1
        )
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).numpy()
    with torch.no_grad():
        emb_np = embed_model.encode(
            valid_concepts, show_progress_bar=False,
            batch_size=64, convert_to_numpy=True,
        )
    cos_sims = np.sum(
        emb_np[u_tensor.numpy()] * emb_np[v_tensor.numpy()], axis=1
    )
    results: List[Dict[str, Any]] = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(
                    ridge.predict([[p_u, p_v, 1.0]])[0]
                )
            except Exception:
                expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = (
            np.exp(-0.5 * semantic_novelty)
            * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        )
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = (
            np.clip((expected_improvement - 50) / 200, 0, 1)
            if expected_improvement > 0 else 0
        )
        D_uv = (
            alpha['gnn'] * gnn_scores[i]
            + alpha['novelty'] * semantic_novelty
            + alpha['gain'] * norm_gain
            + alpha['feas'] * (1.0 - feasibility)
        )
        results.append({
            'concept_u': u_c, 'concept_v': v_c,
            'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty),
            'expected_property_gain': expected_improvement,
            'feasibility_score': float(feasibility),
            'composite_score': float(D_uv),
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    del emb_np
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df.head(min(100, len(df)))


# ============================================================================
# MATHEMATICAL VALIDATION
# ============================================================================
def validate_graph_metrics(
    nx_graph: nx.Graph, valid_concepts: List[str]
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if nx_graph.number_of_nodes() < 3:
        return metrics
    try:
        from networkx.algorithms import community
        partition = list(community.greedy_modularity_communities(nx_graph))
        metrics["modularity"] = community.modularity(nx_graph, partition)
        metrics["n_communities"] = len(partition)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0
    try:
        embed_model = load_embedding_model()
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts, show_progress_bar=False,
                batch_size=64, convert_to_numpy=True,
            )
        if len(valid_concepts) >= 3:
            labels = np.zeros(len(valid_concepts))
            for i, c in enumerate(valid_concepts):
                for idx, comm in enumerate(
                    partition if 'partition' in locals() else [[]]
                ):
                    if c in comm:
                        labels[i] = idx
                        break
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        else:
            metrics["silhouette_score"] = 0.0
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        metrics["silhouette_score"] = 0.0
    weights = [d.get('weight', 1) for _, _, d in nx_graph.edges(data=True)]
    if len(weights) > 10:
        p_values = []
        for w in weights[:50]:
            permuted = np.random.permutation(weights)
            p_values.append(np.sum(permuted >= w) / len(weights))
        metrics["edge_significance_p_mean"] = float(np.mean(p_values))
        metrics["edge_significant_count"] = int(
            sum(1 for p in p_values if p < 0.05)
        )
    else:
        metrics["edge_significance_p_mean"] = 1.0
        metrics["edge_significant_count"] = 0
    try:
        metrics["avg_betweenness"] = np.mean(
            list(nx.betweenness_centrality(nx_graph).values())
        )
        metrics["avg_closeness"] = np.mean(
            list(nx.closeness_centrality(nx_graph).values())
        )
    except Exception:
        pass
    return metrics


@st.cache_data(ttl=3600)
def compute_bootstrap_ci(
    scores: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05
) -> Tuple[float, float, float]:
    if len(scores) < 2:
        return float(np.mean(scores)), 0.0, 0.0
    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(float(np.mean(sample)))
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), ci_low, ci_high


# ============================================================================
# ADVANCED ANALYTICS (CACHED)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def detect_keyword_bursts(
    df_filtered: pd.DataFrame,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    text_columns: List[str],
    burst_threshold: float = 2.0,
) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame()
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 3:
        return pd.DataFrame()
    year_range = sorted(years.unique())
    burst_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 5:
            continue
        concept_years: List[int] = []
        for idx in doc_indices:
            if (
                idx < len(df_filtered)
                and pd.notna(df_filtered.iloc[idx].get("Year"))
            ):
                concept_years.append(int(df_filtered.iloc[idx]["Year"]))
        if len(concept_years) < 3:
            continue
        year_counts = Counter(concept_years)
        counts = [year_counts.get(y, 0) for y in year_range]
        if len(counts) < 3:
            continue
        window = max(2, len(counts) // 5)
        moving_avg = pd.Series(counts).rolling(
            window=window, min_periods=1
        ).mean()
        burst_scores: List[float] = []
        for i in range(window, len(counts)):
            if moving_avg.iloc[i - 1] > 0:
                ratio = counts[i] / max(moving_avg.iloc[i - 1], 0.1)
                burst_scores.append(float(ratio))
        if burst_scores:
            max_burst = max(burst_scores)
            burst_year = year_range[window + burst_scores.index(max_burst)]
            if max_burst >= burst_threshold:
                burst_data.append({
                    "concept": concept,
                    "burst_score": round(max_burst, 2),
                    "burst_year": burst_year,
                    "total_mentions": len(concept_years),
                    "year_range": f"{min(concept_years)}-{max(concept_years)}",
                })
    return pd.DataFrame(burst_data).sort_values(
        "burst_score", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def detect_semantic_drift(
    df_filtered: pd.DataFrame,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    text_columns: List[str],
    early_fraction: float = 0.3,
    late_fraction: float = 0.3,
) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame()
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 4:
        return pd.DataFrame()
    embed_model = load_embedding_model()
    sorted_years = sorted(years.unique())
    n_years = len(sorted_years)
    early_cutoff = sorted_years[int(n_years * early_fraction)]
    late_cutoff = sorted_years[int(n_years * (1 - late_fraction))]
    drift_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 10:
            continue
        early_texts: List[str] = []
        late_texts: List[str] = []
        for idx in doc_indices:
            if idx >= len(df_filtered):
                continue
            row = df_filtered.iloc[idx]
            year = row.get("Year")
            if pd.isna(year):
                continue
            year = int(year)
            text = " ".join([
                str(row.get(col, ""))
                for col in text_columns if pd.notna(row.get(col))
            ])
            if year <= early_cutoff:
                early_texts.append(text)
            elif year >= late_cutoff:
                late_texts.append(text)
        if len(early_texts) < 3 or len(late_texts) < 3:
            continue
        try:
            with torch.no_grad():
                early_emb = embed_model.encode(
                    early_texts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
                late_emb = embed_model.encode(
                    late_texts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
            early_centroid = np.mean(early_emb, axis=0)
            late_centroid = np.mean(late_emb, axis=0)
            drift = 1.0 - cosine_similarity(
                [early_centroid], [late_centroid]
            )[0][0]
            drift_data.append({
                "concept": concept,
                "semantic_drift": round(float(drift), 4),
                "early_papers": len(early_texts),
                "late_papers": len(late_texts),
                "early_period": f"{sorted_years[0]}-{early_cutoff}",
                "late_period": f"{late_cutoff}-{sorted_years[-1]}",
            })
            del early_emb, late_emb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            continue
    return pd.DataFrame(drift_data).sort_values(
        "semantic_drift", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def build_concept_genealogy(
    _nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    if _nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    try:
        pagerank = nx.pagerank(_nx_graph, weight='weight')
    except Exception:
        pagerank = {n: 1.0 for n in _nx_graph.nodes()}
    try:
        betweenness = nx.betweenness_centrality(_nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in _nx_graph.nodes()}
    genealogy_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in _nx_graph:
            continue
        pr = pagerank.get(concept, 0)
        bc = betweenness.get(concept, 0)
        freq = len(concept_abstract_map.get(concept, []))
        degree = _nx_graph.degree(concept)
        if (
            pr > np.percentile(list(pagerank.values()), 75)
            and degree > np.percentile(
                [_nx_graph.degree(n) for n in _nx_graph.nodes()], 75
            )
        ):
            generation = "Foundational (Parent)"
        elif (
            pr < np.percentile(list(pagerank.values()), 25)
            and degree < np.percentile(
                [_nx_graph.degree(n) for n in _nx_graph.nodes()], 25
            )
        ):
            generation = "Emerging (Child)"
        else:
            generation = "Intermediate"
        genealogy_data.append({
            "concept": concept,
            "pagerank": round(pr, 5),
            "betweenness": round(bc, 5),
            "frequency": freq,
            "degree": degree,
            "generation": generation,
        })
    return pd.DataFrame(genealogy_data).sort_values(
        "pagerank", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def detect_cross_domain_bridges(
    _nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    if _nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    category_map = abstract_concepts_to_categories(valid_concepts)
    try:
        betweenness = nx.betweenness_centrality(_nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in _nx_graph.nodes()}
    bridge_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in _nx_graph:
            continue
        neighbors = list(_nx_graph.neighbors(concept))
        if len(neighbors) < 2:
            continue
        own_cat = category_map.get(concept, 'general')
        neighbor_cats = [category_map.get(n, 'general') for n in neighbors]
        unique_cats = set(neighbor_cats)
        if len(unique_cats) < 2:
            continue
        bridge_score = betweenness.get(concept, 0) * len(unique_cats)
        bridge_data.append({
            "concept": concept,
            "bridge_score": round(bridge_score, 4),
            "betweenness": round(betweenness.get(concept, 0), 4),
            "connected_categories": len(unique_cats),
            "categories": ", ".join(sorted(unique_cats)),
            "degree": len(neighbors),
            "own_category": own_cat,
        })
    return pd.DataFrame(bridge_data).sort_values(
        "bridge_score", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_network_motifs(_nx_graph: nx.Graph) -> Dict[str, Any]:
    if _nx_graph.number_of_nodes() < 3:
        return {}
    motifs: Dict[str, Any] = {}
    try:
        triangles = nx.triangles(_nx_graph)
        motifs["total_triangles"] = sum(triangles.values()) // 3
        motifs["avg_triangles_per_node"] = round(
            np.mean(list(triangles.values())), 2
        )
        motifs["nodes_in_triangles"] = sum(
            1 for v in triangles.values() if v > 0
        )
    except Exception:
        motifs["total_triangles"] = 0
    try:
        cliques = list(nx.find_cliques(_nx_graph))
        clique_sizes = [len(c) for c in cliques]
        motifs["total_cliques"] = len(cliques)
        motifs["max_clique_size"] = max(clique_sizes) if clique_sizes else 0
        motifs["avg_clique_size"] = (
            round(np.mean(clique_sizes), 2) if clique_sizes else 0
        )
        motifs["4cliques"] = sum(1 for c in clique_sizes if c >= 4)
    except Exception:
        motifs["total_cliques"] = 0
    try:
        clustering = nx.clustering(_nx_graph)
        stars: List[Tuple[str, int, float]] = []
        for node in _nx_graph.nodes():
            deg = _nx_graph.degree(node)
            clust = clustering.get(node, 0)
            if deg >= 5 and clust < 0.2:
                stars.append((node, deg, clust))
        stars.sort(key=lambda x: x[1], reverse=True)
        motifs["star_motifs"] = len(stars)
        motifs["top_stars"] = stars[:10]
    except Exception:
        motifs["star_motifs"] = 0
    return motifs


# ============================================================================
# CENTRALITY & DEGREE DISTRIBUTION
# ============================================================================
def compute_centrality_comparison(
    nx_graph: nx.Graph, valid_concepts: List[str]
) -> pd.DataFrame:
    if nx_graph.number_of_nodes() < 3:
        return pd.DataFrame()
    centrality_data: List[Dict[str, Any]] = []
    try:
        degree_c = dict(nx_graph.degree())
        betweenness_c = nx.betweenness_centrality(nx_graph, weight='weight')
        closeness_c = nx.closeness_centrality(nx_graph)
        eigenvector_c = nx.eigenvector_centrality(
            nx_graph, weight='weight', max_iter=1000
        )
        pagerank_c = nx.pagerank(nx_graph, weight='weight')
        for concept in valid_concepts:
            if concept not in nx_graph:
                continue
            centrality_data.append({
                "concept": concept,
                "degree": degree_c.get(concept, 0),
                "betweenness": round(betweenness_c.get(concept, 0), 5),
                "closeness": round(closeness_c.get(concept, 0), 5),
                "eigenvector": round(eigenvector_c.get(concept, 0), 5),
                "pagerank": round(pagerank_c.get(concept, 0), 5),
            })
    except Exception as e:
        st.warning(f"Centrality computation error: {e}")
    return pd.DataFrame(centrality_data)


def plot_degree_distribution(
    nx_graph: nx.Graph, theme: Dict = None
) -> go.Figure:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    degrees = [d for n, d in nx_graph.degree()]
    if len(degrees) < 3:
        return go.Figure()
    degree_counts = Counter(degrees)
    x = sorted(degree_counts.keys())
    y = [degree_counts[k] for k in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='Degree Distribution',
        marker=dict(size=10, color=theme.get('highlight_bg', '#ff6b6b')),
    ))
    fig.update_layout(
        title="Degree Distribution (Log-Log)",
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Degree (k)", yaxis_title="Frequency P(k)",
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    return fig


# ============================================================================
# PUBLICATION-READY EXPORTS
# ============================================================================
def export_publication_figure(
    nx_graph, valid_concepts, concept_abstract_map,
    cmap_name="viridis", dpi=300, figsize=(14, 12),
    filename="mpea_graph_pub.png",
) -> bytes:
    try:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5, iterations=200)
        plt.figure(figsize=figsize, dpi=dpi)
        node_colors = [get_mpea_category_color(n) for n in nx_graph.nodes()]
        node_sizes = [
            max(100, min(800, len(concept_abstract_map.get(n, [])) * 20 + 50))
            for n in nx_graph.nodes()
        ]
        nx.draw(
            nx_graph, pos,
            with_labels=True,
            node_color=node_colors,
            edge_color='lightgray',
            node_size=node_sizes,
            font_size=6,
            font_weight='bold',
            edgecolors='white',
            linewidths=1.5,
            width=0.5,
            alpha=0.9,
        )
        plt.title(
            "CoCrFeNi MPEA Quantitative Descriptor Graph",
            fontsize=14, fontweight='bold', pad=20,
        )
        buf = io.BytesIO()
        plt.savefig(
            buf, format='png', dpi=dpi, bbox_inches='tight',
            facecolor='white', edgecolor='none',
        )
        buf.seek(0)
        plt.close()
        return buf.read()
    except Exception as e:
        st.error(f"Publication figure export failed: {e}")
        return b''


def generate_analysis_report(
    nx_graph, valid_concepts, concept_abstract_map,
    top_scores, distill_df, burst_df, drift_df,
    genealogy_df, bridge_df, motifs, val_metrics,
    df_filtered,
) -> str:
    report: List[str] = []
    report.append("# CoCrFeNi MPEA Quantitative Descriptor Graph Analysis Report")
    report.append(
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
    )
    report.append("## 1. Dataset Overview")
    report.append(f"- **Total Records**: {len(df_filtered)}")
    if 'Year' in df_filtered.columns:
        years = df_filtered['Year'].dropna()
        report.append(
            f"- **Year Range**: {int(years.min())} - {int(years.max())}"
        )
    report.append(f"- **Total Concepts**: {len(valid_concepts)}")
    report.append(f"- **Total Edges**: {nx_graph.number_of_edges()}")
    report.append(f"- **Graph Density**: {nx.density(nx_graph):.4f}")
    report.append("")
    report.append("## 2. Top Concepts by Frequency")
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:20]
    for i, c in enumerate(top_concepts, 1):
        freq = len(concept_abstract_map.get(c, []))
        deg = nx_graph.degree(c)
        report.append(f"{i}. **{c}** - Freq: {freq}, Degree: {deg}")
    report.append("")
    report.append("## 3. Concept Distillation Efficiency (Top 15)")
    if not distill_df.empty:
        for _, row in distill_df.head(15).iterrows():
            report.append(
                f"- **{row['concept']}**: Efficiency="
                f"{row['distillation_efficiency']:.3f}, "
                f"Freq={row['frequency']}, "
                f"Coherence={row['coherence_score']:.3f}"
            )
    report.append("")
    report.append("## 4. Research Direction Recommendations (Top 10)")
    if not top_scores.empty:
        for i, (_, row) in enumerate(top_scores.head(10).iterrows(), 1):
            report.append(
                f"{i}. **{row['concept_u']}** + **{row['concept_v']}** - "
                f"Composite Score: {row['composite_score']:.3f}"
            )
    report.append("")
    report.append("## 5. Keyword Burst Detection")
    if not burst_df.empty:
        for _, row in burst_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Burst Score={row['burst_score']:.2f} "
                f"(Year {row['burst_year']})"
            )
    else:
        report.append("No significant keyword bursts detected.")
    report.append("")
    report.append("## 6. Semantic Drift Detection")
    if not drift_df.empty:
        for _, row in drift_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Drift={row['semantic_drift']:.4f} "
                f"({row['early_period']} -> {row['late_period']})"
            )
    else:
        report.append("No significant semantic drift detected.")
    report.append("")
    report.append("## 7. Cross-Domain Bridge Concepts")
    if not bridge_df.empty:
        for _, row in bridge_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Bridge Score={row['bridge_score']:.4f}, "
                f"Connects {row['connected_categories']} categories"
            )
    else:
        report.append("No cross-domain bridges detected.")
    report.append("")
    report.append("## 8. Network Motif Analysis")
    report.append(f"- Total Triangles: {motifs.get('total_triangles', 0)}")
    report.append(f"- Total Cliques: {motifs.get('total_cliques', 0)}")
    report.append(f"- Max Clique Size: {motifs.get('max_clique_size', 0)}")
    report.append(f"- Star Motifs: {motifs.get('star_motifs', 0)}")
    report.append("")
    report.append("## 9. Graph Validation Metrics")
    report.append(f"- Modularity: {val_metrics.get('modularity', 0):.3f}")
    report.append(
        f"- Silhouette Score: {val_metrics.get('silhouette_score', 0):.3f}"
    )
    report.append(f"- Number of Communities: {val_metrics.get('n_communities', 0)}")
    report.append(f"- Avg Betweenness: {val_metrics.get('avg_betweenness', 0):.3f}")
    report.append("")
    report.append("---")
    report.append("*Report generated by CoCrFeNi MPEA Quantitative Descriptor Graph v6.1*")
    return "\n".join(report)


# ============================================================================
# GRAPH EDIT HISTORY (AgNPs pattern: max_history=20)
# ============================================================================
class GraphEditHistory:
    def __init__(self, max_history: int = 20) -> None:
        self.history: deque = deque(maxlen=max_history)
        self.redo_stack: deque = deque(maxlen=max_history)
        self._snapshot_counter = 0

    def save_snapshot(
        self, nx_graph, valid_concepts, concept_to_id,
        id_to_concept, concept_abstract_map,
    ) -> int:
        snapshot = {
            'id': self._snapshot_counter,
            'nx_graph': copy.copy(nx_graph),
            'valid_concepts': list(valid_concepts),
            'concept_to_id': dict(concept_to_id),
            'id_to_concept': dict(id_to_concept),
            'concept_abstract_map': {
                k: list(v) for k, v in concept_abstract_map.items()
            },
            'timestamp': datetime.now().isoformat(),
        }
        self.history.append(snapshot)
        self._snapshot_counter += 1
        self.redo_stack.clear()
        return snapshot['id']

    def undo(self) -> Optional[Dict]:
        if len(self.history) < 2:
            return None
        current = self.history.pop()
        self.redo_stack.append(current)
        previous = self.history[-1]
        return previous

    def redo(self) -> Optional[Dict]:
        if not self.redo_stack:
            return None
        snapshot = self.redo_stack.pop()
        self.history.append(snapshot)
        return snapshot

    def can_undo(self) -> bool:
        return len(self.history) >= 2

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def get_history_summary(self) -> List[str]:
        return [
            f"Snapshot {s['id']} @ {s['timestamp']}" for s in self.history
        ]


# ============================================================================
# THEME CONFIGURATION
# ============================================================================
THEME_PRESETS = {
    "Bright (Default)": {
        "bg": "#ffffff", "font": "#1e293b",
        "tooltip_bg": "rgba(255,255,255,0.95)",
        "tooltip_border": "#cbd5e1", "tooltip_text": "#1e293b",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.45)",
        "edge_semantic": "rgba(251, 146, 60, 0.40)",
        "edge_bridge": "rgba(250, 204, 21, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(34, 197, 94, 0.45)",
        "edge_unknown": "rgba(148, 163, 184, 0.30)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b",
        "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.15)",
        "plotly_bg": "#ffffff", "plotly_paper": "#ffffff",
        "grid_color": "#e2e8f0", "axis_color": "#64748b",
    },
    "Dark": {
        "bg": "#0f172a", "font": "#e2e8f0",
        "tooltip_bg": "rgba(15, 23, 42, 0.95)",
        "tooltip_border": "#334155", "tooltip_text": "#e2e8f0",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.55)",
        "edge_semantic": "rgba(251, 146, 60, 0.50)",
        "edge_bridge": "rgba(250, 204, 21, 0.65)",
        "edge_inferred": "rgba(139, 92, 246, 0.60)",
        "edge_cause": "rgba(239, 68, 68, 0.65)",
        "edge_hypernym": "rgba(34, 197, 94, 0.55)",
        "edge_unknown": "rgba(148, 163, 184, 0.40)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b",
        "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.6)",
        "plotly_bg": "#0f172a", "plotly_paper": "#0f172a",
        "grid_color": "#1e293b", "axis_color": "#94a3b8",
    },
    "Midnight": {
        "bg": "#020617", "font": "#f1f5f9",
        "tooltip_bg": "rgba(2, 6, 23, 0.97)",
        "tooltip_border": "#1e293b", "tooltip_text": "#f1f5f9",
        "edge_cooccurrence": "rgba(99, 102, 241, 0.55)",
        "edge_semantic": "rgba(236, 72, 153, 0.50)",
        "edge_bridge": "rgba(34, 211, 238, 0.65)",
        "edge_inferred": "rgba(168, 85, 247, 0.60)",
        "edge_cause": "rgba(244, 63, 94, 0.65)",
        "edge_hypernym": "rgba(52, 211, 153, 0.55)",
        "edge_unknown": "rgba(71, 85, 105, 0.40)",
        "node_border": "#e2e8f0", "highlight_bg": "#f43f5e",
        "hover_bg": "#22d3ee",
        "shadow_color": "rgba(0,0,0,0.7)",
        "plotly_bg": "#020617", "plotly_paper": "#020617",
        "grid_color": "#0f172a", "axis_color": "#64748b",
    },
    "Warm": {
        "bg": "#fff7ed", "font": "#431407",
        "tooltip_bg": "rgba(255, 247, 237, 0.97)",
        "tooltip_border": "#fdba74", "tooltip_text": "#431407",
        "edge_cooccurrence": "rgba(234, 88, 12, 0.45)",
        "edge_semantic": "rgba(180, 83, 9, 0.40)",
        "edge_bridge": "rgba(202, 138, 4, 0.55)",
        "edge_inferred": "rgba(147, 51, 234, 0.50)",
        "edge_cause": "rgba(220, 38, 38, 0.55)",
        "edge_hypernym": "rgba(22, 163, 74, 0.45)",
        "edge_unknown": "rgba(120, 53, 15, 0.25)",
        "node_border": "#fff7ed", "highlight_bg": "#dc2626",
        "hover_bg": "#f59e0b",
        "shadow_color": "rgba(124, 45, 18, 0.15)",
        "plotly_bg": "#fff7ed", "plotly_paper": "#fff7ed",
        "grid_color": "#fed7aa", "axis_color": "#9a3412",
    },
    "Forest": {
        "bg": "#f0fdf4", "font": "#052e16",
        "tooltip_bg": "rgba(240, 253, 244, 0.97)",
        "tooltip_border": "#86efac", "tooltip_text": "#052e16",
        "edge_cooccurrence": "rgba(22, 163, 74, 0.45)",
        "edge_semantic": "rgba(5, 150, 105, 0.40)",
        "edge_bridge": "rgba(234, 179, 8, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(21, 128, 61, 0.45)",
        "edge_unknown": "rgba(20, 83, 45, 0.25)",
        "node_border": "#f0fdf4", "highlight_bg": "#15803d",
        "hover_bg": "#84cc16",
        "shadow_color": "rgba(20, 83, 45, 0.15)",
        "plotly_bg": "#f0fdf4", "plotly_paper": "#f0fdf4",
        "grid_color": "#bbf7d0", "axis_color": "#166534",
    },
    "Ocean": {
        "bg": "#ecfeff", "font": "#083344",
        "tooltip_bg": "rgba(236, 254, 255, 0.97)",
        "tooltip_border": "#67e8f9", "tooltip_text": "#083344",
        "edge_cooccurrence": "rgba(6, 182, 212, 0.45)",
        "edge_semantic": "rgba(14, 165, 233, 0.40)",
        "edge_bridge": "rgba(99, 102, 241, 0.55)",
        "edge_inferred": "rgba(168, 85, 247, 0.50)",
        "edge_cause": "rgba(244, 63, 94, 0.55)",
        "edge_hypernym": "rgba(13, 148, 136, 0.45)",
        "edge_unknown": "rgba(21, 94, 117, 0.25)",
        "node_border": "#ecfeff", "highlight_bg": "#0ea5e9",
        "hover_bg": "#22d3ee",
        "shadow_color": "rgba(8, 51, 68, 0.15)",
        "plotly_bg": "#ecfeff", "plotly_paper": "#ecfeff",
        "grid_color": "#a5f3fc", "axis_color": "#0e7490",
    },
}

PHYSICS_PRESETS = {
    "Stable (Default)": {
        "damping": 0.55, "gravity": -2500, "spring_length": 140,
        "spring_strength": 0.05, "central_gravity": 0.25,
        "stabilization": 2500,
    },
    "Fluid": {
        "damping": 0.25, "gravity": -1800, "spring_length": 120,
        "spring_strength": 0.05, "central_gravity": 0.30,
        "stabilization": 1500,
    },
    "Tight": {
        "damping": 0.70, "gravity": -4000, "spring_length": 80,
        "spring_strength": 0.08, "central_gravity": 0.20,
        "stabilization": 3000,
    },
    "Off": {
        "damping": 0.99, "gravity": 0, "spring_length": 200,
        "spring_strength": 0.0, "central_gravity": 0.0,
        "stabilization": 0,
    },
}


# ============================================================================
# VISUALIZATION FUNCTIONS (AgNPs Pattern — tempfile + Glassmorphism JS)
# ============================================================================
def get_mpea_category_color(
    concept: str, cmap_colors: Optional[List[str]] = None
) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    category = 'general'
    for pattern, cat in MPEA_DESCRIPTOR_MAPPING.items():
        if re.search(pattern, concept_lower):
            category = cat
            break
    color_map = {
        'compositional_descriptor': '#1f77b4',
        'thermodynamic_parameter': '#ff7f0e',
        'mechanical_property': '#2ca02c',
        'asymmetry_factor': '#d62728',
        'phase_constituent': '#9467bd',
        'processing_route': '#8c564b',
        'general': '#7f7f7f'
    }
    return color_map.get(category, '#7f7f7f')


# -----------------------------------------------------------------------------
# NEW: render_graph_pyvis with edge lightness, color mode, tooltip & legend font sizes
# -----------------------------------------------------------------------------
def render_graph_pyvis(
    nx_graph, concept_abstract_map, physics_enabled=True,
    min_node_size=8, max_node_size=40, cmap_name="viridis",
    custom_labels=None, node_label_size=12, node_label_position="center",
    top_n_nodes=0, theme=None, physics_preset=None,
    show_edge_weights=False, edge_label_mode="hover",
    show_reasoning=False, use_abbreviated_labels=False,
    max_label_length=15,
    node_font_face="Inter, Segoe UI, Roboto, sans-serif",
    edge_label_size=10, edge_label_color=None,
    edge_label_position="middle", enable_node_highlight=True,
    show_definitions=True,
    # NEW parameters
    edge_lightness=0.6,
    edge_color_mode="theme",
    custom_edge_color="#AAAAAA",
    tooltip_font_size=13,
    node_legend_font_size=13,
) -> None:
    """
    Hybrid renderer: Original structure + Edge colors + Hierarchy labels.
    Physics-friendly: No pre-computed x,y, dynamic smoothing, no custom mass.
    """
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(
            degrees.keys(), key=lambda x: degrees[x], reverse=True
        )[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    cmap_colors = get_colormap_colors(
        cmap_name, max(1, len(nx_graph.nodes()))
    )
    net = Network(
        height="780px", width="100%",
        bgcolor=theme['bg'], font_color=theme['font'],
        select_menu=True, notebook=False, cdn_resources='remote',
    )

    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
var options = {{
    "physics": {{
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {{
            "gravitationalConstant": {physics_preset['gravity']},
            "centralGravity": {physics_preset['central_gravity']},
            "springLength": {physics_preset['spring_length']},
            "springConstant": {physics_preset['spring_strength']},
            "damping": {physics_preset['damping']},
            "overlap": 0.15
        }},
        "stabilization": {{
            "enabled": true,
            "iterations": 500,
            "updateInterval": 50,
            "onlyDynamicEdges": true,
            "fit": true
        }}
    }},
    "interaction": {{
        "hover": true,
        "tooltipDelay": 180,
        "hideEdgesOnDrag": false,
        "zoomView": true,
        "dragView": true
    }}
}}
""")
    else:
        net.set_options("""
var options = {
    "physics": { "enabled": false },
    "interaction": {
        "hover": true, "dragNodes": true,
        "dragView": true, "zoomView": true
    }
}
""")

    label_map = {}
    n_counter = 1

    # Track used relationship types for legend
    used_rel_types = {}

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(
            min_node_size + freq * 1.2, min_node_size, max_node_size
        ))
        color = get_mpea_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        original_label = (
            custom_labels.get(node, node) if custom_labels else node
        )

        # Use hierarchy label
        label = get_hierarchy_label(node, style="arrow")

        if (
            use_abbreviated_labels
            and len(original_label) > max_label_length
        ):
            short_label = f"N{n_counter}"
            label_map[short_label] = original_label
            n_counter += 1
            label = short_label
            node_shape = 'circle'
            inside_font_size = max(8, min(int(size * 0.55), 14))
            font_dict = {
                'color': '#ffffff',
                'size': inside_font_size,
                'face': node_font_face,
                'bold': True,
            }
        else:
            label = label  # Use hierarchy label
            node_shape = 'dot'
            font_dict = {
                'color': theme['font'],
                'size': node_label_size,
                'face': node_font_face,
                'strokeWidth': 0,
                'vadjust': -6,
            }

        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        definition = nx_graph.nodes[node].get('definition', '')
        tooltip_content = (
            f"<div style='font-family:{node_font_face};'>"
            f"<b style='font-size:14px;color:{theme['highlight_bg']};'>"
            f"{node}</b><br>"
            f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>"
            f"Type:</span> {concept_type}<br>"
            f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>"
            f"Degree:</span> {degree}<br>"
            f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>"
            f"Frequency:</span> {freq}"
        )
        if show_definitions and definition:
            tooltip_content += (
                f"<br><span style='color:{theme['tooltip_text']};opacity:0.7;'>"
                f"Definition:</span> <i>{definition}</i>"
            )
        if use_abbreviated_labels and label != original_label:
            tooltip_content += (
                f"<br><span style='color:{theme['tooltip_text']};opacity:0.7;'>"
                f"Full Label:</span> {original_label}"
            )
        tooltip_content += "</div>"

        net.add_node(
            node, label=label, size=size,
            color={
                'background': color,
                'border': theme['node_border'],
                'highlight': {
                    'background': theme['highlight_bg'], 'border': '#ffffff'
                },
                'hover': {
                    'background': theme['hover_bg'], 'border': '#ffffff'
                },
            },
            font=font_dict, title=tooltip_content,
            borderWidth=2, borderWidthSelected=3,
            shadow={
                'enabled': True,
                'color': theme['shadow_color'],
                'size': 12, 'x': 4, 'y': 4,
            },
            shape=node_shape,
        )

    # Edge colors: apply lightness and mode
    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)

        # Resolve relationship type for color
        rel_type = RelationshipType.SEMANTIC
        if edge_type != 'unknown':
            try:
                rel_type = RelationshipType(edge_type)
            except ValueError:
                pass

        # Determine base color
        if edge_color_mode == "theme":
            if edge_type == 'unknown':
                base_color = theme['edge_unknown']
            else:
                base_color = get_edge_color(rel_type)
            # Lighten if requested
            if edge_lightness > 0:
                base_color = lighten_hex_color(base_color, edge_lightness)
        elif edge_color_mode == "uniform_grey":
            base_color = lighten_hex_color("#808080", edge_lightness)
        else:  # custom
            base_color = lighten_hex_color(custom_edge_color, edge_lightness)

        # Use theme color as fallback for unknown types (if not already)
        if edge_type == 'unknown' and edge_color_mode == "theme":
            base_color = theme['edge_unknown']

        width = get_edge_width(rel_type) * (0.5 + 0.5 * w)  # scale width by weight
        style = get_edge_style(rel_type)
        dashes = True if style == "dashed" or is_inferred else False

        actual_edge_label_color = (
            edge_label_color if edge_label_color else theme['font']
        )
        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)),
            width=width,
            color={
                'color': base_color,
                'highlight': theme['highlight_bg'],
                'hover': theme['hover_bg'],
                'opacity': 0.85,
            },
            smooth={"type": "dynamic"},
            title=(
                f"<span style='font-family:{node_font_face};'>"
                f"Weight: <b>{w:.2f}</b><br>"
                f"Type: {edge_type}<br>"
                f"Inferred: {is_inferred}</span>"
            ),
            dashes=dashes,
        )

        all_weights = [
            nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()
        ]
        weight_threshold = (
            np.percentile(all_weights, 80) if all_weights else 0
        )
        if (
            edge_label_mode == "all"
            or (edge_label_mode == "threshold" and w >= weight_threshold)
        ):
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {
                'color': actual_edge_label_color,
                'size': int(edge_label_size),
                'background': theme['tooltip_bg'],
                'strokeWidth': 2,
                'strokeColor': theme['node_border'],
                'align': edge_label_position,
                'face': node_font_face,
            }
        net.add_edge(u, v, **edge_kwargs)

        # Track for legend
        if rel_type not in used_rel_types:
            used_rel_types[rel_type] = rel_type.value.replace("_", " ").title()

    # --- Edge-color legend (HTML injected via footer) ---
    if used_rel_types:
        legend_rows = []
        for rt, human in sorted(used_rel_types.items(), key=lambda x: x[1]):
            c = get_edge_color(rt)
            if edge_color_mode == "theme":
                c = lighten_hex_color(c, edge_lightness) if edge_lightness > 0 else c
            elif edge_color_mode == "uniform_grey":
                c = lighten_hex_color("#808080", edge_lightness)
            else:
                c = lighten_hex_color(custom_edge_color, edge_lightness)
            w = get_edge_width(rt)
            s = get_edge_style(rt)
            border = 'border: 1px dashed #888;' if s == "dashed" else 'border: 1px solid transparent;'
            legend_rows.append(
                f'<tr>'
                f'<td style="padding:2px 6px;">'
                f'<span style="display:inline-block;width:{int(20*w)}px;height:3px;'
                f'background:{c};vertical-align:middle;{border}"></span>'
                f'</td>'
                f'<td style="padding:2px 6px;color:#ccc;font-size:11px;">{human}</td>'
                f'</tr>'
            )
        legend_html = (
            '<div style="background:#0d0d1a;border-radius:8px;padding:12px 16px;'
            f'margin-top:8px;max-height:280px;overflow-y:auto;">'
            f'<div style="color:#fff;font-size:13px;font-weight:bold;margin-bottom:6px;">'
            f'Edge Colors ({len(used_rel_types)} relationship types)</div>'
            f'<table style="border-collapse:collapse;">{"".join(legend_rows)}</table>'
            f'</div>'
        )
        net.add_node(
            "__legend__",
            label="",
            shape="dot",
            size=0,
            color="rgba(0,0,0,0)",
            fixed=True,
            x=-500,
            y=-500,
            physics=False,
            title=legend_html,
        )

    try:
        tmp_html = tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False, encoding='utf-8'
        )
        tmp_path = tmp_html.name
        net.write_html(tmp_path, notebook=False)
        tmp_html.close()
        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        if use_abbreviated_labels and label_map:
            label_map_json = json.dumps(label_map)
            hidden_div = (
                f'<div id="hea-label-map-data" style="display:none;">'
                f'{label_map_json}</div>'
            )
            html_content = html_content.replace('</body>', hidden_div + '</body>')
        os.unlink(tmp_path)
    except Exception as e:
        st.error(f"PyVis HTML generation failed: {e}")
        html_content = net.generate_html()

    # Custom CSS: tooltip font size and legend font size
    custom_css = f"""
<style>
body {{
    background: {theme['bg']};
    margin: 0;
    padding: 0;
    font-family: '{node_font_face}', sans-serif;
}}
#mynetwork {{
    border-radius: 16px;
    box-shadow: 0 12px 48px {theme['shadow_color']};
    outline: none;
}}
div.vis-tooltip {{
    background: {theme['tooltip_bg']} !important;
    color: {theme['tooltip_text']} !important;
    border: 1px solid {theme['tooltip_border']} !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    font-family: '{node_font_face}', sans-serif !important;
    font-size: {tooltip_font_size}px !important;
    line-height: 1.5 !important;
    box-shadow: 0 8px 32px {theme['shadow_color']} !important;
    max-width: 320px !important;
    white-space: normal !important;
}}
div.vis-network div.vis-manipulation {{
    background: {theme['tooltip_bg']} !important;
    border-top: 1px solid {theme['tooltip_border']} !important;
    color: {theme['font']} !important;
}}
/* Node legend font size (if abbreviated labels) */
.hea-legend {{
    font-size: {node_legend_font_size}px !important;
}}
</style>
"""
    html_content = html_content.replace('</head>', custom_css + '</head>')

    if enable_node_highlight:
        highlight_js = """
<script>
(function() {
    var checkExist = setInterval(function() {
        if (typeof network !== 'undefined' && network !== null && network.body && network.body.data) {
            clearInterval(checkExist);
            var nodesDS = network.body.data.nodes;
            var edgesDS = network.body.data.edges;
            var savedNodeColors = {};
            var activeNodeId = null;
            var labelMode = 'short';
            var labelMap = {};
            (function initLabelMap() {
                var hidden = document.getElementById('hea-label-map-data');
                if (hidden && hidden.textContent) {
                    try { labelMap = JSON.parse(hidden.textContent); } catch(e) {}
                }
            })();
            function resetAll() {
                var nodeRestores = [];
                for (var nid in savedNodeColors) {
                    nodeRestores.push({id: nid, color: savedNodeColors[nid]});
                }
                if (nodeRestores.length > 0) nodesDS.update(nodeRestores);
                savedNodeColors = {};
                activeNodeId = null;
                var panel = document.getElementById('edge-info-panel');
                if (panel) panel.style.display = 'none';
            }
            function resolveFullName(shortOrId) {
                if (labelMap && labelMap[shortOrId]) return labelMap[shortOrId];
                var n = nodesDS.get(shortOrId);
                if (n && n.title) {
                    var tmp = document.createElement('div');
                    tmp.innerHTML = n.title;
                    var txt = (tmp.textContent || tmp.innerText || '').trim();
                    var firstLine = txt.split('\n')[0];
                    if (firstLine) return firstLine.replace(/<[^>]*>/g,'').trim();
                }
                return shortOrId;
            }
            function formatEdgeRow(e, idx, mode) {
                var typeColor = e.inferred ? '#8b5cf6' : '#0ea5e9';
                var badge = e.inferred ? ' <span style="background:#8b5cf6;color:white;padding:1px 4px;border-radius:3px;font-size:9px;">INFERRED</span>' : '';
                var typeBadge = '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:600;">' + e.type + '</span>';
                var fromName = (mode === 'short') ? e.from : resolveFullName(e.from);
                var toName = (mode === 'short') ? e.to : resolveFullName(e.to);
                return '<div style="padding:8px 10px;margin:4px 0;background:rgba(248,250,252,0.9);border-left:4px solid ' + typeColor + ';border-radius:6px;font-size:12px;">' +
                    '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">' +
                    '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;">' + fromName + '</span>' +
                    '<span style="color:#94a3b8;font-size:13px;">↔</span>' +
                    '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;">' + toName + '</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;gap:8px;padding-left:10px;">' +
                    '<span style="background:#0ea5e9;color:white;font-size:10px;padding:2px 6px;border-radius:4px;font-weight:700;">W: ' + e.weight + '</span>' +
                    typeBadge + badge +
                    '</div></div>';
            }
            function showEdgeInfoPanel(nodeId, connectedEdges) {
                var panel = document.getElementById('edge-info-panel');
                if (!panel) {
                    panel = document.createElement('div');
                    panel.id = 'edge-info-panel';
                    document.body.appendChild(panel);
                }
                panel.style.cssText = 'position:fixed;top:90px;right:20px;width:400px;max-height:calc(100vh - 110px);overflow-y:auto;z-index:9999;' +
                    'background:rgba(255,255,255,0.95);border:1px solid rgba(255,215,0,0.6);border-radius:16px;padding:0;' +
                    'font-family:Inter,Segoe UI,Roboto,sans-serif;box-shadow:0 20px 60px rgba(0,0,0,0.15);backdrop-filter:blur(20px);';
                var nodeData = nodesDS.get(nodeId);
                var nodeName = nodeId;
                var nodeDefinition = ""; var nodeType = ""; var nodeFreq = ""; var nodeDegree = "";
                if (nodeData && nodeData.title) {
                    var tmpDiv = document.createElement("div");
                    tmpDiv.innerHTML = nodeData.title;
                    var tooltipText = tmpDiv.textContent || tmpDiv.innerText || "";
                    var defMatch = tooltipText.match(/Definition:\s*(.+)/i);
                    if (defMatch && defMatch[1]) { nodeDefinition = defMatch[1].trim(); }
                    var typeMatch = tooltipText.match(/Type:\s*(\w+)/i);
                    if (typeMatch && typeMatch[1]) { nodeType = typeMatch[1].trim(); }
                    var freqMatch = tooltipText.match(/Frequency:\s*(\d+)/i);
                    if (freqMatch && freqMatch[1]) { nodeFreq = freqMatch[1].trim(); }
                    var degMatch = tooltipText.match(/Degree:\s*(\d+)/i);
                    if (degMatch && degMatch[1]) { nodeDegree = degMatch[1].trim(); }
                    var nameMatch = tooltipText.match(/^([^\n]+)/);
                    if (nameMatch) nodeName = nameMatch[1].replace(/<[^>]*>/g,'').trim();
                }
                var html = '<div style="padding:16px 20px;background:linear-gradient(135deg,rgba(255,215,0,0.15),rgba(255,183,77,0.1));border-radius:16px 16px 0 0;border-bottom:2px solid rgba(255,215,0,0.4);">';
                html += '<div style="font-size:18px;font-weight:800;color:#1e293b;margin-bottom:8px;">🔬 ' + nodeName + '</div>';
                html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
                if (nodeType) html += '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">' + nodeType + '</span>';
                if (nodeDegree) html += '<span style="background:rgba(168,85,247,0.1);color:#a855f7;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Deg: ' + nodeDegree + '</span>';
                if (nodeFreq) html += '<span style="background:rgba(34,197,94,0.1);color:#22c55e;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Freq: ' + nodeFreq + '</span>';
                html += '</div></div>';
                if (nodeDefinition) {
                    html += '<div style="padding:12px 20px;background:rgba(251,191,36,0.06);border-bottom:1px solid rgba(0,0,0,0.04);">';
                    html += '<div style="font-size:10px;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-bottom:4px;">📖 Definition</div>';
                    html += '<div style="font-size:12px;color:#475569;font-style:italic;line-height:1.4;">' + nodeDefinition + '</div></div>';
                }
                html += '<div style="padding:10px 20px;background:rgba(248,250,252,0.8);border-bottom:1px solid rgba(0,0,0,0.04);display:flex;align-items:center;gap:10px;">';
                html += '<span style="font-size:10px;color:#94a3b8;font-weight:600;">Label Mode</span>';
                html += '<button id="btn-short" onclick="window._heaSetLabelMode(\'short\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:#D32F2F;color:white;">Short</button>';
                html += '<button id="btn-full" onclick="window._heaSetLabelMode(\'full\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:transparent;color:#64748b;">Full</button>';
                html += '</div>';
                html += '<div id="edges-container" style="padding:12px 16px 16px;">';
                var edgeList = [];
                connectedEdges.forEach(function(eId) {
                    var e = edgesDS.get(eId);
                    if (!e) return;
                    var fromNode = nodesDS.get(e.from); var toNode = nodesDS.get(e.to);
                    var fromLabel = fromNode ? (fromNode.label || e.from) : e.from;
                    var toLabel = toNode ? (toNode.label || e.to) : e.to;
                    var w = (typeof e.value === 'number') ? e.value : (e.width || 1);
                    var edgeType = 'unknown', isInferred = false;
                    if (e.title) {
                        var tmpDiv = document.createElement('div'); tmpDiv.innerHTML = e.title;
                        var _txt = tmpDiv.textContent || tmpDiv.innerText || '';
                        var m = _txt.match(/Type:\s*(\w+)/); if (m) edgeType = m[1];
                        if (_txt.indexOf('Inferred: true') !== -1) isInferred = true;
                    }
                    edgeList.push({from: fromLabel, to: toLabel, weight: (typeof w === 'number') ? w.toFixed(2) : String(w), type: edgeType, inferred: isInferred});
                });
                edgeList.sort(function(a,b){ return parseFloat(b.weight)-parseFloat(a.weight); });
                edgeList.forEach(function(e, idx){ html += formatEdgeRow(e, idx, labelMode); });
                html += '</div>';
                panel.innerHTML = html;
                panel.style.display = 'block';
                panel._edgeList = edgeList;
                window._heaSetLabelMode = function(mode) {
                    labelMode = mode;
                    var p = document.getElementById('edge-info-panel');
                    if (!p || !p._edgeList) return;
                    var btnShort = document.getElementById('btn-short');
                    var btnFull = document.getElementById('btn-full');
                    if (mode === 'short') {
                        btnShort.style.background = '#D32F2F'; btnShort.style.color = 'white';
                        btnFull.style.background = 'transparent'; btnFull.style.color = '#64748b';
                    } else {
                        btnFull.style.background = '#D32F2F'; btnFull.style.color = 'white';
                        btnShort.style.background = 'transparent'; btnShort.style.color = '#64748b';
                    }
                    var container = document.getElementById('edges-container');
                    if (container) {
                        var newHtml = '';
                        p._edgeList.forEach(function(e, idx){ newHtml += formatEdgeRow(e, idx, mode); });
                        container.innerHTML = newHtml;
                    }
                };
            }
            network.on("selectNode", function(params) {
                var nodeId = params.nodes[0];
                if (activeNodeId !== null && activeNodeId !== nodeId) resetAll();
                activeNodeId = nodeId;
                var connectedEdges = network.getConnectedEdges(nodeId);
                var connectedNodes = network.getConnectedNodes(nodeId);
                var nodeUpdates = [];
                connectedNodes.forEach(function(nId) {
                    var n = nodesDS.get(nId);
                    if (n && !savedNodeColors[nId]) {
                        savedNodeColors[nId] = JSON.parse(JSON.stringify(n.color));
                        var newColor = JSON.parse(JSON.stringify(n.color));
                        if (typeof newColor === 'string') newColor = {background: newColor, border: '#FFD700'};
                        else newColor.border = '#FFD700';
                        nodeUpdates.push({id: nId, color: newColor, shadow: {enabled: true, color: 'rgba(255,215,0,0.5)', size: 15, x: 0, y: 0}});
                    }
                });
                if (nodeUpdates.length > 0) nodesDS.update(nodeUpdates);
                showEdgeInfoPanel(nodeId, connectedEdges);
            });
            network.on("deselectNode", function(){ resetAll(); });
            network.on("click", function(params){
                if (params.nodes.length === 0 && activeNodeId !== null) resetAll();
            });
        }
    }, 250);
})();
</script>
"""
        html_content = html_content.replace('</body>', highlight_js + '</body>')

    st.components.v1.html(html_content, height=790, scrolling=True)

    if use_abbreviated_labels and label_map:
        st.markdown("---")
        st.markdown("### 🗺️ Node Label Legend")
        st.caption("Hover over nodes in the interactive graph to see their full names and definitions.")
        sorted_legend = sorted(label_map.items(), key=lambda x: int(x[0][1:]))
        cols = st.columns(4)
        for i, (short, full) in enumerate(sorted_legend):
            with cols[i % 4]:
                st.markdown(
                    f"""<div style='padding:8px; border-radius:6px; background-color:{theme.get('tooltip_bg', '#f8fafc')};
border-left:4px solid {theme.get('highlight_bg', '#ff6b6b')}; margin-bottom:6px; font-size:{node_legend_font_size}px;'>
<b style='color:{theme.get('highlight_bg', '#ff6b6b')}; font-size:{node_legend_font_size+1}px;'>{short}</b>:
<span style='font-size:{node_legend_font_size}px; color:{theme.get('font', '#1e293b')};'>{full}</span>
</div>""",
                    unsafe_allow_html=True,
                )

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button(
            "Download Interactive Graph (HTML)",
            data=html_bytes,
            file_name="mpea_concept_graph.html",
            mime="text/html",
        )
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")


# -----------------------------------------------------------------------------
# NEW: render_sunburst_chart with legend font size parameter
# -----------------------------------------------------------------------------
def render_sunburst_chart(
    labels, parents, values, cmap_name="viridis",
    label_size=20, width=900, height=700,
    theme=None, branchvalues="total",
    show_labels=True, show_values=False,
    hover_info="all", color_continuous_scale=None,
    font_family="Arial, sans-serif",
    legend_font_size=12,  # NEW
) -> None:
    """
    Faithful AgNPs pattern: per-node colormap coloring,
    symbol chain legend, full customization.
    """
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]

    parent_map = {labels[i]: parents[i] for i in range(len(labels))}

    def get_depth(label, visited=None):
        if visited is None:
            visited = set()
        if label in visited:
            return 0
        visited.add(label)
        p = parent_map.get(label, "")
        if p == "":
            return 0
        return 1 + get_depth(p, visited)

    depths = [get_depth(l) for l in labels]

    SYMBOL_LIBRARY = ['✦', '★', '●', '■', '▲', '◆', '⬟', '⬢', '◉', '◈',
                      '◇', '○', '□', '△', '◊']

    node_symbols: Dict[str, str] = {}
    for i, lab in enumerate(labels):
        d = depths[i]
        p = parents[i]
        if d == 0:
            node_symbols[lab] = SYMBOL_LIBRARY[0]
        else:
            ancestors: List[str] = []
            current = lab
            visited: Set[str] = set()
            while current != "" and current not in visited:
                visited.add(current)
                parent = parent_map.get(current, "")
                if parent != "" and parent in node_symbols:
                    ancestors.insert(0, node_symbols[parent])
                current = parent
            siblings = [
                labels[j] for j in range(len(labels))
                if parents[j] == p and depths[j] == d
            ]
            sym_idx = siblings.index(lab) if lab in siblings else 0
            own_symbol = SYMBOL_LIBRARY[(d + sym_idx) % len(SYMBOL_LIBRARY)]
            node_symbols[lab] = own_symbol

    display_labels: List[str] = []
    for i, lab in enumerate(labels):
        d = depths[i]
        if show_labels:
            if d == 0:
                display_labels.append(node_symbols[lab])
            else:
                chain: List[str] = []
                current = lab
                visited = set()
                while current != "" and current not in visited:
                    visited.add(current)
                    if current in node_symbols:
                        chain.insert(0, node_symbols[current])
                    current = parent_map.get(current, "")
                combo = "".join(chain[-3:]) if len(chain) > 3 else "".join(chain)
                display_labels.append(combo)
        else:
            display_labels.append(lab)

    unique_ids: List[str] = []
    seen: Dict[str, int] = {}
    for i, lab in enumerate(labels):
        base = f"{lab}_d{depths[i]}"
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base)
            seen[base] = 1

    parent_ids: List[str] = []
    for p in parents:
        if p == "":
            parent_ids.append("")
        else:
            found = False
            for i, lab in enumerate(labels):
                if lab == p:
                    parent_ids.append(unique_ids[i])
                    found = True
                    break
            if not found:
                parent_ids.append("")

    n_nodes = len(labels)
    cmap_to_use = color_continuous_scale or cmap_name or "Spectral"
    plot_colors: List[str] = []
    try:
        cmap_obj = plt.cm.get_cmap(cmap_to_use)
        t_vals = np.linspace(0.05, 0.95, n_nodes)
        rgbas = [cmap_obj(t) for t in t_vals]
        plot_colors = [matplotlib.colors.to_hex(rgba) for rgba in rgbas]
    except Exception:
        try:
            if hasattr(px.colors.sequential, cmap_to_use):
                px_scale = getattr(px.colors.sequential, cmap_to_use)
                plot_colors = [
                    px_scale[int(i * len(px_scale) / n_nodes) % len(px_scale)]
                    for i in range(n_nodes)
                ]
            else:
                raise ValueError("Not a plotly sequential scale")
        except Exception:
            try:
                from plotly.express import colors as px_colors
                qual_palettes = [
                    px_colors.qualitative.Bold,
                    px_colors.qualitative.Vivid,
                    px_colors.qualitative.Safe,
                    px_colors.qualitative.Pastel,
                    px_colors.qualitative.Dark24,
                    px_colors.qualitative.Light24,
                ]
                long_palette: List[str] = []
                for pal in qual_palettes:
                    long_palette.extend(pal)
                plot_colors = [
                    long_palette[i % len(long_palette)] for i in range(n_nodes)
                ]
            except Exception:
                cmap_obj = plt.cm.get_cmap("tab20")
                plot_colors = [
                    matplotlib.colors.to_hex(cmap_obj(i % 20 / 20))
                    for i in range(n_nodes)
                ]

    legend_entries: List[Dict[str, Any]] = []
    for i, lab in enumerate(labels):
        d = depths[i]
        sym = display_labels[i]
        color = plot_colors[i]
        legend_entries.append({
            'symbol': sym,
            'label': lab,
            'depth': d,
            'color': color,
            'value': values[i],
        })
    legend_entries.sort(key=lambda x: (x['depth'], -x['value']))

    bv = (
        branchvalues
        if branchvalues in ["total", "remainder"]
        else "total"
    )
    if show_labels and show_values:
        textinfo = 'label+value'
    elif show_labels:
        textinfo = 'label'
    elif show_values:
        textinfo = 'value'
    else:
        textinfo = 'none'

    sunburst_colors = plot_colors.copy()
    for i in range(len(labels)):
        if depths[i] == 0:
            sunburst_colors[i] = theme.get("plotly_paper", "#f8f9fa")

    fig = go.Figure(go.Sunburst(
        ids=unique_ids,
        labels=display_labels,
        parents=parent_ids,
        values=values,
        customdata=labels,
        branchvalues=bv,
        marker=dict(
            colors=sunburst_colors,
            line=dict(width=0.5, color="rgba(255,255,255,0.25)"),
        ),
        textinfo=textinfo,
        hovertemplate=(
            '<b>%{customdata}</b><br>Value: %{value}<br>'
            'Symbol: %{label}<extra></extra>'
            if hover_info == "all"
            else '<b>%{customdata}</b><extra></extra>'
        ),
        insidetextorientation="radial",
        textfont=dict(
            size=int(label_size), family=font_family, color="white"
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font=dict(color=theme.get("font", "#000000"), family=font_family),
        width=width,
        height=height,
        title=dict(
            text=(
                "<b>Hierarchical Concept Map</b><br>"
                "<sup>★ Parent | ★□ Child | ★□◆ Grandchild — Hover for names</sup>"
            ),
            font=dict(size=16, family=font_family),
        ),
        modebar=dict(
            orientation='h',
            bgcolor='rgba(255,255,255,0.7)',
            color='#333333',
            activecolor='#D32F2F',
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "💡 **Export:** Click the 📷 Camera icon (top-right of chart) "
        "to download high-res PNG/SVG/PDF."
    )

    if st.session_state.get('sunburst_show_legend', True):
        st.markdown("### 📊 Symbol-to-Label Legend")
        depth_names = {
            0: "Root", 1: "Category (Parent)", 2: "Concept (Child)",
            3: "Sub-Concept", 4: "Detail",
        }
        for d in sorted(set([e['depth'] for e in legend_entries])):
            depth_label = depth_names.get(d, f"Level {d}")
            st.markdown(f"**{depth_label}**")
            entries_at_depth = [
                e for e in legend_entries if e['depth'] == d
            ]
            n_cols = min(4, max(1, len(entries_at_depth)))
            cols = st.columns(n_cols)
            for i, entry in enumerate(entries_at_depth):
                with cols[i % n_cols]:
                    st.markdown(
                        f"""<div style='padding:8px; border-radius:6px; background-color:{entry['color']}22;
border-left:4px solid {entry['color']}; margin-bottom:6px; font-size:{legend_font_size}px;'>
<span style='font-size:{legend_font_size+4}px; color:{entry['color']}; margin-right:6px;'>{entry['symbol']}</span>
<span style='font-size:{legend_font_size}px; color:#333; font-weight:500;'>{entry['label']}</span>
<span style='font-size:{legend_font_size-1}px; color:#666; float:right;'>({entry['value']})</span>
</div>""",
                        unsafe_allow_html=True,
                    )


# ... (all other functions unchanged: render_graph_plotly_2d, render_graph_plotly_3d, render_graph_fallback,
#      render_radar_chart, render_tsne_projection, render_community_detection, render_concept_growth,
#      render_bubble_chart, apply_graph_edits, compute_graph_metrics, display_metric_dashboard,
#      render_concept_timeline, render_cooccurrence_heatmap, export_graph, render_reasoning_dashboard,
#      get_memory_usage_mb, split_into_batches, merge_graphs, recompute_edge_weights,
#      extract_doc_metrics, IncrementalGraphBuilder, reset_batch_state, render_batch_processing_controls,
#      run_batch_analysis, render_sidebar, main)
# )
# ============================================================================
# GRAPH METRICS DASHBOARD
# ============================================================================
def compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {}
    metrics: Dict[str, Any] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "clustering": (
            nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
        ),
        "connected_components": nx.number_connected_components(G),
        "avg_clustering": (
            nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
        ),
    }
    try:
        bc = nx.betweenness_centrality(
            G, normalized=True, k=min(100, G.number_of_nodes())
        )
        top_bridges = sorted(
            bc.items(), key=lambda x: x[1], reverse=True
        )[:10]
        metrics["top_bridges"] = top_bridges
        metrics["avg_betweenness"] = np.mean(list(bc.values()))
    except Exception:
        metrics["top_bridges"] = []
    return metrics


def display_metric_dashboard(metrics: Dict, theme=None) -> None:
    if not metrics:
        st.warning("No graph metrics available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"])
    col2.metric("Edges", metrics["edges"])
    col3.metric("Density", f"{metrics['density']:.3f}")
    col4.metric("Avg Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6, col7 = st.columns(3)
    col5.metric("Clustering", f"{metrics['clustering']:.3f}")
    col6.metric("Components", metrics["connected_components"])
    col7.metric(
        "Avg Betweenness", f"{metrics.get('avg_betweenness', 0):.3f}"
    )
    if metrics.get("top_bridges"):
        st.markdown("**Top Bridge Concepts (High Betweenness)**")
        bridge_df = pd.DataFrame(
            metrics["top_bridges"], columns=["Concept", "Bridge Score"]
        )
        st.dataframe(bridge_df, use_container_width=True)


# ============================================================================
# EXTRA VISUALIZATIONS
# ============================================================================
def render_concept_timeline(
    df_filtered, valid_concepts, concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        st.info("No 'Year' data available for timeline visualization.")
        return
    years = df_filtered["Year"].dropna().astype(int)
    if len(years) == 0:
        st.info("No valid year data found.")
        return
    year_range = sorted(years.unique())
    if len(year_range) < 2:
        st.info("Need at least 2 different years for timeline.")
        return
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:10]
    timeline_data: List[Dict[str, Any]] = []
    for year in year_range:
        year_mask = df_filtered["Year"] == year
        year_df = df_filtered[year_mask]
        year_text = ""
        for idx, row in year_df.iterrows():
            for col in df_filtered.columns:
                if pd.notna(row[col]):
                    year_text += " " + str(row[col])
        for concept in top_concepts:
            count = len(re.findall(
                r'\b' + re.escape(concept) + r'\b', year_text, re.I
            ))
            timeline_data.append({
                "Year": year, "Concept": concept, "Count": count,
            })
    if not timeline_data:
        st.info("No timeline data to display.")
        return
    timeline_df = pd.DataFrame(timeline_data)
    fig = px.line(
        timeline_df, x="Year", y="Count", color="Concept",
        title="Concept Frequency Over Time",
        labels={"Count": "Mentions", "Year": "Publication Year"},
        template=(
            "plotly_white" if theme == THEME_PRESETS["Bright (Default)"]
            else "plotly_dark"
        ),
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cooccurrence_heatmap(
    nx_graph, valid_concepts, concept_abstract_map,
    top_n=30, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:top_n]
    if len(top_concepts) < 3:
        st.info("Need at least 3 concepts for heatmap.")
        return
    n = len(top_concepts)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(top_concepts):
        for j, c2 in enumerate(top_concepts):
            if i == j:
                matrix[i][j] = len(concept_abstract_map.get(c1, []))
            elif nx_graph.has_edge(c1, c2):
                matrix[i][j] = nx_graph[c1][c2].get('cooccurrence', 0)
    fig = px.imshow(
        matrix, x=top_concepts, y=top_concepts,
        labels=dict(x="Concept", y="Concept", color="Co-occurrence"),
        title=f"Co-occurrence Heatmap (Top {n} Concepts)",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================
def export_graph(
    nx_graph, concept_abstract_map, export_format: str,
    include_metadata: bool = True,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    if export_format == "GraphML":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '6.1'
                nx_graph.graph['tool'] = 'MPEA-ConceptGraph'
            try:
                nx.write_graphml_lxml(nx_graph, "mpea_graph.graphml")
            except Exception:
                nx.write_graphml(nx_graph, "mpea_graph.graphml")
            with open("mpea_graph.graphml", "rb") as f:
                return f.read(), "application/graphml+xml", "mpea_graph.graphml"
        except Exception as e:
            st.error(f"GraphML export failed: {e}")
            return None, None, None
    elif export_format == "JSON (Full Metadata)":
        data = nx.node_link_data(nx_graph)
        if include_metadata:
            data['metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '6.1',
                'tool': 'MPEA-ConceptGraph',
                'node_count': len(nx_graph.nodes()),
                'edge_count': len(nx_graph.edges()),
                'inferred_edges': sum(
                    1 for u, v, d in nx_graph.edges(data=True)
                    if d.get('inferred', False)
                ),
                'categories': list(set(
                    abstract_concepts_to_categories(
                        list(nx_graph.nodes())
                    ).values()
                )),
            }
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "mpea_graph_full.json"
    elif export_format == "JSON (Compact)":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "mpea_graph.json"
    elif export_format == "CSV (Edges + Metadata)":
        edge_data: List[Dict[str, Any]] = []
        for u, v, data in nx_graph.edges(data=True):
            row = {
                "source": u, "target": v,
                "weight": data.get('weight', 1),
                "cooccurrence": data.get('cooccurrence', 0),
                "semantic_similarity": data.get('semantic', 0),
                "edge_type": data.get('edge_type', 'unknown'),
                "inferred": data.get('inferred', False),
                "confidence": data.get('confidence', 1.0),
                "path": data.get('path', ''),
            }
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "mpea_edges_enhanced.csv"
    elif export_format == "CSV (Nodes + Metadata)":
        node_data: List[Dict[str, Any]] = []
        for node in nx_graph.nodes():
            row = {
                "concept": node,
                "frequency": len(concept_abstract_map.get(node, [])),
                "degree": nx_graph.degree(node),
                "concept_type": nx_graph.nodes[node].get('concept_type', 'general'),
                "definition": nx_graph.nodes[node].get('definition', ''),
                "category": abstract_concepts_to_categories([node]).get(node, 'general'),
            }
            row.update({
                k: v for k, v in nx_graph.nodes[node].items()
                if isinstance(v, (str, int, float, bool))
            })
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "mpea_nodes_enhanced.csv"
    elif export_format == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [
                get_mpea_category_color(n) for n in nx_graph.nodes()
            ]
            nx.draw(
                nx_graph, pos, with_labels=True,
                node_color=node_colors, edge_color='gray',
                node_size=400, font_size=7, font_weight='bold',
                edgecolors='white', linewidths=1,
            )
            buf = io.BytesIO()
            plt.savefig(
                buf, format='png', dpi=300,
                bbox_inches='tight', facecolor='white',
            )
            buf.seek(0)
            plt.close()
            return buf.read(), "image/png", "mpea_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif export_format == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=150)
            node_colors = [
                get_mpea_category_color(n) for n in nx_graph.nodes()
            ]
            nx.draw(
                nx_graph, pos, with_labels=True,
                node_color=node_colors, edge_color='gray',
                node_size=400, font_size=7, font_weight='bold',
                edgecolors='white', linewidths=1,
            )
            buf = io.BytesIO()
            plt.savefig(
                buf, format='svg', bbox_inches='tight', facecolor='white',
            )
            buf.seek(0)
            plt.close()
            return buf.read(), "image/svg+xml", "mpea_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    elif export_format == "GEXF":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '6.1'
            nx.write_gexf(nx_graph, "mpea_graph.gexf")
            with open("mpea_graph.gexf", "rb") as f:
                return f.read(), "application/xml", "mpea_graph.gexf"
        except Exception as e:
            st.error(f"GEXF export failed: {e}")
            return None, None, None
    return None, None, None


# ============================================================================
# REASONING DASHBOARD
# ============================================================================
def render_reasoning_dashboard(
    nx_graph, valid_concepts, ontology, extractor,
) -> None:
    st.subheader("🔍 Ontology-Based Reasoning Insights")
    type_counts: Dict[str, int] = defaultdict(int)
    for c in valid_concepts:
        if c in ontology.concepts:
            type_counts[ontology.concepts[c].concept_type.value] += 1
        else:
            type_counts["unknown"] += 1
    fig = px.pie(
        values=list(type_counts.values()),
        names=list(type_counts.keys()),
        title="Concept Type Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
    inferred_edges = [
        (u, v) for u, v, d in nx_graph.edges(data=True)
        if d.get('inferred', False)
    ]
    observed_edges = [
        (u, v) for u, v, d in nx_graph.edges(data=True)
        if not d.get('inferred', False)
    ]
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed Edges", len(observed_edges))
    col2.metric("Inferred Edges", len(inferred_edges))
    col3.metric(
        "Inference Ratio",
        f"{len(inferred_edges) / max(len(observed_edges), 1):.2f}",
    )
    rel_types: Dict[str, int] = defaultdict(int)
    for u, v, d in nx_graph.edges(data=True):
        rel_types[d.get('edge_type', 'unknown')] += 1
    if rel_types:
        rel_df = pd.DataFrame(
            [(k, v) for k, v in rel_types.items()],
            columns=['Relationship Type', 'Count'],
        )
        rel_df = rel_df.sort_values('Count', ascending=False)
        st.dataframe(rel_df, use_container_width=True)
        fig = px.bar(
            rel_df, x='Relationship Type', y='Count',
            title="Edge Type Distribution",
            color='Relationship Type',
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("🔗 Inferred Process-Parameter-Response Chains")
    process_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.PROCESS
    ]
    property_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.PROPERTY
    ]
    chains_found: List[Dict[str, Any]] = []
    for proc in process_nodes[:5]:
        for prop in property_nodes[:5]:
            paths = ontology.infer_path(proc, prop, max_depth=3)
            if paths:
                chains_found.append({
                    "Process": proc,
                    "Property": prop,
                    "Path Length": len(paths[0]),
                    "Path": " → ".join(paths[0]),
                })
    if chains_found:
        st.dataframe(pd.DataFrame(chains_found), use_container_width=True)
    else:
        st.info(
            "No direct inference chains found. "
            "Build graph with more concepts."
        )
    st.subheader("📚 Synonym Resolution Examples")
    synonym_examples = [
        ("high entropy alloy", "mpea"),
        ("co-cr-fe-ni", "cocrfeni"),
        ("valence electron concentration", "valence_electron_concentration"),
        ("enthalpy of mixing", "enthalpy_of_mixing"),
        ("fcc phase", "fcc_phase"),
    ]
    syn_data: List[Dict[str, Any]] = []
    for original, expected in synonym_examples:
        resolved = ontology.resolve_concept(original)
        syn_data.append({
            "Original": original,
            "Expected": expected,
            "Resolved": resolved,
            "Match": (
                "✅" if resolved == expected
                else ("⚠️" if resolved else "❌")
            ),
        })
    st.dataframe(pd.DataFrame(syn_data), use_container_width=True)
    st.subheader("🏛️ Concept Hierarchy")
    hierarchy_data: List[Dict[str, str]] = []
    for concept in valid_concepts[:20]:
        if concept in ontology.concepts:
            node = ontology.concepts[concept]
            if node.hypernyms:
                for hyp in node.hypernyms:
                    hierarchy_data.append({
                        "Child": concept, "Parent": hyp,
                        "Relation": "is-a",
                    })
            if node.hyponyms:
                for hyp in node.hyponyms:
                    if hyp in valid_concepts:
                        hierarchy_data.append({
                            "Parent": concept, "Child": hyp,
                            "Relation": "has-subtype",
                        })
    if hierarchy_data:
        st.dataframe(
            pd.DataFrame(hierarchy_data), use_container_width=True,
        )
    else:
        st.info(
            "No hierarchical relationships found in current concept set."
        )


# ============================================================================
# BATCH PROCESSING MODE v6.0 (Streamlit Cloud ≤ 1 GB RAM)
# ============================================================================
def get_memory_usage_mb() -> float:
    """Peak RSS memory in MB (Linux: KB, macOS: bytes). 0.0 if unavailable."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return 0.0


def split_into_batches(
    df: pd.DataFrame, batch_size: int
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """Yield (start_positional_index, batch_df) slices of df."""
    total_batches = math.ceil(len(df) / batch_size)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        yield start_idx, df.iloc[start_idx:end_idx]


def merge_graphs(existing_graph: nx.Graph, new_graph: nx.Graph) -> nx.Graph:
    """
    Merge new_graph INTO existing_graph (in-place → no copy → memory-safe).
    - Node 'frequency' values are summed (per-batch doc counts → cumulative).
    - Edge 'cooccurrence' counts are summed, 'semantic' keeps the max,
      'inferred' flags are OR-ed, richer edge_type/confidence/path are kept.
    Call recompute_edge_weights() afterwards for final weights.
    """
    merged = existing_graph
    for node, data in new_graph.nodes(data=True):
        if node in merged:
            merged.nodes[node]["frequency"] = (
                merged.nodes[node].get("frequency", 0)
                + data.get("frequency", 0)
            )
            for attr in ("concept_type", "definition"):
                if not merged.nodes[node].get(attr) and data.get(attr):
                    merged.nodes[node][attr] = data[attr]
        else:
            merged.add_node(node, **data)
    for u, v, data in new_graph.edges(data=True):
        if merged.has_edge(u, v):
            ed = merged[u][v]
            ed["cooccurrence"] = (
                ed.get("cooccurrence", 0) + data.get("cooccurrence", 0)
            )
            ed["semantic"] = max(
                ed.get("semantic", 0) or 0, data.get("semantic", 0) or 0
            )
            ed["inferred"] = bool(ed.get("inferred", False)) or bool(
                data.get("inferred", False)
            )
            if data.get("confidence") is not None:
                ed["confidence"] = max(
                    ed.get("confidence", 0), data["confidence"]
                )
            if data.get("path") and not ed.get("path"):
                ed["path"] = data["path"]
            if (
                ed.get("edge_type", "cooccurrence") == "cooccurrence"
                and data.get("edge_type") not in (None, "cooccurrence")
            ):
                ed["edge_type"] = data["edge_type"]
        else:
            merged.add_edge(u, v, **data)
    return merged


def recompute_edge_weights(nx_graph: nx.Graph, config: Dict) -> None:
    """Same weighting scheme as
    ReasoningEnhancedGraphBuilder._compute_final_weights."""
    cooc_w = config.get("COOCCURRENCE_WEIGHT", 0.7)
    sem_w = config.get("SEMANTIC_WEIGHT", 0.2)
    inf_w = config.get("INFERENCE_WEIGHT", 0.1)
    for _, _, data in nx_graph.edges(data=True):
        cooc = data.get("cooccurrence", 0)
        sem = data.get("semantic", 0) or 0
        inf = 1.0 if data.get("inferred", False) else 0.0
        conf = data.get("confidence", 0.5)
        data["weight"] = cooc_w * cooc + sem_w * sem + inf_w * inf * conf


def extract_doc_metrics(text: str) -> Dict[str, Any]:
    """Regex metric extraction identical to the full-mode pipeline."""
    metrics: Dict[str, Any] = {}
    power_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:w|watt)', text, re.I)
    if power_matches:
        metrics['laser_power_w'] = [float(m) for m in power_matches]
    velocity_matches = re.findall(
        r'(\d+(?:\.\d+)?)\s*(?:mm/s|m/s)', text, re.I
    )
    if velocity_matches:
        metrics['scan_velocity'] = [float(m) for m in velocity_matches]
    temp_matches = re.findall(
        r'(\d+(?:\.\d+)?)\s*(?:k|°c|celsius)', text, re.I
    )
    if temp_matches:
        metrics['temperature'] = [float(m) for m in temp_matches]
    return metrics


class IncrementalGraphBuilder(ReasoningEnhancedGraphBuilder):
    """
    ReasoningEnhancedGraphBuilder subclass that builds a graph from ONE
    document batch. Node frequencies come from the batch itself so that
    merge_graphs() can accumulate them correctly across batches.
    Semantic / inferred / hierarchical edges reuse the parent implementation.
    """

    @timed
    def build_batch_graph(
        self,
        batch_concepts: List[List[str]],
        valid_concepts: List[str],
        concept_to_id: Dict[str, int],
        batch_doc_freq: Dict[str, int],
        embed_model=None,
        config: Dict = None,
    ) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(1000)
        nx_graph = nx.Graph()
        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            definition = self.ontology.get_definition(c)
            nx_graph.add_node(
                c,
                frequency=batch_doc_freq.get(c, 0),
                concept_type=concept_type.value,
                definition=definition,
                degree=0,
            )
        cooccurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for concepts in batch_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if u != v:
                        key = tuple(sorted([u, v]))
                        cooccurrence_map[key] += 1
        for (u, v), count in cooccurrence_map.items():
            nx_graph.add_edge(
                u, v,
                weight=float(count),
                cooccurrence=count,
                semantic=0.0,
                edge_type='cooccurrence',
                inferred=False,
            )
        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(
                nx_graph, valid_concepts, embed_model, config
            )
        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)
        self._add_hierarchical_edges(nx_graph, valid_concepts)
        self._compute_final_weights(nx_graph, config)
        return nx_graph


def reset_batch_state(clear_analysis: bool = False) -> None:
    """Clear incremental batch state (and optionally all analysis results)."""
    st.session_state.batch_state = None
    st.session_state.pop("batch_trigger", None)
    if clear_analysis:
        st.session_state.analysis_data = None
        st.session_state.burst_df = None
        st.session_state.drift_df = None
        st.session_state.genealogy_df = None
        st.session_state.bridge_df = None
        st.session_state.motifs = {}
        st.session_state.edit_history = GraphEditHistory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_batch_processing_controls() -> None:
    """Sidebar UI: batch-mode toggle, batch size, and batch navigation."""
    st.markdown("---")
    st.subheader("📦 Batch Processing (≤1 GB RAM)")
    st.toggle(
        "Enable batch processing",
        key="batch_mode",
        help=(
            "Process documents in small batches with incremental graph "
            "merging and memory cleanup after each batch. Recommended for "
            "Streamlit Cloud free tier (1 GB RAM)."
        ),
    )
    if not st.session_state.get("batch_mode", False):
        return
    st.slider(
        "Batch size (documents)", 100, 2000, 1000, 100,
        key="batch_size",
        help="Smaller batches = lower peak memory but more merge steps.",
    )
    st.slider(
        "GNN epochs (final training)", 10, 50, 40, 5,
        key="batch_gnn_epochs",
        help="GNN is trained ONCE on the final merged graph.",
    )
    bs = st.session_state.get("batch_state")
    if bs:
        total = max(bs.get("total_batches", 1), 1)
        done = bs.get("next_batch", 0)
        st.progress(done / total)
        st.caption(
            f"Batch {done}/{total} • "
            f"{bs.get('docs_processed', len(bs.get('all_texts', {})))} "
            f"docs processed • "
            f"{len(bs.get('all_texts', {}))} texts cached"
        )
    col_next, col_all = st.columns(2)
    with col_next:
        if st.button(
            "▶️ Next batch", use_container_width=True,
            disabled=bool(bs and bs.get("done")),
        ):
            st.session_state["batch_trigger"] = "next"
    with col_all:
        if st.button(
            "⏩ All remaining", use_container_width=True,
            disabled=bool(bs and bs.get("done")),
        ):
            st.session_state["batch_trigger"] = "all"
    if bs:
        if st.button("🗑️ Reset batch state", use_container_width=True):
            reset_batch_state(clear_analysis=True)
            st.success("Batch state cleared!")
            st.rerun()
    else:
        st.caption(
            "Click 🚀 Build Concept Graph (or ▶️ Next batch) to start."
        )


BATCH_TEXT_STORE_CAP = 4000


def run_batch_analysis(
    df_filtered: pd.DataFrame,
    selected_text_cols: List[str],
    ontology: DomainOntology,
    run_mode: str = "all",
) -> None:
    """
    Memory-efficient batch pipeline for Streamlit Cloud (≤ 1 GB RAM).

    run_mode: 'all' → process every remaining batch in this run;
              'next' → process exactly one batch (resumable via sidebar).
    Produces the SAME st.session_state.analysis_data structure as the
    full pipeline, so every downstream tab works unchanged.
    """
    overall_start = time.perf_counter()
    try:
        torch.set_num_threads(2)  # bound CPU/memory spikes on free tier
    except Exception:
        pass
    batch_size = int(st.session_state.get("batch_size", 1000))
    total_docs = len(df_filtered)
    if total_docs == 0:
        st.error("No documents to process.")
        return
    total_batches = math.ceil(total_docs / batch_size)

    data_hash = hashlib.md5(
        (
            f"{total_docs}|{'|'.join(selected_text_cols)}|"
            f"{df_filtered.index.min()}|{df_filtered.index.max()}"
        ).encode("utf-8")
    ).hexdigest()

    bs = st.session_state.get("batch_state")
    if bs is not None and (
        bs.get("data_hash") != data_hash
        or bs.get("batch_size") != batch_size
    ):
        st.info("Dataset or batch size changed — resetting batch state.")
        reset_batch_state(clear_analysis=False)
        bs = None
    if bs is None:
        bs = {
            "data_hash": data_hash,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "next_batch": 0,
            "all_concepts": [],
            "all_metrics": [],
            "all_texts": {},
            "valid_doc_indices": set(),
            "docs_processed": 0,
            "concept_freq": defaultdict(int),
            "concept_abstract_map": defaultdict(list),
            "merged_graph": None,
            "extractor": None,
            "resolver": None,
            "builder": None,
            "done": False,
        }
        st.session_state.batch_state = bs

    if bs["done"]:
        st.success("✅ All batches already processed — see results below.")
        return

    config = get_adaptive_config(total_docs)
    config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)
    config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
    config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
    config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.7)
    config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.2)
    config["INFERENCE_WEIGHT"] = st.session_state.get('inf_weight', 0.1)

    use_ontology = st.session_state.get('use_ontology', True)
    embed_model = load_embedding_model()

    if use_ontology and bs["extractor"] is None:
        with st.spinner("Initializing ontology resolver (one-time)..."):
            resolver = AdvancedConceptResolver(
                ontology, embed_model, cache_max=2000,
            )
            extractor = EnhancedConceptExtractor(
                ontology, resolver,
                store_contexts=False, store_documents=False,
            )
            builder = IncrementalGraphBuilder(ontology, extractor)
            bs["resolver"] = resolver
            bs["extractor"] = extractor
            bs["builder"] = builder
            st.session_state.resolver = resolver
            st.session_state.extractor = extractor
        gc.collect()

    pending = list(range(bs["next_batch"], total_batches))
    if run_mode == "next":
        pending = pending[:1]
    if not pending:
        st.success("✅ Nothing left to process.")
        return

    progress_bar = st.progress(0.0)
    status = st.status("📦 Batch processing running...", expanded=True)

    def _process_one_batch(batch_num: int) -> None:
        start = batch_num * batch_size
        end = min(start + batch_size, total_docs)
        batch_df = df_filtered.iloc[start:end]
        n_this = len(batch_df)
        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        with status:
            st.write(
                f"📦 Batch {batch_num + 1}/{total_batches} — "
                f"docs {start}–{end - 1} ({n_this} docs)"
            )
        batch_concepts: List[List[str]] = []
        batch_metrics: List[Dict] = []
        batch_doc_freq: Dict[str, int] = defaultdict(int)
        extractor = bs["extractor"]

        for local_i, (_, row) in enumerate(batch_df.iterrows()):
            text = " ".join([
                str(row[col]) for col in selected_text_cols
                if col in row and pd.notna(row[col])
            ])
            if use_ontology and extractor is not None:
                concepts = extractor.extract_from_text(text, start + local_i)
            else:
                concepts = extract_concepts_from_text(text)
            batch_concepts.append(concepts)
            batch_metrics.append(extract_doc_metrics(text))
            unique_concepts = set(concepts)
            for c in unique_concepts:
                batch_doc_freq[c] += 1
                bs["concept_freq"][c] += 1
                bs["concept_abstract_map"][c].append(start + local_i)
            has_valid = any(
                bs["concept_freq"].get(c, 0) >= min_freq
                for c in unique_concepts
            )
            if has_valid:
                bs["all_texts"][start + local_i] = (
                    text[:BATCH_TEXT_STORE_CAP]
                )
                bs["valid_doc_indices"].add(start + local_i)
            bs["docs_processed"] += 1
            del text
            if (local_i + 1) % 100 == 0 or (local_i + 1) == n_this:
                frac = (batch_num + (local_i + 1) / n_this) / total_batches
                progress_bar.progress(min(0.90 * frac, 0.90))
                with status:
                    st.write(f"  … {local_i + 1}/{n_this} docs extracted")

        bs["all_concepts"].extend(batch_concepts)
        bs["all_metrics"].extend(batch_metrics)

        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        top_n = config.get("TOP_N_CONCEPTS", 1000)
        batch_unique: Set[str] = set()
        for cs in batch_concepts:
            batch_unique.update(cs)
        batch_valid = [
            c for c in batch_unique
            if bs["concept_freq"].get(c, 0) >= min_freq
        ]
        batch_valid.sort(
            key=lambda c: bs["concept_freq"][c], reverse=True
        )
        batch_valid = batch_valid[:top_n]
        concept_to_id_batch = {c: i for i, c in enumerate(batch_valid)}

        if use_ontology and bs["builder"] is not None:
            batch_graph = bs["builder"].build_batch_graph(
                batch_concepts, batch_valid, concept_to_id_batch,
                batch_doc_freq, embed_model, config,
            )
        else:
            batch_graph = build_hybrid_graph(
                batch_concepts, batch_valid, concept_to_id_batch,
                embed_model, config, ontology,
            )

        if bs["merged_graph"] is None:
            bs["merged_graph"] = batch_graph
        else:
            bs["merged_graph"] = merge_graphs(bs["merged_graph"], batch_graph)
        recompute_edge_weights(bs["merged_graph"], config)
        bs["next_batch"] = batch_num + 1

        g = bs["merged_graph"]
        with status:
            st.write(
                f"✅ Batch {batch_num + 1} done — cumulative graph: "
                f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
                f"| peak RSS ≈ {get_memory_usage_mb():.0f} MB"
            )
        del batch_concepts, batch_metrics, batch_doc_freq
        del batch_graph, batch_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _finalize() -> None:
        merged = bs["merged_graph"]
        if merged is None or merged.number_of_nodes() == 0:
            st.error("No graph could be built from the processed batches.")
            return
        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        top_n = config.get("TOP_N_CONCEPTS", 1000)
        with status:
            st.write("🧩 Finalizing — selecting top concepts...")
        valid_concepts = [
            c for c, f in bs["concept_freq"].items() if f >= min_freq
        ]
        valid_concepts.sort(
            key=lambda c: len(bs["concept_abstract_map"].get(c, [])),
            reverse=True,
        )
        valid_concepts = valid_concepts[:top_n]
        if len(valid_concepts) < 5:
            st.error(
                "Too few concepts extracted. "
                "Try lowering frequency thresholds."
            )
            return
        valid_set = set(valid_concepts)
        drop_nodes = [n for n in merged.nodes() if n not in valid_set]
        merged.remove_nodes_from(drop_nodes)
        del drop_nodes
        concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
        id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
        concept_abstract_map = {
            c: bs["concept_abstract_map"][c] for c in valid_concepts
        }
        progress_bar.progress(0.90)

        with status:
            st.write("🔢 Generating node embeddings...")
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
            node_features = torch.tensor(embeddings, dtype=torch.float32)
            del embeddings
        except Exception:
            node_features = torch.randn(len(valid_concepts), 384)
        gc.collect()

        with status:
            st.write("🧠 Training GraphSAGE (final, once)...")
        pos_pairs, neg_pairs = sample_edges_for_training(
            merged, valid_concepts, concept_to_id, config, memory_safe=True,
        )
        epochs = int(st.session_state.get("batch_gnn_epochs", 40))

        def _gnn_progress(epoch, loss):
            frac = 0.90 + (epoch / max(epochs, 1)) * 0.05
            progress_bar.progress(min(frac, 0.95))
            if epoch % 10 == 0:
                with status:
                    st.write(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

        gnn_model, final_emb, adj_indices, adj_values = train_gnn(
            node_features, merged, concept_to_id,
            pos_pairs, neg_pairs, _gnn_progress, epochs=epochs,
        )
        del pos_pairs, neg_pairs, adj_indices, adj_values
        gc.collect()

        with status:
            st.write("🎯 Scoring research directions...")
        concept_properties: Dict[str, float] = {}
        all_metrics = bs["all_metrics"]
        for concept in valid_concepts:
            values: List[float] = []
            for idx in concept_abstract_map.get(concept, []):
                if idx < len(all_metrics):
                    for metric_values in all_metrics[idx].values():
                        values.extend(metric_values)
            concept_properties[concept] = (
                float(np.median(values)) if values else 0.0
            )
        X_feat: List[List[float]] = []
        y_target: List[float] = []
        for u, v in merged.edges():
            pu = concept_properties.get(u, 0)
            pv = concept_properties.get(v, 0)
            w = merged[u][v].get('weight', 1)
            X_feat.append([pu, pv, w])
            y_target.append(
                max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0
            )
        ridge = None
        if len(X_feat) > 5:
            ridge = Ridge(alpha=1.0).fit(
                np.array(X_feat), np.array(y_target)
            )
        top_scores = compute_research_direction_scores(
            gnn_model, node_features, final_emb, merged,
            valid_concepts, concept_properties, ridge, embed_model,
        )
        del X_feat, y_target, node_features
        gc.collect()

        with status:
            st.write("🧪 Distillation + advanced analytics...")
        distill_df = compute_concept_distillation(
            valid_concepts, concept_abstract_map, bs["all_texts"],
            max_docs_per_concept=30,
        )
        burst_df = None
        drift_df = None
        genealogy_df = None
        bridge_df = None
        motifs: Dict[str, Any] = {}
        try:
            burst_df = detect_keyword_bursts(
                df_filtered, valid_concepts,
                concept_abstract_map, selected_text_cols,
            )
            drift_df = detect_semantic_drift(
                df_filtered, valid_concepts,
                concept_abstract_map, selected_text_cols,
            )
            genealogy_df = build_concept_genealogy(
                merged, valid_concepts, concept_abstract_map,
            )
            bridge_df = detect_cross_domain_bridges(
                merged, valid_concepts, concept_abstract_map,
            )
            motifs = analyze_network_motifs(merged)
        except Exception as e:
            st.warning(f"Some analytics skipped: {e}")
        st.session_state.burst_df = burst_df
        st.session_state.drift_df = drift_df
        st.session_state.genealogy_df = genealogy_df
        st.session_state.bridge_df = bridge_df
        st.session_state.motifs = motifs
        gc.collect()

        analysis_data = {
            "valid_concepts": valid_concepts,
            "concept_to_id": concept_to_id,
            "id_to_concept": id_to_concept,
            "concept_abstract_map": concept_abstract_map,
            "nx_graph": merged,
            "concept_properties": concept_properties,
            "ridge": ridge,
            "top_scores": top_scores,
            "distill_df": distill_df,
            "gnn_model": gnn_model,
            "final_emb": final_emb,
            "embed_model": embed_model,
            "all_metrics": bs["all_metrics"],
            "all_texts": bs["all_texts"],
            "config": config,
            "df_filtered": df_filtered,
            "selected_text_cols": selected_text_cols,
            "batch_info": {
                "mode": "batch",
                "batch_size": batch_size,
                "total_batches": total_batches,
                "total_docs": total_docs,
            },
        }
        if use_ontology:
            analysis_data.update({
                "ontology": ontology,
                "resolver": bs["resolver"],
                "extractor": bs["extractor"],
                "graph_builder": bs["builder"],
                "reasoning_paths": (
                    bs["builder"].reasoning_paths if bs["builder"] else []
                ),
            })
        st.session_state.analysis_data = analysis_data
        st.session_state.edit_history = GraphEditHistory()
        st.session_state.edit_history.save_snapshot(
            merged, valid_concepts, concept_to_id,
            id_to_concept, concept_abstract_map,
        )
        bs["all_concepts"] = []
        bs["valid_doc_indices"] = set()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bs["done"] = True

    try:
        for b in pending:
            _process_one_batch(b)
        if bs["next_batch"] >= total_batches:
            with status:
                st.write("🏁 All batches processed — finalizing...")
            _finalize()
            total_time = time.perf_counter() - overall_start
            progress_bar.progress(1.0)
            status.update(
                label=(
                    f"Batch analysis complete! ({total_time:.1f}s, "
                    f"peak RSS ≈ {get_memory_usage_mb():.0f} MB)"
                ),
                state="complete", expanded=False,
            )
            st.success(
                f"✅ All {total_batches} batches processed in "
                f"{total_time:.1f}s — peak memory ≈ "
                f"{get_memory_usage_mb():.0f} MB"
            )
        else:
            status.update(
                label=(
                    f"Batch {bs['next_batch']}/{total_batches} complete"
                ),
                state="complete", expanded=False,
            )
            st.info(
                f"📦 {total_batches - bs['next_batch']} batch(es) remaining "
                f"— click ▶️ Next batch or ⏩ All remaining in the sidebar."
            )
    except Exception as e:
        st.error(f"Batch pipeline error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# SIDEBAR (AgNPs Pattern — Full Sunburst Customization)
# ============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration v6.1")
        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0,
        )
        theme = THEME_PRESETS[st.session_state['theme']]
        st.subheader("🔬 MPEA Quantitative Descriptor Focus Areas")
        st.markdown(r"- **Compositional Descriptors:** Atomic size difference ($\delta$), Valence Electron Concentration (VEC), Electronegativity ($\Delta \chi$)")
        st.markdown(r"- **Thermodynamic Parameters:** Enthalpy/Entropy of mixing ($\Delta H_{mix}$, $\Delta S_{mix}$), $\Omega$ parameter, Gibbs energy")
        st.markdown("- **Mechanical Properties:** Hardness (HV), Elongation (%), Pugh's ratio (B/G), Cauchy pressure")
        st.markdown("- **Asymmetry Factors:** Melting temp, shear modulus, and enthalpy asymmetries (Key predictive features)")
        st.markdown("- **Phase Constituents:** FCC, BCC, Intermetallic (IM), Solid Solution (SS), Laves phase")
        st.markdown("- **Processing Routes:** Casting, Wrought, Sintering, Annealing")
        st.subheader("🧠 NLP Reasoning Options")
        st.session_state['use_ontology'] = st.checkbox(
            "Use ontology-based resolution", value=True,
            help="Maps synonyms like 'HEA', 'high-entropy alloy' to canonical concepts",
        )
        st.session_state['use_embedding_resolution'] = st.checkbox(
            "Use embedding-based semantic equivalence", value=True,
            help="Detects semantic similarity >0.85 even for unseen variants",
        )
        st.session_state['use_relationship_extraction'] = st.checkbox(
            "Extract cause-effect relationships", value=True,
            help="Identifies causal links between laser parameters and microstructure",
        )
        st.session_state['use_inference'] = st.checkbox(
            "Enable reasoning-based edge inference", value=True,
            help="Infers process→parameter→response chains even when not co-occurring",
        )
        st.session_state['context_window'] = st.slider(
            "Context window (chars)", 20, 200, 50,
            help="Window size for context-based disambiguation",
        )
        st.subheader("📊 Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:",
            ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"],
            index=0,
        )
        st.session_state['show_edge_weights'] = st.toggle(
            "Show edge weights", value=False,
            help="Display numerical weight labels on graph edges.",
        )
        st.session_state['edge_label_mode'] = st.selectbox(
            "Edge label mode:", ["hover", "threshold", "all"], index=0,
            help="hover=tooltip only, threshold=top 20% edges, all=all edges",
        )
        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:",
            options=list(SUPPORTED_COLORMAPS.keys()),
            index=0,
        )
        st.subheader("⚡ Physics & Layout")
        st.session_state['physics_preset'] = st.selectbox(
            "Physics preset:",
            options=list(PHYSICS_PRESETS.keys()),
            index=0,
        )
        preset = PHYSICS_PRESETS[st.session_state['physics_preset']]
        st.session_state['physics_enabled'] = st.checkbox(
            "Enable physics", value=(preset["gravity"] != 0),
        )
        with st.expander("Advanced Physics Overrides"):
            st.session_state['adv_damping'] = st.slider(
                "Damping", 0.05, 0.95, preset["damping"], step=0.05,
            )
            st.session_state['adv_gravity'] = st.slider(
                "Repulsion", -8000, -500, preset["gravity"], step=100,
            )
            st.session_state['adv_spring_length'] = st.slider(
                "Spring length", 40, 300, preset["spring_length"], step=10,
            )
            st.session_state['adv_spring_strength'] = st.slider(
                "Spring strength", 0.01, 0.20,
                preset["spring_strength"], step=0.01,
            )
            st.session_state['adv_central_gravity'] = st.slider(
                "Central gravity", 0.0, 0.5,
                preset["central_gravity"], step=0.05,
            )
            st.session_state['adv_stabilization'] = st.slider(
                "Stabilization iter", 0, 5000,
                preset["stabilization"], step=250,
            )
        base_preset = PHYSICS_PRESETS[
            st.session_state['physics_preset']
        ].copy()
        if st.session_state.get('adv_damping') is not None:
            base_preset["damping"] = st.session_state['adv_damping']
            base_preset["gravity"] = st.session_state['adv_gravity']
            base_preset["spring_length"] = st.session_state['adv_spring_length']
            base_preset["spring_strength"] = st.session_state['adv_spring_strength']
            base_preset["central_gravity"] = st.session_state['adv_central_gravity']
            base_preset["stabilization"] = st.session_state['adv_stabilization']
        st.session_state['effective_physics'] = base_preset
        st.subheader("📏 Display Limits")
        col_all1, col_slider1 = st.columns([0.3, 0.7])
        with col_all1:
            all_graph = st.checkbox("All", value=True, key="all_graph_chk")
        with col_slider1:
            st.session_state['top_n_graph'] = st.slider(
                "Max nodes", 10, 500, 200, step=10,
                disabled=all_graph, key="top_n_graph_slider",
            )
        if all_graph:
            st.session_state['top_n_graph'] = 0
        col_all2, col_slider2 = st.columns([0.3, 0.7])
        with col_all2:
            all_sun = st.checkbox("All", value=True, key="all_sun_chk")
        with col_slider2:
            st.session_state['top_n_sunburst'] = st.slider(
                "Max children/category", 10, 100, 40, step=10,
                disabled=all_sun, key="top_n_sunburst_slider",
            )
        if all_sun:
            st.session_state['top_n_sunburst'] = 0
        col_all3, col_slider3 = st.columns([0.3, 0.7])
        with col_all3:
            all_radar = st.checkbox("All", value=True, key="all_radar_chk")
        with col_slider3:
            st.session_state['top_n_radar'] = st.slider(
                "Top K for radar", 5, 30, 15,
                disabled=all_radar, key="top_n_radar_slider",
            )
        if all_radar:
            st.session_state['top_n_radar'] = 0
        st.subheader("🔧 Graph Parameters")
        st.session_state['min_freq'] = st.slider(
            "Min concept frequency", 1, 20, 1,
        )
        st.session_state['min_words'] = st.slider(
            "Min words per concept", 2, 5, 2,
        )
        st.session_state['sim_threshold'] = st.slider(
            "Semantic threshold", 0.6, 0.95, 0.85, step=0.05,
        )
        st.session_state['cooc_weight'] = st.slider(
            "Co-occurrence weight", 0.5, 1.0, 0.7, step=0.1,
        )
        st.session_state['sem_weight'] = st.slider(
            "Semantic weight", 0.0, 0.5, 0.2, step=0.1,
        )
        st.session_state['inf_weight'] = st.slider(
            "Inference weight", 0.0, 0.3, 0.1, step=0.05,
        )
        
        # Batch Processing Controls
        render_batch_processing_controls()

        st.subheader("📈 Statistics")
        st.session_state['bootstrap_samples'] = st.slider(
            "Bootstrap samples", 100, 2000, 500, step=100,
        )
        st.session_state['alpha_level'] = st.selectbox(
            "Significance alpha", [0.01, 0.05, 0.10], index=1,
        )

        st.markdown("---")
        st.subheader("🎨 Visualization Customization")
        st.session_state['enable_node_highlight'] = st.checkbox(
            "🔍 Enable Node Selection Highlight & Descriptions",
            value=False,
            help=(
                "When enabled, clicking a node highlights connected nodes "
                "with gold borders and overlays edge weights/relationship descriptions."
            ),
        )
        with st.expander("Node & Label Settings"):
            st.session_state['node_label_size'] = st.slider(
                "Node label font size", 8, 24, 12, step=1,
                help="Font size for node labels in the graph",
            )
            st.session_state['node_label_position'] = st.selectbox(
                "Node label position",
                ["center", "top", "bottom", "left", "right"],
                index=0,
                help="Where to place node labels relative to nodes",
            )
            st.session_state['node_font_face'] = st.selectbox(
                "Node font family",
                [
                    "Inter, Segoe UI, Roboto, sans-serif",
                    "Arial, Helvetica, sans-serif",
                    "Georgia, serif",
                    "Courier New, monospace",
                    "Times New Roman, serif",
                ],
                index=0,
            )
            # NEW: Node legend font size
            st.session_state['node_legend_font_size'] = st.slider(
                "Node legend font size", 8, 20, 13, step=1,
                help="Font size for the abbreviated node legend below the graph.",
            )
        st.session_state['use_abbreviated_labels'] = st.checkbox(
            "Use short labels (N1, N2...) for long names",
            value=False,
            help="Replaces long node labels with N1, N2... and generates a legend below the graph.",
        )
        if st.session_state['use_abbreviated_labels']:
            st.session_state['max_label_length'] = st.slider(
                "Max label length before abbreviation",
                min_value=2, max_value=50, value=15, step=1,
                help="Labels longer than this threshold will be replaced by N1, N2, etc.",
            )
        else:
            st.session_state['max_label_length'] = 15
        st.session_state['show_definitions'] = st.checkbox(
            "📖 Show concept definitions in tooltips",
            value=True,
            help="When enabled, hovering over a node displays its ontology definition in the tooltip.",
        )
        # NEW: Tooltip font size
        st.session_state['tooltip_font_size'] = st.slider(
            "Tooltip font size", 10, 20, 13, step=1,
            help="Font size for hover tooltips in the interactive graph.",
        )
        with st.expander("Edge Label Settings"):
            st.session_state['edge_label_size'] = st.slider(
                "Edge label font size", 6, 18, 10, step=1,
                help="Font size for edge weight labels",
            )
            st.session_state['edge_label_color'] = st.color_picker(
                "Edge label color", value="#000000",
                help="Color for edge weight labels (default matches theme)",
            )
            st.session_state['edge_label_position'] = st.selectbox(
                "Edge label position",
                ["middle", "top", "bottom", "from", "to"],
                index=0,
                help="Where to place edge labels along the edge",
            )
        edge_color_value = st.session_state.get('edge_label_color')
        if not edge_color_value or edge_color_value == '':
            edge_color_value = '#000000'
        st.session_state['edge_label_color'] = edge_color_value

        # NEW: Edge color customization
        with st.expander("Edge Color Customization"):
            st.session_state['edge_color_mode'] = st.selectbox(
                "Edge color mode",
                ["theme", "uniform_grey", "custom"],
                index=0,
                help="theme: based on relationship type (lightened), uniform_grey: single grey, custom: your pick",
            )
            if st.session_state['edge_color_mode'] == "custom":
                st.session_state['custom_edge_color'] = st.color_picker(
                    "Custom edge color", value="#AAAAAA",
                )
            else:
                st.session_state['custom_edge_color'] = "#AAAAAA"
            st.session_state['edge_lightness'] = st.slider(
                "Edge lightness (0=original, 1=white)", 0.0, 1.0, 0.6, step=0.05,
                help="Higher values make edges lighter, improving node visibility.",
            )

        st.markdown("---")
        st.subheader("✏️ Graph Editing")
        with st.expander("Remove Nodes"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                nodes_to_remove = st.multiselect(
                    "Select nodes to remove:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="remove_nodes_select",
                )
                st.session_state['nodes_to_remove'] = nodes_to_remove
            else:
                st.info("Build graph first to edit nodes.")
                st.session_state['nodes_to_remove'] = []
        with st.expander("Merge Nodes"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                nodes_to_merge = st.multiselect(
                    "Select nodes to merge:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="merge_nodes_select",
                )
                merge_name = st.text_input(
                    "New merged concept name:", key="merge_name_input",
                )
                st.session_state['nodes_to_merge'] = nodes_to_merge
                st.session_state['merge_name'] = merge_name
            else:
                st.info("Build graph first to merge nodes.")
                st.session_state['nodes_to_merge'] = []
                st.session_state['merge_name'] = ""
        with st.expander("Add Edge"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                all_concepts = st.session_state['analysis_data']['valid_concepts']
                edge_u = st.selectbox(
                    "Source concept:", options=all_concepts, key="edge_u_select",
                )
                edge_v = st.selectbox(
                    "Target concept:", options=all_concepts, key="edge_v_select",
                )
                edge_weight = st.number_input(
                    "Edge weight:", min_value=0.1, max_value=10.0,
                    value=1.0, step=0.1, key="edge_weight_input",
                )
                st.session_state['new_edge'] = (
                    (edge_u, edge_v) if edge_u != edge_v else None
                )
                st.session_state['new_edge_weight'] = edge_weight
            else:
                st.info("Build graph first to add edges.")
                st.session_state['new_edge'] = None
                st.session_state['new_edge_weight'] = 1.0
        with st.expander("Filter by Degree/Frequency"):
            st.session_state['filter_min_degree'] = st.slider(
                "Min degree", 0, 20, 0, key="filter_degree_slider",
            )
            st.session_state['filter_min_freq'] = st.slider(
                "Min frequency", 0, 50, 0, key="filter_freq_slider",
            )
        if (
            st.session_state.get('analysis_data')
            and st.session_state['analysis_data'].get('valid_concepts')
        ):
            if st.button("Apply Graph Edits", key="apply_edits_btn"):
                st.session_state['apply_edits'] = True
        if (
            st.session_state.get('analysis_data')
            and st.session_state.get('edit_history')
        ):
            col_undo, col_redo = st.columns(2)
            with col_undo:
                if (
                    st.button("↩️ Undo", key="undo_btn")
                    and st.session_state['edit_history'].can_undo()
                ):
                    snapshot = st.session_state['edit_history'].undo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Undo applied!")
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
            with col_redo:
                if (
                    st.button("↪️ Redo", key="redo_btn")
                    and st.session_state['edit_history'].can_redo()
                ):
                    snapshot = st.session_state['edit_history'].redo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Redo applied!")
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()

        st.markdown("---")
        st.subheader("☀️ Sunburst Chart Customization")
        st.session_state['sunburst_cmap'] = st.selectbox(
            "Colormap:",
            options=[
                "viridis", "plasma", "inferno", "magma", "cividis",
                "turbo", "rainbow", "hsv", "coolwarm", "RdBu", "Spectral",
                "tab10", "tab20", "Pastel1", "Set1", "Set2", "Set3",
                "YlOrRd", "PuBuGn", "GnBu", "YlGnBu",
            ],
            index=0,
            help="Choose color scheme for sunburst categories",
            key="sunburst_cmap_select",
        )
        st.session_state['sunburst_font_family'] = st.selectbox(
            "Sunburst font family",
            [
                "Arial, sans-serif",
                "Inter, Segoe UI, Roboto, sans-serif",
                "Georgia, serif",
                "Courier New, monospace",
                "Times New Roman, serif",
            ],
            index=0,
            help="Font family for sunburst chart labels",
            key="sunburst_font_family_select",
        )
        col_labels, col_values = st.columns(2)
        with col_labels:
            st.session_state['sunburst_show_labels'] = st.checkbox(
                "Show symbols", value=True,
                help="Display symbol combinations inside chart segments",
                key="sunburst_show_labels_chk",
            )
        with col_values:
            st.session_state['sunburst_show_values'] = st.checkbox(
                "Show values", value=False,
                help="Display numerical values inside chart segments",
                key="sunburst_show_values_chk",
            )
        st.session_state['sunburst_hover_info'] = st.selectbox(
            "Hover information:",
            options=["all", "minimal", "none"],
            index=0,
            help="Amount of information shown on hover tooltip",
            key="sunburst_hover_select",
        )
        st.session_state['sunburst_branchvalues'] = st.selectbox(
            "Branch values mode:", ["total", "remainder"], index=0,
            help="How to calculate branch sizes: total=sum of children, remainder=parent minus children",
            key="sunburst_branch_mode",
        )
        col_w, col_h = st.columns(2)
        with col_w:
            st.session_state['sunburst_width'] = st.slider(
                "Chart width (px)", 600, 1400, 900, step=50,
                key="sunburst_width_slider",
            )
        with col_h:
            st.session_state['sunburst_height'] = st.slider(
                "Chart height (px)", 500, 1200, 700, step=50,
                key="sunburst_height_slider",
            )
        st.session_state['sunburst_label_size'] = st.slider(
            "Symbol font size", 8, 30, 20, step=1,
            help="Size of symbols inside sunburst slices",
            key="sunburst_label_size_slider",
        )
        # NEW: Sunburst legend font size
        st.session_state['sunburst_legend_font_size'] = st.slider(
            "Sunburst legend font size", 8, 20, 12, step=1,
            help="Font size for the symbol-to-label legend below the sunburst chart.",
            key="sunburst_legend_font_size",
        )
        st.session_state['sunburst_show_legend'] = st.checkbox(
            "Show symbol legend", value=True,
            help="Display symbol-to-label mapping table below chart",
            key="sunburst_show_legend_chk",
        )
        if (
            st.session_state.get('analysis_data')
            and st.session_state['analysis_data'].get('valid_concepts')
        ):
            all_cats = list(set(
                abstract_concepts_to_categories(
                    st.session_state['analysis_data']['valid_concepts']
                ).values()
            ))
            st.session_state['sunburst_categories'] = st.multiselect(
                "Filter categories:", options=all_cats,
                default=all_cats, key="sunburst_cat_filter",
            )
        else:
            st.info("Build graph first to filter categories.")
            st.session_state['sunburst_categories'] = []

        st.markdown("---")
        with st.expander("⚡ Performance Monitor"):
            if st.button("Show Timing Report"):
                report = PerformanceMonitor.get_report()
                if report:
                    st.code(report, language="text")
                else:
                    st.info("No timing data yet. Run analysis first.")
            if st.button("Reset Timings"):
                PerformanceMonitor.reset()
                st.success("Timing data reset!")

        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"Device: {gpu_info}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main() -> None:
    st.title(
        "🔬 CoCrFeNi MPEA Quantitative Descriptor Graph v6.1"
    )
    st.caption(
        "Multi-level reasoning concept graph for numerical/quantitative description of CoCrFeNi MPEAs | "
        "Focus: Thermodynamic, Compositional, and Mechanical Descriptors | "
        "Memory-Safe | Batch Processing (≤1 GB) | Interactive Visualization | "
        "Ontology-aware resolution"
    )

    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology

    render_sidebar()

    # AgNPs pattern: Initialize ALL session_state keys
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None
    if "apply_edits" not in st.session_state:
        st.session_state.apply_edits = False
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = GraphEditHistory()
    if "burst_df" not in st.session_state:
        st.session_state.burst_df = None
    if "drift_df" not in st.session_state:
        st.session_state.drift_df = None
    if "genealogy_df" not in st.session_state:
        st.session_state.genealogy_df = None
    if "bridge_df" not in st.session_state:
        st.session_state.bridge_df = None
    if "motifs" not in st.session_state:
        st.session_state.motifs = {}

    # --- LOAD JSON DATA ---
    st.header("📁 Data Loading")
    st.info(f"Place JSON/BibTeX/CSV files in: `{JSON_METADATA_DIR}`")
    with st.spinner("Scanning json_metadatabase..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)

    if not file_records:
        st.warning("No .json/.bib/.csv files found in the directory.")
        st.info(
            "Please place your metadata files in the `json_metadatabase/` folder."
        )
        return
    successful_files = [f for f in file_records if f[1]]
    if not successful_files:
        st.error(
            "Files found but none could be parsed. Check error messages above."
        )
        return
    st.success(
        f"Loaded {len(successful_files)} file(s) | {len(df)} record(s)"
    )
    file_names = [f[0] for f in successful_files]
    selected_files = st.multiselect(
        "Filter by source file", file_names, default=file_names,
    )
    if selected_files:
        df_filtered = df[df["_source_file"].isin(selected_files)].copy()
    else:
        df_filtered = df.copy()
    st.write(f"Working with **{len(df_filtered)}** records")
    with st.expander("Preview Data Structure"):
        st.dataframe(df_filtered.head(5), use_container_width=True)
        st.markdown("**Available columns:**")
        st.write(list(df_filtered.columns))

    # --- TEXT COLUMN SELECTION ---
    text_cols = [
        c for c in df_filtered.columns
        if any(
            k in c.lower()
            for k in ['abstract', 'title', 'summary', 'text', 'content', 'description']
        )
    ]
    if not text_cols:
        text_cols = [
            c for c in df_filtered.columns if df_filtered[c].dtype == 'object'
        ]
    selected_text_cols = st.multiselect(
        "Select text columns for concept extraction:",
        options=text_cols,
        default=text_cols[:2] if len(text_cols) >= 2 else text_cols,
    )
    if not selected_text_cols:
        st.error("Please select at least one text column.")
        return

    # --- RUN ANALYSIS ---
    build_clicked = st.button(
        "🚀 Build Concept Graph with Reasoning",
        type="primary", use_container_width=True,
    )
    batch_trigger = st.session_state.pop("batch_trigger", None)
    batch_mode_on = st.session_state.get("batch_mode", False)
    if batch_mode_on and (build_clicked or batch_trigger):
        run_batch_analysis(
            df_filtered=df_filtered,
            selected_text_cols=selected_text_cols,
            ontology=ontology,
            run_mode=(batch_trigger or "all"),
        )
    elif build_clicked:
        progress_bar = st.progress(0.0)
        status = st.status(
            "Initializing advanced NLP analysis...", expanded=True,
        )
        overall_start = time.perf_counter()
        try:
            with status:
                st.write("Preparing text corpus...")
                all_texts: List[str] = []
                for idx, row in df_filtered.iterrows():
                    text = " ".join([
                        str(row[col]) for col in selected_text_cols
                        if col in row and pd.notna(row[col])
                    ])
                    all_texts.append(text)
                num_abstracts = len(all_texts)
                st.write(f"Prepared {num_abstracts} documents")
                progress_bar.progress(0.05)

                st.write("Loading embedding model...")
                embed_model = load_embedding_model()
                st.success("Embedding model loaded")
                progress_bar.progress(0.10)

                config = get_adaptive_config(num_abstracts)
                config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)
                config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
                config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
                config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.7)
                config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.2)
                config["INFERENCE_WEIGHT"] = st.session_state.get('inf_weight', 0.1)
                st.write(f"Adaptive config: {config}")
                progress_bar.progress(0.15)

                use_ontology = st.session_state.get('use_ontology', True)
                use_embedding = st.session_state.get('use_embedding_resolution', True)
                use_inference = st.session_state.get('use_inference', True)

                if use_ontology:
                    st.write("Initializing ontology-based concept resolver...")
                    resolver = AdvancedConceptResolver(ontology, embed_model)
                    extractor = EnhancedConceptExtractor(ontology, resolver)
                    st.session_state.resolver = resolver
                    st.session_state.extractor = extractor
                    st.success("Ontology and resolver initialized")
                else:
                    st.write("Using legacy extraction (no ontology)...")
                    resolver = None
                    extractor = None
                progress_bar.progress(0.20)

                st.write("Extracting concepts from abstracts (Parallel)...")
                all_concepts: List[Optional[List[str]]] = [None] * len(df_filtered)
                all_metrics: List[Optional[Dict]] = [None] * len(df_filtered)

                def _process_single_row(idx, row):
                    text = " ".join([
                        str(row[col]) for col in selected_text_cols
                        if col in row and pd.notna(row[col])
                    ])
                    concepts = extractor.extract_from_text(text, idx)
                    metrics: Dict[str, Any] = {}
                    power_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:w|watt)', text, re.I
                    )
                    if power_matches:
                        metrics['laser_power_w'] = [float(m) for m in power_matches]
                    velocity_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:mm/s|m/s)', text, re.I
                    )
                    if velocity_matches:
                        metrics['scan_velocity'] = [float(m) for m in velocity_matches]
                    temp_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:k|°c|celsius)', text, re.I
                    )
                    if temp_matches:
                        metrics['temperature'] = [float(m) for m in temp_matches]
                    return idx, concepts, metrics

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(_process_single_row, idx, row): idx
                        for idx, row in df_filtered.iterrows()
                    }
                    completed = 0
                    total = len(futures)
                    for future in as_completed(futures):
                        idx, concepts, metrics = future.result()
                        all_concepts[idx] = concepts
                        all_metrics[idx] = metrics
                        completed += 1
                        if completed % 10 == 0 or completed == total:
                            progress_bar.progress(
                                0.20 + (completed / total) * 0.15
                            )
                            status.write(
                                f"Extracted {completed}/{total} documents..."
                            )

                all_concepts = [
                    c if c is not None else [] for c in all_concepts
                ]
                all_metrics = [
                    m if m is not None else {} for m in all_metrics
                ]

                if use_ontology and extractor is not None:
                    concept_freq = extractor.get_concept_frequencies()
                    valid_concepts = [
                        c for c, f in concept_freq.items()
                        if f >= config.get("MIN_CONCEPT_FREQ", 2)
                    ]
                    concept_abstract_map: Dict[str, List[int]] = defaultdict(list)
                    for doc_idx, concepts in enumerate(all_concepts):
                        for c in set(concepts):
                            concept_abstract_map[c].append(doc_idx)
                else:
                    concept_freq: Dict[str, int] = defaultdict(int)
                    for concepts in all_concepts:
                        for c in concepts:
                            concept_freq[c] += 1
                    valid_concepts = [
                        c for c, f in concept_freq.items()
                        if f >= config.get("MIN_CONCEPT_FREQ", 2)
                    ]
                    concept_abstract_map = defaultdict(list)
                    for doc_idx, concepts in enumerate(all_concepts):
                        for c in set(concepts):
                            concept_abstract_map[c].append(doc_idx)

                st.write(f"✅ Extraction complete. Found {len(valid_concepts)} valid concepts.")
                progress_bar.progress(0.35)

                valid_concepts = sorted(
                    valid_concepts,
                    key=lambda c: concept_abstract_map.get(c, []).__len__(),
                    reverse=True,
                )
                top_n = config.get("TOP_N_CONCEPTS", 1000)
                if len(valid_concepts) > top_n:
                    valid_concepts = valid_concepts[:top_n]
                concept_to_id = {
                    c: i for i, c in enumerate(valid_concepts)
                }
                id_to_concept = {
                    i: c for i, c in enumerate(valid_concepts)
                }
                st.write(f"**{len(valid_concepts)}** valid concepts retained")
                progress_bar.progress(0.45)

                if len(valid_concepts) < 5:
                    st.error(
                        "Too few concepts extracted. "
                        "Try lowering frequency thresholds."
                    )
                    return

                st.write("Building concept graph...")
                if use_ontology and use_inference:
                    graph_builder = ReasoningEnhancedGraphBuilder(
                        ontology, extractor
                    )
                    nx_graph = graph_builder.build_graph(
                        all_concepts, valid_concepts,
                        concept_to_id, embed_model, config,
                    )
                else:
                    nx_graph = build_hybrid_graph(
                        all_concepts, valid_concepts,
                        concept_to_id, embed_model, config, ontology,
                    )
                pos_pairs, neg_pairs = sample_edges_for_training(
                    nx_graph, valid_concepts, concept_to_id, config,
                )
                st.write(
                    f"Graph: {len(valid_concepts)} nodes, "
                    f"{nx_graph.number_of_edges()} edges"
                )
                progress_bar.progress(0.55)

                st.write("Generating node embeddings...")
                try:
                    with torch.no_grad():
                        embeddings = embed_model.encode(
                            valid_concepts, show_progress_bar=False,
                            batch_size=64, convert_to_numpy=True,
                        )
                    node_features = torch.tensor(
                        embeddings, dtype=torch.float32,
                    )
                except Exception:
                    node_features = torch.randn(len(valid_concepts), 384)
                st.write(f"Node features: {node_features.shape}")
                progress_bar.progress(0.65)

                st.write("Training GraphSAGE...")

                def training_progress(epoch, loss):
                    progress = 0.65 + (epoch / 50) * 0.15
                    progress_bar.progress(min(1.0, progress))
                    if epoch % 10 == 0:
                        status.write(
                            f"Epoch {epoch}/50 | Loss: {loss:.4f}"
                        )

                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id,
                    pos_pairs, neg_pairs, training_progress,
                )
                st.success("GNN training complete")
                progress_bar.progress(0.80)

                st.write("Scoring research directions...")
                concept_properties: Dict[str, float] = {}
                for concept in valid_concepts:
                    doc_indices = concept_abstract_map.get(concept, [])
                    values: List[float] = []
                    for idx in doc_indices:
                        if idx < len(all_metrics):
                            metric_dict = all_metrics[idx]
                            if metric_dict is not None:
                                for metric_values in metric_dict.values():
                                    values.extend(metric_values)
                    concept_properties[concept] = (
                        float(np.median(values)) if values else 0.0
                    )
                X_feat: List[List[float]] = []
                y_target: List[float] = []
                for u, v in nx_graph.edges():
                    pu = concept_properties.get(u, 0)
                    pv = concept_properties.get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(
                        max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0
                    )
                ridge = None
                if len(X_feat) > 5:
                    ridge = Ridge(alpha=1.0).fit(
                        np.array(X_feat), np.array(y_target)
                    )
                top_scores = compute_research_direction_scores(
                    gnn_model, node_features, final_emb, nx_graph,
                    valid_concepts, concept_properties, ridge, embed_model,
                )
                st.write(f"Scored {len(top_scores)} novel pairs")
                progress_bar.progress(0.90)

                st.write("Computing distillation metrics...")
                distill_df = compute_concept_distillation(
                    valid_concepts, concept_abstract_map, all_texts,
                )

                st.write("Running advanced analytics...")
                burst_df = detect_keyword_bursts(
                    df_filtered, valid_concepts,
                    concept_abstract_map, selected_text_cols,
                )
                drift_df = detect_semantic_drift(
                    df_filtered, valid_concepts,
                    concept_abstract_map, selected_text_cols,
                )
                genealogy_df = build_concept_genealogy(
                    nx_graph, valid_concepts, concept_abstract_map,
                )
                bridge_df = detect_cross_domain_bridges(
                    nx_graph, valid_concepts, concept_abstract_map,
                )
                motifs = analyze_network_motifs(nx_graph)

                st.session_state.burst_df = burst_df
                st.session_state.drift_df = drift_df
                st.session_state.genealogy_df = genealogy_df
                st.session_state.bridge_df = bridge_df
                st.session_state.motifs = motifs

                total_time = time.perf_counter() - overall_start
                st.success(f"Analysis complete in {total_time:.1f}s!")
                progress_bar.progress(1.00)
                status.update(
                    label=f"Analysis complete! ({total_time:.1f}s)",
                    state="complete", expanded=False,
                )

                analysis_data = {
                    "valid_concepts": valid_concepts,
                    "concept_to_id": concept_to_id,
                    "id_to_concept": id_to_concept,
                    "concept_abstract_map": concept_abstract_map,
                    "nx_graph": nx_graph,
                    "concept_properties": concept_properties,
                    "ridge": ridge,
                    "top_scores": top_scores,
                    "distill_df": distill_df,
                    "gnn_model": gnn_model,
                    "final_emb": final_emb,
                    "embed_model": embed_model,
                    "all_metrics": all_metrics,
                    "all_texts": all_texts,
                    "config": config,
                    "df_filtered": df_filtered,
                    "selected_text_cols": selected_text_cols,
                }
                if use_ontology:
                    analysis_data.update({
                        "ontology": ontology,
                        "resolver": resolver,
                        "extractor": extractor,
                        "graph_builder": graph_builder if use_inference else None,
                        "reasoning_paths": graph_builder.reasoning_paths if use_inference else [],
                    })
                st.session_state.analysis_data = analysis_data

                st.session_state.edit_history = GraphEditHistory()
                st.session_state.edit_history.save_snapshot(
                    nx_graph, valid_concepts, concept_to_id,
                    id_to_concept, concept_abstract_map,
                )
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            return
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- APPLY GRAPH EDITS ---
    if (
        st.session_state.get('apply_edits')
        and st.session_state.analysis_data is not None
    ):
        data = st.session_state.analysis_data
        st.session_state.edit_history.save_snapshot(
            data["nx_graph"], data["valid_concepts"],
            data["concept_to_id"], data["id_to_concept"],
            data["concept_abstract_map"],
        )
        (
            nx_graph, valid_concepts, concept_to_id,
            id_to_concept, concept_abstract_map, edited,
        ) = apply_graph_edits(
            data["nx_graph"], data["valid_concepts"],
            data["concept_to_id"], data["id_to_concept"],
            data["concept_abstract_map"],
            nodes_to_remove=st.session_state.get('nodes_to_remove', []),
            nodes_to_merge=st.session_state.get('nodes_to_merge', []),
            merge_name=st.session_state.get('merge_name', None),
            new_edge=st.session_state.get('new_edge', None),
            new_edge_weight=st.session_state.get('new_edge_weight', 1.0),
            min_degree=st.session_state.get('filter_min_degree', 0),
            min_freq=st.session_state.get('filter_min_freq', 0),
        )
        if edited:
            st.session_state.analysis_data["nx_graph"] = nx_graph
            st.session_state.analysis_data["valid_concepts"] = valid_concepts
            st.session_state.analysis_data["concept_to_id"] = concept_to_id
            st.session_state.analysis_data["id_to_concept"] = id_to_concept
            st.session_state.analysis_data["concept_abstract_map"] = concept_abstract_map
            st.success("Graph edits applied successfully!")
            st.session_state['apply_edits'] = False
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

    # --- DISPLAY RESULTS ---
    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        valid_concepts = data["valid_concepts"]
        concept_abstract_map = data["concept_abstract_map"]
        nx_graph = data["nx_graph"]
        top_scores = data["top_scores"]
        distill_df = data["distill_df"]
        df_filtered = data.get("df_filtered", pd.DataFrame())
        selected_text_cols = data.get("selected_text_cols", [])
        cmap = st.session_state.get('cmap_name', 'viridis')
        top_n_graph = st.session_state.get('top_n_graph', 200)

        has_reasoning = "ontology" in data
        tab_names = [
            "📊 Visualization", "🧪 Distillation", "🎯 Research Directions",
            "✅ Validation", "📥 Export", "📈 Extra Viz",
            "🔬 Advanced Analytics",
        ]
        if has_reasoning:
            tab_names.append("🧠 Reasoning Dashboard")
        tabs = st.tabs(tab_names)
        tab_idx = 0

        with tabs[tab_idx]:
            st.subheader("Interactive Concept Graph")
            if nx_graph.number_of_nodes() == 0:
                st.warning("No nodes to display.")
            elif nx_graph.number_of_edges() == 0:
                st.warning("No edges - building semantic fallback")
                nx_graph = nx.complete_graph(len(valid_concepts))
                nx_graph = nx.relabel_nodes(
                    nx_graph, {i: valid_concepts[i] for i in range(len(valid_concepts))}
                )
            viz_choice = st.session_state.get('viz_backend', 'PyVis (Interactive)')
            physics = st.session_state.get('physics_enabled', True)
            physics_preset = st.session_state.get(
                'effective_physics', PHYSICS_PRESETS["Stable (Default)"]
            )
            theme = THEME_PRESETS.get(
                st.session_state.get('theme', 'Bright (Default)'),
                THEME_PRESETS["Bright (Default)"],
            )
            top_n = st.session_state.get('top_n_graph', 0)
            show_weights = st.session_state.get('show_edge_weights', False)
            edge_label_mode = st.session_state.get('edge_label_mode', 'hover')

            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(
                    nx_graph, concept_abstract_map,
                    physics_enabled=physics,
                    cmap_name=cmap,
                    top_n_nodes=top_n,
                    theme=theme,
                    physics_preset=physics_preset,
                    show_edge_weights=show_weights,
                    edge_label_mode=edge_label_mode,
                    node_label_size=st.session_state.get('node_label_size') or 12,
                    node_label_position=st.session_state.get('node_label_position') or 'center',
                    node_font_face=st.session_state.get('node_font_face') or 'Inter, Segoe UI, Roboto, sans-serif',
                    edge_label_size=st.session_state.get('edge_label_size') or 10,
                    edge_label_color=st.session_state.get('edge_label_color') or None,
                    edge_label_position=st.session_state.get('edge_label_position') or 'middle',
                    use_abbreviated_labels=st.session_state.get('use_abbreviated_labels', False),
                    max_label_length=st.session_state.get('max_label_length', 15),
                    enable_node_highlight=st.session_state.get('enable_node_highlight', False),
                    show_definitions=st.session_state.get('show_definitions', True),
                    # NEW parameters
                    edge_lightness=st.session_state.get('edge_lightness', 0.6),
                    edge_color_mode=st.session_state.get('edge_color_mode', 'theme'),
                    custom_edge_color=st.session_state.get('custom_edge_color', '#AAAAAA'),
                    tooltip_font_size=st.session_state.get('tooltip_font_size', 13),
                    node_legend_font_size=st.session_state.get('node_legend_font_size', 13),
                )
            elif viz_choice == "Plotly 2D":
                render_graph_plotly_2d(
                    nx_graph, concept_abstract_map,
                    cmap_name=cmap,
                    top_n_nodes=top_n,
                    theme=theme,
                    show_edge_weights=show_weights,
                    node_label_size=st.session_state.get('node_label_size') or 10,
                )
            elif viz_choice == "Plotly 3D":
                render_graph_plotly_3d(
                    nx_graph, concept_abstract_map,
                    cmap_name=cmap, top_n_nodes=top_n,
                    theme=theme, show_edge_weights=show_weights,
                )
            else:
                render_graph_fallback(
                    nx_graph, concept_abstract_map,
                    theme=theme, show_edge_weights=show_weights,
                )
            with st.expander("Graph Metrics"):
                metrics = compute_graph_metrics(nx_graph)
                display_metric_dashboard(metrics, theme=theme)
            with st.expander("Domain Hierarchy (Sunburst)"):
                cat_filter = st.session_state.get('sunburst_categories', [])
                bv_mode = st.session_state.get('sunburst_branchvalues', 'total')
                if cat_filter:
                    filtered_concepts = [
                        c for c in valid_concepts
                        if abstract_concepts_to_categories([c]).get(c, 'general') in cat_filter
                    ]
                    filtered_map = {
                        c: concept_abstract_map[c]
                        for c in filtered_concepts if c in concept_abstract_map
                    }
                else:
                    filtered_concepts = valid_concepts
                    filtered_map = concept_abstract_map
                labels, parents, values = build_category_hierarchy(
                    filtered_concepts, filtered_map,
                    top_n_per_category=st.session_state.get('top_n_sunburst', 0),
                )
                render_sunburst_chart(
                    labels, parents, values,
                    cmap_name=st.session_state.get('sunburst_cmap', cmap),
                    theme=theme,
                    branchvalues=bv_mode,
                    label_size=st.session_state.get('sunburst_label_size') or 20,
                    width=st.session_state.get('sunburst_width') or 900,
                    height=st.session_state.get('sunburst_height') or 700,
                    show_labels=st.session_state.get('sunburst_show_labels', True),
                    show_values=st.session_state.get('sunburst_show_values', False),
                    hover_info=st.session_state.get('sunburst_hover_info', 'all'),
                    font_family=st.session_state.get(
                        'sunburst_font_family',
                        st.session_state.get('node_font_face', 'Inter, Segoe UI, Roboto, sans-serif'),
                    ),
                    legend_font_size=st.session_state.get('sunburst_legend_font_size', 12),  # NEW
                )
            with st.expander("Concept Radar"):
                radar_k = st.session_state.get('top_n_radar', 15)
                if radar_k == 0:
                    radar_k = min(15, len(distill_df))
                render_radar_chart(
                    distill_df, top_k=radar_k, cmap_name=cmap, theme=theme,
                )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Concept Distillation Efficiency")
            top_n = st.slider(
                "Show Top N", 10, min(200, len(distill_df)), 50,
                key="distill_top_n",
            )
            display_df = distill_df.head(top_n)
            st.dataframe(display_df, use_container_width=True)
            st.markdown("**Efficiency vs Frequency:**")
            chart_df = display_df.set_index('concept')[['distillation_efficiency']]
            st.bar_chart(chart_df)
            st.markdown("**Multi-Metric Comparison:**")
            metric_cols = [
                c for c in [
                    'frequency', 'tfidf_weight',
                    'semantic_density', 'coherence_score',
                ]
                if c in display_df.columns
            ]
            if metric_cols:
                compare_df = display_df[['concept'] + metric_cols].set_index('concept')
                st.line_chart(compare_df)

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Top Research Direction Recommendations")
            if top_scores.empty:
                st.info(
                    "No novel pairs scored. "
                    "The graph may be too dense or too sparse."
                )
            else:
                st.write(f"Top {len(top_scores)} novel concept pairs:")
                st.dataframe(
                    top_scores[[
                        'concept_u', 'concept_v', 'composite_score',
                        'gnn_affinity', 'semantic_novelty',
                        'expected_property_gain', 'feasibility_score',
                    ]].head(20),
                    use_container_width=True,
                )
                csv_scores = top_scores.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Scores (CSV)", data=csv_scores,
                    file_name="mpea_research_directions.csv", mime="text/csv",
                )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Mathematical Validation")
            val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Modularity", f"{val_metrics.get('modularity', 0):.3f}"
            )
            col2.metric(
                "Silhouette",
                f"{val_metrics.get('silhouette_score', 0):.3f}",
            )
            col3.metric(
                "Communities", val_metrics.get('n_communities', 0)
            )
            col4.metric(
                "Significant Edges",
                val_metrics.get('edge_significant_count', 0),
            )
            if not top_scores.empty:
                n_boot = st.session_state.get('bootstrap_samples', 500)
                alpha = st.session_state.get('alpha_level', 0.05)
                mean_score, ci_low, ci_high = compute_bootstrap_ci(
                    top_scores['composite_score'].values,
                    n_bootstrap=n_boot, alpha=alpha,
                )
                st.success(
                    f"Composite Score: `{mean_score:.3f}` | "
                    f"{int((1 - alpha) * 100)}% CI: "
                    f"`[{ci_low:.3f}, {ci_high:.3f}]`"
                )
                X_feat: List[List[float]] = []
                y_target: List[float] = []
                for u, v in nx_graph.edges():
                    pu = data["concept_properties"].get(u, 0)
                    pv = data["concept_properties"].get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(
                        max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0
                    )
                if data["ridge"] is not None and len(X_feat) > 5:
                    y_pred = data["ridge"].predict(np.array(X_feat))
                    st.markdown("### Ridge Regression (Property Prediction)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R2", f"{r2_score(y_target, y_pred):.3f}")
                    c2.metric(
                        "MAE", f"{mean_absolute_error(y_target, y_pred):.2f}"
                    )
                    c3.metric(
                        "RMSE",
                        f"{np.sqrt(mean_squared_error(y_target, y_pred)):.2f}",
                    )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Export & Post-Processing")
            export_format = st.selectbox("Format:", [
                "GraphML", "JSON (Full Metadata)", "JSON (Compact)",
                "CSV (Edges + Metadata)", "CSV (Nodes + Metadata)",
                "PNG", "SVG", "GEXF",
            ])
            include_metadata = st.checkbox(
                "Include metadata in export", value=True,
            )
            if st.button("Generate Export"):
                result = export_graph(
                    nx_graph, concept_abstract_map,
                    export_format, include_metadata,
                )
                if result[0]:
                    data_bytes, mime, filename = result
                    st.download_button(
                        "💾 Save File", data=data_bytes,
                        file_name=filename, mime=mime,
                    )
            st.markdown("---")
            st.subheader("Publication-Ready Figure")
            pub_dpi = st.slider("DPI", 150, 600, 300, step=50)
            pub_figsize = st.selectbox(
                "Figure size:",
                [(10, 8), (12, 10), (14, 12), (16, 14)],
                index=2,
            )
            if st.button("Generate Publication Figure"):
                pub_bytes = export_publication_figure(
                    nx_graph, valid_concepts, concept_abstract_map,
                    cmap_name=cmap, dpi=pub_dpi, figsize=pub_figsize,
                )
                if pub_bytes:
                    st.download_button(
                        "📥 Download Publication PNG",
                        data=pub_bytes,
                        file_name="mpea_graph_publication.png",
                        mime="image/png",
                    )
            st.markdown("---")
            st.subheader("Automated Analysis Report")
            if st.button("Generate Markdown Report"):
                burst_df = st.session_state.get('burst_df', pd.DataFrame())
                drift_df = st.session_state.get('drift_df', pd.DataFrame())
                genealogy_df = st.session_state.get('genealogy_df', pd.DataFrame())
                bridge_df = st.session_state.get('bridge_df', pd.DataFrame())
                motifs = st.session_state.get('motifs', {})
                report = generate_analysis_report(
                    nx_graph, valid_concepts, concept_abstract_map,
                    top_scores, distill_df, burst_df, drift_df,
                    genealogy_df, bridge_df, motifs, val_metrics, df_filtered,
                )
                st.download_button(
                    "📄 Download Report (Markdown)",
                    data=report.encode('utf-8'),
                    file_name="mpea_laser_analysis_report.md",
                    mime="text/markdown",
                )
                with st.expander("Preview Report"):
                    st.markdown(report)
            concept_list_df = pd.DataFrame({
                'concept': valid_concepts,
                'frequency': [
                    len(concept_abstract_map.get(c, [])) for c in valid_concepts
                ],
                'degree': [nx_graph.degree(c) for c in valid_concepts],
                'category': [
                    abstract_concepts_to_categories([c]).get(c, 'general')
                    for c in valid_concepts
                ],
                'concept_type': [
                    nx_graph.nodes[c].get('concept_type', 'general')
                    for c in valid_concepts
                ],
                'definition': [
                    nx_graph.nodes[c].get('definition', '')
                    for c in valid_concepts
                ],
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📋 Download Concept List (CSV)",
                data=csv_concepts,
                file_name="mpea_concepts_enhanced.csv", mime="text/csv",
            )
            with st.expander("📖 Concept Definitions & Meanings"):
                defs_df = concept_list_df[
                    concept_list_df['definition'] != ''
                ][['concept', 'definition', 'category']]
                if not defs_df.empty:
                    st.dataframe(defs_df, use_container_width=True)
                else:
                    st.info(
                        "No definitions available. "
                        "Enable ontology-based resolution to see concept definitions."
                    )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Extra Visualizations")
            theme = THEME_PRESETS.get(
                st.session_state.get('theme', 'Bright (Default)'),
                THEME_PRESETS["Bright (Default)"],
            )
            with st.expander("Concept Timeline", expanded=True):
                render_concept_timeline(
                    df_filtered, valid_concepts,
                    concept_abstract_map, theme=theme,
                )
            with st.expander("Co-occurrence Heatmap"):
                heatmap_n = st.slider(
                    "Top N concepts for heatmap", 5, 50, 25,
                    key="heatmap_n_slider",
                )
                render_cooccurrence_heatmap(
                    nx_graph, valid_concepts, concept_abstract_map,
                    top_n=heatmap_n, theme=theme,
                )
            with st.expander("t-SNE Projection"):
                embed_model = data.get("embed_model")
                if embed_model:
                    render_tsne_projection(
                        valid_concepts, concept_abstract_map,
                        embed_model, theme=theme,
                    )
                else:
                    st.info("Embedding model not available. Rebuild the graph.")
            with st.expander("Community Detection"):
                render_community_detection(
                    nx_graph, valid_concepts,
                    concept_abstract_map, theme=theme,
                )
            with st.expander("Concept Growth Rate"):
                render_concept_growth(
                    df_filtered, valid_concepts,
                    concept_abstract_map, theme=theme,
                )
            with st.expander("Bubble Chart (Importance)"):
                render_bubble_chart(
                    nx_graph, valid_concepts,
                    concept_abstract_map, distill_df, theme=theme,
                )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Advanced Analytics")
            with st.expander("Keyword Burst Detection", expanded=True):
                burst_df = st.session_state.get('burst_df')
                if burst_df is not None and not burst_df.empty:
                    st.dataframe(burst_df.head(20), use_container_width=True)
                    fig = px.bar(
                        burst_df.head(15), x='concept', y='burst_score',
                        color='burst_year',
                        title=(
                            "Keyword Bursts "
                            "(Sudden Spikes in Publication Frequency)"
                        ),
                        labels={
                            'burst_score': 'Burst Score',
                            'concept': 'Concept',
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No burst data available. "
                        "Build graph with temporal data."
                    )
            with st.expander("Semantic Drift Detection"):
                drift_df = st.session_state.get('drift_df')
                if drift_df is not None and not drift_df.empty:
                    st.dataframe(drift_df.head(20), use_container_width=True)
                    fig = px.bar(
                        drift_df.head(15), x='concept', y='semantic_drift',
                        title=(
                            "Semantic Drift "
                            "(Contextual Meaning Shift Over Time)"
                        ),
                        labels={
                            'semantic_drift': 'Drift Score',
                            'concept': 'Concept',
                        },
                        color='semantic_drift',
                        color_continuous_scale='RdYlBu_r',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No drift data available. "
                        "Build graph with temporal data spanning multiple years."
                    )
            with st.expander("Concept Genealogy"):
                genealogy_df = st.session_state.get('genealogy_df')
                if genealogy_df is not None and not genealogy_df.empty:
                    st.dataframe(
                        genealogy_df.head(20), use_container_width=True,
                    )
                    gen_counts = genealogy_df['generation'].value_counts()
                    fig = px.pie(
                        values=gen_counts.values, names=gen_counts.index,
                        title="Concept Generations Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No genealogy data available.")
            with st.expander("Cross-Domain Bridge Detection"):
                bridge_df = st.session_state.get('bridge_df')
                if bridge_df is not None and not bridge_df.empty:
                    st.dataframe(
                        bridge_df.head(20), use_container_width=True,
                    )
                    fig = px.scatter(
                        bridge_df.head(30),
                        x='betweenness', y='connected_categories',
                        size='bridge_score', color='own_category',
                        hover_data=['concept', 'categories'],
                        title="Cross-Domain Bridge Concepts",
                        labels={
                            'betweenness': 'Betweenness Centrality',
                            'connected_categories': 'Categories Connected',
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No bridge data available.")
            with st.expander("Network Motif Analysis"):
                motifs = st.session_state.get('motifs', {})
                if motifs:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(
                        "Triangles", motifs.get('total_triangles', 0)
                    )
                    col2.metric("Cliques", motifs.get('total_cliques', 0))
                    col3.metric(
                        "Max Clique Size", motifs.get('max_clique_size', 0)
                    )
                    col4.metric(
                        "Star Motifs", motifs.get('star_motifs', 0)
                    )
                    if motifs.get('top_stars'):
                        st.markdown(
                            "**Top Star Motifs (Central Hubs):**"
                        )
                        star_df = pd.DataFrame(
                            motifs['top_stars'],
                            columns=['Concept', 'Degree', 'Clustering'],
                        )
                        st.dataframe(
                            star_df, use_container_width=True,
                        )
                else:
                    st.info("No motif data available.")
            with st.expander("Centrality Comparison & Degree Distribution"):
                centrality_df = compute_centrality_comparison(
                    nx_graph, valid_concepts,
                )
                if not centrality_df.empty:
                    st.dataframe(
                        centrality_df.head(20), use_container_width=True,
                    )
                    corr_cols = [
                        'degree', 'betweenness', 'closeness',
                        'eigenvector', 'pagerank',
                    ]
                    available = [
                        c for c in corr_cols if c in centrality_df.columns
                    ]
                    if len(available) >= 2:
                        corr_matrix = centrality_df[available].corr()
                        fig = px.imshow(
                            corr_matrix, text_auto=True, aspect="auto",
                            title="Centrality Correlation Matrix",
                            color_continuous_scale='RdBu_r',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    fig = plot_degree_distribution(nx_graph, theme=theme)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No centrality data available.")

        if has_reasoning:
            tab_idx += 1
            with tabs[tab_idx]:
                ontology_data = data.get("ontology")
                extractor_data = data.get("extractor")
                if ontology_data and extractor_data:
                    render_reasoning_dashboard(
                        nx_graph, valid_concepts, ontology_data, extractor_data,
                    )
                else:
                    st.info(
                        "Reasoning data not available. "
                        "Rebuild graph with ontology enabled."
                    )


if __name__ == "__main__":
    main()
