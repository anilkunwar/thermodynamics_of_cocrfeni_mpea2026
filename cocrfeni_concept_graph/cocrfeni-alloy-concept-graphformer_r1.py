#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sodium-Ion Battery Quantitative Descriptor Graph v6.2 (SIB Edition) + Microtransformer #2
==========================================================================================
Multi-level reasoning concept graph for numerical/quantitative description
of Sodium-Ion Batteries (SIBs).
Focus: Electrochemical, Compositional, and Performance Descriptors.

This is a TRUE architectural port of the AgNP-Sustainability-ConceptGraph codebase,
preserving every memory-safe pattern, visualization pattern, and session-state management
pattern from the working AgNPs code. The domain ontology and extraction patterns have been
replaced with those for Sodium-Ion Battery quantitative descriptors.

NEW in v6.2 — Microtransformer #2: KG-RAG Extractor with LatentMoE:
- Encodes graph traversals (node-edge-node sequences) to extract phase stability
  and electromechanical links.
- Uses Latent Mixture of Experts (l-MoE_acc) with 32 specialized latent domains.
- Interactive UI to select any source/target concept, automatically finds a path
  via the graph or ontology inference.
- Visualizes per-token expert routing as heatmaps and bar charts.
- Supports ONNX export for edge deployment (Ubuntu/Lubuntu).

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run:
    streamlit run sib_concept_graph_v6.2_mt.py

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
    page_title="Sodium-Ion Battery Quantitative Descriptor Graph v6.2 + MT",
    page_icon="🔋",
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
# ENHANCED ONTOLOGY & NLP REASONING SYSTEM (SODIUM-ION BATTERY)
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
    MEASURES = "measures"
    OBSERVES = "observes"
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
    RelationshipType.MEASURES:          "#6B8E23",   # olive drab
    RelationshipType.OBSERVES:          "#808000",   # olive
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



def lighten_hex_color(hex_color: str, factor: float) -> str:
    """Lighten a hex color by mixing with white."""
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
    """Comprehensive ontology for Sodium-Ion Battery Quantitative Descriptors."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self._build_ontology()

    def _build_ontology(self) -> None:
        # === CATHODE MATERIALS ===
        self._add_concept("layered_oxide_cathode", ConceptType.MATERIAL,
            synonyms={"na_mno2", "namno2", "na_x_mno2", "p2_na_mno2", "o3_na_mno2", "layered oxide"},
            definition="Sodium transition metal oxide cathodes (e.g., NaₓMnO₂, NaₓCoO₂) with layered structure")
        self._add_concept("polyanionic_cathode", ConceptType.MATERIAL,
            synonyms={"na3v2(po4)3", "nvp", "na3v2(po4)2f3", "nvpf", "na3v2(po4)3", "polyanion"},
            definition="Polyanionic compound cathodes with NASICON or phosphate frameworks (e.g., Na₃V₂(PO₄)₃)")
        self._add_concept("prussian_blue_analogue", ConceptType.MATERIAL,
            synonyms={"pba", "prussian blue", "na2mnfe(cn)6", "hexacyanoferrate", "pba cathode"},
            definition="Prussian blue analogues (PBAs) with open framework for sodium intercalation")
        self._add_concept("nasicon_cathode", ConceptType.MATERIAL,
            synonyms={"nasicon", "na superionic conductor", "na3zr2si2po12", "na3v2(po4)3", "nasicon-type"},
            definition="NASICON-type cathodes with 3D framework for fast sodium ion transport")

        # === ANODE MATERIALS ===
        self._add_concept("hard_carbon", ConceptType.MATERIAL,
            synonyms={"hc", "hard carbon anode", "disordered carbon", "non-graphitizable carbon"},
            definition="Hard carbon with disordered structure, the most common sodium-ion battery anode")
        self._add_concept("sodium_metal", ConceptType.MATERIAL,
            synonyms={"na metal", "sodium anode", "metallic sodium", "na foil"},
            definition="Pure sodium metal anode for high energy density, requires stable electrolyte")
        self._add_concept("alloying_anode", ConceptType.MATERIAL,
            synonyms={"sn anode", "sb anode", "bi anode", "tin anode", "antimony anode", "alloy anode"},
            definition="Alloying-type anode materials (Sn, Sb, Bi) with high capacity but large volume change")
        self._add_concept("intercalation_anode", ConceptType.MATERIAL,
            synonyms={"tio2", "na2ti3o7", "layered titanium oxide", "naxmoo2"},
            definition="Intercalation anode materials (e.g., TiO₂, Na₂Ti₃O₇) with stable cycling")

        # === ELECTROLYTES ===
        self._add_concept("liquid_electrolyte", ConceptType.MATERIAL,
            synonyms={"organic electrolyte", "naclo4 in ec/dec", "na pf6", "aqueous electrolyte", "liquid sodium electrolyte"},
            definition="Liquid electrolyte (organic solvent with sodium salt) for sodium-ion batteries")
        self._add_concept("solid_electrolyte", ConceptType.MATERIAL,
            synonyms={"solid sodium electrolyte", "nasicon", "na3ps4", "na3zr2si2po12", "sulfide electrolyte"},
            definition="Solid-state electrolyte for all-solid-state sodium batteries (ceramic, sulfide, polymer)")
        self._add_concept("polymer_electrolyte", ConceptType.MATERIAL,
            synonyms={"peo", "polyethylene oxide", "gel polymer", "polymer electrolyte", "quasi-solid"},
            definition="Polymer-based electrolyte (PEO, PAN) with sodium salt, often gel or solid")
        self._add_concept("quasi_solid_electrolyte", ConceptType.MATERIAL,
            synonyms={"gel electrolyte", "quasi-solid", "semi-solid", "in-situ polymerized"},
            definition="Quasi-solid electrolyte blending polymer and liquid for improved interface")

        # === ELECTROCHEMICAL PROPERTIES ===
        self._add_concept("specific_capacity", ConceptType.PROPERTY,
            synonyms={"capacity", "mah/g", "specific charge", "gravimetric capacity"},
            definition="Specific capacity (mAh/g) of electrode material, a key performance metric")
        self._add_concept("energy_density", ConceptType.PROPERTY,
            synonyms={"wh/kg", "specific energy", "volumetric energy density", "wh/l"},
            definition="Energy density (Wh/kg) of the full cell or electrode")
        self._add_concept("coulombic_efficiency", ConceptType.PROPERTY,
            synonyms={"ce", "coloumbic efficiency", "charge-discharge efficiency", "reversibility"},
            definition="Coulombic efficiency (%), the ratio of discharge to charge capacity")
        self._add_concept("cycle_life", ConceptType.PROPERTY,
            synonyms={"cycling stability", "retention", "capacity retention", "long-term cycling"},
            definition="Cycle life (number of cycles before capacity drops below 80%)")
        self._add_concept("rate_capability", ConceptType.PROPERTY,
            synonyms={"rate performance", "high rate", "rate capability", "c-rate"},
            definition="Ability to maintain capacity at high charge/discharge rates (C-rate)")
        self._add_concept("ionic_conductivity", ConceptType.PROPERTY,
            synonyms={"na+ conductivity", "s/cm", "ionic transport", "bulk conductivity", "grain boundary conductivity"},
            definition="Ionic conductivity (S/cm) of electrolyte or electrode, critical for rate performance")
        self._add_concept("voltage_plateau", ConceptType.PROPERTY,
            synonyms={"discharge voltage", "charge voltage", "voltage profile", "operating voltage"},
            definition="Voltage plateau (V) during discharge/charge, determining energy density")

        # === PHENOMENA ===
        self._add_concept("dendrite_growth", ConceptType.PHENOMENON,
            synonyms={"sodium dendrite", "dendrite formation", "mossy sodium", "dendritic sodium"},
            definition="Formation of sodium dendrites during plating, causing short circuits and safety issues")
        self._add_concept("sei_formation", ConceptType.PHENOMENON,
            synonyms={"solid electrolyte interphase", "sei layer", "passivation film", "interface layer"},
            definition="Solid-electrolyte interphase (SEI) formed on anode, crucial for cycle life")
        self._add_concept("sodium_plating_stripping", ConceptType.PHENOMENON,
            synonyms={"na plating", "sodium stripping", "plating/stripping", "electrodeposition"},
            definition="Electrochemical deposition and dissolution of sodium metal")
        self._add_concept("intercalation", ConceptType.PHENOMENON,
            synonyms={"na+ insertion", "sodium intercalation", "deintercalation", "host-guest"},
            definition="Insertion/extraction of Na+ ions into host electrode structure")
        self._add_concept("conversion_reaction", ConceptType.PHENOMENON,
            synonyms={"conversion", "alloying/dealloying", "conversion electrode"},
            definition="Electrochemical conversion reaction (e.g., metal oxide + Na -> Na2O + metal)")

        # === METHODS ===
        self._add_concept("cyclic_voltammetry", ConceptType.METHOD,
            synonyms={"cv", "cyclic voltammogram", "voltammetry"},
            definition="Cyclic voltammetry (CV) for electrochemical characterization")
        self._add_concept("electrochemical_impedance_spectroscopy", ConceptType.METHOD,
            synonyms={"eis", "nyquist plot", "impedance spectroscopy"},
            definition="Electrochemical impedance spectroscopy (EIS) for interface and kinetics")
        self._add_concept("galvanostatic_cycling", ConceptType.METHOD,
            synonyms={"constant current", "cccv", "galvanostatic", "charge-discharge cycling"},
            definition="Galvanostatic cycling at constant current")
        self._add_concept("operando_characterization", ConceptType.METHOD,
            synonyms={"in situ xrd", "operando xrd", "in situ raman", "real-time characterization"},
            definition="Operando characterization during battery operation (XRD, Raman, etc.)")

        # === PARAMETERS ===
        self._add_concept("current_density", ConceptType.PARAMETER,
            synonyms={"ma/g", "a/g", "c-rate", "charge current", "discharge current"},
            definition="Current density (mA/g or A/g) applied during cycling")
        self._add_concept("cut_off_voltage", ConceptType.PARAMETER,
            synonyms={"voltage window", "v", "upper cut-off", "lower cut-off"},
            definition="Cut-off voltage (V) window for charge/discharge")
        self._add_concept("temperature", ConceptType.PARAMETER,
            synonyms={"celsius", "kelvin", "operating temperature", "thermal"},
            definition="Temperature (°C or K) during battery operation or testing")

        # === PROCESSING ===
        self._add_concept("slurry_coating", ConceptType.PROCESS,
            synonyms={"electrode coating", "doctor blade", "tape casting", "slurry"},
            definition="Slurry coating process for electrode fabrication")
        self._add_concept("cell_assembly", ConceptType.PROCESS,
            synonyms={"coin cell", "pouch cell", "swagelok", "cell fabrication"},
            definition="Assembly of battery cell (coin, pouch, Swagelok)")

        # === GENERAL SIB ===
        self._add_concept("sodium_ion_battery", ConceptType.MATERIAL,
            synonyms={"sib", "na-ion battery", "sodium battery", "na battery"},
            definition="Sodium-ion battery (SIB) system")
        self._add_concept("all_solid_state_sodium_battery", ConceptType.MATERIAL,
            synonyms={"asssb", "solid-state sodium", "all-solid-state na battery"},
            definition="All-solid-state sodium battery with solid electrolyte")

        # === NEW CONCEPTS v6.2+ (Extended Ontology) ===
        self._add_concept("mxene", ConceptType.MATERIAL,
            synonyms={"mxenes", "ti3c2tx", "tinbased mxene", "v2ctz", "2d transition metal carbide", "ti3c2", "v2c", "nb2c", "mo2c"},
            definition="MXenes, 2D transition metal carbides/nitrides (e.g., Ti3C2Tx, V2CTz), used as high-rate anodes or cathodes in SIBs")
        self._add_concept("organic_cathode", ConceptType.MATERIAL,
            synonyms={"organic electrode", "pdtca", "ppta", "naphthalene tetracarboxylic dianhydride", "conjugated carbonyl compound", "organic cathode material", "p-type organic cathode", "n-type organic cathode"},
            definition="Organic cathode materials offering structural flexibility and high theoretical capacity via carbonyl or conjugated groups")
        self._add_concept("sodium_ion_capacitor", ConceptType.MATERIAL,
            synonyms={"na-ion capacitor", "sib capacitor", "supercapacitor", "hybrid sodium capacitor", "sihc", "sodium ion hybrid capacitor"},
            definition="Sodium-ion hybrid capacitor (SIHC) bridging the gap between batteries and supercapacitors")
        self._add_concept("conversion_anode", ConceptType.MATERIAL,
            synonyms={"conversion reaction anode", "metal sulfide anode", "metal oxide anode", "fes2", "cos2", "mos2"},
            definition="Conversion-type anode materials reacting with sodium to form new phases, offering high capacity")
        self._add_concept("solid_polymer_electrolyte", ConceptType.MATERIAL,
            synonyms={"spe", "solid polymer electrolyte", "pan based electrolyte", "pan na", "pmma electrolyte"},
            definition="Solid polymer electrolyte with sodium salt for flexible all-solid-state SIBs")
        self._add_concept("aqueous_electrolyte", ConceptType.MATERIAL,
            synonyms={"aqueous na electrolyte", "water based electrolyte", "na2so4 electrolyte", "naoh electrolyte"},
            definition="Aqueous electrolyte using water as solvent, offering intrinsic safety and low cost")
        self._add_concept("interface_engineering", ConceptType.PROCESS,
            synonyms={"surface coating", "interfacial layer", "artificial sei", "al2o3 coating", "carbon coating"},
            definition="Surface and interface engineering strategies to stabilize electrode-electrolyte interfaces")
        self._add_concept("pre_sodiation", ConceptType.PROCESS,
            synonyms={"pre-sodiation", "sodium compensation", "sodium pre-loading", "sacrificial salt"},
            definition="Pre-sodiation techniques to compensate for initial sodium loss and improve ICE")
        self._add_concept("thermal_runaway", ConceptType.PHENOMENON,
            synonyms={"thermal abuse", "overheating", "battery fire", "thermal stability", "self heating"},
            definition="Thermal runaway and safety-related thermal phenomena in sodium-ion batteries")
        self._add_concept("volume_expansion", ConceptType.PHENOMENON,
            synonyms={"structural change", "lattice expansion", "pulverization", "mechanical stress"},
            definition="Volume expansion and mechanical degradation of electrodes during sodiation/desodiation")
        self._add_concept("full_cell", ConceptType.MODEL,
            synonyms={"full sodium ion cell", "sodium ion full cell", "practical cell", "anode free cell"},
            definition="Practical full-cell configuration pairing cathode and anode with limited sodium source")

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
        # === MATERIALS → PROPERTIES ===
        causal_chains = [
            ("hard_carbon", RelationshipType.INFLUENCES, "specific_capacity", 0.85),
            ("hard_carbon", RelationshipType.INFLUENCES, "cycle_life", 0.75),
            ("sodium_metal", RelationshipType.INFLUENCES, "energy_density", 0.90),
            ("sodium_metal", RelationshipType.INFLUENCES, "dendrite_growth", 0.80),
            ("layered_oxide_cathode", RelationshipType.INFLUENCES, "specific_capacity", 0.80),
            ("polyanionic_cathode", RelationshipType.INFLUENCES, "cycle_life", 0.85),
            ("prussian_blue_analogue", RelationshipType.INFLUENCES, "rate_capability", 0.80),
            ("solid_electrolyte", RelationshipType.INFLUENCES, "ionic_conductivity", 0.90),
            ("solid_electrolyte", RelationshipType.INFLUENCES, "dendrite_growth", -0.70),
            ("liquid_electrolyte", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.80),
            ("polymer_electrolyte", RelationshipType.INFLUENCES, "cycle_life", 0.75),
            ("quasi_solid_electrolyte", RelationshipType.INFLUENCES, "energy_density", 0.70),
            # Properties → Performance
            ("specific_capacity", RelationshipType.CAUSES, "energy_density", 0.95),
            ("coulombic_efficiency", RelationshipType.CAUSES, "cycle_life", 0.90),
            ("rate_capability", RelationshipType.INFLUENCES, "specific_capacity", 0.80),
            ("ionic_conductivity", RelationshipType.INFLUENCES, "rate_capability", 0.85),
            ("voltage_plateau", RelationshipType.INFLUENCES, "energy_density", 0.90),
            # Phenomena → Performance
            ("dendrite_growth", RelationshipType.CAUSES, "cycle_life", -0.85),
            ("dendrite_growth", RelationshipType.CAUSES, "coulombic_efficiency", -0.80),
            ("sei_formation", RelationshipType.INFLUENCES, "cycle_life", 0.70),
            ("sodium_plating_stripping", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.75),
            ("intercalation", RelationshipType.INFLUENCES, "specific_capacity", 0.80),
            # Methods → Phenomena
            ("cyclic_voltammetry", RelationshipType.DETECTS, "intercalation", 0.85),
            ("electrochemical_impedance_spectroscopy", RelationshipType.DETECTS, "sei_formation", 0.80),
            ("galvanostatic_cycling", RelationshipType.MEASURES, "specific_capacity", 0.90),
            ("operando_characterization", RelationshipType.OBSERVES, "dendrite_growth", 0.75),
            # Parameters → Performance
            ("current_density", RelationshipType.INFLUENCES, "rate_capability", 0.85),
            ("cut_off_voltage", RelationshipType.CONSTRAINS, "specific_capacity", 0.70),
            ("temperature", RelationshipType.INFLUENCES, "ionic_conductivity", 0.80),
            # Processing → Cell
            ("slurry_coating", RelationshipType.PROCESSES, "cell_assembly", 0.85),
            ("cell_assembly", RelationshipType.FORMS, "sodium_ion_battery", 0.95),
            # Generic
            ("sodium_ion_battery", RelationshipType.HYPONYM, "electrochemical_energy_storage", 1.0),
            ("all_solid_state_sodium_battery", RelationshipType.HYPONYM, "sodium_ion_battery", 0.9),
            # === NEW RELATIONSHIPS v6.2+ ===
            ("mxene", RelationshipType.INFLUENCES, "rate_capability", 0.90),
            ("mxene", RelationshipType.INFLUENCES, "ionic_conductivity", 0.85),
            ("mxene", RelationshipType.INFLUENCES, "cycle_life", 0.70),
            ("organic_cathode", RelationshipType.INFLUENCES, "specific_capacity", 0.80),
            ("organic_cathode", RelationshipType.INFLUENCES, "cycle_life", -0.60),
            ("organic_cathode", RelationshipType.INFLUENCES, "energy_density", 0.75),
            ("sodium_ion_capacitor", RelationshipType.HYPONYM, "sodium_ion_battery", 0.9),
            ("sodium_ion_capacitor", RelationshipType.INFLUENCES, "energy_density", 0.70),
            ("sodium_ion_capacitor", RelationshipType.INFLUENCES, "rate_capability", 0.95),
            ("conversion_anode", RelationshipType.INFLUENCES, "specific_capacity", 0.90),
            ("conversion_anode", RelationshipType.INFLUENCES, "volume_expansion", -0.85),
            ("solid_polymer_electrolyte", RelationshipType.INFLUENCES, "ionic_conductivity", 0.65),
            ("solid_polymer_electrolyte", RelationshipType.INFLUENCES, "dendrite_growth", -0.60),
            ("aqueous_electrolyte", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.80),
            ("aqueous_electrolyte", RelationshipType.INFLUENCES, "thermal_runaway", -0.90),
            ("interface_engineering", RelationshipType.STABILIZES, "sei_formation", 0.85),
            ("interface_engineering", RelationshipType.INFLUENCES, "cycle_life", 0.80),
            ("pre_sodiation", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.90),
            ("pre_sodiation", RelationshipType.PRESERVES, "energy_density", 0.75),
            ("thermal_runaway", RelationshipType.CAUSES, "cycle_life", -0.95),
            ("volume_expansion", RelationshipType.CAUSES, "cycle_life", -0.80),
            ("full_cell", RelationshipType.HYPONYM, "sodium_ion_battery", 0.95),
            ("full_cell", RelationshipType.DEPENDS_ON, "coulombic_efficiency", 0.85),
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
    # --- Root domain ---
    "sodium_ion_battery": (None, 0),
    "all_solid_state_sodium_battery": (None, 0),

    # --- Tier 1: Materials ---
    "layered_oxide_cathode":          ("Cathode Materials", 1),
    "polyanionic_cathode":            ("Cathode Materials", 1),
    "prussian_blue_analogue":         ("Cathode Materials", 1),
    "nasicon_cathode":                ("Cathode Materials", 1),
    "hard_carbon":                    ("Anode Materials", 1),
    "sodium_metal":                   ("Anode Materials", 1),
    "alloying_anode":                 ("Anode Materials", 1),
    "intercalation_anode":            ("Anode Materials", 1),
    "liquid_electrolyte":             ("Electrolytes", 1),
    "solid_electrolyte":              ("Electrolytes", 1),
    "polymer_electrolyte":            ("Electrolytes", 1),
    "quasi_solid_electrolyte":        ("Electrolytes", 1),

    # --- Tier 1: Properties ---
    "specific_capacity":              ("Electrochemical Properties", 1),
    "energy_density":                 ("Electrochemical Properties", 1),
    "coulombic_efficiency":           ("Electrochemical Properties", 1),
    "cycle_life":                     ("Electrochemical Properties", 1),
    "rate_capability":                ("Electrochemical Properties", 1),
    "ionic_conductivity":             ("Electrochemical Properties", 1),
    "voltage_plateau":                ("Electrochemical Properties", 1),

    # --- Tier 1: Phenomena ---
    "dendrite_growth":                ("Phenomena", 1),
    "sei_formation":                  ("Phenomena", 1),
    "sodium_plating_stripping":       ("Phenomena", 1),
    "intercalation":                  ("Phenomena", 1),
    "conversion_reaction":            ("Phenomena", 1),

    # --- Tier 1: Methods ---
    "cyclic_voltammetry":             ("Characterization Methods", 1),
    "electrochemical_impedance_spectroscopy": ("Characterization Methods", 1),
    "galvanostatic_cycling":          ("Characterization Methods", 1),
    "operando_characterization":      ("Characterization Methods", 1),

    # --- Tier 1: Parameters ---
    "current_density":                ("Parameters", 1),
    "cut_off_voltage":                ("Parameters", 1),
    "temperature":                    ("Parameters", 1),

    # --- Tier 1: Processing ---
    "slurry_coating":                 ("Processing", 1),
    "cell_assembly":                  ("Processing", 1),

    # === NEW HIERARCHIES v6.2+ ===
    "mxene":                          ("Anode Materials", 1),
    "organic_cathode":                ("Cathode Materials", 1),
    "sodium_ion_capacitor":           (None, 0),
    "conversion_anode":               ("Anode Materials", 1),
    "solid_polymer_electrolyte":      ("Electrolytes", 1),
    "aqueous_electrolyte":            ("Electrolytes", 1),
    "interface_engineering":          ("Processing", 1),
    "pre_sodiation":                  ("Processing", 1),
    "thermal_runaway":                ("Phenomena", 1),
    "volume_expansion":               ("Phenomena", 1),
    "full_cell":                      (None, 0),
}


def get_hierarchy_label(concept_key: str,
                        style: str = "arrow") -> str:
    """
    Build a human-readable hierarchy label for a concept.

    Parameters
    ----------
    concept_key : str
        The canonical name key used in the ontology (e.g. "specific_capacity").
    style : str
        "arrow"   → "Electrochemical Properties → Specific Capacity"
        "bracket" → "Electrochemical Properties [Specific Capacity]"
        "dot"     → "Electrochemical Properties · Specific Capacity"
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
    ["Sodium-Ion Battery", "Electrochemical Properties", "Specific Capacity"].

    This is directly usable as the `ids` / `labels` / `parents` arrays
    for a Plotly sunburst chart.
    """
    leaf = concept_key.replace("_", " ").title()
    entry = _HIERARCHY_PARENTS.get(concept_key)

    if entry is None or entry[0] is None:
        return ["Sodium-Ion Battery", leaf]

    parent_label = entry[0]
    return ["Sodium-Ion Battery", parent_label, leaf]


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
    root_id = "Sodium-Ion Battery"
    ids.append(root_id)
    labels.append("Sodium-Ion Battery")
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
        electrochemical_indicators = [
            'capacity', 'voltage', 'current', 'cycle', 'efficiency', 'density',
            'conductivity', 'impedance'
        ]
        if any(ind in context_lower for ind in electrochemical_indicators):
            if 'capacity' in text or 'density' in text:
                return "specific_capacity" if 'capacity' in text else "energy_density"
            if 'efficiency' in text:
                return "coulombic_efficiency"
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
# ENHANCED CONCEPT EXTRACTOR (SIB-focused)
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
        # Sodium-ion battery specific patterns
        self.cathode_patterns = [
            r'\blayered\s+oxide\b', r'\bna_mno2\b', r'\bnamno2\b',
            r'\bnvp\b', r'\bna3v2(po4)3\b', r'\bnvpf\b',
            r'\bprussian\s+blue\s+analogue\b', r'\bpba\b',
            r'\bnasicon\b'
        ]
        self.anode_patterns = [
            r'\bhard\s+carbon\b', r'\bhard\s+carbon\s+anode\b',
            r'\bsodium\s+metal\b', r'\bna\s+metal\b',
            r'\balloying\s+anode\b', r'\bsn\s+anode\b', r'\bsb\s+anode\b'
        ]
        self.electrolyte_patterns = [
            r'\bliquid\s+electrolyte\b', r'\bnaclo4\b', r'\bna\s*pf6\b',
            r'\bsolid\s+electrolyte\b', r'\bnasicon\b', r'\bna3ps4\b',
            r'\bpolymer\s+electrolyte\b', r'\bpeo\b',
            r'\bquasi[- ]solid\s+electrolyte\b'
        ]
        self.property_patterns = [
            r'\bspecific\s+capacity\b', r'\bmah/g\b',
            r'\benergy\s+density\b', r'\bwh/kg\b',
            r'\bcoulombic\s+efficiency\b', r'\bce\b',
            r'\bcycle\s+life\b', r'\brate\s+capability\b',
            r'\bionic\s+conductivity\b', r'\bvoltage\s+plateau\b'
        ]
        self.phenomena_patterns = [
            r'\bdendrite\s+growth\b', r'\bsodium\s+dendrite\b',
            r'\bsei\s+formation\b', r'\bsolid\s+electrolyte\s+interphase\b',
            r'\bplating/stripping\b', r'\bsodium\s+plating\b',
            r'\bintercalation\b', r'\bconversion\s+reaction\b'
        ]
        self.method_patterns = [
            r'\bcyclic\s+voltammetry\b', r'\bcv\b',
            r'\belectrochemical\s+impedance\s+spectroscopy\b', r'\beis\b',
            r'\bgalvanostatic\s+cycling\b', r'\bcccv\b',
            r'\boperando\b', r'\bin\s+ situ\s+ xrd\b'
        ]
        self.param_patterns = [
            r'\bcurrent\s+density\b', r'\bma/g\b', r'\ba/g\b',
            r'\bcut[ -]off\s+voltage\b', r'\bvoltage\s+window\b',
            r'\btemperature\b', r'\bcelsius\b', r'\bkelvin\b'
        ]
        self.processing_patterns = [
            r'\bslurry\s+coating\b', r'\bdoctor\s+blade\b',
            r'\bcoin\s+cell\b', r'\bpouch\s+cell\b', r'\bcell\s+assembly\b'
        ]
        # === NEW PATTERNS v6.2+ ===
        self.mxene_patterns = [
            r'\bmxene[s]?\b', r'\bti3c2tx\b', r'\bv2ctz\b',
            r'\b2d\s+transition\s+metal\s+carbide\b'
        ]
        self.organic_cathode_patterns = [
            r'\borganic\s+cathode\b', r'\borganic\s+electrode\b',
            r'\bpdtca\b', r'\bppta\b', r'\bconjugated\s+carbonyl\b'
        ]
        self.capacitor_patterns = [
            r'\bsodium[-\s]?ion\s+capacitor\b', r'\bsupercapacitor\b',
            r'\bhybrid\s+sodium\s+capacitor\b', r'\bsihc\b'
        ]
        self.conversion_anode_patterns = [
            r'\bconversion\s+anode\b', r'\bfes2\b', r'\bcos2\b', r'\bmos2\b'
        ]
        self.aqueous_patterns = [
            r'\baqueous\s+electrolyte\b', r'\bwater\s+based\s+electrolyte\b',
            r'\bna2so4\s+electrolyte\b'
        ]
        self.interface_patterns = [
            r'\bsurface\s+coating\b', r'\bartificial\s+sei\b',
            r'\bal2o3\s+coating\b', r'\bcarbon\s+coating\b'
        ]
        self.pre_sodiation_patterns = [
            r'\bpre[-\s]?sodiation\b', r'\bsodium\s+compensation\b',
            r'\bsacrificial\s+salt\b'
        ]
        self.thermal_patterns = [
            r'\bthermal\s+runaway\b', r'\bthermal\s+abuse\b',
            r'\bbattery\s+fire\b'
        ]
        self.volume_patterns = [
            r'\bvolume\s+expansion\b', r'\bpulverization\b',
            r'\bstructural\s+change\b'
        ]
        self.full_cell_patterns = [
            r'\bfull\s+cell\b', r'\banode[-\s]?free\s+cell\b',
            r'\bpractical\s+cell\b'
        ]

        self.all_patterns = (
            self.cathode_patterns + self.anode_patterns +
            self.electrolyte_patterns + self.property_patterns +
            self.phenomena_patterns + self.method_patterns +
            self.param_patterns + self.processing_patterns +
            self.mxene_patterns + self.organic_cathode_patterns +
            self.capacitor_patterns + self.conversion_anode_patterns +
            self.aqueous_patterns + self.interface_patterns +
            self.pre_sodiation_patterns + self.thermal_patterns +
            self.volume_patterns + self.full_cell_patterns
        )
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.all_patterns
        ]
        self.compiled_param_patterns = []  # kept for compatibility
        self.compiled_cause_patterns = [
            re.compile(r'\b(increase|decrease|enhance|reduce)\w*\s+(?:in|of)\s+([\w\s-]+?)\s+(?:lead[s]?|result[s]?|cause[s]?)\s+(?:to|in)?\s+([\w\s-]+?)\b', re.I),
        ]

    @timed
    def extract_from_text(self, text: str, doc_id: int = 0, allowed_concepts: Optional[Set[str]] = None) -> List[str]:
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
                        if allowed_concepts is not None and canonical not in allowed_concepts:
                            continue
                        concepts.add(canonical)
                    else:
                        if allowed_concepts is not None:
                            continue   # skip unresolved concepts in focused mode
                        concepts.add(concept)

        # 2. Localized Context Window Extraction (Prevents Memory issue)
        context_concepts = self._extract_from_context_windows(text)
        if allowed_concepts is not None:
            context_concepts = {c for c in context_concepts if c in allowed_concepts}
        concepts.update(context_concepts)

        # 3. Batch resolve remaining raw concepts (limit to 50 to prevent OOM)
        raw_concepts = set()
        for c in concepts:
            if c not in self.ontology.concepts and not self.resolver.resolve(c):
                raw_concepts.add(c)
        if raw_concepts:
            raw_list = list(raw_concepts)[:50]
            resolved_map = self.resolver.resolve_batch(raw_list, context="")
            for raw, canonical in resolved_map.items():
                if canonical:
                    if allowed_concepts is not None and canonical not in allowed_concepts:
                        continue
                    concepts.add(canonical)
                else:
                    if allowed_concepts is not None:
                        continue
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
        material_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.MATERIAL
        ]
        property_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.PROPERTY
        ]
        for mat in material_nodes:
            for prop in property_nodes:
                if not nx_graph.has_edge(mat, prop):
                    paths = self.ontology.infer_path(mat, prop, max_depth=2)
                    if paths:
                        avg_confidence = 0.6
                        nx_graph.add_edge(
                            mat, prop,
                            weight=avg_confidence,
                            cooccurrence=0,
                            semantic=avg_confidence,
                            edge_type='bridge',
                            inferred=True,
                            path=" -> ".join(paths[0]),
                        )
                        self.inferred_edges.add((mat, prop))
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


def build_query_whitelist(st_session):
    if not st_session.get('query_focused_build', False):
        return None
    analysis = st_session.get('last_query_analysis')
    if analysis is None:
        st.warning("No query analysis available – falling back to full graph.")
        return None
    whitelist = set(analysis.explicitly_mentioned)
    whitelist.update(analysis.inferred_concepts)
    whitelist.update(st_session.get('last_query_dynamic_concepts', set()))
    whitelist.update(st_session.get('last_query_bridge_concepts', {}).keys())
    return whitelist


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
# SODIUM-ION BATTERY KEYWORDS
# ============================================================================
CATHODE_KEYWORDS = [
    "layered oxide", "na_mno2", "namno2", "p2 na_mno2", "o3 na_mno2",
    "polyanionic", "na3v2(po4)3", "nvp", "nvpf", "prussian blue", "pba",
    "nasicon", "na3zr2si2po12",
    "organic cathode", "organic electrode", "pdtca", "ppta", "conjugated carbonyl"
]
ANODE_KEYWORDS = [
    "hard carbon", "hard carbon anode", "sodium metal", "na metal",
    "alloying anode", "sn anode", "sb anode", "intercalation anode", "tio2",
    "mxene", "mxenes", "ti3c2tx", "v2ctz", "conversion anode", "fes2", "cos2", "mos2"
]
ELECTROLYTE_KEYWORDS = [
    "liquid electrolyte", "naclo4", "na pf6", "solid electrolyte",
    "na3ps4", "polymer electrolyte", "peo", "quasi-solid electrolyte",
    "aqueous electrolyte", "water based electrolyte", "na2so4 electrolyte",
    "solid polymer electrolyte", "spe", "pan based electrolyte"
]
PROPERTY_KEYWORDS = [
    "specific capacity", "mah/g", "energy density", "wh/kg",
    "coulombic efficiency", "ce", "cycle life", "rate capability",
    "ionic conductivity", "voltage plateau",
    "thermal stability", "safety"
]
PHENOMENA_KEYWORDS = [
    "dendrite growth", "sodium dendrite", "sei formation",
    "solid electrolyte interphase", "plating/stripping",
    "sodium plating", "intercalation", "conversion reaction",
    "thermal runaway", "thermal abuse", "battery fire",
    "volume expansion", "pulverization", "structural change"
]
METHOD_KEYWORDS = [
    "cyclic voltammetry", "cv", "electrochemical impedance spectroscopy",
    "eis", "galvanostatic cycling", "cccv", "operando", "in situ xrd"
]
PARAM_KEYWORDS = [
    "current density", "ma/g", "a/g", "cut-off voltage",
    "voltage window", "temperature"
]
PROCESSING_KEYWORDS = [
    "slurry coating", "doctor blade", "coin cell", "pouch cell",
    "cell assembly",
    "interface engineering", "surface coating", "artificial sei", "al2o3 coating", "carbon coating",
    "pre-sodiation", "sodium compensation", "sacrificial salt"
]
ALL_DOMAIN_KEYWORDS = (
    CATHODE_KEYWORDS + ANODE_KEYWORDS + ELECTROLYTE_KEYWORDS +
    PROPERTY_KEYWORDS + PHENOMENA_KEYWORDS + METHOD_KEYWORDS +
    PARAM_KEYWORDS + PROCESSING_KEYWORDS
)
SIB_PATTERNS = [
    r'\blayered\s+oxide\b', r'\bna_mno2\b', r'\bnamno2\b',
    r'\bnvp\b', r'\bna3v2(po4)3\b', r'\bnvpf\b',
    r'\bprussian\s+blue\s+analogue\b', r'\bpba\b',
    r'\bnasicon\b',
    r'\bhard\s+carbon\b', r'\bsodium\s+metal\b',
    r'\balloying\s+anode\b', r'\bsn\s+anode\b', r'\bsb\s+anode\b',
    r'\bliquid\s+electrolyte\b', r'\bnaclo4\b', r'\bna\s*pf6\b',
    r'\bsolid\s+electrolyte\b', r'\bna3ps4\b',
    r'\bpolymer\s+electrolyte\b', r'\bpeo\b',
    r'\bquasi[- ]solid\s+electrolyte\b',
    r'\bspecific\s+capacity\b', r'\bmah/g\b',
    r'\benergy\s+density\b', r'\bwh/kg\b',
    r'\bcoulombic\s+efficiency\b', r'\bce\b',
    r'\bcycle\s+life\b', r'\brate\s+capability\b',
    r'\bionic\s+conductivity\b', r'\bvoltage\s+plateau\b',
    r'\bdendrite\s+growth\b', r'\bsodium\s+dendrite\b',
    r'\bsei\s+formation\b', r'\bsolid\s+electrolyte\s+interphase\b',
    r'\bplating/stripping\b', r'\bsodium\s+plating\b',
    r'\bintercalation\b', r'\bconversion\s+reaction\b',
    r'\bcyclic\s+voltammetry\b', r'\bcv\b',
    r'\belectrochemical\s+impedance\s+spectroscopy\b', r'\beis\b',
    r'\bgalvanostatic\s+cycling\b', r'\bcccv\b',
    r'\boperando\b', r'\bin\s+ situ\s+ xrd\b',
    r'\bcurrent\s+density\b', r'\bma/g\b', r'\ba/g\b',
    r'\bcut[ -]off\s+voltage\b', r'\bvoltage\s+window\b',
    r'\btemperature\b',
    r'\bslurry\s+coating\b', r'\bdoctor\s+blade\b',
    r'\bcoin\s+cell\b', r'\bpouch\s+cell\b', r'\bcell\s+assembly\b',
    # === NEW PATTERNS v6.2+ ===
    r'\bmxene[s]?\b', r'\bti3c2tx\b', r'\bv2ctz\b',
    r'\borganic\s+cathode\b', r'\borganic\s+electrode\b', r'\bpdtca\b', r'\bppta\b',
    r'\bsodium[-\s]?ion\s+capacitor\b', r'\bsupercapacitor\b', r'\bhybrid\s+sodium\s+capacitor\b',
    r'\bconversion\s+anode\b', r'\bfes2\b', r'\bcos2\b', r'\bmos2\b',
    r'\baqueous\s+electrolyte\b', r'\bwater\s+based\s+electrolyte\b',
    r'\bsolid\s+polymer\s+electrolyte\b', r'\bspe\b',
    r'\bsurface\s+coating\b', r'\bartificial\s+sei\b', r'\bal2o3\s+coating\b',
    r'\bpre[-\s]?sodiation\b', r'\bsodium\s+compensation\b',
    r'\bthermal\s+runaway\b', r'\bthermal\s+abuse\b',
    r'\bvolume\s+expansion\b', r'\bpulverization\b',
    r'\bfull\s+cell\b', r'\banode[-\s]?free\s+cell\b'
]
SIB_DESCRIPTOR_MAPPING = {
    r'layered oxide|na_mno2|namno2|polyanionic|na3v2(po4)3|nvp|nvpf|prussian blue|pba|nasicon|organic cathode|organic electrode|pdtca|ppta|conjugated carbonyl': 'cathode_material',
    r'hard carbon|sodium metal|alloying anode|sn anode|sb anode|intercalation anode|tio2|mxene|mxenes|ti3c2tx|v2ctz|conversion anode|fes2|cos2|mos2': 'anode_material',
    r'liquid electrolyte|naclo4|na pf6|solid electrolyte|na3ps4|polymer electrolyte|peo|quasi-solid|aqueous electrolyte|water based|solid polymer electrolyte|spe|pan based': 'electrolyte',
    r'specific capacity|mah/g|energy density|wh/kg|coulombic efficiency|ce|cycle life|rate capability|ionic conductivity|voltage plateau|thermal stability|safety': 'electrochemical_property',
    r'dendrite growth|sodium dendrite|sei formation|solid electrolyte interphase|plating/stripping|sodium plating|intercalation|conversion reaction|thermal runaway|thermal abuse|battery fire|volume expansion|pulverization|structural change': 'phenomenon',
    r'cyclic voltammetry|cv|electrochemical impedance spectroscopy|eis|galvanostatic cycling|cccv|operando|in situ xrd': 'method',
    r'current density|ma/g|a/g|cut-off voltage|voltage window|temperature': 'parameter',
    r'slurry coating|doctor blade|coin cell|pouch cell|cell assembly|interface engineering|surface coating|artificial sei|al2o3 coating|carbon coating|pre-sodiation|sodium compensation|sacrificial salt': 'processing',
    r'sodium ion capacitor|supercapacitor|hybrid sodium capacitor|sihc': 'general',
    r'full cell|anode free cell|practical cell': 'general'
}


def is_valid_sib_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw.lower() in concept_lower for kw in ALL_DOMAIN_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in SIB_PATTERNS)
    generic = {
        'study', 'analysis', 'effect', 'role', 'investigation', 'research',
        'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
        'new', 'recent', 'various', 'different', 'significant', 'important',
        'report', 'demonstrate', 'show', 'result', 'data', 'find', 'present',
        'propose', 'develop', 'investigate', 'discuss', 'conclude', 'battery',
        'cell', 'electrode', 'material', 'system', 'sample', 'specimen'
    }
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic


def normalize_sib_concept(concept: str) -> str:
    concept = concept.lower().strip()
    # Replace synonyms with canonical forms
    concept = re.sub(r'\bna_mno2\b|\bnamno2\b', 'layered_oxide_cathode', concept)
    concept = re.sub(r'\bna3v2(po4)3\b|\bnvp\b|\bnvpf\b', 'polyanionic_cathode', concept)
    concept = re.sub(r'\bprussian\s+blue\s+analogue\b|\bpba\b', 'prussian_blue_analogue', concept)
    concept = re.sub(r'\bnasicon\b', 'nasicon_cathode', concept)
    concept = re.sub(r'\bhard\s+carbon\b', 'hard_carbon', concept)
    concept = re.sub(r'\bsodium\s+metal\b', 'sodium_metal', concept)
    concept = re.sub(r'\balloying\s+anode\b|\bsn\s+anode\b|\bsb\s+anode\b', 'alloying_anode', concept)
    concept = re.sub(r'\bliquid\s+electrolyte\b', 'liquid_electrolyte', concept)
    concept = re.sub(r'\bsolid\s+electrolyte\b', 'solid_electrolyte', concept)
    concept = re.sub(r'\bpolymer\s+electrolyte\b|\bpeo\b', 'polymer_electrolyte', concept)
    concept = re.sub(r'\bquasi[- ]solid\s+electrolyte\b', 'quasi_solid_electrolyte', concept)
    concept = re.sub(r'\bspecific\s+capacity\b', 'specific_capacity', concept)
    concept = re.sub(r'\benergy\s+density\b', 'energy_density', concept)
    concept = re.sub(r'\bcoulombic\s+efficiency\b|\bce\b', 'coulombic_efficiency', concept)
    concept = re.sub(r'\bcycle\s+life\b', 'cycle_life', concept)
    concept = re.sub(r'\brate\s+capability\b', 'rate_capability', concept)
    concept = re.sub(r'\bionic\s+conductivity\b', 'ionic_conductivity', concept)
    concept = re.sub(r'\bvoltage\s+plateau\b', 'voltage_plateau', concept)
    concept = re.sub(r'\bdendrite\s+growth\b', 'dendrite_growth', concept)
    concept = re.sub(r'\bsei\s+formation\b|\bsolid\s+electrolyte\s+interphase\b', 'sei_formation', concept)
    concept = re.sub(r'\bsodium\s+plating\b|\bplating/stripping\b', 'sodium_plating_stripping', concept)
    concept = re.sub(r'\bintercalation\b', 'intercalation', concept)
    concept = re.sub(r'\bconversion\s+reaction\b', 'conversion_reaction', concept)
    concept = re.sub(r'\bcyclic\s+voltammetry\b|\bcv\b', 'cyclic_voltammetry', concept)
    concept = re.sub(r'\belectrochemical\s+impedance\s+spectroscopy\b|\beis\b', 'electrochemical_impedance_spectroscopy', concept)
    concept = re.sub(r'\bgalvanostatic\s+cycling\b|\bcccv\b', 'galvanostatic_cycling', concept)
    concept = re.sub(r'\boperando\b|\bin\s+situ\s+xrd\b', 'operando_characterization', concept)
    concept = re.sub(r'\bcurrent\s+density\b', 'current_density', concept)
    concept = re.sub(r'\bcut[ -]off\s+voltage\b|\bvoltage\s+window\b', 'cut_off_voltage', concept)
    concept = re.sub(r'\btemperature\b', 'temperature', concept)
    concept = re.sub(r'\bslurry\s+coating\b', 'slurry_coating', concept)
    concept = re.sub(r'\bcoin\s+cell\b|\bpouch\s+cell\b|\bcell\s+assembly\b', 'cell_assembly', concept)
    # === NEW NORMALIZATIONS v6.2+ ===
    concept = re.sub(r'\bmxene[s]?\b|\bti3c2tx\b|\bv2ctz\b', 'mxene', concept)
    concept = re.sub(r'\borganic\s+cathode\b|\borganic\s+electrode\b|\bconjugated\s+carbonyl\b', 'organic_cathode', concept)
    concept = re.sub(r'\bsodium[-\s]?ion\s+capacitor\b|\bsupercapacitor\b|\bhybrid\s+sodium\s+capacitor\b', 'sodium_ion_capacitor', concept)
    concept = re.sub(r'\bconversion\s+anode\b|\bfes2\b|\bcos2\b|\bmos2\b', 'conversion_anode', concept)
    concept = re.sub(r'\bsolid\s+polymer\s+electrolyte\b|\bspe\b|\bpan\s+based\s+electrolyte\b', 'solid_polymer_electrolyte', concept)
    concept = re.sub(r'\baqueous\s+electrolyte\b|\bwater\s+based\s+electrolyte\b', 'aqueous_electrolyte', concept)
    concept = re.sub(r'\bsurface\s+coating\b|\bartificial\s+sei\b|\bal2o3\s+coating\b|\bcarbon\s+coating\b', 'interface_engineering', concept)
    concept = re.sub(r'\bpre[-\s]?sodiation\b|\bsodium\s+compensation\b|\bsacrificial\s+salt\b', 'pre_sodiation', concept)
    concept = re.sub(r'\bthermal\s+runaway\b|\bthermal\s+abuse\b|\bbattery\s+fire\b', 'thermal_runaway', concept)
    concept = re.sub(r'\bvolume\s+expansion\b|\bpulverization\b|\bstructural\s+change\b', 'volume_expansion', concept)
    concept = re.sub(r'\bfull\s+cell\b|\banode[-\s]?free\s+cell\b|\bpractical\s+cell\b', 'full_cell', concept)
    return concept


def extract_concepts_from_text(text: str) -> List[str]:
    concepts: Set[str] = set()
    text_lower = text.lower()
    for pattern in SIB_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    noun_pattern = (
        r'\b(?:[a-z]+(?:[-\s]?[a-z]+){0,2}[-\s]?)?'
        r'(?:capacity|density|efficiency|conductivity|voltage|plateau|dendrite|sei|intercalation|conversion|cycling|impedance|coating|cell)\b'
    )
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_sib_concept(concept):
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
                if is_valid_sib_concept(concept):
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
        # Extract numeric metrics typical for battery literature
        current_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ma/g|a/g|ma\s*g-1)', combined_text, re.I)
        if current_matches:
            metrics['current_density_ma_g'] = [float(m) for m in current_matches]
        cap_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mah/g|mah\s*g-1)', combined_text, re.I)
        if cap_matches:
            metrics['capacity_mah_g'] = [float(m) for m in cap_matches]
        density_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:wh/kg|wh\s*kg-1)', combined_text, re.I)
        if density_matches:
            metrics['energy_density_wh_kg'] = [float(m) for m in density_matches]
        voltage_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:v)', combined_text, re.I)
        if voltage_matches:
            metrics['voltage_v'] = [float(m) for m in voltage_matches]
        temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', combined_text, re.I)
        if temp_matches:
            metrics['temperature'] = [float(m) for m in temp_matches]
        all_metrics.append(metrics)
        concepts = extract_concepts_from_text(combined_text)
        normalized = [normalize_sib_concept(c) for c in concepts]
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
            if c not in seen_in_doc and is_valid_sib_concept(c):
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
        for pattern, category in SIB_DESCRIPTOR_MAPPING.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            if any(re.search(p, concept, re.I) for p in [r'cathode', r'oxide', r'phosphate', r'prussian']):
                concept_to_abstract[concept] = 'cathode_material'
            elif any(re.search(p, concept, re.I) for p in [r'anode', r'carbon', r'metal', r'alloying']):
                concept_to_abstract[concept] = 'anode_material'
            elif any(re.search(p, concept, re.I) for p in [r'electrolyte', r'nasicon', r'polymer']):
                concept_to_abstract[concept] = 'electrolyte'
            elif any(re.search(p, concept, re.I) for p in [r'capacity', r'density', r'efficiency', r'conductivity', r'voltage']):
                concept_to_abstract[concept] = 'electrochemical_property'
            elif any(re.search(p, concept, re.I) for p in [r'dendrite', r'sei', r'intercalation', r'conversion']):
                concept_to_abstract[concept] = 'phenomenon'
            elif any(re.search(p, concept, re.I) for p in [r'cv', r'eis', r'galvanostatic', r'operando']):
                concept_to_abstract[concept] = 'method'
            elif any(re.search(p, concept, re.I) for p in [r'current', r'voltage', r'temperature']):
                concept_to_abstract[concept] = 'parameter'
            elif any(re.search(p, concept, re.I) for p in [r'coating', r'cell', r'assembly']):
                concept_to_abstract[concept] = 'processing'
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
        return pd.DataFrame(columns=["concept", "burst_score", "burst_year", "total_mentions", "year_range"])
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 3:
        return pd.DataFrame(columns=["concept", "burst_score", "burst_year", "total_mentions", "year_range"])
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
    if not burst_data:

        return pd.DataFrame(columns=["concept", "burst_score", "burst_year", "total_mentions", "year_range"])

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
        return pd.DataFrame(columns=["concept", "semantic_drift", "early_papers", "late_papers", "early_period", "late_period"])
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 4:
        return pd.DataFrame(columns=["concept", "semantic_drift", "early_papers", "late_papers", "early_period", "late_period"])
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
    if not drift_data:

        return pd.DataFrame(columns=["concept", "semantic_drift", "early_papers", "late_papers", "early_period", "late_period"])

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
    if not genealogy_data:

        return pd.DataFrame(columns=["concept", "pagerank", "betweenness", "frequency", "degree", "generation"])

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
    if not bridge_data:

        return pd.DataFrame(columns=["concept", "bridge_score", "betweenness", "connected_categories", "categories", "degree", "own_category"])

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
    filename="sib_graph_pub.png",
) -> bytes:
    try:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5, iterations=200)
        plt.figure(figsize=figsize, dpi=dpi)
        node_colors = [get_sib_category_color(n) for n in nx_graph.nodes()]
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
            "Sodium-Ion Battery Quantitative Descriptor Graph",
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
    report.append("# Sodium-Ion Battery Quantitative Descriptor Graph Analysis Report")
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
    report.append("*Report generated by Sodium-Ion Battery Quantitative Descriptor Graph v6.2 + Microtransformer #2*")
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
def get_sib_category_color(
    concept: str, cmap_colors: Optional[List[str]] = None
) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    category = 'general'
    for pattern, cat in SIB_DESCRIPTOR_MAPPING.items():
        if re.search(pattern, concept_lower):
            category = cat
            break
    color_map = {
        'cathode_material': '#1f77b4',
        'anode_material': '#ff7f0e',
        'electrolyte': '#2ca02c',
        'electrochemical_property': '#d62728',
        'phenomenon': '#9467bd',
        'method': '#8c564b',
        'parameter': '#e377c2',
        'processing': '#7f7f7f',
        'general': '#bcbd22'
    }
    return color_map.get(category, '#bcbd22')

# Alias for compatibility
get_mpea_category_color = get_sib_category_color




# ============================================================================
# PYVIS RENDERER — colored edges + hierarchy labels
# ============================================================================

# Concept-type → node-color map for Pyvis
_NODE_TYPE_COLORS = {
    ConceptType.MATERIAL:       "#E74C3C",   # red
    ConceptType.PROCESS:        "#3498DB",   # blue
    ConceptType.PROPERTY:       "#2ECC71",   # green
    ConceptType.PHENOMENON:     "#F39C12",   # orange
    ConceptType.METHOD:         "#9B59B6",   # purple
    ConceptType.PARAMETER:      "#1ABC9C",   # teal
    ConceptType.MICROSTRUCTURE: "#E67E22",   # dark orange
    ConceptType.MODEL:          "#2980B9",   # dark blue
    ConceptType.GENERAL:        "#95A5A6",   # grey
}


def render_pyvis_graph(
    nx_graph, concept_abstract_map, physics_enabled=True,
    cmap_name="viridis", top_n_nodes=0, theme=None, physics_preset=None,
    show_edge_weights=False, edge_label_mode="hover",
    node_label_size=12, node_label_position="center",
    node_font_face="Inter, Segoe UI, Roboto, sans-serif",
    edge_label_size=10, edge_label_color=None,
    edge_label_position="middle",
    use_abbreviated_labels=False, max_label_length=15,
    enable_node_highlight=True, show_definitions=True, ontology=None,
    edge_lightness=0.6, edge_color_mode="theme",
    custom_edge_color="#AAAAAA", tooltip_font_size=13,
    node_legend_font_size=13
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))
    
    net = Network(height="780px", width="100%", bgcolor=theme['bg'], font_color=theme['font'],
                  select_menu=True, notebook=False, cdn_resources='remote')

    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
        var options = {{
            "physics": {{
                "enabled": true, "solver": "barnesHut",
                "barnesHut": {{
                    "gravitationalConstant": {physics_preset['gravity']},
                    "centralGravity": {physics_preset['central_gravity']},
                    "springLength": {physics_preset['spring_length']},
                    "springConstant": {physics_preset['spring_strength']},
                    "damping": {physics_preset['damping']}, "overlap": 0.15
                }},
                "stabilization": {{ "enabled": true, "iterations": 500, "updateInterval": 50, "onlyDynamicEdges": true, "fit": true }}
            }},
            "interaction": {{ "hover": true, "tooltipDelay": 180, "hideEdgesOnDrag": false, "zoomView": true, "dragView": true }}
        }}
        """)
    else:
        net.set_options("""var options = { "physics": { "enabled": false }, "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true } }""")

    label_map = {}
    n_counter = 1
    used_rel_types = {}

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(8 + freq * 1.2, 8, 40))
        color = get_sib_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        
        original_label = node
        label = get_hierarchy_label(node, style="arrow") if node in _HIERARCHY_PARENTS else node
        
        if use_abbreviated_labels and len(original_label) > max_label_length:
            short_label = f"N{n_counter}"
            label_map[short_label] = original_label
            n_counter += 1
            label = short_label

        node_shape = 'circle'
        inside_font_size = max(8, min(int(node_label_size), 14))
        font_dict = {'color': '#ffffff', 'size': inside_font_size, 'face': node_font_face, 'bold': True}
        
        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        definition = nx_graph.nodes[node].get('definition', '')
        
        _def_display = ""
        if show_definitions and definition:
            _def_display = definition[:180] + "..." if len(definition) > 180 else definition
            
        _full_label_display = ""
        if use_abbreviated_labels and label != original_label:
            _full_label_display = original_label

        # FIX 1: Ensure the tooltip ALWAYS starts with the exact canonical node name on its own line
        # This prevents vis.js from merging it with other text and causing parsing issues
        tooltip_content = (
            f"{node}\n"
            f"Type: {concept_type}\n"
            f"Degree: {degree}\n"
            f"Frequency: {freq}"
            + (f"\nDefinition: {_def_display}" if _def_display else "")
            + (f"\nFull Label: {_full_label_display}" if _full_label_display else "")
        )

        net.add_node(node, label=label, size=size,
                     color={'background': color, 'border': theme['node_border'],
                            'highlight': {'background': theme['highlight_bg'], 'border': '#ffffff'},
                            'hover': {'background': theme['hover_bg'], 'border': '#ffffff'}},
                     font=font_dict, title=tooltip_content, borderWidth=2, borderWidthSelected=3,
                     shadow={'enabled': True, 'color': theme['shadow_color'], 'size': 12, 'x': 4, 'y': 4},
                     shape=node_shape, mass=max(1, 1 + freq * 0.05))

    all_weights = [nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()]
    weight_threshold = float(np.percentile(all_weights, 80)) if all_weights else 0.0

    for u, v in nx_graph.edges():
        w = float(nx_graph[u][v].get('weight', 1))
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)
        rel_type = RelationshipType.SEMANTIC
        if edge_type != 'unknown':
            try: rel_type = RelationshipType(edge_type)
            except ValueError: pass

        if edge_color_mode == "theme":
            base_color = theme['edge_unknown'] if edge_type == 'unknown' else get_edge_color(rel_type)
            if edge_lightness > 0:
                base_color = lighten_hex_color(base_color, edge_lightness)
        elif edge_color_mode == "uniform_grey":
            base_color = lighten_hex_color("#808080", edge_lightness)
        else:
            base_color = lighten_hex_color(custom_edge_color, edge_lightness)

        width = float(get_edge_width(rel_type) * (0.5 + 0.5 * w))
        style = get_edge_style(rel_type)
        dashes = True if style == "dashed" or is_inferred else False

        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)), width=width,
            color={'color': base_color, 'highlight': theme['highlight_bg'], 'hover': theme['hover_bg'], 'opacity': 0.85},
            smooth={"type": "dynamic"},
            title=f"Weight: {w:.2f}\nType: {edge_type}\nInferred: {is_inferred}",
            dashes=dashes
        )
        if edge_label_mode == "all" or (edge_label_mode == "threshold" and w >= weight_threshold):
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {'color': edge_label_color or theme['font'], 'size': int(edge_label_size),
                                   'background': theme['tooltip_bg'], 'strokeWidth': 2, 'strokeColor': theme['node_border'],
                                   'align': edge_label_position, 'face': node_font_face}
        net.add_edge(u, v, **edge_kwargs)
        if rel_type not in used_rel_types:
            used_rel_types[rel_type] = rel_type.value.replace("_", " ").title()

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
            w_leg = get_edge_width(rt)
            s_leg = get_edge_style(rt)
            border = 'border: 1px dashed #888;' if s_leg == "dashed" else 'border: 1px solid transparent;'
            legend_rows.append(f'<tr><td style="padding:2px 6px;"><span style="display:inline-block;width:{int(20*w_leg)}px;height:3px;background:{c};vertical-align:middle;{border}"></span></td><td style="padding:2px 6px;color:#ccc;font-size:11px;">{human}</td></tr>')
        legend_html = f'<div style="background:#0d0d1a;border-radius:8px;padding:12px 16px;margin-top:8px;max-height:280px;overflow-y:auto;"><div style="color:#fff;font-size:13px;font-weight:bold;margin-bottom:6px;">Edge Colors ({len(used_rel_types)} types)</div><table style="border-collapse:collapse;">{"".join(legend_rows)}</table></div>'
        net.add_node("__legend__", label="", shape="dot", size=0, color="rgba(0,0,0,0)", fixed=True, x=-500, y=-500, physics=False, title=legend_html)

    try:
        tmp_html = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
        tmp_path = tmp_html.name
        net.write_html(tmp_path, notebook=False)
        tmp_html.close()
        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        if use_abbreviated_labels and label_map:
            label_map_json = json.dumps(label_map)
            html_content = html_content.replace('</body>', f'<div id="hea-label-map-data" style="display:none;">{label_map_json}</div></body>')
        os.unlink(tmp_path)
    except Exception as e:
        st.error(f"PyVis HTML generation failed: {e}")
        html_content = net.generate_html()

    # FIX 2: Enhanced CSS to prevent any text truncation in tooltips or panels
    custom_css = f"""
    <style>
    body {{ background: {theme['bg']}; margin: 0; padding: 0; font-family: '{node_font_face}', sans-serif; }}
    #mynetwork {{ border-radius: 16px; box-shadow: 0 12px 48px {theme['shadow_color']}; outline: none; }}
    
    div.vis-tooltip {{
        max-width: 540px !important;
        width: auto !important;
        max-height: 280px !important;
        height: auto !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        z-index: 10000 !important;
        white-space: pre-wrap !important; /* Preserves newlines from our tooltip format */
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        line-height: 1.45 !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
    }}
    div.vis-tooltip > div {{
        max-width: 520px !important;
        width: auto !important;
        max-height: 260px !important;
        overflow: auto !important;
        white-space: pre-wrap !important;
    }}
    .hea-legend {{ font-size: {node_legend_font_size}px !important; }}
    
    /* FIX 3: Explicitly prevent truncation in the edge info panel header */
    #edge-info-panel > div:first-child > div:first-child {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
    }}
    </style>
    """

    if '</head>' in html_content:
        html_content = html_content.replace('</head>', custom_css + '</head>')
    elif '<head>' in html_content:
        html_content = re.sub(r'</head\s*>', custom_css + r'\g<0>', html_content, flags=re.I)
    else:
        if '<body>' in html_content:
            html_content = html_content.replace('<body>', '<body>' + custom_css)
        else:
            html_content = custom_css + html_content

    if 'div.vis-tooltip' not in html_content:
        st.warning("Tooltip CSS injection failed — tooltips may render with default (clipped) styling.")

    if enable_node_highlight:
        # FIX 4: Robust JS that uses nodeId directly instead of parsing truncated tooltips
        highlight_js = r"""
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
                        if (hidden && hidden.textContent) { try { labelMap = JSON.parse(hidden.textContent); } catch(e) {} }
                    })();

                    function resetAll() {
                        var nodeRestores = [];
                        for (var nid in savedNodeColors) { nodeRestores.push({id: nid, color: savedNodeColors[nid]}); }
                        if (nodeRestores.length > 0) nodesDS.update(nodeRestores);
                        savedNodeColors = {}; activeNodeId = null;
                        var panel = document.getElementById('edge-info-panel'); if (panel) panel.style.display = 'none';
                    }

                    function resolveFullName(shortOrId) {
                        if (labelMap && labelMap[shortOrId]) return labelMap[shortOrId];
                        return shortOrId;
                    }

                    function formatEdgeRow(e, idx, mode) {
                        var typeColor = e.inferred ? '#8b5cf6' : '#0ea5e9';
                        var badge = e.inferred ? ' <span style="background:#8b5cf6;color:white;padding:1px 4px;border-radius:3px;font-size:9px;">INFERRED</span>' : '';
                        var typeBadge = '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:600;">' + e.type + '</span>';
                        var fromName = (mode === 'short') ? e.from : resolveFullName(e.from);
                        var toName = (mode === 'short') ? e.to : resolveFullName(e.to);
                        return '<div style="padding:8px 10px;margin:4px 0;background:rgba(248,250,252,0.9);border-left:4px solid ' + typeColor + ';border-radius:6px;font-size:12px;">' +
                            '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;flex-wrap:wrap;">' +
                            '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;word-break:break-word;">' + fromName + '</span>' +
                            '<span style="color:#94a3b8;font-size:13px;">↔</span>' +
                            '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;word-break:break-word;">' + toName + '</span></div>' +
                            '<div style="display:flex;align-items:center;gap:8px;padding-left:10px;">' +
                            '<span style="background:#0ea5e9;color:white;font-size:10px;padding:2px 6px;border-radius:4px;font-weight:700;">W: ' + e.weight + '</span>' +
                            typeBadge + badge + '</div></div>';
                    }

                    function showEdgeInfoPanel(nodeId, connectedEdges) {
                        var panel = document.getElementById('edge-info-panel');
                        if (!panel) { panel = document.createElement('div'); panel.id = 'edge-info-panel'; document.body.appendChild(panel); }
                        panel.style.cssText = 'position:fixed;top:90px;right:20px;width:400px;max-height:calc(100vh - 110px);overflow-y:auto;z-index:9990;' +
                            'background:rgba(255,255,255,0.95);border:1px solid rgba(255,215,0,0.6);border-radius:16px;padding:0;' +
                            'font-family:Inter,Segoe UI,Roboto,sans-serif;box-shadow:0 20px 60px rgba(0,0,0,0.15);backdrop-filter:blur(20px);';

                        var nodeData = nodesDS.get(nodeId);
                        
                        // FIX 5: Use nodeId directly as the primary name to avoid tooltip parsing truncation
                        var nodeName = nodeId; 
                        var nodeDefinition = ""; var nodeType = ""; var nodeFreq = ""; var nodeDegree = "";
                        
                        if (nodeData && nodeData.title) {
                            var tooltipText = nodeData.title;
                            var defMatch = tooltipText.match(/Definition:\s*(.+)/i); if (defMatch && defMatch[1]) { nodeDefinition = defMatch[1].trim(); }
                            var typeMatch = tooltipText.match(/Type:\s*(\w+)/i); if (typeMatch && typeMatch[1]) { nodeType = typeMatch[1].trim(); }
                            var freqMatch = tooltipText.match(/Frequency:\s*(\d+)/i); if (freqMatch && freqMatch[1]) { nodeFreq = freqMatch[1].trim(); }
                            var degMatch = tooltipText.match(/Degree:\s*(\d+)/i); if (degMatch && degMatch[1]) { nodeDegree = degMatch[1].trim(); }
                        }

                        var html = '<div style="padding:16px 20px;background:linear-gradient(135deg,rgba(255,215,0,0.15),rgba(255,183,77,0.1));border-radius:16px 16px 0 0;border-bottom:2px solid rgba(255,215,0,0.4);">';
                        // FIX 6: Added word-break and white-space normal to header to prevent truncation
                        html += '<div style="font-size:18px;font-weight:800;color:#1e293b;margin-bottom:8px;word-break:break-word;white-space:normal;overflow:visible;">🔋 ' + nodeName + '</div>';
                        html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
                        if (nodeType) html += '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">' + nodeType + '</span>';
                        if (nodeDegree) html += '<span style="background:rgba(168,85,247,0.1);color:#a855f7;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Deg: ' + nodeDegree + '</span>';
                        if (nodeFreq) html += '<span style="background:rgba(34,197,94,0.1);color:#22c55e;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Freq: ' + nodeFreq + '</span>';
                        html += '</div></div>';
                        
                        if (nodeDefinition) {
                            html += '<div style="padding:12px 20px;background:rgba(251,191,36,0.06);border-bottom:1px solid rgba(0,0,0,0.04);">';
                            html += '<div style="font-size:10px;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-bottom:4px;">📖 Definition</div>';
                            html += '<div style="font-size:12px;color:#475569;font-style:italic;line-height:1.4;word-break:break-word;">' + nodeDefinition + '</div></div>';
                        }
                        
                        html += '<div style="padding:10px 20px;background:rgba(248,250,252,0.8);border-bottom:1px solid rgba(0,0,0,0.04);display:flex;align-items:center;gap:10px;">';
                        html += '<span style="font-size:10px;color:#94a3b8;font-weight:600;">Label Mode</span>';
                        html += '<button id="btn-short" onclick="window._heaSetLabelMode(\'short\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:#D32F2F;color:white;">Short</button>';
                        html += '<button id="btn-full" onclick="window._heaSetLabelMode(\'full\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:transparent;color:#64748b;">Full</button>';
                        html += '</div>';
                        
                        html += '<div id="edges-container" style="padding:12px 16px 16px;">';
                        var edgeList = [];
                        connectedEdges.forEach(function(eId) {
                            var e = edgesDS.get(eId); if (!e) return;
                            var fromNode = nodesDS.get(e.from); var toNode = nodesDS.get(e.to);
                            var fromLabel = fromNode ? (fromNode.label || e.from) : e.from;
                            var toLabel = toNode ? (toNode.label || e.to) : e.to;
                            var w = (typeof e.value === 'number') ? e.value : (e.width || 1);
                            var edgeType = 'unknown', isInferred = false;
                            if (e.title) {
                                var _txt = e.title;
                                var m = _txt.match(/Type:\s*(\w+)/); if (m) edgeType = m[1];
                                if (_txt.indexOf('Inferred: true') !== -1) isInferred = true;
                            }
                            edgeList.push({from: fromLabel, to: toLabel, weight: (typeof w === 'number') ? w.toFixed(2) : String(w), type: edgeType, inferred: isInferred});
                        });
                        edgeList.sort(function(a,b){ return parseFloat(b.weight)-parseFloat(a.weight); });
                        edgeList.forEach(function(e, idx){ html += formatEdgeRow(e, idx, labelMode); });
                        html += '</div>';
                        
                        panel.innerHTML = html; panel.style.display = 'block'; panel._edgeList = edgeList;
                        window._heaSetLabelMode = function(mode) {
                            labelMode = mode; var p = document.getElementById('edge-info-panel');
                            if (!p || !p._edgeList) return;
                            var btnShort = document.getElementById('btn-short'); var btnFull = document.getElementById('btn-full');
                            if (mode === 'short') { btnShort.style.background = '#D32F2F'; btnShort.style.color = 'white'; btnFull.style.background = 'transparent'; btnFull.style.color = '#64748b'; }
                            else { btnFull.style.background = '#D32F2F'; btnFull.style.color = 'white'; btnShort.style.background = 'transparent'; btnShort.style.color = '#64748b'; }
                            var container = document.getElementById('edges-container');
                            if (container) { var newHtml = ''; p._edgeList.forEach(function(e, idx){ newHtml += formatEdgeRow(e, idx, mode); }); container.innerHTML = newHtml; }
                        };
                    }

                    network.on("selectNode", function(params) {
                        var nodeId = params.nodes[0];
                        if (nodeId === "__legend__") { network.unselectAll(); return; }
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
                                if (typeof newColor === 'string') newColor = {background: newColor, border: '#FFD700'}; else newColor.border = '#FFD700';
                                nodeUpdates.push({id: nId, color: newColor, shadow: {enabled: true, color: 'rgba(255,215,0,0.5)', size: 15, x: 0, y: 0}});
                            }
                        });
                        if (nodeUpdates.length > 0) nodesDS.update(nodeUpdates);
                        showEdgeInfoPanel(nodeId, connectedEdges);
                    });
                    network.on("deselectNode", function(){ resetAll(); });
                    network.on("click", function(params){ if (params.nodes.length === 0 && activeNodeId !== null) resetAll(); });
                }
            }, 250);
        })();
        </script>
        """
        html_content = html_content.replace('</body>', highlight_js + '</body>')

    st.components.v1.html(html_content, height=950, scrolling=True)

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button(
            "📥 Download Interactive Graph (HTML)",
            data=html_bytes,
            file_name="sib_concept_graph.html",
            mime="text/html"
        )
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")

    if use_abbreviated_labels and label_map:
        st.markdown("---")
        st.markdown("### 🗺️ Node Label Legend")
        sorted_legend = sorted(label_map.items(), key=lambda x: int(x[0][1:]))
        cols = st.columns(4)
        for i, (short, full) in enumerate(sorted_legend):
            with cols[i % 4]:
                st.markdown(f"""<div class='hea-legend' style='padding:8px; border-radius:6px; background-color:{theme.get('tooltip_bg', '#f8fafc')}; border-left:4px solid {theme.get('highlight_bg', '#ff6b6b')}; margin-bottom:6px;'>
<b style='color:{theme.get('highlight_bg', '#ff6b6b')}; font-size:{node_legend_font_size+1}px;'>{short}</b>: <span style='font-size:{node_legend_font_size}px; color:{theme.get('font', '#1e293b')}; word-break:break-word;'>{full}</span></div>""", unsafe_allow_html=True)

def render_graph_plotly_2d(
    nx_graph, concept_abstract_map, cmap_name="viridis",
    custom_labels=None, top_n_nodes=0, node_label_size=10,
    theme=None, show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(
            degrees.keys(), key=lambda x: degrees[x], reverse=True
        )[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(nx_graph, k=1.5, iterations=50, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    edge_hover: List[Optional[str]] = []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)
        edge_hover.extend([
            (
                f"<b>{u} + {v}</b><br>"
                f"Weight: {w:.2f}<br>"
                f"Type: {edge_type}<br>"
                f"Inferred: {is_inferred}"
            )
        ] * 2 + [None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color=theme['edge_unknown']),
        hoverinfo='text', hovertext=edge_hover, name='Connections',
    )
    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_size: List[int] = []
    node_color: List[str] = []
    node_labels: List[str] = []
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        node_text.append(
            f"{node}<br>Type: {concept_type}<br>"
            f"Degree: {deg}<br>Frequency: {freq}"
        )
        node_size.append(max(8, min(35, deg * 2.5 + 10)))
        node_color.append(cmap_colors[i])
        node_labels.append(
            custom_labels.get(node, node) if custom_labels else node
        )
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(
            size=node_size, color=node_color,
            line=dict(width=2, color=theme['node_border']),
        ),
        text=node_labels, textposition="bottom center",
        textfont=dict(size=node_label_size, color=theme['font']),
        hovertext=node_text, hoverinfo='text', name='Concepts',
    )
    fig_data = [edge_trace, node_trace]
    if show_edge_weights:
        for u, v in nx_graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = nx_graph[u][v].get('weight', 1)
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            fig_data.append(go.Scatter(
                x=[mid_x], y=[mid_y], mode='text',
                text=[f"{w:.1f}"],
                textfont=dict(size=8, color=theme['font']),
                hoverinfo='skip', showlegend=False,
            ))
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            plot_bgcolor=theme['plotly_bg'],
            paper_bgcolor=theme['plotly_paper'],
            font=dict(color=theme['font']),
            xaxis=dict(
                showgrid=True, gridcolor=theme['grid_color'],
                zeroline=False, showticklabels=False,
                linecolor=theme['axis_color'],
            ),
            yaxis=dict(
                showgrid=True, gridcolor=theme['grid_color'],
                zeroline=False, showticklabels=False,
                linecolor=theme['axis_color'],
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_graph_plotly_3d(
    nx_graph, concept_abstract_map, cmap_name="viridis",
    top_n_nodes=0, theme=None, show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(nx_graph.nodes()) < 3:
        st.info("3D view requires >=3 nodes.")
        return
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(
            degrees.keys(), key=lambda x: degrees[x], reverse=True
        )[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos_3d = nx.spring_layout(nx_graph, dim=3, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    edge_z: List[Optional[float]] = []
    for u, v in nx_graph.edges():
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(width=2, color=theme['edge_unknown']),
        hoverinfo='skip',
    )
    node_x: List[float] = []
    node_y: List[float] = []
    node_z: List[float] = []
    node_text: List[str] = []
    node_size: List[int] = []
    node_color: List[str] = []
    node_labels: List[str] = []
    for i, node in enumerate(nx_graph.nodes()):
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        node_text.append(
            f"{node}<br>Type: {concept_type}<br>"
            f"Degree: {deg}<br>Frequency: {freq}"
        )
        node_size.append(max(6, min(25, deg * 2 + 8)))
        node_color.append(cmap_colors[i])
        node_labels.append(node)
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers+text',
        marker=dict(size=node_size, color=node_color, opacity=0.9),
        text=node_labels, textposition="top center",
        textfont=dict(size=8, color=theme['font']),
        hovertext=node_text, hoverinfo='text',
    )
    fig_data = [edge_trace, node_trace]
    if show_edge_weights:
        for u, v in nx_graph.edges():
            x0, y0, z0 = pos_3d[u]
            x1, y1, z1 = pos_3d[v]
            w = nx_graph[u][v].get('weight', 1)
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            mid_z = (z0 + z1) / 2
            fig_data.append(go.Scatter3d(
                x=[mid_x], y=[mid_y], z=[mid_z], mode='text',
                text=[f"{w:.1f}"],
                textfont=dict(size=7, color=theme['font']),
                hoverinfo='skip', showlegend=False,
            ))
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    gridcolor=theme['grid_color'],
                    linecolor=theme['axis_color'],
                ),
                yaxis=dict(
                    showbackground=False,
                    gridcolor=theme['grid_color'],
                    linecolor=theme['axis_color'],
                ),
                zaxis=dict(
                    showbackground=False,
                    gridcolor=theme['grid_color'],
                    linecolor=theme['axis_color'],
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False,
            paper_bgcolor=theme['plotly_paper'],
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_graph_fallback(
    nx_graph, concept_abstract_map, theme=None, show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    st.markdown(f"### Graph Summary (Text View)")
    st.markdown(f"- **Nodes**: {len(nx_graph.nodes())}")
    st.markdown(f"- **Edges**: {len(nx_graph.edges())}")
    if len(nx_graph.edges()) > 0:
        edge_list = [
            (
                u, v,
                nx_graph[u][v].get('weight', 1),
                nx_graph[u][v].get('edge_type', 'unknown'),
                nx_graph[u][v].get('inferred', False),
            )
            for u, v in nx_graph.edges()
        ]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**Top 20 Strongest Connections:**")
        for i, (u, v, w, etype, inferred) in enumerate(edge_list[:20], 1):
            inferred_badge = (
                "<span style='background:#8b5cf6;color:white;"
                "padding:1px 5px;border-radius:4px;font-size:11px;'>"
                "INFERRED</span>"
                if inferred else ""
            )
            st.markdown(
                f"{i}. `{u}` + `{v}` {inferred_badge} "
                f"(weight: {w:.2f}, type: {etype})",
                unsafe_allow_html=True,
            )
    if len(concept_abstract_map) > 0:
        freq_data = [
            (c, len(concept_abstract_map.get(c, [])))
            for c in nx_graph.nodes()
        ]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**Top Concepts by Frequency:**")
        st.dataframe(
            pd.DataFrame(
                freq_data[:15], columns=["Concept", "Abstract Count"]
            ),
            use_container_width=True,
        )


# ============================================================================
# SUNBURST & RADAR CHARTS (AgNPs Pattern — Duplicate Prevention)
# ============================================================================


# ============================================================================
# PLOTLY SUNBURST — full hierarchy labels
# ============================================================================

# Category → color map (tier-1 colors for the sunburst segments)
_SUNBURST_CATEGORY_COLORS = {
    "Cathode Materials":          "#E74C3C",
    "Anode Materials":            "#E67E22",
    "Electrolytes":               "#2ECC71",
    "Electrochemical Properties": "#F1C40F",
    "Phenomena":                  "#3498DB",
    "Characterization Methods":   "#9B59B6",
    "Parameters":                 "#1ABC9C",
    "Processing":                 "#2980B9",
}


def build_category_hierarchy(
    valid_concepts: List[str],
    concept_abstract_map: Dict,
    top_n_per_category: int = 40,
) -> Tuple[List, List, List]:
    """
    Faithful AgNPs/MPEA pattern: 2-level hierarchy with DUPLICATE PREVENTION.
    - Root (center): "Sodium-Ion Battery"
    - Ring 1: Categories
    - Ring 2: Concepts (NEVER repeating category names)
    """
    category_map = abstract_concepts_to_categories(valid_concepts)
    all_category_names = set(category_map.values())

    hierarchy: Dict[str, Dict] = {}
    for cat in all_category_names:
        hierarchy[cat] = {"children": [], "count": 0}

    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        freq = len(concept_abstract_map.get(concept, []))

        # ★ KEY FIX: Skip if the concept IS a category name
        if concept in all_category_names:
            hierarchy.setdefault(category, {"children": [], "count": 0})
            hierarchy[category]["count"] += freq
            continue

        hierarchy.setdefault(category, {"children": [], "count": 0})
        hierarchy[category]["children"].append((concept, freq))
        hierarchy[category]["count"] += freq

    labels: List[str] = []
    parents: List[str] = []
    values: List[int] = []

    root_label = "Sodium-Ion Battery"
    total = sum(h["count"] for h in hierarchy.values())
    labels.append(root_label)
    parents.append("")
    values.append(total)

    for category, data in sorted(hierarchy.items()):
        children = data["children"]
        children.sort(key=lambda x: x[1], reverse=True)

        if top_n_per_category > 0 and len(children) > top_n_per_category:
            children = children[:top_n_per_category]

        cat_child_sum = sum(freq for _, freq in children)
        cat_display = category.replace('_', ' ').title()  # <-- FORMAT

        labels.append(cat_display)
        parents.append(root_label)
        values.append(cat_child_sum if cat_child_sum > 0 else data["count"])

        for concept, freq in children:
            # ★ SAFETY: Never add a concept that duplicates any category name
            if concept in all_category_names:
                continue
            concept_display = concept.replace('_', ' ').title()  # <-- FORMAT

            labels.append(concept_display)
            parents.append(cat_display)  # <-- REFERENCE FORMATTED PARENT
            values.append(max(freq, 1))

    return labels, parents, values


def render_sunburst_chart(
    labels, parents, values, cmap_name="viridis",
    label_size=20, width=900, height=700,
    theme=None, branchvalues="total",
    show_labels=True, show_values=False,
    hover_info="all", color_continuous_scale=None,
    font_family="Arial, sans-serif",
    legend_font_size=12,
) -> None:
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    if len(labels) != len(parents) or len(labels) != len(values):
        st.error("Sunburst data mismatch.")
        return

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]

    parent_map = {labels[i]: parents[i] for i in range(len(labels))}

    def get_depth(label, visited=None):
        if visited is None: visited = set()
        if label in visited: return 0
        visited.add(label)
        p = parent_map.get(label, "")
        if p == "": return 0
        return 1 + get_depth(p, visited)

    depths = [get_depth(l) for l in labels]
    SYMBOL_LIBRARY = ['✦', '★', '●', '■', '▲', '◆', '⬟', '⬢', '◉', '◈', '◇', '○', '□', '△', '◊']
    node_symbols = {}
    for i, lab in enumerate(labels):
        d = depths[i]
        p = parents[i]
        if d == 0:
            node_symbols[lab] = SYMBOL_LIBRARY[0]
        else:
            siblings = [labels[j] for j in range(len(labels)) if parents[j] == p and depths[j] == d]
            sym_idx = siblings.index(lab) if lab in siblings else 0
            node_symbols[lab] = SYMBOL_LIBRARY[(d + sym_idx) % len(SYMBOL_LIBRARY)]

    display_labels = []
    for i, lab in enumerate(labels):
        if show_labels:
            chain = []
            current = lab
            visited = set()
            while current != "" and current not in visited:
                visited.add(current)
                if current in node_symbols: chain.insert(0, node_symbols[current])
                current = parent_map.get(current, "")
            combo = "".join(chain[-3:]) if len(chain) > 3 else "".join(chain)
            display_labels.append(combo)
        else:
            display_labels.append(lab)


    # --- ADD THIS BLOCK: Generate Unique IDs ---
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
    # ------------------------------------------
    n_nodes = len(labels)
    cmap_to_use = color_continuous_scale or cmap_name or "Spectral"
    plot_colors: List[str] = []

    # Strategy 1: matplotlib colormap
    color_success = False
    try:
        cmap_obj = plt.cm.get_cmap(cmap_to_use)
        t_vals = np.linspace(0.05, 0.95, n_nodes)
        rgbas = [cmap_obj(t) for t in t_vals]
        plot_colors = [matplotlib.colors.to_hex(rgba) for rgba in rgbas]
        color_success = True
    except Exception:
        pass

    # Strategy 2: Plotly sequential scale
    if not color_success:
        try:
            if hasattr(px.colors.sequential, cmap_to_use):
                px_scale = getattr(px.colors.sequential, cmap_to_use)
                plot_colors = [
                    px_scale[int(i * len(px_scale) / n_nodes) % len(px_scale)]
                    for i in range(n_nodes)
                ]
                color_success = True
        except Exception:
            pass

    # Strategy 3: Plotly qualitative palettes
    if not color_success:
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
            color_success = True
        except Exception:
            pass

    # Strategy 4: tab20 fallback
    if not color_success:
        try:
            cmap_obj = plt.cm.get_cmap("tab20")
            plot_colors = [
                matplotlib.colors.to_hex(cmap_obj(i % 20 / 20))
                for i in range(n_nodes)
            ]
        except Exception:
            plot_colors = ["#ff6b6b"] * n_nodes

    sunburst_colors = plot_colors.copy()
    for i in range(len(labels)):
        if depths[i] == 0:
            sunburst_colors[i] = theme.get("plotly_paper", "#f8f9fa")

    bv = branchvalues if branchvalues in ["total", "remainder"] else "total"
    textinfo = 'label+value' if show_labels and show_values else 'label' if show_labels else 'value' if show_values else 'none'

    fig = go.Figure(go.Sunburst(
        ids=unique_ids,               # <-- ADD
        labels=display_labels,
        parents=parent_ids,           # <-- CHANGE from parents to parent_ids
        values=values,
        customdata=labels,
        branchvalues=bv,
        marker=dict(colors=sunburst_colors, line=dict(width=0.5, color="rgba(255,255,255,0.25)")),
        textinfo=textinfo,
        hovertemplate='<b>%{customdata}</b><br>Value: %{value}<extra></extra>' if hover_info == "all" else '<b>%{customdata}</b><extra></extra>' if hover_info == "minimal" else '<extra></extra>',
        insidetextorientation="radial",
        textfont=dict(size=int(label_size), family=font_family, color="white")
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font=dict(color=theme.get("font", "#000000"), family=font_family),
        width=int(width), height=int(height),
        title=dict(text=f"<b>Hierarchical Concept Map</b><br><sup>★ Parent | ★□ Child | ★□◆ Grandchild — Hover for names</sup>", font=dict(size=16, family=font_family))
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get('sunburst_show_legend', True):
        st.markdown("### 📊 Symbol-to-Label Legend")
        legend_entries = [{'symbol': display_labels[i], 'label': labels[i], 'depth': depths[i], 'color': plot_colors[i], 'value': values[i]} for i in range(len(labels))]
        legend_entries.sort(key=lambda x: (x['depth'], -x['value']))
        for d in sorted(set([e['depth'] for e in legend_entries])):
            st.markdown(f"**{'Root' if d == 0 else 'Category' if d == 1 else 'Concept'}**")
            entries = [e for e in legend_entries if e['depth'] == d]
            cols = st.columns(min(4, max(1, len(entries))))
            for i, entry in enumerate(entries):
                with cols[i % len(cols)]:
                    st.markdown(f"""<div style='padding:8px; border-radius:6px; background-color:{entry['color']}22; border-left:4px solid {entry['color']}; margin-bottom:6px; font-size:{legend_font_size}px;'>
                    <span style='font-size:{legend_font_size+4}px; color:{entry['color']}; margin-right:6px;'>{entry['symbol']}</span>
                    <span style='font-size:{legend_font_size}px; color:{theme.get("font", "#333")}; font-weight:500;'>{entry['label']}</span>
                    <span style='font-size:{legend_font_size-1}px; color:#666; float:right;'>({entry['value']:.0f})</span></div>""", unsafe_allow_html=True)
def render_radar_chart(
    distill_df, top_k=15, cmap_name="viridis", theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if distill_df.empty or top_k == 0:
        st.info("No data available for radar chart.")
        return
    df = distill_df.head(top_k).copy()
    if df.empty:
        return
    metrics = [
        'frequency', 'tfidf_weight', 'semantic_density', 'coherence_score',
    ]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        st.info("No metric columns available for radar chart.")
        return
    for m in available_metrics:
        max_val = df[m].max()
        if max_val > 0:
            df[f'{m}_norm'] = df[m] / max_val
        else:
            df[f'{m}_norm'] = 0
    fig = go.Figure()
    plot_df = df.head(min(top_k, 10))
    for i, row in plot_df.iterrows():
        values = [row[f'{m}_norm'] for m in available_metrics]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            fill='toself',
            name=row['concept'][:25],
            opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        showlegend=True,
        title=f"Concept Radar Chart (Top {min(top_k, 10)})",
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tsne_projection(
    valid_concepts: List[str], concept_abstract_map: Dict[str, List[int]],
    embed_model, theme: Dict = None, n_components: int = 2,
    perplexity: int = 30,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 10:
        st.info("Need at least 10 concepts for t-SNE projection.")
        return
    try:
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts, show_progress_bar=False,
                batch_size=64, convert_to_numpy=True,
            )
        actual_perplexity = min(perplexity, len(valid_concepts) - 1)
        tsne = TSNE(
            n_components=n_components, random_state=42,
            perplexity=actual_perplexity,
        )
        coords = tsne.fit_transform(embeddings)
        category_map = abstract_concepts_to_categories(valid_concepts)
        categories = [category_map.get(c, 'general') for c in valid_concepts]
        freqs = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
        if n_components == 2:
            fig = px.scatter(
                x=coords[:, 0], y=coords[:, 1],
                color=categories, size=freqs,
                hover_name=valid_concepts,
                title="t-SNE Projection of Concept Embeddings",
                labels={'color': 'Category', 'size': 'Frequency'},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            fig = px.scatter_3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                color=categories, size=freqs,
                hover_name=valid_concepts,
                title="3D t-SNE Projection of Concept Embeddings",
                labels={'color': 'Category', 'size': 'Frequency'},
            )
        fig.update_layout(
            paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
            font_color=theme.get("font", "#000000"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        del embeddings, coords
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        st.error(f"t-SNE projection failed: {e}")


def render_community_detection(
    nx_graph, valid_concepts, concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(nx_graph.nodes()) < 3:
        st.info("Need at least 3 nodes for community detection.")
        return
    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(nx_graph))
        node_to_comm: Dict[str, int] = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        pos = nx.spring_layout(nx_graph, seed=42)
        cmap_colors = get_colormap_colors(
            "tab20", max(len(communities), 1)
        )
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        for u, v in nx_graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.8, color=theme['edge_unknown']),
            hoverinfo='none',
        )
        node_traces: List[go.Scatter] = []
        for i, comm in enumerate(communities):
            comm_nodes = list(comm)
            node_x: List[float] = []
            node_y: List[float] = []
            node_text: List[str] = []
            node_size: List[int] = []
            for node in comm_nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                deg = nx_graph.degree(node)
                freq = len(concept_abstract_map.get(node, []))
                node_text.append(
                    f"{node}<br>Community {i}<br>"
                    f"Degree: {deg}<br>Freq: {freq}"
                )
                node_size.append(max(10, min(30, deg * 2 + 8)))
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=cmap_colors[i % len(cmap_colors)],
                    line=dict(width=1.5, color='white'),
                ),
                text=comm_nodes, textposition="bottom center",
                textfont=dict(size=8, color=theme['font']),
                hovertext=node_text, hoverinfo='text',
                name=f"Community {i} ({len(comm_nodes)})",
            )
            node_traces.append(node_trace)
        fig = go.Figure(
            data=[edge_trace] + node_traces,
            layout=go.Layout(
                showlegend=True, hovermode='closest',
                title=f"Community Detection ({len(communities)} communities)",
                margin=dict(b=0, l=0, r=0, t=40),
                plot_bgcolor=theme['plotly_bg'],
                paper_bgcolor=theme['plotly_paper'],
                font=dict(color=theme['font']),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        comm_data: List[Dict[str, Any]] = []
        for i, comm in enumerate(communities):
            comm_data.append({
                "Community": i,
                "Size": len(comm),
                "Top Concepts": ", ".join(
                    sorted(
                        comm,
                        key=lambda c: len(concept_abstract_map.get(c, [])),
                        reverse=True,
                    )[:5]
                ),
            })
        st.dataframe(pd.DataFrame(comm_data), use_container_width=True)
    except Exception as e:
        st.warning(f"Community detection failed: {e}")


def render_concept_growth(
    df_filtered, valid_concepts, concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        st.info("No 'Year' data available for growth analysis.")
        return
    years = df_filtered["Year"].dropna().astype(int)
    if len(years) == 0:
        st.info("No valid year data found.")
        return
    mid_year = int(years.median())
    early_df = df_filtered[df_filtered["Year"] <= mid_year]
    recent_df = df_filtered[df_filtered["Year"] > mid_year]
    if len(early_df) == 0 or len(recent_df) == 0:
        st.info("Need data from both early and recent periods.")
        return
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:15]
    growth_data: List[Dict[str, Any]] = []
    for concept in top_concepts:
        early_count = 0
        recent_count = 0
        for idx, row in early_df.iterrows():
            text = " ".join([
                str(row[col]) for col in df_filtered.columns
                if pd.notna(row[col])
            ])
            early_count += len(re.findall(
                r'\b' + re.escape(concept) + r'\b', text, re.I
            ))
        for idx, row in recent_df.iterrows():
            text = " ".join([
                str(row[col]) for col in df_filtered.columns
                if pd.notna(row[col])
            ])
            recent_count += len(re.findall(
                r'\b' + re.escape(concept) + r'\b', text, re.I
            ))
        growth_rate = (
            ((recent_count - early_count) / max(early_count, 1)) * 100
            if early_count > 0 else 0
        )
        growth_data.append({
            "Concept": concept,
            "Early Count": early_count,
            "Recent Count": recent_count,
            "Growth Rate (%)": growth_rate,
        })
    growth_df = pd.DataFrame(growth_data).sort_values(
        "Growth Rate (%)", ascending=False
    )
    fig = px.bar(
        growth_df, x="Concept", y="Growth Rate (%)",
        color="Growth Rate (%)", color_continuous_scale="RdYlGn",
        title=(
            f"Concept Growth Rate "
            f"(Early <={mid_year} vs Recent >{mid_year})"
        ),
        labels={"Growth Rate (%)": "Growth Rate (%)"},
        template=(
            "plotly_white" if theme == THEME_PRESETS["Bright (Default)"]
            else "plotly_dark"
        ),
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(growth_df, use_container_width=True)


def render_bubble_chart(
    nx_graph, valid_concepts, concept_abstract_map, distill_df, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 3:
        st.info("Need at least 3 concepts for bubble chart.")
        return
    category_map = abstract_concepts_to_categories(valid_concepts)
    bubble_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        degree = nx_graph.degree(concept) if concept in nx_graph else 0
        freq = len(concept_abstract_map.get(concept, []))
        efficiency = distill_df[
            distill_df['concept'] == concept
        ]['distillation_efficiency'].values
        efficiency = (
            float(efficiency[0]) if len(efficiency) > 0 else 0.0
        )
        category = category_map.get(concept, 'general')
        bubble_data.append({
            "Concept": concept, "Degree": degree,
            "Frequency": freq,
            "Distillation Efficiency": efficiency,
            "Category": category,
        })
    bubble_df = pd.DataFrame(bubble_data)
    fig = px.scatter(
        bubble_df, x="Degree", y="Frequency",
        size="Distillation Efficiency", color="Category",
        hover_data=["Concept"],
        title="Concept Importance Bubble Chart",
        size_max=50,
        template=(
            "plotly_white" if theme == THEME_PRESETS["Bright (Default)"]
            else "plotly_dark"
        ),
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# INTERACTIVE GRAPH EDITING (WITH UNDO/REDO)
# ============================================================================
def apply_graph_edits(
    nx_graph, valid_concepts, concept_to_id, id_to_concept,
    concept_abstract_map,
    nodes_to_remove=None, nodes_to_merge=None, merge_name=None,
    new_edge=None, new_edge_weight=1.0, min_degree=0, min_freq=0,
):
    edited = False
    if nodes_to_remove:
        for node in nodes_to_remove:
            if node in nx_graph:
                nx_graph.remove_node(node)
                edited = True
        valid_concepts = [
            c for c in valid_concepts if c not in nodes_to_remove
        ]
        for node in nodes_to_remove:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
    if nodes_to_merge and merge_name and len(nodes_to_merge) >= 2:
        merged_edges: Dict[str, Dict[str, Any]] = {}
        merged_freq = 0
        merged_abstracts: Set[int] = set()
        for node in nodes_to_merge:
            if node in nx_graph:
                for neighbor in list(nx_graph.neighbors(node)):
                    if neighbor not in nodes_to_merge:
                        w = nx_graph[node][neighbor].get('weight', 1)
                        cooc = nx_graph[node][neighbor].get('cooccurrence', 0)
                        sem = nx_graph[node][neighbor].get('semantic', 0)
                        etype = nx_graph[node][neighbor].get('edge_type', 'unknown')
                        if neighbor in merged_edges:
                            merged_edges[neighbor]['weight'] += w
                            merged_edges[neighbor]['cooccurrence'] += cooc
                            merged_edges[neighbor]['semantic'] += sem
                        else:
                            merged_edges[neighbor] = {
                                'weight': w, 'cooccurrence': cooc,
                                'semantic': sem, 'edge_type': etype,
                            }
                merged_freq += nx_graph.nodes[node].get('frequency', 0)
                if node in concept_abstract_map:
                    merged_abstracts.update(concept_abstract_map[node])
                nx_graph.remove_node(node)
        nx_graph.add_node(merge_name, frequency=merged_freq)
        for neighbor, edge_data in merged_edges.items():
            nx_graph.add_edge(merge_name, neighbor, **edge_data)
        concept_abstract_map[merge_name] = list(merged_abstracts)
        valid_concepts = [
            c for c in valid_concepts if c not in nodes_to_merge
        ]
        if merge_name not in valid_concepts:
            valid_concepts.append(merge_name)
        for node in nodes_to_merge:
            if node in concept_abstract_map and node != merge_name:
                del concept_abstract_map[node]
        edited = True
    if new_edge and len(new_edge) == 2:
        u, v = new_edge
        if (
            u in nx_graph and v in nx_graph
            and not nx_graph.has_edge(u, v)
        ):
            nx_graph.add_edge(
                u, v, weight=new_edge_weight,
                cooccurrence=0, semantic=0, edge_type='manual',
            )
            edited = True
    if min_degree > 0:
        low_degree = [
            n for n in nx_graph.nodes() if nx_graph.degree(n) < min_degree
        ]
        for node in low_degree:
            nx_graph.remove_node(node)
        valid_concepts = [c for c in valid_concepts if c not in low_degree]
        for node in low_degree:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
        edited = True
    if min_freq > 0:
        low_freq = [
            n for n in nx_graph.nodes()
            if nx_graph.nodes[n].get('frequency', 0) < min_freq
        ]
        for node in low_freq:
            nx_graph.remove_node(node)
        valid_concepts = [c for c in valid_concepts if c not in low_freq]
        for node in low_freq:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
        edited = True
    valid_concepts = sorted(set(valid_concepts))
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    return (
        nx_graph, valid_concepts, concept_to_id,
        id_to_concept, concept_abstract_map, edited,
    )


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
                nx_graph.graph['version'] = '6.2'
                nx_graph.graph['tool'] = 'SIB-ConceptGraph+MT'
            try:
                nx.write_graphml_lxml(nx_graph, "sib_graph.graphml")
            except Exception:
                nx.write_graphml(nx_graph, "sib_graph.graphml")
            with open("sib_graph.graphml", "rb") as f:
                return f.read(), "application/graphml+xml", "sib_graph.graphml"
        except Exception as e:
            st.error(f"GraphML export failed: {e}")
            return None, None, None
    elif export_format == "JSON (Full Metadata)":
        data = nx.node_link_data(nx_graph)
        if include_metadata:
            data['metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '6.2',
                'tool': 'SIB-ConceptGraph+MT',
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
        return json_str.encode('utf-8'), "application/json", "sib_graph_full.json"
    elif export_format == "JSON (Compact)":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "sib_graph.json"
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
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "sib_edges_enhanced.csv"
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
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "sib_nodes_enhanced.csv"
    elif export_format == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [
                get_sib_category_color(n) for n in nx_graph.nodes()
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
            return buf.read(), "image/png", "sib_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif export_format == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=150)
            node_colors = [
                get_sib_category_color(n) for n in nx_graph.nodes()
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
            return buf.read(), "image/svg+xml", "sib_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    elif export_format == "GEXF":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '6.2'
            nx.write_gexf(nx_graph, "sib_graph.gexf")
            with open("sib_graph.gexf", "rb") as f:
                return f.read(), "application/xml", "sib_graph.gexf"
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
    st.subheader("🔗 Inferred Material-Property Chains")
    material_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.MATERIAL
    ]
    property_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.PROPERTY
    ]
    chains_found: List[Dict[str, Any]] = []
    for mat in material_nodes[:5]:
        for prop in property_nodes[:5]:
            paths = ontology.infer_path(mat, prop, max_depth=3)
            if paths:
                chains_found.append({
                    "Material": mat,
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
        ("sodium-ion battery", "sodium_ion_battery"),
        ("hard carbon anode", "hard_carbon"),
        ("specific capacity", "specific_capacity"),
        ("energy density", "energy_density"),
        ("coulombic efficiency", "coulombic_efficiency"),
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
    current_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ma/g|a/g|ma\s*g-1)', text, re.I)
    if current_matches:
        metrics['current_density_ma_g'] = [float(m) for m in current_matches]
    cap_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mah/g|mah\s*g-1)', text, re.I)
    if cap_matches:
        metrics['capacity_mah_g'] = [float(m) for m in cap_matches]
    density_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:wh/kg|wh\s*kg-1)', text, re.I)
    if density_matches:
        metrics['energy_density_wh_kg'] = [float(m) for m in density_matches]
    voltage_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:v)', text, re.I)
    if voltage_matches:
        metrics['voltage_v'] = [float(m) for m in voltage_matches]
    temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', text, re.I)
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
# ★★★ MICROTRANSFORMER #2: KG-RAG EXTRACTOR (LatentMoE) — SIB EDITION ★★★
# ============================================================================
# === MICROTRANSFORMER #2 ADDITIONS: Constants and Model ===
# Mapping string relationships to integers for the Transformer edge embeddings
RELATIONSHIP_TO_IDX = {rel.name: i for i, rel in enumerate(RelationshipType)}
NUM_EDGE_TYPES = len(RelationshipType)

# 32 Specialized Latent Experts for Sodium‑Ion Battery Domains
SIB_EXPERT_LABELS = [
    "Anode Bottleneck", "Cathode Instability", "SEI Chemistry", "Solid-State Interface",
    "Energy Density", "Moisture Sensitivity", "Pre-Sodiation", "Interface Engineering",
    "Hard Carbon", "Layered Oxide", "Polyanionic", "Prussian Blue",
    "Dendrite Growth", "Volume Expansion", "Thermal Runaway", "Phase Transition",
    "Ionic Conductivity", "Electronic Conductivity", "Diffusion Barrier", "Intercalation",
    "Conversion Reaction", "Alloying", "Plating/Stripping", "Electrolyte Decomposition",
    "Cyclic Voltammetry", "Electrochemical Impedance", "Galvanostatic Cycling", "Operando",
    "Current Density", "Cut-off Voltage", "Temperature", "Full Cell Design"
]

class LatentMoEKGExtractor(nn.Module):
    """
    Microtransformer #2: KG-RAG Extractor.
    Uses Latent Mixture of Experts (l-MoE_acc) for efficient graph traversal encoding.
    Parameter Budget: ~500K. Designed for ONNX edge deployment on Lubuntu/Ubuntu.
    """
    def __init__(self, num_nodes, num_edge_types, d_model=96, latent_dim=24,
                 n_experts=32, top_k=4, num_heads=4, num_layers=2):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        self.edge_embedding = nn.Embedding(num_edge_types, d_model)

        # Latent MoE Components
        self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.up_proj = nn.Linear(latent_dim, d_model, bias=False)

        self.router = nn.Linear(latent_dim, n_experts)
        self.experts = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(n_experts)])
        self.top_k = top_k
        self.n_experts = n_experts

        # Lightweight Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True,
            dim_feedforward=d_model * 2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, node_seq, edge_seq):
        # node_seq: (batch, seq_len), edge_seq: (batch, seq_len - 1)
        node_emb = self.node_embedding(node_seq)          # (batch, seq_len, d_model)

        if edge_seq.size(1) > 0:
            edge_emb = self.edge_embedding(edge_seq)      # (batch, seq_len-1, d_model)
            node_emb[:, 1:, :] = node_emb[:, 1:, :] + edge_emb

        batch_size, seq_len, d_model = node_emb.shape

        # 1. Down‑project to latent space
        latent_repr = self.down_proj(node_emb)            # (batch, seq_len, latent_dim)

        # Flatten for per‑token routing
        flat_latent = latent_repr.view(batch_size * seq_len, -1)  # (N, latent_dim)

        # 2. Routing: compute softmax over experts for each token
        router_logits = self.router(flat_latent)          # (N, n_experts)
        routing_weights = F.softmax(router_logits, dim=-1) # (N, n_experts)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Compute expert outputs for all tokens
        # Stack all experts' outputs: (n_experts, N, latent_dim)
        expert_outputs = torch.stack([self.experts[i](flat_latent) for i in range(self.n_experts)], dim=0)

        # Gather the top‑k experts' outputs for each token
        token_indices = torch.arange(batch_size * seq_len, device=flat_latent.device).unsqueeze(1).expand(-1, self.top_k)
        selected = expert_outputs[topk_indices, token_indices, :]   # (N, top_k, latent_dim)

        # Weighted sum over top‑k experts
        weighted = topk_weights.unsqueeze(-1) * selected            # (N, top_k, latent_dim)
        moe_output_flat = weighted.sum(dim=1)                       # (N, latent_dim)

        # Reshape back to (batch, seq_len, latent_dim)
        moe_output = moe_output_flat.view(batch_size, seq_len, -1)

        # 4. Up‑project and add residual
        node_emb = node_emb + self.up_proj(moe_output)   # (batch, seq_len, d_model)

        # 5. Transformer contextualization
        out = self.transformer(node_emb)
        out = self.output_proj(out)
        return out, routing_weights.view(batch_size, seq_len, -1)  # return routing_weights reshaped

# === MICROTRANSFORMER #2 ADDITIONS: Helper functions ===
def plotly_continuous_scale(cmap_key: str, n: int = 12) -> List[str]:
    """Any registered matplotlib cmap (jet/turbo/rainbow/inferno/…) → Plotly color list."""
    return get_colormap_colors(cmap_key, n)

def apply_mt_chart_style(fig, theme: Dict):
    """Uniform fonts / backgrounds / colorbar styling for microtransformer charts."""
    fam   = st.session_state.get("mt_font_family", "Inter, Segoe UI, Roboto, sans-serif")
    tsize = int(st.session_state.get("mt_font_size", 11))
    fig.update_layout(
        font=dict(family=fam, size=tsize, color=theme["font"]),
        title_font=dict(family=fam, size=int(st.session_state.get("mt_title_size", 15))),
        paper_bgcolor=theme["plotly_paper"], plot_bgcolor=theme["plotly_bg"],
    )
    for ax in (fig.update_xaxes, fig.update_yaxes):
        ax(
            showgrid=st.session_state.get("mt_show_grid", False),
            gridcolor=theme["grid_color"],
            tickfont=dict(family=fam, size=tsize, color=theme["axis_color"]),
            title_font=dict(family=fam, size=tsize + 1),
        )
    # update_colorbars doesn't exist on Figure — use layout-level coloraxis
    cbar_title = st.session_state.get("mt_cbar_title", "Weight")
    fig.update_layout(
        coloraxis=dict(
            colorbar=dict(
                title=dict(text=cbar_title,
                             font=dict(family=fam, size=tsize + 1,
                                         color=theme["font"])),
                tickfont=dict(family=fam, size=max(8, tsize - 1),
                                color=theme["axis_color"]),
                thickness=st.session_state.get("mt_cbar_thick", 14),
                outlinewidth=0,
                len=st.session_state.get("mt_cbar_len", 0.8),
            )
        )
    )
    return fig

# === MICROTRANSFORMER #2 ADDITIONS: UI Renderer ===
def render_microtransformer_kg_rag_tab(analysis_data: Dict, ontology: Any):
    st.subheader("🧠 Microtransformer #2: KG-RAG Extractor (LatentMoE)")
    st.markdown("""
    This microtransformer encodes graph traversals (node-edge-node sequences) to extract
    phase stability and electromechanical links. It uses **Latent Mixture of Experts (l-MoE_acc)**
    to route tokens through 32 specialized latent domains in a compressed 24-dimensional space.
    """)

    if not analysis_data or "nx_graph" not in analysis_data:
        st.info("Please build the concept graph first.")
        return

    ontology = ensure_ontology_populated()  # defined below
    nx_graph = analysis_data["nx_graph"]
    concept_to_id = analysis_data["concept_to_id"]
    num_nodes = len(concept_to_id)

    if num_nodes < 2:
        st.error("Graph has fewer than 2 nodes. Cannot build model.")
        return

    # ─── Build unified concept list ─────────────────────────────────────
    type_order = [
        ConceptType.MATERIAL, ConceptType.PARAMETER, ConceptType.PHENOMENON,
        ConceptType.PROPERTY, ConceptType.PROCESS, ConceptType.METHOD,
        ConceptType.MICROSTRUCTURE, ConceptType.GENERAL,
    ]
    type_labels = {
        ConceptType.MATERIAL: "📦 Materials",
        ConceptType.PARAMETER: "🎛️ Parameters (doping, T, grain size...)",
        ConceptType.PHENOMENON: "⚡ Phenomena (scattering, convergence...)",
        ConceptType.PROPERTY: "📊 Properties (S, σ, κ, ZT...)",
        ConceptType.PROCESS: "🔥 Processes (SPS, hot pressing...)",
        ConceptType.METHOD: "🔬 Methods (ZEM-3, XRD, TEM...)",
        ConceptType.MICROSTRUCTURE: "🏗️ Microstructure",
        ConceptType.GENERAL: "📋 General",
    }

    graph_concepts = set(concept_to_id.keys())
    ontology_concepts = set(ontology.concepts.keys())
    all_available = sorted(graph_concepts | ontology_concepts)

    # Mark which concepts are actually in the graph
    in_graph = {c for c in all_available if c in concept_to_id}

    # Build grouped options
    grouped_options = []
    for ctype in type_order:
        members = [c for c in all_available if ontology.get_concept_type(c) == ctype]
        if members:
            grouped_options.append(f"--- {type_labels.get(ctype, ctype.value)} ---")
            for m in members:
                tag = " ✅" if m in in_graph else " ⚠️ not in graph"
                grouped_options.append(f"{m}{tag}")

    separator_set = {s for s in grouped_options if s.startswith("--- ")}

    def strip_tag(option_str: str) -> str:
        """Remove the ✅/⚠️ tag from an option string."""
        for marker in [" ✅", " ⚠️ not in graph"]:
            if option_str.endswith(marker):
                return option_str[:-len(marker)]
        return option_str

    # ─── NLP QUERY PARSING (sets defaults) ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔬 Expert Routing Activation Analysis")
    with st.expander("🎨 Chart Customization (colormap, fonts, colorbar)", expanded=False):
        _cmaps = list(SUPPORTED_COLORMAPS.keys())
        st.selectbox("Colormap:", options=_cmaps,
                     index=_cmaps.index(st.session_state.get("cmap_name", "viridis"))
                     if st.session_state.get("cmap_name", "viridis") in _cmaps else 0,
                     key="mt_cmap",
                     help="Sequential (viridis/inferno/turbo) best for heatmaps; jet/rainbow are popular but not colorblind-safe.")
        st.selectbox("Font family (labels, ticks, colorbar):",
                     ["Inter, Segoe UI, Roboto, sans-serif", "Arial, Helvetica, sans-serif",
                      "Georgia, serif", "Courier New, monospace", "Times New Roman, serif"],
                     key="mt_font_family")
        c1, c2, c3, c4 = st.columns(4)
        c1.slider("Tick font size", 8, 20, 11, key="mt_font_size")
        c2.slider("Title font size", 10, 26, 15, key="mt_title_size")
        c3.slider("Colorbar length", 0.3, 1.0, 0.8, 0.05, key="mt_cbar_len")
        c4.slider("Colorbar thickness (px)", 6, 40, 14, key="mt_cbar_thick")
        st.text_input("Colorbar title", value="Weight", key="mt_cbar_title")
        st.checkbox("Show gridlines", value=False, key="mt_show_grid")
        st.number_input("Torch seed (reproducible demo)", 0, 9999, 42, key="mt_seed")


    # ─── SIB-specific query patterns for NLP suggestion ────────────────
    QUERY_CONCEPT_MAP = [
        (r"\bdoping\b.*\bcapacity\b", "doping_concentration", "specific_capacity"),
        (r"\bhard\s+carbon\b.*\bcapacity\b", "hard_carbon", "specific_capacity"),
        (r"\bhard\s+carbon\b.*\bice\b", "hard_carbon", "coulombic_efficiency"),
        (r"\bsei\b.*\bcoulombic\b", "sei_formation", "coulombic_efficiency"),
        (r"\bdendrite\b.*\bcycle\s+life\b", "dendrite_growth", "cycle_life"),
        (r"\bsolid\s+electrolyte\b.*\bionic\s+conductivity\b", "solid_electrolyte", "ionic_conductivity"),
        (r"\binterface\b.*\benergy\s+density\b", "interface_engineering", "energy_density"),
        (r"\bphase\s+transition\b.*\bcapacity\b", "phase_transition", "specific_capacity"),
    ]

    quick_query = st.text_input(
        "Or type a natural-language question:",
        value="How does doping affect specific capacity?",
        key="mt_quick_query",
        placeholder="e.g., How does doping affect specific capacity?",
    )

    nlp_src, nlp_tgt = None, None
    if quick_query.strip():
        q_lower = quick_query.lower()
        for pattern, src_concept, tgt_concept in QUERY_CONCEPT_MAP:
            if re.search(pattern, q_lower):
                if src_concept in all_available:
                    nlp_src = src_concept
                if tgt_concept in all_available:
                    nlp_tgt = tgt_concept
                break

    # Find dropdown indices for NLP results (to set defaults)
    def find_option_index(concept_name: str) -> int:
        for i, opt in enumerate(grouped_options):
            if strip_tag(opt) == concept_name:
                return i
        return 0

    # Set default indices based on NLP parse
    if nlp_src and nlp_tgt:
        default_src_idx = find_option_index(nlp_src)
        default_tgt_idx = find_option_index(nlp_tgt)
    else:
        # Fallback defaults
        default_src_idx = find_option_index("doping_concentration")
        default_tgt_idx = find_option_index("specific_capacity")

    # ─── DROPDOWNS (use NLP results as defaults) ────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        src_option = st.selectbox(
            "Source Concept (any type):",
            options=grouped_options,
            index=default_src_idx,
            key="mt_src_select",
        )
    with col2:
        tgt_option = st.selectbox(
            "Target Concept (any type):",
            options=grouped_options,
            index=default_tgt_idx,
            key="mt_tgt_select",
        )

    selected_src = None if src_option in separator_set else strip_tag(src_option)
    selected_tgt = None if tgt_option in separator_set else strip_tag(tgt_option)

    # ─── STATUS BANNERS ────────────────────────────────────────────────
    if nlp_src and nlp_tgt:
        st.success(
            f"🧠 NLP parsed: **{nlp_src}** → **{nlp_tgt}** "
            f"(dropdowns auto-set; click button to run)"
        )

    if selected_src and selected_tgt:
        src_in_graph = selected_src in concept_to_id
        tgt_in_graph = selected_tgt in concept_to_id
        if not src_in_graph or not tgt_in_graph:
            missing = []
            if not src_in_graph:
                missing.append(f"`{selected_src}`")
            if not tgt_in_graph:
                missing.append(f"`{selected_tgt}`")
            st.warning(
                f"{' and '.join(missing)} not extracted from your documents. "
                f"The ontology knows about {'them' if len(missing) > 1 else 'it'}, "
                f"so path inference will still work, but node embeddings will be generic."
            )

    if selected_src and selected_tgt and selected_src == selected_tgt:
        st.error("Source and target must be different concepts.")
        selected_src = None

    # ─── BUILD MODEL ──────────────────────────────────────────────────
    torch.manual_seed(int(st.session_state.get("mt_seed", 42)))
    kg_model = LatentMoEKGExtractor(num_nodes, NUM_EDGE_TYPES)
    kg_model.eval()

    # ─── RUN INFERENCE ──────────────────────────────────────────────────
    if st.button("⚡ Run LatentMoE Inference on Path", type="primary"):
        if not selected_src or not selected_tgt:
            st.warning("Please select both a source and a target concept.")
        elif selected_src == selected_tgt:
            st.warning("Source and target must be different.")
        else:
            path = None

            # 1. Try graph path (both must be in graph)
            if selected_src in concept_to_id and selected_tgt in concept_to_id:
                if selected_src in nx_graph and selected_tgt in nx_graph:
                    try:
                        path = nx.shortest_path(
                            nx_graph, source=selected_src, target=selected_tgt
                        )
                        st.success(f"✅ Path found in graph: {' → '.join(path)}")
                    except nx.NetworkXNoPath:
                        pass

            # 2. Try ontology inference (works even if concepts aren't in graph)
            if path is None:
                ontology_paths = ontology.infer_path(
                    selected_src, selected_tgt, max_depth=3
                )
                if ontology_paths:
                    path = ontology_paths[0]
                    st.info(
                        f"🔍 Ontology inference path: {' → '.join(path)}"
                    )
                else:
                    st.error(
                        f"No path found between `{selected_src}` and "
                        f"`{selected_tgt}` in either the graph or the ontology."
                    )
                    src_related = ontology.get_related_concepts(selected_src)
                    if src_related:
                        st.markdown(
                            f"**Concepts connected to `{selected_src}`:** "
                            + ", ".join(
                                f"`{t}` ({r.value}, {c:.2f})"
                                for t, r, c in sorted(src_related, key=lambda x: -x[2])[:8]
                            )
                        )
                    path = None

            if path is None:
                st.session_state.pop("mt_last_run", None)
            else:
                st.success(f"🚀 Processing Path: {' → '.join(path)}")

                # ─── Tokenize path ───────────────────────────────────────
                node_indices = []
                ontology_only_tokens = []
                for n in path:
                    if n in concept_to_id:
                        node_indices.append(concept_to_id[n])
                    else:
                        node_indices.append(0)
                        ontology_only_tokens.append(n)

                node_seq = torch.tensor([node_indices], dtype=torch.long)

                # ─── Build edge sequence ─────────────────────────────────
                edge_indices = []
                for i in range(len(path) - 1):
                    edge_data = nx_graph.get_edge_data(path[i], path[i + 1])
                    if edge_data:
                        rel_str = edge_data.get('edge_type', 'semantic').upper()
                        edge_indices.append(RELATIONSHIP_TO_IDX.get(rel_str, 0))
                    else:
                        rel = next(
                            (r for r in ontology.relationships
                             if r.source == path[i] and r.target == path[i + 1]),
                            None,
                        )
                        if rel:
                            edge_indices.append(RELATIONSHIP_TO_IDX.get(rel.rel_type.name, 0))
                        else:
                            edge_indices.append(RELATIONSHIP_TO_IDX.get("INFLUENCES", 0))

                edge_seq = torch.tensor([edge_indices], dtype=torch.long)

                if ontology_only_tokens:
                    st.caption(
                        f"⚠️ Tokens {ontology_only_tokens} are ontology-only (not in graph). "
                        f"Their node embeddings are generic (ID 0). Expert routing still works "
                        f"because it depends on the router network, not the node embeddings."
                    )

                # ─── Forward pass ────────────────────────────────────────
                with torch.no_grad():
                    out, routing_weights = kg_model(node_seq, edge_seq)

                avg_weights = routing_weights.mean(dim=1).squeeze(0).numpy()

                # ─── Persist for restyling ───────────────────────────────
                st.session_state["mt_last_run"] = {
                    "path": path,
                    "routing": routing_weights.squeeze(0).numpy(),   # (seq_len, 32)
                    "avg": avg_weights,
                }

    # ─── RENDER (every rerun; reads current customization) ─────────────
    run = st.session_state.get("mt_last_run")
    if not run:
        st.info("Press ⚡ Run once — afterwards you can restyle freely.")
    else:
        path        = run["path"]
        routing_np  = run["routing"]
        avg_weights = run["avg"]
        token_labels = [n.replace("_", " ").title() for n in path]
        theme = THEME_PRESETS[st.session_state.get("theme", "Bright (Default)")]
        scale = plotly_continuous_scale(st.session_state.get("mt_cmap", "viridis"))

        st.success(f"🚀 Processing Path: {' → '.join(path)}")

        # ─── Per-token routing heatmap ──────────────────────────
        st.markdown("#### 📊 Per-Token Expert Routing Heatmap")
        per_token_df = pd.DataFrame(routing_np, index=token_labels, columns=SIB_EXPERT_LABELS)
        fig_heat = px.imshow(
            per_token_df.T,
            labels=dict(x="Path Token", y="Expert Domain"),
            color_continuous_scale=scale, aspect="auto", height=400,
        )
        st.plotly_chart(apply_mt_chart_style(fig_heat, theme), use_container_width=True)

        # ─── Averaged bar chart ──────────────────────────────────
        df_experts = pd.DataFrame({
            "Expert Domain": SIB_EXPERT_LABELS,
            "Activation Weight": avg_weights,
        }).sort_values("Activation Weight", ascending=False)

        fig = px.bar(
            df_experts,
            x="Expert Domain",
            y="Activation Weight",
            title=f"LatentMoE Expert Routing: {' → '.join(token_labels)}",
            color="Activation Weight",
            color_continuous_scale=scale,
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(apply_mt_chart_style(fig, theme), use_container_width=True)

        # ─── Top experts metric cards ────────────────────────────
        st.markdown("**Top Activated Experts:**")
        top_experts = df_experts.head(4)
        cols = st.columns(4)
        for i, (_, row) in enumerate(top_experts.iterrows()):
            cols[i].metric(
                label=str(row["Expert Domain"]),
                value=f"{float(row['Activation Weight']):.3f}",
            )

        # ─── Scientific interpretation ────────────────────────────
        st.markdown("#### 🔬 Scientific Interpretation")
        interpretation_map = {
            "Anode Bottleneck": "Graphite incompatibility for Na+ intercalation — model detects hard carbon as alternative.",
            "Cathode Instability": "Phase transitions and structural degradation in layered oxides.",
            "SEI Chemistry": "Sodium salt solubility and SEI dissolution issues.",
            "Solid-State Interface": "Interfacial contact loss and void formation in solid electrolytes.",
            "Energy Density": "Lower Na potential and heavier mass reduce specific energy.",
            "Moisture Sensitivity": "Hygroscopic cathode materials cause slurry gelation.",
            "Pre-Sodiation": "Compensates initial Na loss to improve ICE.",
            "Interface Engineering": "Surface coatings to stabilize electrode-electrolyte interfaces.",
            "Hard Carbon": "The most common SIB anode material.",
            "Layered Oxide": "High capacity but suffer from structural instability.",
            "Polyanionic": "Stable framework materials with good cycle life.",
            "Prussian Blue": "Open framework for fast Na+ transport.",
            "Dendrite Growth": "Metal anode failure mechanism.",
            "Volume Expansion": "Mechanical stress during alloying/conversion.",
            "Thermal Runaway": "Safety concern under abuse conditions.",
            "Phase Transition": "Structural rearrangements during cycling.",
            "Ionic Conductivity": "Limiting factor for rate performance.",
            "Electronic Conductivity": "Affects power capability.",
            "Diffusion Barrier": "Sluggish Na+ migration in electrodes.",
            "Intercalation": "Host-guest insertion mechanism.",
            "Conversion Reaction": "High capacity but large voltage hysteresis.",
            "Alloying": "Sn, Sb anodes with high capacity but large volume change.",
            "Plating/Stripping": "Sodium metal deposition and dissolution.",
            "Electrolyte Decomposition": "SEI formation from electrolyte reduction.",
            "Cyclic Voltammetry": "Electrochemical characterization.",
            "Electrochemical Impedance": "Interface resistance and kinetics.",
            "Galvanostatic Cycling": "Constant current charge/discharge.",
            "Operando": "Real-time characterization during operation.",
            "Current Density": "Affects rate capability and overpotential.",
            "Cut-off Voltage": "Defines the electrochemical window.",
            "Temperature": "Influences kinetics and safety.",
            "Full Cell Design": "Practical cell configuration and N/P ratio."
        }
        top3_names = df_experts.head(3)["Expert Domain"].tolist()
        for expert_name in top3_names:
            interp = interpretation_map.get(expert_name)
            if interp:
                st.info(f"**{expert_name}**: {interp}")

        # ─── Path reasoning chain ───────────────────────────────
        st.markdown("#### 🔗 Reasoning Chain Along Path")
        for i in range(len(path) - 1):
            src_name = path[i].replace("_", " ").title()
            tgt_name = path[i + 1].replace("_", " ").title()
            rel_desc = "unknown"
            for r in ontology.relationships:
                if r.source == path[i] and r.target == path[i + 1]:
                    rel_desc = f"{r.rel_type.value} (confidence: {r.confidence:.2f})"
                    break
            token_experts = pd.DataFrame({
                "Expert": SIB_EXPERT_LABELS,
                "Weight": routing_np[i],
            }).sort_values("Weight", ascending=False)
            top2 = ", ".join(
                f"{row['Expert']} ({row['Weight']:.3f})"
                for _, row in token_experts.head(2).iterrows()
            )
            st.markdown(
                f"**Step {i+1}**: `{src_name}` --[{rel_desc}]--> `{tgt_name}`  "
                f"*(top experts: {top2})*"
            )

    st.markdown("---")
    st.markdown("#### 📤 Edge Deployment (Ubuntu/Lubuntu ONNX Export)")
    if st.button("📦 Export to ONNX"):
        with st.spinner("Exporting and quantizing model..."):
            dummy_nodes = torch.tensor([[1, 2, 3]], dtype=torch.long)
            dummy_edges = torch.tensor([[0, 0]], dtype=torch.long)
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    onnx_path = os.path.join(tmp_dir, "kg_microtransformer.onnx")
                    torch.onnx.export(
                        kg_model,
                        (dummy_nodes, dummy_edges),
                        onnx_path,
                        input_names=["node_seq", "edge_seq"],
                        output_names=["embeddings", "routing_weights"],
                        dynamic_axes={
                            "node_seq": {0: "batch", 1: "seq_len"},
                            "edge_seq": {0: "batch", 1: "seq_len"},
                        },
                        opset_version=14,
                    )
                    with open(onnx_path, "rb") as f:
                        onnx_bytes = f.read()
                    st.success(
                        f"✅ ONNX Export Successful! File size: {len(onnx_bytes)/1024:.1f} KB"
                    )
                    st.download_button(
                        label="⬇️ Download kg_microtransformer.onnx",
                        data=onnx_bytes,
                        file_name="kg_microtransformer.onnx",
                        mime="application/octet-stream",
                    )
            except Exception as e:
                st.error(f"ONNX Export failed: {e}")

# ============================================================================
# SIDEBAR (AgNPs Pattern — Full Sunburst Customization)
# ============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration v6.2 + MT")
        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0,
        )

        st.subheader("🔍 Query-Focused Graph Mode")
        query_focused_enabled = st.checkbox("Build graph only for current query concepts", key="query_focused_build")
        if query_focused_enabled:
            whitelist = st.session_state.get('last_query_whitelist', set())
            if whitelist:
                st.success(f"Will extract {len(whitelist)} focused concepts")
                with st.expander("Preview whitelisted concepts"):
                    st.write(sorted(whitelist))
            else:
                st.info("Ask a question in the 🤖 LLM-Guided Q&A tab to generate a whitelist.")
        theme = THEME_PRESETS[st.session_state['theme']]
        st.subheader("🔬 Sodium-Ion Battery Focus Areas")
        st.markdown("- **Cathode Materials:** Layered oxides (NaMnO₂), Polyanionic (Na₃V₂(PO₄)₃), Prussian blue analogues, NASICON")
        st.markdown("- **Anode Materials:** Hard carbon, Sodium metal, Alloying (Sn, Sb), Intercalation (TiO₂)")
        st.markdown("- **Electrolytes:** Liquid, Solid (NASICON, sulfide), Polymer, Quasi-solid")
        st.markdown("- **Electrochemical Properties:** Specific capacity, Energy density, Coulombic efficiency, Cycle life, Rate capability, Ionic conductivity")
        st.markdown("- **Phenomena:** Dendrite growth, SEI formation, Plating/stripping, Intercalation")
        st.markdown("- **Methods:** CV, EIS, Galvanostatic cycling, Operando characterization")
        st.markdown("- **Parameters:** Current density, Cut-off voltage, Temperature")
        st.markdown("- **Processing:** Slurry coating, Cell assembly")
        st.subheader("🧠 NLP Reasoning Options")
        st.session_state['use_ontology'] = st.checkbox(
            "Use ontology-based resolution", value=True,
            help="Maps synonyms like 'SIB', 'Na-ion battery' to canonical concepts",
        )
        st.session_state['use_embedding_resolution'] = st.checkbox(
            "Use embedding-based semantic equivalence", value=True,
            help="Detects semantic similarity >0.85 even for unseen variants",
        )
        st.session_state['use_relationship_extraction'] = st.checkbox(
            "Extract cause-effect relationships", value=True,
            help="Identifies causal links between material parameters and performance",
        )
        st.session_state['use_inference'] = st.checkbox(
            "Enable reasoning-based edge inference", value=True,
            help="Infers material→property chains even when not co-occurring",
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
                "Node label font size", 8, 50, 25, step=1,
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
            st.slider(
                "Node legend font size", 8, 50, 25, step=1,
                help="Font size for the abbreviated node legend below the graph.",
                key="node_legend_font_size",
            )
        st.session_state['use_abbreviated_labels'] = st.checkbox(
            "Use short labels (N1, N2...) for long names",
            value=False,
            help="Replaces long node labels with N1, N2... and generates a legend below the graph.",
        )
        if st.session_state['use_abbreviated_labels']:
            st.session_state['max_label_length'] = st.slider(
                "Max label length before abbreviation",
                min_value=2, max_value=50, value=30, step=1,
                help="Labels longer than this threshold will be replaced by N1, N2, etc.",
            )
        else:
            st.session_state['max_label_length'] = 30
        st.session_state['show_definitions'] = st.checkbox(
            "📖 Show concept definitions in tooltips",
            value=True,
            help="When enabled, hovering over a node displays its ontology definition in the tooltip.",
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
        with st.expander("Edge Color Customization"):
            st.selectbox(
                "Edge color mode",
                ["theme", "uniform_grey", "custom"],
                index=0,
                help="theme: based on relationship type (lightened), uniform_grey: single grey, custom: your pick",
                key="edge_color_mode",
            )
            if st.session_state['edge_color_mode'] == "custom":
                st.color_picker(
                    "Custom edge color", value="#AAAAAA",
                    key="custom_edge_color",
                )
            else:
                st.session_state['custom_edge_color'] = "#AAAAAA"
            st.slider(
                "Edge lightness (0=original, 1=white)", 0.0, 1.0, 0.6, step=0.05,
                help="Higher values make edges lighter, improving node visibility.",
                key="edge_lightness",
            )
        edge_color_value = st.session_state.get('edge_label_color')
        if not edge_color_value or edge_color_value == '':
            edge_color_value = '#000000'
        st.session_state['edge_label_color'] = edge_color_value

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
        # ADD THIS MISSING SLIDER:
        st.slider(
            "Sunburst legend font size", 8, 50, 24, step=1,
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

        # LLM Query Panel – always visible (ontology is always available)
        ontology = st.session_state.ontology
        expander = st.session_state.qa_expander
        # Pass a dummy graph if not built yet (subgraph extraction won't be used until after build)
        full_graph = st.session_state.analysis_data.get("nx_graph") if st.session_state.get('analysis_data') else nx.Graph()
        render_llm_query_panel(ontology, expander, full_graph)
        # Mutation controls and history are useful even without a graph (they track added concepts)
        render_mutation_controls(expander)
        render_query_history()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# ============================================================================
# ★★★ LLM-GUIDED QUERY ANALYSIS & GRAPHRAG INTEGRATOR (v6.2) ★★★
# ============================================================================
import json
import re
import copy
import tempfile
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st

# ============================================================================
# 0. LOCAL LLM MODEL REGISTRY (< 1B parameters for Streamlit Cloud)
# ============================================================================
LOCAL_LLM_REGISTRY: Dict[str, Optional[str]] = {
    "Fallback (Rule-based, no LLM)": None,
    "DistilGPT-2 (82M, fastest)": "distilgpt2",
    "GPT-Neo-125M (125M)": "EleutherAI/gpt-neo-125M",
    "Pythia-410M (410M, balanced)": "EleutherAI/pythia-410m",
    "BLOOM-560M (560M, multilingual)": "bigscience/bloom-560m",
    "Qwen2-0.5B-Instruct (500M, best JSON)": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2.5-0.5B-Instruct (500M, newest)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B-Chat (1.1B, chat-optimized)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# ============================================================================
# 1. QUERY ANALYSIS DATA STRUCTURES
# ============================================================================
class SIBCoreProblem(Enum):
    ANODE_BOTTLENECK = "anode_bottleneck"
    CATHODE_INSTABILITY = "cathode_instability"
    SEI_CHEMISTRY = "sei_chemistry"
    SOLID_STATE_INTERFACE = "solid_state_interface"
    LOW_ENERGY_DENSITY = "low_energy_density"
    MOISTURE_MANUFACTURING = "moisture_manufacturing"
    GENERAL = "general"
    MULTI_PROBLEM = "multi_problem"

@dataclass
class SIBProblemDefinition:
    problem_id: SIBCoreProblem
    title: str
    scientific_description: str
    root_cause: str
    key_concepts: List[str]
    key_relationships: List[Tuple[str, str, str]]
    solution_directions: List[str]
    relevant_materials: List[str]
    relevant_phenomena: List[str]
    relevant_properties: List[str]
    example_queries: List[str]
    visualization_focus: List[str]

    def get_ontology_concepts(self) -> Set[str]:
        concepts = set(self.key_concepts + self.relevant_materials + 
                       self.relevant_phenomena + self.relevant_properties)
        for src, _, tgt in self.key_relationships:
            concepts.update([src, tgt])
        return concepts

# Pre-defined SIB Problem Definitions
SIB_PROBLEM_DEFINITIONS: Dict[SIBCoreProblem, SIBProblemDefinition] = {
    SIBCoreProblem.ANODE_BOTTLENECK: SIBProblemDefinition(
        problem_id=SIBCoreProblem.ANODE_BOTTLENECK, title="The Anode Bottleneck: Graphite Incompatibility",
        scientific_description="Sodium ions (Na⁺, radius ~1.02 Å) cannot effectively intercalate into standard graphite layers.",
        root_cause="Na⁺ ionic radius is ~34% larger than Li⁺, preventing stable intercalation.",
        key_concepts=["hard_carbon", "alloying_anode", "intercalation_anode", "sodium_metal", "initial_coulombic_efficiency", "volume_expansion"],
        key_relationships=[("hard_carbon", "INFLUENCES", "specific_capacity"), ("alloying_anode", "CAUSES", "volume_expansion")],
        solution_directions=["Optimize hard carbon microstructure", "Develop alloying anodes with nanostructuring", "Apply pre-sodiation techniques"],
        relevant_materials=["hard_carbon", "alloying_anode", "mxene", "conversion_anode"],
        relevant_phenomena=["intercalation", "volume_expansion", "sei_formation"],
        relevant_properties=["specific_capacity", "coulombic_efficiency", "cycle_life"],
        example_queries=["Why can't sodium intercalate into graphite like lithium does?", "How can we improve the initial Coulombic efficiency of hard carbon?"],
        visualization_focus=["anode_materials_subgraph", "ice_analysis"]
    ),
    SIBCoreProblem.CATHODE_INSTABILITY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.CATHODE_INSTABILITY, title="Cathode Structural Instability and Volume Change",
        scientific_description="The larger Na⁺ ion causes significant mechanical stress during insertion/extraction.",
        root_cause="Na⁺ induces ~55% larger lattice parameter changes than Li⁺ during intercalation.",
        key_concepts=["layered_oxide_cathode", "polyanionic_cathode", "prussian_blue_analogue", "phase_transition", "volume_change", "elemental_doping"],
        key_relationships=[("layered_oxide_cathode", "CAUSES", "phase_transition"), ("elemental_dobing", "STABILIZES", "layered_oxide_cathode")],
        solution_directions=["Elemental doping to stabilize layered structures", "Create concentration-gradient cathodes", "Design single-crystal cathodes"],
        relevant_materials=["layered_oxide_cathode", "polyanionic_cathode", "prussian_blue_analogue"],
        relevant_phenomena=["phase_transition", "volume_expansion", "structural_degradation"],
        relevant_properties=["specific_capacity", "cycle_life", "rate_capability"],
        example_queries=["What causes the P2 to O2 phase transition in NaₓMnO₂ cathodes?", "How does elemental doping stabilize layered oxide cathodes?"],
        visualization_focus=["cathode_phase_diagram", "doping_effects"]
    ),
    SIBCoreProblem.SEI_CHEMISTRY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.SEI_CHEMISTRY, title="Electrolyte and Interphase (SEI) Chemistry",
        scientific_description="The Solid Electrolyte Interphase (SEI) is a protective layer formed on the anode from electrolyte decomposition.",
        root_cause="Sodium salts have higher solubility in the SEI matrix than lithium salts, preventing stable passivation.",
        key_concepts=["sei_formation", "liquid_electrolyte", "interface_engineering", "artificial_sei", "electrolyte_decomposition", "concentrated_electrolyte"],
        key_relationships=[("liquid_electrolyte", "CAUSES", "sei_formation"), ("interface_engineering", "STABILIZES", "sei_formation")],
        solution_directions=["Design concentrated electrolytes", "Use fluorinated solvents/additives", "Create artificial SEI layers"],
        relevant_materials=["liquid_electrolyte", "solid_electrolyte", "polymer_electrolyte"],
        relevant_phenomena=["sei_formation", "electrolyte_decomposition"],
        relevant_properties=["coulombic_efficiency", "cycle_life", "ionic_conductivity"],
        example_queries=["Why is the SEI in SIBs less stable than in LIBs?", "What role do fluorinated additives play in SEI stabilization?"],
        visualization_focus=["sei_composition_map", "electrolyte_comparison"]
    ),
    SIBCoreProblem.SOLID_STATE_INTERFACE: SIBProblemDefinition(
        problem_id=SIBCoreProblem.SOLID_STATE_INTERFACE, title="Solid-State and Semi-Solid Interface Challenges",
        scientific_description="Moving to solid-state SIBs introduces severe interfacial problems.",
        root_cause="Rigid solid electrolytes cannot accommodate volume changes, causing contact loss and dendrite penetration.",
        key_concepts=["solid_electrolyte", "quasi_solid_electrolyte", "interface_contact", "interfacial_resistance", "dendrite_growth", "void_formation"],
        key_relationships=[("solid_electrolyte", "CAUSES", "interfacial_resistance"), ("dendrite_growth", "CAUSES", "short_circuit")],
        solution_directions=["Apply interfacial coating layers", "Design compliant interlayers", "Use quasi-solid/gel electrolytes"],
        relevant_materials=["solid_electrolyte", "quasi_solid_electrolyte", "solid_polymer_electrolyte"],
        relevant_phenomena=["dendrite_growth", "void_formation", "delamination"],
        relevant_properties=["ionic_conductivity", "interfacial_resistance", "cycle_life"],
        example_queries=["What causes void formation at the solid electrolyte-anode interface?", "What strategies suppress sodium dendrite growth?"],
        visualization_focus=["interface_schematic", "dendrite_penetration"]
    ),
    SIBCoreProblem.LOW_ENERGY_DENSITY: SIBProblemDefinition(
        problem_id=SIBCoreProblem.LOW_ENERGY_DENSITY, title="Lower Energy Density Challenge",
        scientific_description="Sodium has a lower standard reduction potential and is heavier than lithium.",
        root_cause="Na/Na⁺ potential is 0.33 V higher than Li/Li⁺, and Na atomic mass is 3.3× that of Li.",
        key_concepts=["energy_density", "specific_capacity", "voltage_plateau", "high_voltage_cathode", "full_cell", "n_p_ratio"],
        key_relationships=[("specific_capacity", "CAUSES", "energy_density"), ("high_voltage_cathode", "INFLUENCES", "energy_density")],
        solution_directions=["Develop high-voltage cathodes (>4.0 V)", "Optimize full-cell design (N/P ratio)", "Explore sodium metal anodes"],
        relevant_materials=["layered_oxide_cathode", "hard_carbon", "sodium_metal", "full_cell"],
        relevant_phenomena=["intercalation", "conversion_reaction"],
        relevant_properties=["energy_density", "specific_capacity", "voltage_plateau"],
        example_queries=["What is the theoretical energy density limit for SIBs vs LIBs?", "Can SIBs ever achieve energy density competitive with LIBs for EVs?"],
        visualization_focus=["energy_density_comparison", "ragone_plot"]
    ),
    SIBCoreProblem.MOISTURE_MANUFACTURING: SIBProblemDefinition(
        problem_id=SIBCoreProblem.MOISTURE_MANUFACTURING, title="Moisture Sensitivity and Manufacturing Challenges",
        scientific_description="Many high-performance sodium cathode materials and salts (NaPF₆) are highly hygroscopic.",
        root_cause="Sodium cathode materials react with atmospheric H₂O/CO₂ to form surface alkaline species disrupting slurry rheology.",
        key_concepts=["moisture_sensitivity", "hygroscopic_materials", "surface_alkalinity", "slurry_gelation", "aqueous_processing", "dry_room_requirements"],
        key_relationships=[("moisture_sensitivity", "CAUSES", "surface_alkalinity"), ("surface_alkalinity", "CAUSES", "slurry_gelation")],
        solution_directions=["Develop moisture-stable cathode compositions", "Apply surface washing treatments", "Use aqueous binders"],
        relevant_materials=["layered_oxide_cathode", "prussian_blue_analogue"],
        relevant_phenomena=["surface_alkalinity", "slurry_gelation"],
        relevant_properties=["coulombic_efficiency", "manufacturing_yield"],
        example_queries=["Why are sodium cathode materials more moisture-sensitive than lithium cathodes?", "What surface washing treatments can stabilize sodium cathodes?"],
        visualization_focus=["moisture_degradation_schematic", "slurry_rheology"]
    ),
    SIBCoreProblem.GENERAL: SIBProblemDefinition(
        problem_id=SIBCoreProblem.GENERAL, title="General SIB Inquiry", scientific_description="General inquiry about sodium-ion batteries.",
        root_cause="N/A", key_concepts=["sodium_ion_battery"], key_relationships=[], solution_directions=[],
        relevant_materials=[], relevant_phenomena=[], relevant_properties=[], example_queries=["What is a sodium-ion battery?"], visualization_focus=["general_overview"]
    ),
    SIBCoreProblem.MULTI_PROBLEM: SIBProblemDefinition(
        problem_id=SIBCoreProblem.MULTI_PROBLEM, title="Multi-Problem SIB Inquiry", scientific_description="Inquiry spanning multiple core SIB problems.",
        root_cause="N/A", key_concepts=[], key_relationships=[], solution_directions=[],
        relevant_materials=[], relevant_phenomena=[], relevant_properties=[], example_queries=[], visualization_focus=["multi_problem_comparison"]
    )
}

@dataclass
class ConceptPriority:
    concept_name: str
    concept_type: str
    composite_score: float
    direct_score: float
    problem_affinity_score: float
    causal_path_score: float
    is_explicitly_mentioned: bool
    is_inferred: bool
    inference_reason: str = ""
    # NEW FIELDS
    ppr_score: float = 0.0
    qc_pmi: float = 0.0
    semantic_resonance: float = 0.0
    cde: float = 0.0
    causal_proximity: float = 0.0

    def to_dict(self) -> Dict:
        return {**self.__dict__, "score": round(self.composite_score, 3)}

@dataclass
class QueryAnalysisResult:
    original_query: str
    normalized_query: str
    primary_problem: SIBCoreProblem
    secondary_problems: List[SIBCoreProblem]
    problem_confidences: Dict[str, float]
    explicitly_mentioned: List[str]
    inferred_concepts: List[str]
    all_relevant_concepts: List[str]
    concept_priorities: Dict[str, ConceptPriority] = field(default_factory=dict)
    query_type: str = "general"
    emphasis_direction: str = "cause"
    comparison_pairs: List[Tuple[str, str]] = field(default_factory=list)
    subgraph_depth: int = 2
    priority_threshold: float = 0.3
    focus_nodes: List[str] = field(default_factory=list)
    bridge_nodes: List[str] = field(default_factory=list)
    suggested_layout: str = "force"
    highlight_paths: List[List[str]] = field(default_factory=list)
    visualization_focus: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def get_top_concepts(self, n: int = 10) -> List[ConceptPriority]:
        return sorted(self.concept_priorities.values(), key=lambda x: x.composite_score, reverse=True)[:n]

    def get_concepts_above_threshold(self, threshold: float = None) -> List[str]:
        thresh = threshold or self.priority_threshold
        return [name for name, cp in self.concept_priorities.items() if cp.composite_score >= thresh]

# ============================================================================
# 2. LLM QUERY ANALYZERS (Abstract + Implementations)
# ============================================================================
class LLMQueryAnalyzer(ABC):
    @abstractmethod
    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult: pass
    @abstractmethod
    def is_available(self) -> bool: pass

class FallbackAnalyzer(LLMQueryAnalyzer):
    PROBLEM_KEYWORDS = {
        SIBCoreProblem.ANODE_BOTTLENECK: {"anode", "hard carbon", "graphite", "intercalation", "alloying", "ice", "initial coulombic efficiency"},
        SIBCoreProblem.CATHODE_INSTABILITY: {"cathode", "layered oxide", "phase transition", "p2", "o2", "o3", "structural", "degradation", "doping"},
        SIBCoreProblem.SEI_CHEMISTRY: {"sei", "solid electrolyte interphase", "electrolyte", "interface", "passivation", "decomposition", "fluorinated"},
        SIBCoreProblem.SOLID_STATE_INTERFACE: {"solid state", "solid electrolyte", "nasicon", "sulfide", "contact", "dendrite", "void", "delamination"},
        SIBCoreProblem.LOW_ENERGY_DENSITY: {"energy density", "wh/kg", "specific energy", "voltage", "capacity", "full cell", "n/p ratio"},
        SIBCoreProblem.MOISTURE_MANUFACTURING: {"moisture", "humidity", "hygroscopic", "surface alkalinity", "slurry", "coating", "manufacturing", "dry room"},
    }
    def is_available(self) -> bool: return True

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        q = query.lower().strip()
        problem_scores = {p: sum(1 for kw in kws if kw in q) for p, kws in self.PROBLEM_KEYWORDS.items()}
        primary = max(problem_scores, key=problem_scores.get) if sum(problem_scores.values()) > 0 else SIBCoreProblem.GENERAL
        secondary = [p for p, s in sorted(problem_scores.items(), key=lambda x: -x[1]) if s > 0 and p != primary][:2]

        explicitly_mentioned = []
        for canonical, node in ontology.concepts.items():
            if canonical.replace("_", " ") in q or any(syn.replace("_", " ") in q for syn in node.synonyms):
                explicitly_mentioned.append(canonical)

        inferred = []
        if primary != SIBCoreProblem.GENERAL:
            pdef = SIB_PROBLEM_DEFINITIONS[primary]
            for concept in pdef.get_ontology_concepts():
                if concept not in explicitly_mentioned and concept in ontology.concepts:
                    inferred.append(concept)

        all_relevant = list(dict.fromkeys(explicitly_mentioned + inferred))
        priorities = {}
        pdef = SIB_PROBLEM_DEFINITIONS.get(primary, SIB_PROBLEM_DEFINITIONS[SIBCoreProblem.GENERAL])
        problem_concept_set = pdef.get_ontology_concepts()

        for concept in all_relevant:
            is_explicit = concept in explicitly_mentioned
            priorities[concept] = ConceptPriority(
                concept_name=concept, concept_type=ontology.get_concept_type(concept).value,
                composite_score=(1.0 if is_explicit else 0.6) * 0.5 + (1.0 if concept in problem_concept_set else 0.4) * 0.5,
                direct_score=1.0 if is_explicit else 0.6, problem_affinity_score=1.0 if concept in problem_concept_set else 0.4,
                causal_path_score=0.5, is_explicitly_mentioned=is_explicit, is_inferred=not is_explicit,
                inference_reason="problem_affinity" if not is_explicit else "explicit_mention"
            )

        query_type = "general"
        if any(w in q for w in ["compare", "vs", "versus", "difference"]): query_type = "comparison"
        elif any(w in q for w in ["why", "cause", "reason", "lead to"]): query_type = "causal"
        elif any(w in q for w in ["how", "improve", "enhance", "optimize", "strategy"]): query_type = "solution"

        highlight_paths = [[src, tgt] for src, rel, tgt in pdef.key_relationships if src in ontology.concepts and tgt in ontology.concepts]
        total = max(sum(problem_scores.values()), 1)
        
        return QueryAnalysisResult(
            original_query=query, normalized_query=q, primary_problem=primary, secondary_problems=secondary,
            problem_confidences={p.value: s / total for p, s in problem_scores.items()},
            explicitly_mentioned=explicitly_mentioned, inferred_concepts=inferred, all_relevant_concepts=all_relevant,
            concept_priorities=priorities, query_type=query_type, emphasis_direction="cause" if query_type == "causal" else "neutral",
            subgraph_depth=2, priority_threshold=0.3, focus_nodes=explicitly_mentioned[:5], bridge_nodes=inferred[:3],
            suggested_layout="force" if query_type != "comparison" else "bisected", highlight_paths=highlight_paths,
            visualization_focus=pdef.visualization_focus, reasoning_chain=[f"Query normalized: '{q}'", f"Primary problem: {primary.value}"],
            confidence=min(sum(problem_scores.values()) / 3.0, 1.0)
        )

class OpenAIQueryAnalyzer(LLMQueryAnalyzer):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self._client = None
        self._pending_new_concepts = []
        self._pending_new_relationships = []

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                st.warning("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool: return bool(self.api_key) and self._get_client() is not None

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        client = self._get_client()
        if client is None: return FallbackAnalyzer().analyze_query(query, ontology)

        concept_list = list(ontology.concepts.keys())[:50]
        system_prompt = """You are an expert Sodium-Ion Battery (SIB) researcher. Analyze the user's query and return ONLY valid JSON with:
        1. "primary_problem": One of: anode_bottleneck, cathode_instability, sei_chemistry, solid_state_interface, low_energy_density, moisture_manufacturing, general, multi_problem
        2. "explicitly_mentioned": List of canonical concept names from the query (use snake_case)
        3. "inferred_concepts": List of additional relevant concepts the query implies
        4. "query_type": One of: causal, comparison, solution, definition, general
        5. "highlight_paths": List of [source, target] concept pairs to highlight
        6. "reasoning_chain": List of strings explaining analysis steps
        7. "new_concepts": List of objects with "name" (snake_case), "type" (material/property/phenomenon/process/method/parameter), "definition", "synonyms" (list)
        8. "new_relationships": List of [source, relationship_type, target, confidence] for NEW relationships between EXISTING concepts."""
        
        try:
            response = client.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Analyze: '{query}'. Available concepts: {', '.join(concept_list)}"}],
                temperature=0.1, max_tokens=1500, response_format={"type": "json_object"}
            )
            parsed = json.loads(response.choices[0].message.content)
            self._pending_new_concepts = parsed.get("new_concepts", [])
            self._pending_new_relationships = parsed.get("new_relationships", [])
            
            # Map to QueryAnalysisResult (simplified for brevity, mirrors Fallback logic but uses LLM output)
            problem_map = {p.value: p for p in SIBCoreProblem}
            primary = problem_map.get(parsed.get("primary_problem", "general"), SIBCoreProblem.GENERAL)
            explicitly_mentioned = [c for c in parsed.get("explicitly_mentioned", []) if c in ontology.concepts]
            inferred = [c for c in parsed.get("inferred_concepts", []) if c in ontology.concepts and c not in explicitly_mentioned]
            
            priorities = {c: ConceptPriority(c, ontology.get_concept_type(c).value, 0.9 if c in explicitly_mentioned else 0.6, 1.0 if c in explicitly_mentioned else 0.5, 0.8, 0.5, c in explicitly_mentioned, c not in explicitly_mentioned, "llm_inferred") for c in list(dict.fromkeys(explicitly_mentioned + inferred))}
            
            return QueryAnalysisResult(
                original_query=query, normalized_query=query.lower().strip(), primary_problem=primary, secondary_problems=[],
                problem_confidences={}, explicitly_mentioned=explicitly_mentioned, inferred_concepts=inferred, all_relevant_concepts=list(dict.fromkeys(explicitly_mentioned + inferred)),
                concept_priorities=priorities, query_type=parsed.get("query_type", "general"), emphasis_direction="cause",
                subgraph_depth=2, priority_threshold=0.3, focus_nodes=explicitly_mentioned[:5], bridge_nodes=inferred[:3],
                suggested_layout="bisected" if parsed.get("query_type") == "comparison" else "force",
                highlight_paths=[[p[0], p[1]] for p in parsed.get("highlight_paths", []) if len(p) >= 2],
                visualization_focus=SIB_PROBLEM_DEFINITIONS[primary].visualization_focus, reasoning_chain=parsed.get("reasoning_chain", ["LLM analysis completed"]), confidence=0.85
            )
        except Exception as e:
            st.warning(f"OpenAI analysis failed ({e}), falling back to rule-based.")
            return FallbackAnalyzer().analyze_query(query, ontology)

class LocalLLMQueryAnalyzer(LLMQueryAnalyzer):
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False
        self._pending_new_concepts = []
        self._pending_new_relationships = []

    def _load_model(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            st.info(f"⏳ Loading local model: `{self.model_name}`… (first run may take 1–2 min)")

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Memory-efficient loading for Streamlit Cloud (≤1 GB RAM)
            load_kwargs: Dict[str, Any] = {}
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                try:
                    load_kwargs["load_in_8bit"] = True
                except Exception:
                    pass
            else:
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = None

            model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,  # Reduced from 1500 to save memory & time
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            self._loaded = True
            st.success(f"✅ Model `{self.model_name}` loaded!")
        except Exception as e:
            st.warning(f"⚠️ Failed to load local model `{self.model_name}`: {e}")
            self._loaded = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_available(self) -> bool:
        self._load_model()
        return self._loaded

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        if not self.is_available():
            return FallbackAnalyzer().analyze_query(query, ontology)
        prompt = (
            f"[INST] You are an SIB expert. Analyze: '{query}'. "
            "Return ONLY valid JSON with: primary_problem, explicitly_mentioned "
            "(snake_case list), inferred_concepts (list), query_type, highlight_paths "
            "(list of [src, tgt]), reasoning_chain (list). [/INST]"
        )
        try:
            result = self._pipeline(prompt)[0]["generated_text"]
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Reuse OpenAI parser logic for consistency
                fake_openai = OpenAIQueryAnalyzer()
                fake_openai._pending_new_concepts = parsed.get("new_concepts", [])
                fake_openai._pending_new_relationships = parsed.get("new_relationships", [])
                return fake_openai.analyze_query(query, ontology)  # Delegate parsing
        except Exception as e:
            st.warning(f"Local LLM parsing failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return FallbackAnalyzer().analyze_query(query, ontology)

class LLMQueryAnalyzerFactory:
    def __init__(self):
        self._openai_cache: Optional[OpenAIQueryAnalyzer] = None
        self._local_cache: Dict[str, LocalLLMQueryAnalyzer] = {}
        self._fallback = FallbackAnalyzer()

    def get_analyzer(self, mode: str = "auto", api_key: str = None, local_model: str = None) -> LLMQueryAnalyzer:
        if mode == "openai":
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            return self._openai_cache
        elif mode == "local":
            model = local_model
            if model is None:
                return self._fallback
            if model not in self._local_cache:
                self._local_cache[model] = LocalLLMQueryAnalyzer(model)
            return self._local_cache[model]
        elif mode == "fallback":
            return self._fallback
        else:  # auto
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            if self._openai_cache.is_available():
                return self._openai_cache
            model = local_model
            if model is None:
                return self._fallback
            if model not in self._local_cache:
                self._local_cache[model] = LocalLLMQueryAnalyzer(model)
            if self._local_cache[model].is_available():
                return self._local_cache[model]
            return self._fallback

# ============================================================================
# 3. DYNAMIC ONTOLOGY EXPANDER
# ============================================================================
class DynamicOntologyExpander:
    REL_STR_TO_ENUM = {r.value: r for r in RelationshipType}
    for _k, _v in list(REL_STR_TO_ENUM.items()): REL_STR_TO_ENUM[_k.upper()] = _v
    TYPE_STR_TO_ENUM = {t.value: t for t in ConceptType}

    def __init__(self, ontology: Any):
        self.ontology = ontology
        self.mutation_log: List[Dict[str, Any]] = []
        self.session_concepts_added: Set[str] = set()
        self.session_relationships_added: List[Tuple[str, str, RelationshipType, float]] = []
        self.query_bridge_concepts: Dict[str, str] = {}
        self.priority_overrides: Dict[str, float] = {}
        self._base_concept_count = len(ontology.concepts)
        self._base_rel_count = len(ontology.relationships)

    @property
    def stats(self) -> Dict[str, int]:
        return {"base_concepts": self._base_concept_count, "base_relationships": self._base_rel_count,
                "concepts_added": len(self.session_concepts_added), "relationships_added": len(self.session_relationships_added),
                "bridge_concepts": len(self.query_bridge_concepts), "total_mutations": len(self.mutation_log)}

    def apply_query_analysis(self, analysis: QueryAnalysisResult, analyzer: LLMQueryAnalyzer = None) -> Dict[str, Any]:
        changes = {"concepts_added": [], "relationships_added": [], "bridges_created": []}
        for concept_name, priority in analysis.concept_priorities.items():
            if concept_name in self.ontology.concepts:
                self.priority_overrides[concept_name] = priority.composite_score

        new_concepts_raw = getattr(analyzer, '_pending_new_concepts', []) if hasattr(analyzer, '_pending_new_concepts') else []
        new_rels_raw = getattr(analyzer, '_pending_new_relationships', []) if hasattr(analyzer, '_pending_new_relationships') else []

        for concept_data in new_concepts_raw:
            result = self._add_concept_from_llm(concept_data, analysis.original_query)
            if result: changes["concepts_added"].append(result)
        for rel_data in new_rels_raw:
            result = self._add_relationship_from_llm(rel_data, analysis.original_query)
            if result: changes["relationships_added"].append(result)

        for concept in analysis.inferred_concepts:
            if concept not in self.ontology.concepts:
                bridge_result = self._create_bridge_concept(concept, analysis.original_query, analysis.primary_problem)
                if bridge_result: changes["bridges_created"].append(bridge_result)
        
        self.ontology._build_synonym_index()
        return changes

    def _add_concept_from_llm(self, concept_data: Dict, source_query: str) -> Optional[Dict]:
        name = concept_data.get("name", "").strip().lower().replace(" ", "_")
        if not name or name in self.ontology.concepts or name in self.session_concepts_added: return None
        concept_type = self.TYPE_STR_TO_ENUM.get(concept_data.get("type", "general"), ConceptType.GENERAL)
        synonyms = set(s.lower().strip() for s in concept_data.get("synonyms", []) if isinstance(s, str))
        definition = concept_data.get("definition", f"LLM-inferred concept from query: {source_query}")
        
        self.ontology._add_concept(name, concept_type, synonyms=synonyms, definition=definition)
        self.ontology.synonym_to_canonical[name.lower()] = name
        for syn in synonyms: self.ontology.synonym_to_canonical[syn] = name
        self.session_concepts_added.add(name)
        
        for rel_tuple in concept_data.get("relate_to", []):
            if len(rel_tuple) >= 2:
                target, rel_type_str = rel_tuple[0], rel_tuple[1] if len(rel_tuple) > 1 else "influences"
                conf = float(rel_tuple[2]) if len(rel_tuple) > 2 else 0.7
                rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
                if target in self.ontology.concepts:
                    self.ontology._add_relationship(name, rel_enum, target, conf)
                    self.session_relationships_added.append((name, target, rel_enum, conf))
        
        self.mutation_log.append({"type": "add_concept", "concept": name, "concept_type": concept_type.value, "source_query": source_query})
        return {"name": name, "type": concept_type.value, "synonyms": list(synonyms)}

    def _add_relationship_from_llm(self, rel_data: List, source_query: str) -> Optional[Dict]:
        if len(rel_data) < 3: return None
        source, rel_type_str, target = str(rel_data[0]).strip().lower().replace(" ", "_"), str(rel_data[1]).upper(), str(rel_data[2]).strip().lower().replace(" ", "_")
        confidence = float(rel_data[3]) if len(rel_data) > 3 else 0.7
        if source not in self.ontology.concepts or target not in self.ontology.concepts: return None
        
        rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
        self.ontology._add_relationship(source, rel_enum, target, confidence)
        self.session_relationships_added.append((source, target, rel_enum, confidence))
        self.mutation_log.append({"type": "add_relationship", "source": source, "target": target, "rel_type": rel_enum.value, "source_query": source_query})
        return {"source": source, "target": target, "rel_type": rel_enum.value, "confidence": confidence}

    def _create_bridge_concept(self, missing_concept: str, source_query: str, problem: SIBCoreProblem) -> Optional[Dict]:
        bridge_name = f"query_bridge_{missing_concept.replace(' ', '_').lower()}"
        if bridge_name in self.ontology.concepts: return None
        pdef = SIB_PROBLEM_DEFINITIONS.get(problem, SIB_PROBLEM_DEFINITIONS[SIBCoreProblem.GENERAL])
        self.ontology._add_concept(bridge_name, ConceptType.GENERAL, synonyms={missing_concept.lower()}, definition=f"Query-inferred bridge: '{missing_concept}'")
        self.ontology.synonym_to_canonical[bridge_name] = bridge_name
        self.ontology.synonym_to_canonical[missing_concept.lower()] = bridge_name
        
        connected = []
        for key_concept in pdef.key_concepts[:3]:
            if key_concept in self.ontology.concepts:
                self.ontology._add_relationship(bridge_name, RelationshipType.BRIDGE, key_concept, 0.5)
                self.session_relationships_added.append((bridge_name, key_concept, RelationshipType.BRIDGE, 0.5))
                connected.append(key_concept)
        self.session_concepts_added.add(bridge_name)
        self.query_bridge_concepts[bridge_name] = source_query
        self.mutation_log.append({"type": "create_bridge", "bridge_name": bridge_name, "original_term": missing_concept, "connected_to": connected})
        return {"bridge": bridge_name, "for": missing_concept, "connected_to": connected}

    def get_priority_boosted_scores(self, base_priorities: Dict[str, ConceptPriority]) -> Dict[str, ConceptPriority]:
        boosted = {}
        for name, priority in base_priorities.items():
            boost = self.priority_overrides.get(name, 0.0)
            if boost > 0:
                bp = copy.deepcopy(priority)
                bp.composite_score = min(bp.composite_score + boost * 0.2, 1.0)
                bp.causal_path_score = boost * 0.2
                boosted[name] = bp
            else:
                boosted[name] = priority
        return boosted

    def undo_last_mutation(self) -> Optional[Dict]:
        if not self.mutation_log: return None
        mutation = self.mutation_log.pop()
        if mutation["type"] == "add_concept":
            name = mutation["concept"]
            if name in self.ontology.concepts:
                del self.ontology.concepts[name]
                self.session_concepts_added.discard(name)
                self.ontology.relationships = [r for r in self.ontology.relationships if r.source != name and r.target != name]
        elif mutation["type"] == "add_relationship":
            self.ontology.relationships = [r for r in self.ontology.relationships if not (r.source == mutation["source"] and r.target == mutation["target"] and r.rel_type.value == mutation["rel_type"])]
        elif mutation["type"] == "create_bridge":
            bridge_name = mutation["bridge_name"]
            if bridge_name in self.ontology.concepts:
                del self.ontology.concepts[bridge_name]
                self.session_concepts_added.discard(bridge_name)
                self.query_bridge_concepts.pop(bridge_name, None)
        self.ontology._build_synonym_index()
        return mutation

    def reset_to_base(self) -> Dict[str, int]:
        for name in list(self.session_concepts_added):
            if name in self.ontology.concepts: del self.ontology.concepts[name]
        self.ontology.relationships = self.ontology.relationships[:self._base_rel_count]
        self.session_concepts_added.clear()
        self.session_relationships_added.clear()
        self.query_bridge_concepts.clear()
        self.priority_overrides.clear()
        self.mutation_log.clear()
        self.ontology._build_synonym_index()
        return {"concepts_removed": len(self.session_concepts_added), "relationships_removed": len(self.ontology.relationships) - self._base_rel_count}

# ============================================================================
# 4. PRIORITY-GUIDED SUBGRAPH EXTRACTOR & VISUALIZER
# ============================================================================
class PriorityGuidedSubgraphExtractor:
    def __init__(self, full_graph: nx.Graph, ontology: Any, expander: DynamicOntologyExpander):
        self.full_graph = full_graph
        self.ontology = ontology
        self.expander = expander

    def extract(self, analysis: QueryAnalysisResult, query_embedding: np.ndarray = None) -> nx.Graph:
        """
        Extract a query-centric subgraph using:
        - Personalized PageRank (topological centering)
        - Query-Conditioned PMI (local co-occurrence)
        - Semantic Resonance (contextual relevance)
        """
        # 1. Seed nodes from analysis
        raw_seed_nodes = set(analysis.focus_nodes + analysis.get_concepts_above_threshold())
        # Filter to nodes actually present in full_graph
        seed_nodes = {n for n in raw_seed_nodes if n in self.full_graph}
        if not seed_nodes:
            # Fallback: use all nodes with priority_score > 0.3
            seed_nodes = {n for n, d in self.full_graph.nodes(data=True)
                          if d.get("priority_score", 0) >= 0.3}

        # 2. Personalized PageRank (PPR)
        personalization = {n: 1.0 if n in seed_nodes else 0.0 for n in self.full_graph.nodes()}
        try:
            ppr_scores = nx.pagerank(self.full_graph, personalization=personalization, alpha=0.85)
        except Exception:
            ppr_scores = {n: 1.0/len(self.full_graph) for n in self.full_graph.nodes()}

        # 3. Query-Conditioned Corpus (D_Q) – placeholder for future extension
        qc_pmi = {}

        # 4. Combine scores: priority_score = 0.6*PPR + 0.4*SRS (if available)
        for node in self.full_graph.nodes():
            ppr = ppr_scores.get(node, 0.0)
            srs = self._compute_semantic_resonance(node, query_embedding) if query_embedding is not None else 0.5
            combined = 0.6 * ppr + 0.4 * srs
            self.full_graph.nodes[node]["priority_score"] = combined
            self.full_graph.nodes[node]["ppr_score"] = ppr
            self.full_graph.nodes[node]["semantic_resonance"] = srs

            # Preserve explicit/inferred flags from analysis
            if node in analysis.concept_priorities:
                cp = analysis.concept_priorities[node]
                self.full_graph.nodes[node]["is_explicit"] = cp.is_explicitly_mentioned
                self.full_graph.nodes[node]["is_inferred"] = cp.is_inferred
            elif node in self.expander.session_concepts_added:
                self.full_graph.nodes[node]["is_explicit"] = False
                self.full_graph.nodes[node]["is_inferred"] = True
                self.full_graph.nodes[node]["is_llm_added"] = True
            else:
                self.full_graph.nodes[node]["is_explicit"] = False
                self.full_graph.nodes[node]["is_inferred"] = False

        # 5. Subgraph extraction: keep nodes with priority_score > threshold (e.g., 0.1)
        threshold = 0.1
        selected_nodes = {n for n, d in self.full_graph.nodes(data=True)
                          if d.get("priority_score", 0) >= threshold}
        # Also include explicit seeds if they fell below threshold
        selected_nodes.update(seed_nodes)

        # Also include 1-hop neighbors of selected nodes if they have high degree
        for node in list(selected_nodes):
            for neighbor in self.full_graph.neighbors(node):
                if self.full_graph.degree(neighbor) > 2:
                    selected_nodes.add(neighbor)

        subgraph = self.full_graph.subgraph(selected_nodes).copy()
        return subgraph

    def _compute_semantic_resonance(self, concept: str, query_emb: np.ndarray) -> float:
        """Simplified SRS: cosine similarity between query embedding and concept embedding."""
        embed_model = st.session_state.get('embed_model')
        if embed_model is None:
            return 0.5
        try:
            concept_emb = embed_model.encode(concept, convert_to_numpy=True)
            sim = np.dot(query_emb, concept_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(concept_emb) + 1e-8)
            return float(np.clip(sim, 0, 1))
        except Exception:
            return 0.5

class QueryDrivenVisualizer:
    def __init__(self, ontology: Any):
        self.ontology = ontology
        self.type_colors = {"material": "#FF6B6B", "property": "#4ECDC4", "phenomenon": "#FFE66D", "method": "#95E1D3", "parameter": "#F38181", "process": "#AA96DA", "model": "#FCBAD3", "general": "#A8D8EA"}

    def render_pyvis(self, subgraph: nx.Graph, analysis: QueryAnalysisResult, height: str = "700px",
                     physics_enabled: bool = True,
                     gravity: float = -800.0,
                     central_gravity: float = 0.1,
                     spring_length: float = 120,
                     spring_strength: float = 0.02,
                     damping: float = 0.95) -> str:
        from pyvis.network import Network
        net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="remote")
        if physics_enabled:
            net.barnes_hut(
                gravity=gravity,
                central_gravity=central_gravity,
                spring_length=spring_length,
                spring_strength=spring_strength,
                damping=damping,
                overlap=0.1
            )
        else:
            net.set_options('{"physics": {"enabled": false}, "interaction": {"hover": true, "dragNodes": true, "dragView": true, "zoomView": true}}')
        for node, attrs in subgraph.nodes(data=True):
            concept_type = attrs.get("concept_type", "general")
            priority = attrs.get("priority_score", 0.2)
            is_explicit = attrs.get("is_explicit", False)
            is_llm_added = attrs.get("is_llm_added", False)
            size = 15 + priority * 35
            color = self.type_colors.get(concept_type, "#A8D8EA")
            if is_explicit: border_width, border_color, shape = 4, "#FF0000", "dot"
            elif is_llm_added: border_width, border_color, shape = 3, "#00FF00", "diamond"
            else: border_width, border_color, shape = 1, "#666666", "dot"
            title = "<b>" + node + "</b><br>Type: " + concept_type + "<br>Priority: " + str(round(priority, 2))
            if is_llm_added: title += "<br>⚠️ LLM-inferred concept"
            defn = attrs.get("definition", "")
            if defn: title += "<br><i>" + defn[:150] + "...</i>"
            net.add_node(node, label=node.replace("_", " ").title(), size=size, color=color, border_width=border_width, border_color=border_color, shape=shape, title=title, font={"size": 10 + priority * 6})
        for u, v, attrs in subgraph.edges(data=True):
            color = attrs.get("color", "#888888")
            width = attrs.get("width", 1.0)
            highlighted = any(len(p) >= 2 and ((p[0] == u and p[1] == v) or (p[1] == u and p[0] == v)) for p in analysis.highlight_paths)
            if highlighted: color, width = "#FF0000", max(width, 4.0)
            net.add_edge(u, v, color=color, width=width, dashes=attrs.get("style") == "dashed" or attrs.get("inferred", False), title=u + " → " + v + "<br>Type: " + attrs.get('edge_type','unknown'), arrows="to")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            return Path(f.name).read_text(encoding='utf-8')
class GraphRAGAnswerGenerator:
    def __init__(self, analyzer: LLMQueryAnalyzer):
        self.analyzer = analyzer

    def generate_ground_response(self, query: str, analysis: QueryAnalysisResult, subgraph: nx.Graph, concept_abstract_map: Dict[str, List[int]], all_texts: Union[List[str], Dict[int, str]], max_docs_per_concept: int = 2) -> str:
        top_nodes = sorted(subgraph.nodes(data=True), key=lambda x: x[1].get("priority_score", 0.0), reverse=True)[:5]
        evidence_snippets = []
        for node, attrs in top_nodes:
            doc_indices = concept_abstract_map.get(node, [])[:max_docs_per_concept]
            for idx in doc_indices:
                if isinstance(all_texts, dict):
                    text = all_texts.get(idx, "")
                else:
                    text = all_texts[idx] if 0 <= idx < len(all_texts) else ""
                if text:
                    clean_text = re.sub(r'\s+', ' ', text).strip()[:400]
                    evidence_snippets.append("- **" + node + "**: " + clean_text + "...")
        nl = chr(10)
        prompt = "You are an expert Sodium-Ion Battery (SIB) researcher. Answer the user's query based *strictly* on the provided graph context and evidence snippets." + nl
        prompt += "User Query: " + repr(query) + nl
        prompt += "Identified Core Problem: " + analysis.primary_problem.value.replace("_", " ").title() + nl
        prompt += "Key Graph Concepts: " + ", ".join([n for n, _ in top_nodes]) + nl
        prompt += "Evidence Snippets from Literature:" + nl
        if evidence_snippets:
            prompt += nl.join(evidence_snippets) + nl
        else:
            prompt += "No direct text snippets found. Rely on your general SIB knowledge but note the lack of specific retrieved context." + nl
        prompt += "Instructions:" + nl
        prompt += "1. Provide a direct, scientifically accurate answer (2-3 paragraphs)." + nl
        prompt += "2. Explicitly mention how the key concepts interact (e.g., causal chains like 'A influences B')." + nl
        prompt += "3. If the retrieved evidence is insufficient, state what specific data is missing."
        if isinstance(self.analyzer, OpenAIQueryAnalyzer) and self.analyzer.is_available():
            return self._call_llm_for_answer(prompt, self.analyzer, query, analysis, top_nodes, evidence_snippets)
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _call_llm_for_answer(self, prompt: str, analyzer: LLMQueryAnalyzer, query: str, analysis: QueryAnalysisResult, top_nodes, evidence_snippets) -> str:
        client = analyzer._get_client()
        if client:
            try:
                response = client.chat.completions.create(
                    model=analyzer.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800
                )
                return response.choices[0].message.content
            except Exception as e:
                fallback_text = self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)
                return "⚠️ LLM API Error: " + str(e) + chr(10) + chr(10) + fallback_text
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _generate_fallback_answer(self, query: str, analysis: Optional[QueryAnalysisResult], top_nodes, snippets: List[str]) -> str:
        nl = chr(10)
        fallback_text = "### Analysis of: '" + query + "'" + nl + nl
        if analysis is not None:
            primary = getattr(analysis, 'primary_problem', None)
            fallback_text += "**Core Problem Identified:** " + (primary.value.replace('_', ' ').title() if primary else 'Unknown') + nl + nl
        else:
            fallback_text += "**Core Problem Identified:** (analysis unavailable)" + nl + nl
        fallback_text += "**Key Concepts in Focus:**" + nl
        fallback_text += nl.join(["- **" + node + "** (" + attrs.get("concept_type", "general") + "): Priority Score " + str(round(attrs.get("priority_score", 0), 2)) for node, attrs in top_nodes])
        if snippets:
            fallback_text += nl + "**Retrieved Evidence Context:**" + nl + nl.join(snippets[:3]) + nl
        else:
            fallback_text += nl + "*Note: No direct text snippets were linked to these concepts in the current dataset.*" + nl
        fallback_text += nl + "**System Reasoning Chain:**" + nl
        if analysis is not None:
            reasoning_chain = getattr(analysis, 'reasoning_chain', [])
            fallback_text += nl.join(["- " + step for step in reasoning_chain])
        else:
            fallback_text += "- No reasoning chain available (analysis was None)." + nl
        return fallback_text

class QuerySessionManager:
    SESSION_KEY = "sib_query_session"
    @classmethod
    def init_session(cls) -> Dict[str, Any]:
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = {"query_history": [], "analysis_history": [], "mutation_history": [], "analyzer_mode": "auto", "total_concepts_added": 0, "total_relationships_added": 0}
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def record_query(cls, query: str, analysis: QueryAnalysisResult, mutations: Dict[str, Any]) -> None:
        session = cls.init_session()
        session["query_history"].append(query)
        session["analysis_history"].append({"query": query, "primary_problem": analysis.primary_problem.value, "query_type": analysis.query_type, "concepts_found": len(analysis.all_relevant_concepts), "explicit": len(analysis.explicitly_mentioned), "inferred": len(analysis.inferred_concepts), "confidence": analysis.confidence, "timestamp": datetime.now().isoformat()})
        session["mutation_history"].append({"query": query, "concepts_added": len(mutations.get("concepts_added", [])), "relationships_added": len(mutations.get("relationships_added", [])), "bridges_created": len(mutations.get("bridges_created", [])), "timestamp": datetime.now().isoformat()})
        session["total_concepts_added"] += len(mutations.get("concepts_added", []))
        session["total_relationships_added"] += len(mutations.get("relationships_added", []))

    @classmethod
    def get_session(cls) -> Dict[str, Any]: return cls.init_session()
    @classmethod
    def clear_session(cls) -> None:
        if cls.SESSION_KEY in st.session_state: del st.session_state[cls.SESSION_KEY]

# ============================================================================
# 7. STREAMLIT UI INTEGRATORS
# ============================================================================
def render_llm_query_panel(ontology: Any, expander: DynamicOntologyExpander, full_graph: nx.Graph) -> Optional[QueryAnalysisResult]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 LLM-Guided Query")
    st.sidebar.caption("Ask a question to dynamically expand the ontology and focus the graph")

    session = QuerySessionManager.get_session()
    mode = st.sidebar.selectbox("Analysis Engine", ["auto", "fallback", "openai", "local"], index=["auto", "fallback", "openai", "local"].index(session.get("analyzer_mode", "auto")), key="llm_mode_select")
    session["analyzer_mode"] = mode

    api_key = None
    if mode in ("auto", "openai"):
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value=os.environ.get("OPENAI_API_KEY", ""), key="openai_key_input")

    # Local model dropdown (memory-safe options for Streamlit Cloud)
    local_model = None
    if mode in ("auto", "local"):
        st.sidebar.markdown("#### 🖥️ Local LLM Model")
        st.sidebar.caption("⚠️ Streamlit Cloud ≈1 GB RAM. Pick a small model or use Fallback.")

        model_display_names = list(LOCAL_LLM_REGISTRY.keys())
        selected_display = st.sidebar.selectbox(
            "Select model:",
            options=model_display_names,
            index=0,  # Default: Fallback
            key="local_model_select",
        )
        local_model = LOCAL_LLM_REGISTRY[selected_display]
        st.session_state['selected_local_model'] = local_model

        if local_model and "TinyLlama" in local_model:
            st.sidebar.warning("⚠️ TinyLlama (1.1B) may OOM on free tier. Use DistilGPT-2 or GPT-Neo-125M for safety.")
        elif local_model and ("0.5B" in selected_display or "560M" in selected_display or "410M" in selected_display):
            st.sidebar.info("ℹ️ 400–500M models work on free tier but load slowly. DistilGPT-2 (82M) is fastest.")

    example_queries = [q for pdef in SIB_PROBLEM_DEFINITIONS.values() for q in pdef.example_queries[:1]]
    selected_example = st.sidebar.selectbox("Or select an example:", [""] + example_queries, key="example_query_select")
    query = st.sidebar.text_area("Your SIB question:", value=selected_example, height=100, key="llm_query_input", placeholder="e.g., Why can't sodium intercalate into graphite like lithium does?")
    
    submitted = st.sidebar.button("🚀 Analyze & Expand Ontology", type="primary", key="llm_submit")
    if not submitted or not query.strip(): return None

    factory = LLMQueryAnalyzerFactory()
    analyzer = factory.get_analyzer(mode=mode, api_key=api_key, local_model=local_model)

    if isinstance(analyzer, OpenAIQueryAnalyzer): st.sidebar.info("🤖 Using **OpenAI GPT-4o-mini**")
    elif isinstance(analyzer, LocalLLMQueryAnalyzer): st.sidebar.info("🖥️ Using **Local LLM**")
    else: st.sidebar.info("📋 Using **Rule-based fallback**")

    with st.sidebar.spinner("Analyzing query..."):
        analysis = analyzer.analyze_query(query, ontology)
    with st.sidebar.spinner("Expanding ontology..."):
        mutations = expander.apply_query_analysis(analysis, analyzer)

    # Store whitelist for query-focused rebuild
    whitelist = set(analysis.explicitly_mentioned)
    whitelist.update(analysis.inferred_concepts)
    whitelist.update(expander.session_concepts_added)
    whitelist.update(expander.query_bridge_concepts.keys())
    st.session_state['last_query_analysis'] = analysis
    st.session_state['last_query_text'] = query
    st.session_state['last_query_whitelist'] = whitelist
    st.session_state['last_query_dynamic_concepts'] = expander.session_concepts_added
    st.session_state['last_query_bridge_concepts'] = expander.query_bridge_concepts

    QuerySessionManager.record_query(query, analysis, mutations)

    st.sidebar.success(f"✅ Analysis complete (confidence: {analysis.confidence:.0%})")
    st.sidebar.caption(f"Primary problem: **{analysis.primary_problem.value}**")
    st.sidebar.caption(f"Explicit concepts: {len(analysis.explicitly_mentioned)} | Inferred: {len(analysis.inferred_concepts)}")
    if mutations["concepts_added"]:
        st.sidebar.warning(f"🆕 {len(mutations['concepts_added'])} new concept(s) added")
        for c in mutations["concepts_added"]: st.sidebar.markdown(f"  - `{c['name']}` ({c['type']})")
    if mutations["bridges_created"]:
        st.sidebar.info(f"🌉 {len(mutations['bridges_created'])} bridge concept(s) created")
        for b in mutations["bridges_created"]: st.sidebar.markdown(f"  - `{b['bridge']}` ← `{b['for']}`")
    return analysis

def render_mutation_controls(expander: DynamicOntologyExpander) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Ontology Mutations")
    stats = expander.stats
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Concepts +", stats["concepts_added"])
    col2.metric("Relations +", stats["relationships_added"])
    if stats["total_mutations"] > 0:
        with st.sidebar.expander("📋 Mutation Log", expanded=False):
            for i, mut in enumerate(expander.mutation_log[-10:], 1):
                if mut["type"] == "add_concept": st.sidebar.markdown(f"{i}. ➕ `{mut['concept']}`")
                elif mut["type"] == "add_relationship": st.sidebar.markdown(f"{i}. 🔗 `{mut['source']}` → `{mut['target']}`")
                elif mut["type"] == "create_bridge": st.sidebar.markdown(f"{i}. 🌉 `{mut['bridge_name']}`")
        col_undo, col_reset = st.sidebar.columns(2)
        if col_undo.button("↩️ Undo Last", key="undo_mutation"):
            undone = expander.undo_last_mutation()
            if undone: st.sidebar.toast(f"Undone: {undone['type']}"); st.rerun()
        if col_reset.button("🔄 Reset All", key="reset_mutations"):
            result = expander.reset_to_base()
            st.sidebar.toast(f"Reset: {result['concepts_removed']} concepts, {result['relationships_removed']} relations removed")
            st.rerun()

def render_query_history() -> None:
    session = QuerySessionManager.get_session()
    if not session["query_history"]: return
    st.sidebar.markdown("---")
    with st.sidebar.expander("📜 Query History", expanded=False):
        for i, entry in enumerate(reversed(session["analysis_history"][-10:]), 1):
            st.sidebar.markdown(f"**{i}.** {entry['query'][:60]}...")
            st.sidebar.caption(f"  Problem: {entry['primary_problem']} | Type: {entry['query_type']} | Concepts: {entry['concepts_found']}")

def render_analysis_details(analysis: QueryAnalysisResult) -> None:
    st.markdown("## 📊 Query Analysis Results")
    with st.expander("🧠 Reasoning Chain", expanded=True):
        for step in analysis.reasoning_chain: st.markdown(f"→ {step}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Primary Problem", analysis.primary_problem.value.replace("_", " "))
    col2.metric("Query Type", analysis.query_type)
    col3.metric("Confidence", f"{analysis.confidence:.0%}")
    
    st.markdown("### Concept Priority Rankings")
    top = analysis.get_top_concepts(15)
    if top:
        df = pd.DataFrame([cp.to_dict() for cp in top])
        def highlight_row(row):
            if row.get("explicit", False): return ["background-color: #d4edda"] * len(row)
            elif row.get("inferred", False): return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

def render_llm_qa_tab(analysis_data: Dict, ontology: Any):
    st.subheader("🤖 LLM-Guided Graph Q&A")
    st.markdown("Ask a specific scientific question. The system will dynamically expand the ontology, extract a relevant subgraph, and generate a grounded answer using retrieved literature snippets.")
    
    if "qa_factory" not in st.session_state: st.session_state.qa_factory = LLMQueryAnalyzerFactory()
    if "qa_expander" not in st.session_state: st.session_state.qa_expander = DynamicOntologyExpander(ontology)
    if "qa_generator" not in st.session_state: st.session_state.qa_generator = GraphRAGAnswerGenerator(st.session_state.qa_factory.get_analyzer("auto"))

    factory = st.session_state.qa_factory
    expander = st.session_state.qa_expander
    generator = st.session_state.qa_generator

    col1, col2 = st.columns([3, 1])
    with col1: query = st.text_input("Enter your SIB research question:", placeholder="e.g., Why does hard carbon suffer from low initial Coulombic efficiency, and how does pre-sodiation help?")
    with col2: mode = st.selectbox("Engine", ["auto", "openai", "local", "fallback"], index=0)
        
    if st.button("🔍 Analyze & Answer", type="primary"):
        if not query.strip(): st.warning("Please enter a query."); return
            
        local_model = st.session_state.get('selected_local_model')
        analyzer = factory.get_analyzer(mode=mode, local_model=local_model)
        generator.analyzer = analyzer
        
        with st.spinner("🧠 Analyzing query and expanding ontology..."):
            analysis = analyzer.analyze_query(query, ontology)
            mutations = expander.apply_query_analysis(analysis, analyzer)

            # Store whitelist for query-focused rebuild
            whitelist = set(analysis.explicitly_mentioned)
            whitelist.update(analysis.inferred_concepts)
            whitelist.update(expander.session_concepts_added)
            whitelist.update(expander.query_bridge_concepts.keys())
            st.session_state['last_query_analysis'] = analysis
            st.session_state['last_query_text'] = query
            st.session_state['last_query_whitelist'] = whitelist
            st.session_state['last_query_dynamic_concepts'] = expander.session_concepts_added
            st.session_state['last_query_bridge_concepts'] = expander.query_bridge_concepts

            if st.session_state.get('query_focused_build'):
                st.success(f"✅ Query analysis complete. Whitelist contains {len(whitelist)} concepts.")
                if st.button("🔧 Rebuild Graph for This Query", type="primary", key="rebuild_for_query_btn"):
                    st.session_state['force_rebuild'] = True
                    st.rerun()

        with st.spinner("🕸️ Extracting priority-guided subgraph..."):
            full_graph = analysis_data["nx_graph"]
            extractor = PriorityGuidedSubgraphExtractor(full_graph, ontology, expander)
            embed_model = analysis_data.get("embed_model")
            if embed_model is not None:
                st.session_state['embed_model'] = embed_model
            query_embedding = None
            if embed_model is not None:
                try:
                    with torch.no_grad():
                        query_embedding = embed_model.encode(query, convert_to_numpy=True)
                except Exception:
                    pass
            subgraph = extractor.extract(analysis, query_embedding)
            
        with st.spinner("📚 Retrieving evidence and generating answer..."):
            answer = generator.generate_ground_response(
                query=query, analysis=analysis, subgraph=subgraph,
                concept_abstract_map=analysis_data["concept_abstract_map"],
                all_texts=analysis_data.get("all_texts", []), # Handles both v6.1 List and Dict formats
                max_docs_per_concept=2
            )
            
        st.markdown("### 💡 Generated Answer")
        st.markdown(answer)
        st.markdown("---")
        st.markdown("---")
        st.markdown("### 🕸️ Focused Subgraph Visualization")
        with st.expander("⚙️ Subgraph Physics Settings (Prevent Jiggling)", expanded=False):
            phys_preset = st.selectbox(
                "Physics Preset",
                ["Stable (No Jiggle)", "Fluid", "Tight", "Off"],
                index=0,
                key="subgraph_phys_preset",
                help="'Stable' uses high damping to stop oscillation. 'Off' freezes the layout."
            )
            presets = {
                "Stable (No Jiggle)": {"gravity": -800, "central_gravity": 0.1, "spring_length": 120, "spring_strength": 0.02, "damping": 0.95},
                "Fluid": {"gravity": -500, "central_gravity": 0.2, "spring_length": 150, "spring_strength": 0.04, "damping": 0.8},
                "Tight": {"gravity": -2000, "central_gravity": 0.3, "spring_length": 80, "spring_strength": 0.08, "damping": 0.6},
                "Off": {"gravity": 0, "central_gravity": 0, "spring_length": 100, "spring_strength": 0, "damping": 0.99},
            }
            p = presets[phys_preset]
            col1, col2 = st.columns(2)
            with col1:
                grav = st.slider("Gravity (Repulsion)", -5000, 0, p["gravity"], step=100, key="sub_grav")
                spring_len = st.slider("Spring Length", 50, 300, p["spring_length"], step=10, key="sub_slen")
                damp = st.slider("Damping (Anti-jiggle)", 0.1, 0.99, p["damping"], step=0.01, key="sub_damp")
            with col2:
                cent_grav = st.slider("Central Gravity", 0.0, 1.0, p["central_gravity"], step=0.05, key="sub_cgrav")
                spring_str = st.slider("Spring Strength", 0.0, 0.5, p["spring_strength"], step=0.01, key="sub_sstr")
                phys_on = st.checkbox("Enable Physics", value=(phys_preset != "Off"), key="sub_phys_on")
        visualizer = QueryDrivenVisualizer(ontology)
        html = visualizer.render_pyvis(
            subgraph, analysis,
            physics_enabled=phys_on,
            gravity=grav,
            central_gravity=cent_grav,
            spring_length=spring_len,
            spring_strength=spring_str,
            damping=damp
        )
        st.components.v1.html(html, height=600, scrolling=True)
        with st.expander("🔧 Behind the Scenes: Ontology Mutations & Reasoning"):
            st.markdown("**Reasoning Chain:**")
            for step in analysis.reasoning_chain: st.markdown("- " + step)
            if mutations.get("concepts_added") or mutations.get("bridges_created"):
                st.markdown("**Dynamic Ontology Updates:**")
                for c in mutations.get("concepts_added", []): st.markdown("➕ Added Concept: `" + c['name'] + "` (" + c['type'] + ")")
                for b in mutations.get("bridges_created", []): st.markdown("🌉 Created Bridge: `" + b['bridge'] + "` for `" + b['for'] + "`")

# ============================================================================
# ONTOLOGY SAFETY HELPER (used by microtransformer)
# ============================================================================
def ensure_ontology_populated() -> "DomainOntology":
    """
    Returns a fully populated DomainOntology instance, re‑initialising it if necessary.
    """
    if "ontology" not in st.session_state or not st.session_state.ontology.concepts:
        st.session_state.ontology = DomainOntology()
    # Double‑check that we have at least one material and one property
    ontology = st.session_state.ontology
    has_material = any(node.concept_type == ConceptType.MATERIAL for node in ontology.concepts.values())
    has_property = any(node.concept_type == ConceptType.PROPERTY for node in ontology.concepts.values())
    if not has_material or not has_property:
        # Force rebuild (should not normally happen)
        st.session_state.ontology = DomainOntology()
    return st.session_state.ontology

# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    st.title(
        "🔋 Sodium-Ion Battery Quantitative Descriptor Graph v6.2 + Microtransformer #2"
    )
    st.caption(
        "Multi-level reasoning concept graph for numerical/quantitative description of Sodium-Ion Batteries | "
        "Memory-Safe | Batch Processing (≤1 GB) | Interactive Visualization | "
        "Ontology-aware resolution | LLM-Guided Q&A | KG-RAG Microtransformer"
    )

    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology

    # Initialize LLM Q&A session state
    if 'qa_factory' not in st.session_state:
        st.session_state.qa_factory = LLMQueryAnalyzerFactory()
    if 'qa_expander' not in st.session_state:
        st.session_state.qa_expander = DynamicOntologyExpander(ontology)
    if 'qa_generator' not in st.session_state:
        st.session_state.qa_generator = GraphRAGAnswerGenerator(st.session_state.qa_factory.get_analyzer("auto"))

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
    force_rebuild = st.session_state.pop("force_rebuild", False)

    # Determine if we should run the pipeline
    should_build = build_clicked or force_rebuild

    if batch_mode_on and (should_build or batch_trigger):
        if force_rebuild and st.session_state.get('query_focused_build'):
            st.warning("Query-focused build is not yet supported in batch mode. Running standard batch analysis.")
        run_batch_analysis(
            df_filtered=df_filtered,
            selected_text_cols=selected_text_cols,
            ontology=ontology,
            run_mode=(batch_trigger or "all"),
        )
    elif should_build:
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

                # Query-focused whitelist support
                whitelist = build_query_whitelist(st.session_state)
                if whitelist is not None:
                    if len(whitelist) <= 15:
                        config["MIN_CONCEPT_FREQ"] = 1
                        st.info("Frequency threshold lowered to 1 for focused query.")
                    else:
                        config["MIN_CONCEPT_FREQ"] = 2
                        st.info(f"Query-focused build: {len(whitelist)} concepts whitelisted. MIN_CONCEPT_FREQ set to {config['MIN_CONCEPT_FREQ']}.")

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

                def _process_single_row(idx, row, allowed_concepts=None):
                    text = " ".join([
                        str(row[col]) for col in selected_text_cols
                        if col in row and pd.notna(row[col])
                    ])
                    concepts = extractor.extract_from_text(text, idx, allowed_concepts=allowed_concepts)
                    metrics: Dict[str, Any] = {}
                    current_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:ma/g|a/g|ma\s*g-1)', text, re.I
                    )
                    if current_matches:
                        metrics['current_density_ma_g'] = [float(m) for m in current_matches]
                    cap_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:mah/g|mah\s*g-1)', text, re.I
                    )
                    if cap_matches:
                        metrics['capacity_mah_g'] = [float(m) for m in cap_matches]
                    density_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:wh/kg|wh\s*kg-1)', text, re.I
                    )
                    if density_matches:
                        metrics['energy_density_wh_kg'] = [float(m) for m in density_matches]
                    voltage_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:v)', text, re.I
                    )
                    if voltage_matches:
                        metrics['voltage_v'] = [float(m) for m in voltage_matches]
                    temp_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', text, re.I
                    )
                    if temp_matches:
                        metrics['temperature'] = [float(m) for m in temp_matches]
                    return idx, concepts, metrics

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(_process_single_row, idx, row, whitelist): idx
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
        tab_names.append("🤖 LLM-Guided Q&A")
        # === MICROTRANSFORMER TAB ===
        tab_names.append("🧠 Microtransformer #2")   # <-- ADD THIS LINE
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
                render_pyvis_graph(
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
                    branchvalues=st.session_state.get('sunburst_branchvalues', 'total'),
                    label_size=st.session_state.get('sunburst_label_size') or 20,
                    width=st.session_state.get('sunburst_width') or 900,
                    height=st.session_state.get('sunburst_height') or 700,
                    show_labels=st.session_state.get('sunburst_show_labels', True),
                    show_values=st.session_state.get('sunburst_show_values', False),
                    hover_info=st.session_state.get('sunburst_hover_info', 'all'),
                    font_family=st.session_state.get('sunburst_font_family', 'Inter, Segoe UI, Roboto, sans-serif'),
                    legend_font_size=st.session_state.get('sunburst_legend_font_size', 12),
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
                    file_name="sib_research_directions.csv", mime="text/csv",
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
                        file_name="sib_graph_publication.png",
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
                    file_name="sib_analysis_report.md",
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
                file_name="sib_concepts_enhanced.csv", mime="text/csv",
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


        # LLM-Guided Q&A Tab
        tab_idx += 1
        with tabs[tab_idx]:
            if st.session_state.analysis_data is not None and "ontology" in st.session_state.analysis_data:
                render_llm_qa_tab(st.session_state.analysis_data, st.session_state.analysis_data["ontology"])
            else:
                st.info("Please build the concept graph with ontology enabled first.")

        # === MICROTRANSFORMER #2 TAB ===
        tab_idx += 1
        with tabs[tab_idx]:
            if st.session_state.analysis_data is not None:
                render_microtransformer_kg_rag_tab(st.session_state.analysis_data, st.session_state.ontology)
            else:
                st.info("Please build the concept graph first to initialize the Microtransformer.")


if __name__ == "__main__":
    main()
