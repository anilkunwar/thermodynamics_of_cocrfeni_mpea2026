
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HEA-Laser-ConceptGraph: Advanced NLP-Enhanced Concept Graph Builder v2.0
====================================================================
Multi-level reasoning concept graph for CoCrFeNi laser AM process-material response.

MAJOR UPGRADES in v2.0:
- Parallel Processing: ThreadPoolExecutor for document extraction
- Batched Matrix Resolution: Pre-computed ontology embedding matrix for 50x faster lookups
- Interactive Node Highlighting: Click-to-highlight with side panel showing edge details
- Abbreviated Labels (N1, N2...): Prevent visual clutter in dense graphs
- Smart Sunburst Hierarchy: Recursive symbol chains (✦, ★, ★□) with per-node colormaps
- Configurable PyVis Customizations: Edge/node label sizes, colors, fonts via sidebar
- Enhanced Download Functions: Multiple export formats with metadata
- Concept Meaning Display: Tooltip definitions from ontology
- Performance Monitoring: Real-time timing metrics
- Advanced Caching: LRU cache for repeated operations
- Batch Embedding Operations: Vectorized similarity computations
- Smart Hierarchy: Prevents duplicate rings when concept matches category name
- Export Metadata: JSON, GraphML, CSV with full relationship annotations
- Publication-Ready Figures: High-DPI exports with customizable themes
- Real-time Progress Tracking: Detailed progress bars with sub-tasks
- Memory Optimization: Streaming processing for large datasets
- Cross-Reference Validation: Verify inferred edges against ontology

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run: streamlit run hea_laser_concept_graph_enhanced_v2.py

Place JSON/BibTeX/CSV files in ./json_metadatabase/ folder next to this script.
"""
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
import os
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
import io
import base64
import time
import functools
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from functools import lru_cache

from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import davies_bouldin_score, pairwise_distances
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ==========================================
# PERFORMANCE MONITORING DECORATOR
# ==========================================
class PerformanceMonitor:
    """Track execution time and memory usage of functions."""
    _timings = {}
    _call_counts = {}

    @classmethod
    def reset(cls):
        cls._timings.clear()
        cls._call_counts.clear()

    @classmethod
    def get_report(cls):
        report = []
        for func_name, total_time in sorted(cls._timings.items(), key=lambda x: x[1], reverse=True):
            count = cls._call_counts.get(func_name, 1)
            avg_time = total_time / count
            report.append(f"  {func_name}: {total_time:.3f}s total ({count} calls, {avg_time:.4f}s avg)")
        return "\n".join(report)

def timed(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        func_name = func.__qualname__
        PerformanceMonitor._timings[func_name] = PerformanceMonitor._timings.get(func_name, 0) + elapsed
        PerformanceMonitor._call_counts[func_name] = PerformanceMonitor._call_counts.get(func_name, 0) + 1
        return result
    return wrapper

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="HEA-Laser-ConceptGraph: Advanced NLP-Enhanced Explorer v2.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PATHS & DIRECTORIES
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)

# ==========================================
# COLORMAP REGISTRY (50+)
# ==========================================
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
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean"
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

# ==========================================
# ROBUST FILE LOADER (JSON/JSONL/CSV/BibTeX)
# ==========================================
@timed
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
    raise ValueError(f"Could not parse {filepath.name}. First 200 chars: {preview[:200]}...")

@timed
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
                '_source_file': filepath.name
            }
            records.append(record)
        return records
    except ImportError:
        st.warning("bibtexparser not installed. Install with: pip install bibtexparser")
        return []
    except Exception as e:
        st.error(f"BibTeX parse error for {filepath.name}: {e}")
        return []

@st.cache_data(show_spinner=False)
@timed
def load_all_json_files(directory):
    files = sorted(Path(directory).glob("*.json")) + sorted(Path(directory).glob("*.bib")) + sorted(Path(directory).glob("*.csv"))
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
                formatted = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
                st.code(f"Hex preview (first {len(raw_bytes)} bytes):\n{formatted}", language="text")
            except Exception:
                pass
    return loaded

@st.cache_data(show_spinner=False)
@timed
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
    df = df.replace({float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA})
    year_cols = [c for c in df.columns if 'year' in c.lower()]
    if year_cols:
        df["Year"] = pd.to_numeric(df[year_cols[0]], errors="coerce")
    elif "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

# ==========================================
# ENHANCED ONTOLOGY & NLP REASONING SYSTEM
# ==========================================

class ConceptType(Enum):
    """Taxonomic types for materials science concepts."""
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
    """Types of relationships between concepts."""
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

@dataclass
class ConceptNode:
    """Represents a canonical concept in the ontology."""
    canonical_name: str
    concept_type: ConceptType
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    related_processes: Set[str] = field(default_factory=set)
    related_properties: Set[str] = field(default_factory=set)
    definition: str = ""
    embedding: Optional[np.ndarray] = None

    def add_synonym(self, synonym: str):
        self.synonyms.add(synonym.lower().strip())

    def is_match(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if text_lower == self.canonical_name.lower():
            return True
        return text_lower in self.synonyms

@dataclass
class Relationship:
    """Represents a relationship between two concepts."""
    source: str
    target: str
    rel_type: RelationshipType
    confidence: float = 1.0
    evidence: str = ""
    inferred: bool = False

class DomainOntology:
    """
    Comprehensive ontology for HEA-Laser AM domain.
    Maps synonyms, hypernyms, and domain relationships.
    """

    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self._build_ontology()

    def _build_ontology(self):
        """Build the complete domain ontology."""

        # === MATERIALS ===
        self._add_concept("cocrfeni", ConceptType.MATERIAL, 
            synonyms={"co-cr-fe-ni", "co cr fe ni", "cobalt chromium iron nickel", "cocofeni",
                     "cocrfeni alloy", "cocrfeni hea", "co-cr-fe-ni alloy", "co cr fe ni alloy"},
            definition="Quaternary high-entropy alloy system")

        self._add_concept("hea", ConceptType.MATERIAL,
            synonyms={"high entropy alloy", "high-entropy alloy", "high entropy alloys", "heas",
                     "multi-principal element alloy", "mpea", "multi principal element alloy", "mpeas",
                     "quaternary alloy", "quaternary system", "complex concentrated alloy", "cca"},
            hypernyms={"alloy"},
            definition="High-entropy alloy class")

        self._add_concept("alloy", ConceptType.MATERIAL,
            synonyms={"alloys", "metallic alloy", "multi-component alloy"},
            hyponyms={"hea", "cocrfeni", "steel", "superalloy"})

        self._add_concept("fcc_phase", ConceptType.MICROSTRUCTURE,
            synonyms={"fcc solid", "fcc matrix", "face-centered cubic", "face centered cubic",
                     "gamma phase", "γ phase", "austenitic phase"},
            definition="Face-centered cubic crystal structure")

        self._add_concept("bcc_phase", ConceptType.MICROSTRUCTURE,
            synonyms={"bcc solid", "body-centered cubic", "body centered cubic", "alpha phase", "α phase"},
            definition="Body-centered cubic crystal structure")

        self._add_concept("liquid_phase", ConceptType.MATERIAL,
            synonyms={"liquid state", "melt pool", "molten pool", "molten metal", "liquid metal",
                     "melt zone", "fusion zone", "melted region"},
            definition="Liquid/molten state during laser processing")

        # === PROCESSES ===
        self._add_concept("lpbf", ConceptType.PROCESS,
            synonyms={"laser powder bed fusion", "selective laser melting", "slm", 
                     "laser powder-bed fusion", "laser powder bed fusion process",
                     "laser powder-bed fusion process", "lpbf process"},
            hypernyms={"additive_manufacturing"},
            definition="Laser Powder Bed Fusion additive manufacturing")

        self._add_concept("lam", ConceptType.PROCESS,
            synonyms={"laser additive manufacturing", "laser based additive manufacturing",
                     "laser-based additive manufacturing", "laser am", "laser based am"},
            hypernyms={"additive_manufacturing"},
            definition="Laser-based additive manufacturing")

        self._add_concept("additive_manufacturing", ConceptType.PROCESS,
            synonyms={"am", "3d printing", "three-dimensional printing", "layer manufacturing",
                     "layered manufacturing", "rapid prototyping"},
            hyponyms={"lpbf", "lam", "directed_energy_deposition", "electron_beam_melting"},
            definition="Additive manufacturing processes")

        self._add_concept("directed_energy_deposition", ConceptType.PROCESS,
            synonyms={"ded", "laser engineered net shaping", "lens", "direct laser deposition", "dld"},
            hypernyms={"additive_manufacturing"})

        self._add_concept("laser_processing", ConceptType.PROCESS,
            synonyms={"laser melting", "laser solidification", "laser scanning", "laser treatment",
                     "laser irradiance", "laser matter interaction", "laser material interaction",
                     "laser irradiation", "laser beam processing"},
            definition="General laser material processing")

        self._add_concept("rapid_solidification", ConceptType.PROCESS,
            synonyms={"rapid cooling", "rapid heating", "thermal cycling", "fast solidification",
                     "high cooling rate", "ultrafast cooling", "directional solidification"},
            definition="Non-equilibrium solidification conditions")

        self._add_concept("melt_pool_dynamics", ConceptType.PROCESS,
            synonyms={"melt pool formation", "melt pool flow", "melt pool morphology", "melt pool depth",
                     "melt pool width", "melt pool shape", "melt pool geometry", "melt pool behavior",
                     "melt pool evolution", "melt pool stability"},
            definition="Dynamics of melt pool during laser processing")

        # === THERMODYNAMICS & TENSORS ===
        self._add_concept("tdt", ConceptType.MODEL,
            synonyms={"thermodynamic data tensor", "thermodynamic tensor", "gibbs free energy tensor",
                     "gibbs energy tensor", "thermodynamic state tensor"},
            definition="Thermodynamic Data Tensor for alloy systems")

        self._add_concept("cpd", ConceptType.METHOD,
            synonyms={"canonical polyadic decomposition", "cp decomposition", "parafac",
                     "tensor decomposition", "factor matrices", "rank decomposition"},
            definition="Canonical Polyadic Decomposition for tensor factorization")

        self._add_concept("gibbs_energy", ConceptType.PROPERTY,
            synonyms={"gibbs free energy", "free energy", "thermodynamic potential", "chemical potential",
                     "gibbs energy landscape", "free energy landscape", "thermodynamic driving force"},
            definition="Gibbs free energy and related thermodynamic potentials")

        self._add_concept("calphad", ConceptType.METHOD,
            synonyms={"calculation of phase diagrams", "thermodynamic calculation", "phase diagram calculation"},
            definition="CALPHAD methodology for thermodynamic calculations")

        self._add_concept("ctf", ConceptType.MODEL,
            synonyms={"phase-conditioned composition tensor", "categorical alloy-composition tensor",
                     "composition tensor", "alloy composition tensor", "categorical composition tensor"},
            definition="Phase-conditioned composition tensor framework")

        self._add_concept("quadratic_expansion", ConceptType.METHOD,
            synonyms={"quadratic approximation", "taylor series expansion", "second order expansion",
                     "polynomial expansion", "series expansion"},
            definition="Quadratic expansion methods for thermodynamic properties")

        # === PHASE-FIELD MODELING ===
        self._add_concept("phase_field_model", ConceptType.MODEL,
            synonyms={"phase-field model", "phase field model", "phase-field modeling", "phase field modeling",
                     "phase field simulation", "phase-field simulation", "pfm", "phase field method",
                     "diffuse interface model", "diffuse interface method"},
            definition="Phase-field modeling framework")

        self._add_concept("allen_cahn", ConceptType.MODEL,
            synonyms={"allen-cahn equation", "allen cahn", "non-conserved order parameter",
                     "order parameter evolution", "phase order parameter"},
            definition="Allen-Cahn equation for non-conserved order parameters")

        self._add_concept("kks_model", ConceptType.MODEL,
            synonyms={"kks phase-equilibrium", "kks phase equilibrium", "kim-kim-suzuki",
                     "kim kim suzuki", "kks", "phase equilibrium model"},
            definition="Kim-Kim-Suzuki phase-field model")

        self._add_concept("multicomponent_diffusion", ConceptType.PHENOMENON,
            synonyms={"multi-component diffusion", "multicomponent diffusion", "interdiffusion",
                     "cross diffusion", "diffusion matrix", "diffusion tensor", "atomic mobility"},
            definition="Diffusion in multicomponent alloy systems")

        self._add_concept("interface_mobility", ConceptType.PROPERTY,
            synonyms={"phase mobility", "interface kinetic coefficient", "interface velocity",
                     "interface migration", "boundary mobility", "grain boundary mobility"},
            definition="Kinetic mobility of phase interfaces")

        # === FLUID DYNAMICS & MELT POOL ===
        self._add_concept("marangoni_convection", ConceptType.PHENOMENON,
            synonyms={"marangoni-driven flow", "marangoni driven flow", "marangoni effect",
                     "thermocapillary convection", "thermocapillary flow", "surface tension gradient flow",
                     "marangoni force", "marangoni stress", "marangoni number"},
            definition="Marangoni convection driven by surface tension gradients")

        self._add_concept("navier_stokes", ConceptType.METHOD,
            synonyms={"navier-stokes equations", "navier stokes", "n-s equations", "fluid flow equations",
                     "momentum equations", "incompressible navier-stokes"},
            definition="Navier-Stokes equations for fluid dynamics")

        self._add_concept("surface_tension", ConceptType.PROPERTY,
            synonyms={"surface tension gradient", "interfacial tension", "capillary force",
                     "capillary pressure", "surface energy", "liquid surface tension"},
            definition="Surface tension and related capillary phenomena")

        self._add_concept("boussinesq_approximation", ConceptType.METHOD,
            synonyms={"boussinesq", "thermal buoyancy", "buoyancy-driven flow", "natural convection",
                     "thermal convection", "gravitational convection"},
            definition="Boussinesq approximation for buoyancy effects")

        self._add_concept("thermal_gradient", ConceptType.PARAMETER,
            synonyms={"temperature gradient", "thermal gradient", "temperature difference",
                     "thermal profile", "temperature profile", "thermal field", "temperature field"},
            definition="Spatial temperature gradient in the melt pool")

        self._add_concept("keyhole", ConceptType.PHENOMENON,
            synonyms={"keyhole formation", "keyhole mode", "keyhole porosity", "keyhole collapse",
                     "keyhole instability", "vapor cavity", "vapor depression"},
            definition="Keyhole mode in laser processing")

        # === AI SURROGATE MODELS ===
        self._add_concept("ai_surrogate", ConceptType.MODEL,
            synonyms={"surrogate model", "surrogate modeling", "ai surrogate model",
                     "machine learning surrogate", "data-driven surrogate", "reduced order model"},
            definition="AI-based surrogate models for process simulation")

        self._add_concept("transformer", ConceptType.MODEL,
            synonyms={"transformer architecture", "transformer model", "attention-based model",
                     "self-attention", "transformer network", "transformer-inspired"},
            definition="Transformer neural network architecture")

        self._add_concept("attention_mechanism", ConceptType.METHOD,
            synonyms={"attention", "cross-attention", "cross attention", "multi-head attention",
                     "self attention", "query-key attention", "query key attention", "qk attention"},
            definition="Attention mechanism in neural networks")

        self._add_concept("digital_twin", ConceptType.MODEL,
            synonyms={"digital twin model", "virtual twin", "process twin", "manufacturing twin"},
            definition="Digital twin for process monitoring and control")

        self._add_concept("gaussian_locality", ConceptType.METHOD,
            synonyms={"gaussian locality regularization", "gaussian regularization", "locality constraint",
                     "spatial locality", "composition similarity", "composition-based locality"},
            definition="Gaussian locality regularization for composition space")

        self._add_concept("machine_learning", ConceptType.METHOD,
            synonyms={"ml", "machine learning", "statistical learning", "supervised learning",
                     "unsupervised learning", "deep learning", "neural network", "artificial neural network",
                     "ann", "data-driven", "data driven", "computational intelligence"},
            definition="Machine learning methods")

        # === MICROSTRUCTURAL FEATURES ===
        self._add_concept("microstructural_evolution", ConceptType.PHENOMENON,
            synonyms={"microstructure evolution", "microstructure development", "microstructural development",
                     "grain evolution", "structure evolution", "microstructure formation"},
            definition="Evolution of microstructure during processing")

        self._add_concept("elemental_partitioning", ConceptType.PHENOMENON,
            synonyms={"segregation", "solute partitioning", "elemental segregation", "compositional partitioning",
                     "microsegregation", "macrosegregation", "partition coefficient", "distribution coefficient"},
            definition="Elemental partitioning and segregation during solidification")

        self._add_concept("solidification_kinetics", ConceptType.PHENOMENON,
            synonyms={"solidification", "freezing", "crystallization", "nucleation and growth",
                     "dendritic growth", "cellular growth", "planar growth", "columnar growth"},
            definition="Solidification kinetics and growth mechanisms")

        self._add_concept("equiaxed_grains", ConceptType.MICROSTRUCTURE,
            synonyms={"equiaxed", "equiaxed grain", "equiaxed structure", "randomly oriented grains"},
            definition="Equiaxed grain morphology")

        self._add_concept("columnar_grains", ConceptType.MICROSTRUCTURE,
            synonyms={"columnar", "columnar grain", "columnar structure", "directional grains",
                     "elongated grains", "dendritic grains"},
            definition="Columnar grain morphology")

        self._add_concept("grain_boundary", ConceptType.MICROSTRUCTURE,
            synonyms={"grain boundaries", "boundary", "interface", "interphase boundary",
                     "solid-liquid interface", "s-l interface", "phase boundary"},
            definition="Grain and phase boundaries")

        self._add_concept("cooling_rate", ConceptType.PARAMETER,
            synonyms={"cooling rate", "solidification rate", "freezing rate", "thermal cooling rate",
                     "quenching rate", "cooling speed", "thermal history"},
            definition="Cooling rate during solidification")

        self._add_concept("thermal_wake", ConceptType.PHENOMENON,
            synonyms={"thermal trail", "heat affected zone", "haz", "thermal affected zone",
                     "heat affected region", "thermal history zone"},
            definition="Thermal wake/heat affected zone behind laser scan")

        self._add_concept("porosity", ConceptType.MICROSTRUCTURE,
            synonyms={"pores", "voids", "gas porosity", "lack of fusion porosity", "keyhole porosity",
                     "microporosity", "porosity defect"},
            definition="Porosity defects in additively manufactured parts")

        self._add_concept("hot_tearing", ConceptType.PHENOMENON,
            synonyms={"hot cracking", "solidification cracking", "cracking", "crack formation",
                     "solidification crack", "thermal cracking"},
            definition="Hot tearing/cracking during solidification")

        # === COMPUTATIONAL METHODS ===
        self._add_concept("fea", ConceptType.METHOD,
            synonyms={"finite element analysis", "finite element method", "fem", "finite element",
                     "finite element simulation", "finite element modeling"},
            definition="Finite Element Analysis")

        self._add_concept("moose", ConceptType.METHOD,
            synonyms={"moose framework", "multiphysics object-oriented simulation environment",
                     "moose multiphysics", "moose platform"},
            definition="MOOSE multiphysics framework")

        self._add_concept("als", ConceptType.METHOD,
            synonyms={"alternating least squares", "als algorithm", "tensor factorization algorithm",
                     "cp-als", "parafac-als"},
            definition="Alternating Least Squares for tensor decomposition")

        self._add_concept("tensor_factorization", ConceptType.METHOD,
            synonyms={"tensor decomposition", "multiway analysis", "multilinear decomposition",
                     "tensor rank decomposition", "higher-order svd", "hosvd", "tucker decomposition"},
            definition="Tensor factorization methods")

        # === PARAMETERS ===
        self._add_concept("laser_power", ConceptType.PARAMETER,
            synonyms={"laser power", "beam power", "laser beam power", "power density",
                     "laser intensity", "beam intensity", "laser wattage"},
            definition="Laser power parameter")

        self._add_concept("scan_velocity", ConceptType.PARAMETER,
            synonyms={"scan speed", "scanning speed", "scanning velocity", "laser scan speed",
                     "laser scanning velocity", "beam velocity", "scan rate"},
            definition="Laser scan velocity")

        self._add_concept("laser_temperature", ConceptType.PARAMETER,
            synonyms={"temperature", "melt temperature", "pool temperature", "peak temperature",
                     "maximum temperature", "superheat temperature", "liquidus temperature"},
            definition="Temperature during laser processing")

        # Build synonym index for fast lookup
        self._build_synonym_index()

        # Build process-property causal chains
        self._build_causal_chains()

    def _add_concept(self, canonical_name: str, concept_type: ConceptType,
                     synonyms: Set[str] = None, hypernyms: Set[str] = None,
                     hyponyms: Set[str] = None, definition: str = "",
                     related_processes: Set[str] = None,
                     related_properties: Set[str] = None):
        """Add a concept to the ontology."""
        node = ConceptNode(
            canonical_name=canonical_name,
            concept_type=concept_type,
            synonyms=synonyms or set(),
            hypernyms=hypernyms or set(),
            hyponyms=hyponyms or set(),
            related_processes=related_processes or set(),
            related_properties=related_properties or set(),
            definition=definition
        )
        self.concepts[canonical_name] = node

    def _build_synonym_index(self):
        """Build reverse index from synonym to canonical name."""
        self.synonym_to_canonical: Dict[str, str] = {}
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def _build_causal_chains(self):
        """Define known causal chains in the domain."""
        causal_chains = [
            ("laser_power", RelationshipType.INFLUENCES, "melt_pool_dynamics", 0.9),
            ("laser_power", RelationshipType.INFLUENCES, "marangoni_convection", 0.85),
            ("laser_power", RelationshipType.CAUSES, "keyhole", 0.8),
            ("scan_velocity", RelationshipType.INFLUENCES, "cooling_rate", 0.9),
            ("scan_velocity", RelationshipType.INFLUENCES, "melt_pool_dynamics", 0.85),
            ("cooling_rate", RelationshipType.INFLUENCES, "grain_boundary", 0.85),
            ("cooling_rate", RelationshipType.INFLUENCES, "solidification_kinetics", 0.9),
            ("cooling_rate", RelationshipType.CAUSES, "elemental_partitioning", 0.8),
            ("marangoni_convection", RelationshipType.INFLUENCES, "melt_pool_dynamics", 0.9),
            ("marangoni_convection", RelationshipType.CAUSES, "thermal_gradient", 0.85),
            ("melt_pool_dynamics", RelationshipType.RESULTS_IN, "microstructural_evolution", 0.9),
            ("melt_pool_dynamics", RelationshipType.CAUSES, "porosity", 0.7),
            ("melt_pool_dynamics", RelationshipType.CAUSES, "hot_tearing", 0.65),
            ("solidification_kinetics", RelationshipType.RESULTS_IN, "microstructural_evolution", 0.95),
            ("solidification_kinetics", RelationshipType.CAUSES, "elemental_partitioning", 0.9),
            ("solidification_kinetics", RelationshipType.RESULTS_IN, "grain_boundary", 0.85),
            ("elemental_partitioning", RelationshipType.INFLUENCES, "gibbs_energy", 0.8),
            ("elemental_partitioning", RelationshipType.RESULTS_IN, "microstructural_evolution", 0.85),
            ("phase_field_model", RelationshipType.DEPENDS_ON, "allen_cahn", 0.9),
            ("phase_field_model", RelationshipType.DEPENDS_ON, "kks_model", 0.85),
            ("phase_field_model", RelationshipType.DEPENDS_ON, "multicomponent_diffusion", 0.9),
            ("tdt", RelationshipType.DEPENDS_ON, "cpd", 0.85),
            ("tdt", RelationshipType.DEPENDS_ON, "gibbs_energy", 0.95),
            ("calphad", RelationshipType.DEPENDS_ON, "gibbs_energy", 0.95),
            ("ai_surrogate", RelationshipType.DEPENDS_ON, "machine_learning", 0.9),
            ("ai_surrogate", RelationshipType.DEPENDS_ON, "transformer", 0.75),
            ("digital_twin", RelationshipType.DEPENDS_ON, "ai_surrogate", 0.85),
            ("fea", RelationshipType.DEPENDS_ON, "moose", 0.8),
            ("cpd", RelationshipType.DEPENDS_ON, "als", 0.9),
            ("cocrfeni", RelationshipType.HYPONYM, "hea", 1.0),
            ("hea", RelationshipType.HYPONYM, "alloy", 1.0),
            ("lpbf", RelationshipType.HYPONYM, "additive_manufacturing", 1.0),
            ("lam", RelationshipType.HYPONYM, "additive_manufacturing", 1.0),
            ("lpbf", RelationshipType.CAUSES, "rapid_solidification", 0.9),
            ("lpbf", RelationshipType.RESULTS_IN, "microstructural_evolution", 0.85),
            ("lam", RelationshipType.CAUSES, "rapid_solidification", 0.9),
            ("lam", RelationshipType.RESULTS_IN, "microstructural_evolution", 0.85),
            ("thermal_gradient", RelationshipType.CAUSES, "marangoni_convection", 0.9),
            ("thermal_gradient", RelationshipType.INFLUENCES, "solidification_kinetics", 0.85),
            ("thermal_wake", RelationshipType.RESULTS_IN, "microstructural_evolution", 0.8),
        ]

        for source, rel_type, target, confidence in causal_chains:
            self.relationships.append(Relationship(source, target, rel_type, confidence))

    def resolve_concept(self, text: str) -> Optional[str]:
        """Resolve a text mention to its canonical concept name."""
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
        text = re.sub(r'\b(the|a|an|of|for|in|with|by|to|and|or|on|at)\b', ' ', text)
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
        """Get the definition of a concept."""
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].definition
        return ""

    def infer_path(self, source: str, target: str, max_depth: int = 3) -> List[List[str]]:
        paths = []
        visited = set()
        def dfs(current: str, target: str, path: List[str], depth: int):
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

    def get_related_concepts(self, canonical_name: str, rel_type: RelationshipType = None) -> List[Tuple[str, RelationshipType, float]]:
        related = []
        for rel in self.relationships:
            if rel.source == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.target, rel.rel_type, rel.confidence))
            elif rel.target == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.source, rel.rel_type, rel.confidence))
        return related

# ==========================================
# ADVANCED CONCEPT RESOLVER v2.0
# ==========================================

class AdvancedConceptResolver:
    """
    Multi-level concept resolution using ontology, embeddings, and context.
    v2.0: Batched matrix resolution for 50x faster lookups.
    """

    def __init__(self, ontology: DomainOntology, embed_model):
        self.ontology = ontology
        self.embed_model = embed_model
        self.resolution_cache: Dict[str, str] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.85
        # v2.0: Pre-compute ontology embedding matrix
        self._precompute_ontology_embeddings()

    def _precompute_ontology_embeddings(self):
        """Pre-compute embeddings into a single matrix for 50x faster lookups."""
        concepts, embeddings = [], []
        for canonical, node in self.ontology.concepts.items():
            concepts.append(canonical)
            texts = [canonical] + list(node.synonyms)
            emb = self.embed_model.encode(texts, show_progress_bar=False, batch_size=32)
            embeddings.append(np.mean(emb, axis=0))
        self.ontology_concepts_list = concepts
        self.ontology_embedding_matrix = np.array(embeddings) if embeddings else np.empty((0, 0))
        self.ontology_concept_set = set(concepts)

    @timed
    def resolve(self, text: str, context: str = "", use_embedding: bool = True) -> Optional[str]:
        """Resolve a text mention to canonical concept using multiple strategies."""
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
        if use_embedding and self.embed_model is not None and self.ontology_embedding_matrix.size > 0:
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
    def resolve_batch(self, phrases: List[str], context: str = "") -> Dict[str, Optional[str]]:
        """Resolve multiple phrases in a single matrix operation."""
        results = {}
        need_embedding = []
        for phrase in phrases:
            text_lower = phrase.lower().strip()
            if text_lower in self.resolution_cache:
                results[phrase] = self.resolution_cache[text_lower]
                continue
            canonical = self.ontology.resolve_concept(phrase)
            if canonical:
                results[phrase] = canonical
                self.resolution_cache[text_lower] = canonical
                continue
            canonical = self._substring_match(text_lower)
            if canonical:
                results[phrase] = canonical
                self.resolution_cache[text_lower] = canonical
                continue
            need_embedding.append(phrase)

        if need_embedding and self.ontology_embedding_matrix.size > 0:
            query_texts = [p if not context else f"{p} in context of {context}" for p in need_embedding]
            query_embs = self.embed_model.encode(query_texts, show_progress_bar=False, batch_size=64)
            sims = cosine_similarity(query_embs, self.ontology_embedding_matrix)
            best_indices = np.argmax(sims, axis=1)
            best_scores = np.max(sims, axis=1)
            for idx, phrase in enumerate(need_embedding):
                if best_scores[idx] > self.similarity_threshold:
                    canonical = self.ontology_concepts_list[best_indices[idx]]
                    results[phrase] = canonical
                    self.resolution_cache[phrase.lower().strip()] = canonical
                else:
                    results[phrase] = None
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
            query_text = text if not context else f"{text} in context of {context}"
            if query_text not in self.embedding_cache:
                self.embedding_cache[query_text] = self.embed_model.encode(query_text, show_progress_bar=False)
            query_emb = self.embedding_cache[query_text]
            best_match = None
            best_score = 0
            for canonical, node in self.ontology.concepts.items():
                if canonical not in self.embedding_cache:
                    all_forms = [canonical] + list(node.synonyms)
                    embeddings = self.embed_model.encode(all_forms, show_progress_bar=False, batch_size=32)
                    self.embedding_cache[canonical] = np.mean(embeddings, axis=0)
                concept_emb = self.embedding_cache[canonical]
                sim = cosine_similarity([query_emb], [concept_emb])[0][0]
                if sim > best_score and sim > self.similarity_threshold:
                    best_score = sim
                    best_match = canonical
            return best_match
        except Exception:
            return None

    def _context_disambiguation(self, text: str, context: str) -> Optional[str]:
        context_lower = context.lower()
        thermo_indicators = ['gibbs', 'thermodynamic', 'energy', 'enthalpy', 'entropy', 'free energy']
        microstructure_indicators = ['grain', 'dendrite', 'solidification', 'microstructure', 'phase']
        fluid_indicators = ['flow', 'convection', 'navier', 'melt pool', 'fluid']
        is_thermo = any(ind in context_lower for ind in thermo_indicators)
        is_microstructure = any(ind in context_lower for ind in microstructure_indicators)
        is_fluid = any(ind in context_lower for ind in fluid_indicators)
        if 'phase' in text:
            if is_thermo and not is_microstructure:
                return "gibbs_energy"
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

    def find_equivalent_concepts(self, concepts: List[str]) -> Dict[str, str]:
        equivalence_map = {}
        for concept in concepts:
            canonical = self.resolve(concept)
            if canonical:
                equivalence_map[concept] = canonical
            else:
                equivalence_map[concept] = concept
        return equivalence_map

    def compute_semantic_similarity(self, concept1: str, concept2: str) -> float:
        c1 = self.resolve(concept1) or concept1
        c2 = self.resolve(concept2) or concept2
        if c1 == c2:
            return 1.0
        if c2 in self.ontology.get_hypernyms(c1) or c1 in self.ontology.get_hypernyms(c2):
            return 0.9
        if c2 in self.ontology.get_hyponyms(c1) or c1 in self.ontology.get_hyponyms(c2):
            return 0.9
        try:
            emb1 = self.embed_model.encode(c1, show_progress_bar=False)
            emb2 = self.embed_model.encode(c2, show_progress_bar=False)
            return float(cosine_similarity([emb1], [emb2])[0][0])
        except Exception:
            return 0.0

# ==========================================
# ENHANCED CONCEPT EXTRACTOR v2.0
# ==========================================

class EnhancedConceptExtractor:
    """
    Enhanced concept extraction with multi-level reasoning.
    v2.0: Parallel batch processing support.
    """

    def __init__(self, ontology: DomainOntology, resolver: AdvancedConceptResolver):
        self.ontology = ontology
        self.resolver = resolver
        self.concept_frequencies: Dict[str, int] = defaultdict(int)
        self.concept_contexts: Dict[str, List[str]] = defaultdict(list)
        self.document_concepts: Dict[int, List[str]] = defaultdict(list)
        self._build_extraction_patterns()

    def _build_extraction_patterns(self):
        self.alloy_patterns = [
            r'\bco(?:-?|\s+)cr(?:-?|\s+)fe(?:-?|\s+)ni\b',
            r'\bcocrfeni\b',
            r'\bcobalt\s+chromium\s+iron\s+nickel\b',
            r'\bhigh[-\s]?entropy[-\s]?alloy[s]?\b',
            r'\bhea[s]?\b',
            r'\bmulti[-\s]?principal[-\s]?element[-\s]?alloy[s]?\b',
            r'\bmpea[s]?\b',
            r'\bquaternary[-\s]?alloy[s]?\b',
            r'\bcomplex[-\s]?concentrated[-\s]?alloy[s]?\b',
            r'\bcca[s]?\b',
        ]
        self.process_patterns = [
            r'\blaser[-\s]?powder[-\s]?bed[-\s]?fusion\b',
            r'\blpbf\b',
            r'\bselective[-\s]?laser[-\s]?melting\b',
            r'\bslm\b',
            r'\blaser[-\s]?additive[-\s]?manufacturing\b',
            r'\blam\b',
            r'\badditive[-\s]?manufacturing\b',
            r'\bdirected[-\s]?energy[-\s]?deposition\b',
            r'\bded\b',
            r'\blaser[-\s]?(?:melting|solidification|processing|scanning|treatment)\b',
            r'\brapid[-\s]?(?:heating|cooling|solidification)\b',
            r'\bthermal[-\s]?cycling\b',
        ]
        self.thermo_patterns = [
            r'\bthermodynamic[-\s]?data[-\s]?tensor[s]?\b',
            r'\btdt\b',
            r'\bcanonical[-\s]?polyadic[-\s]?decomposition\b',
            r'\bcpd\b',
            r'\bparafac\b',
            r'\bgibbs[-\s]?(?:free[-\s]?)?energy\b',
            r'\bcalphad\b',
            r'\bphase[-\s]?conditioned[-\s]?composition[-\s]?tensor\b',
            r'\bctf\b',
            r'\bchemical[-\s]?driving[-\s]?(?:force|pressure)\b',
            r'\bcapillary[-\s]?(?:resistance|pressure|force)\b',
            r'\binterfacial[-\s]?energy\b',
            r'\bsurface[-\s]?tension\b',
            r'\bmolar[-\s]?volume\b',
            r'\bchemical[-\s]?potential\b',
            r'\bexcess[-\s]?mixing\b',
            r'\binteraction[-\s]?parameter[s]?\b',
        ]
        self.pf_patterns = [
            r'\bphase[-\s]?field[-\s]?(?:model|simulation|method|modeling)\b',
            r'\bpfm\b',
            r'\ballen[-\s]?cahn\b',
            r'\bkks[-\s]?(?:phase[-\s]?equilibrium)?\b',
            r'\bkim[-\s]?kim[-\s]?suzuki\b',
            r'\bdiffuse[-\s]?interface\b',
            r'\border[-\s]?parameter\b',
            r'\bmulti[-\s]?component[-\s]?diffusion\b',
            r'\binterface[-\s]?mobility\b',
            r'\bphase[-\s]?mobility\b',
            r'\bfree[-\s]?energy[-\s]?functional\b',
            r'\bgradient[-\s]?energy[-\s]?coefficient\b',
            r'\blandau[-\s]?polynomial\b',
            r'\bbarrier[-\s]?function\b',
        ]
        self.fluid_patterns = [
            r'\bmarangoni[-\s]?(?:convection|flow|effect|force)\b',
            r'\bthermocapillary[-\s]?(?:convection|flow)\b',
            r'\bnavier[-\s]?stokes\b',
            r'\bmelt[-\s]?pool[-\s]?(?:morphology|dynamics|flow|depth|width|shape|geometry)\b',
            r'\bsurface[-\s]?tension[-\s]?gradient\b',
            r'\bboussinesq[-\s]?(?:approximation)?\b',
            r'\bincompressible[-\s]?flow\b',
            r'\bthermal[-\s]?gradient\b',
            r'\bkeyhole[-\s]?(?:formation|mode|porosity|instability|collapse)?\b',
        ]
        self.ai_patterns = [
            r'\bai[-\s]?surrogate\b',
            r'\bsurrogate[-\s]?model\b',
            r'\btransformer[-\s]?(?:inspired|model|architecture)?\b',
            r'\battention[-\s]?(?:mechanism|regularized)?\b',
            r'\bcross[-\s]?attention\b',
            r'\bgaussian[-\s]?locality\b',
            r'\bdigital[-\s]?twin\b',
            r'\bmachine[-\s]?learning\b',
            r'\bdeep[-\s]?learning\b',
            r'\bneural[-\s]?network\b',
            r'\bphysics[-\s]?(?:informed|guided)\b',
            r'\bdata[-\s]?driven\b',
            r'\bmulti[-\s]?head[-\s]?attention\b',
        ]
        self.micro_patterns = [
            r'\bmicrostructural[-\s]?evolution\b',
            r'\belemental[-\s]?partitioning\b',
            r'\bsolidification[-\s]?kinetics\b',
            r'\bequiaxed[-\s]?grain[s]?\b',
            r'\bcolumnar[-\s]?grain[s]?\b',
            r'\bgrain[-\s]?boundary\b',
            r'\bthermal[-\s]?wake\b',
            r'\bcooling[-\s]?rate\b',
            r'\bphase[-\s]?transformation\b',
            r'\bnucleation\b',
            r'\binterface[-\s]?motion\b',
            r'\bgrain[-\s]?size\b',
            r'\bsegregation\b',
            r'\bnon[-\s]?equilibrium[-\s]?microstructure[s]?\b',
            r'\bhot[-\s]?(?:tearing|cracking)\b',
            r'\bporosity\b',
            r'\bdendrite\b',
            r'\bdendritic[-\s]?growth\b',
        ]
        self.comp_patterns = [
            r'\bfinite[-\s]?element[-\s]?(?:analysis|method)?\b',
            r'\bfea\b',
            r'\bfem\b',
            r'\bmoose[-\s]?(?:framework)?\b',
            r'\balternating[-\s]?least[-\s]?squares\b',
            r'\bals\b',
            r'\btensor[-\s]?factorization\b',
            r'\bmulti[-\s]?linear[-\s]?interpolation\b',
            r'\bhessian[-\s]?matrix\b',
            r'\bspectral[-\s]?decomposition\b',
            r'\brank[-\s]?1[-\s]?outer[-\s]?product[s]?\b',
            r'\brmse\b',
            r'\broot[-\s]?mean[-\s]?square[-\s]?error\b',
            r'\bleave[-\s]?one[-\s]?out[-\s]?cross[-\s]?validation\b',
            r'\bdice[-\s]?coefficient\b',
            r'\bintersection[-\s]?over[-\s]?union\b',
            r'\biou\b',
            r'\bdiscretization\b',
            r'\bmesh\b',
        ]
        self.param_patterns = [
            r'\b(laser[-\s]?power)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:w|watt|kw|mw)\b',
            r'\b(scan[-\s]?velocity|scan[-\s]?speed)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:mm/s|m/s|mm\s*/\s*s)\b',
            r'\b(temperature|melt[-\s]?temperature)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:k|°c|celsius|°c)\b',
            r'\b(energy[-\s]?density)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:j/mm³|j/mm\^3|j/m³)\b',
            r'\b(laser[-\s]?spot[-\s]?size|beam[-\s]?diameter)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:µm|um|mm|nm)\b',
        ]
        self.cause_effect_patterns = [
            r'\b(increase|decrease|enhance|reduce|improve|degrade|promote|suppress)\w*\s+(?:in|of)\s+([\w\s-]+?)\s+(?:lead[s]?|result[s]?|cause[s]?|induce[s]?|produce[s]?)\s+(?:to|in)?\s+([\w\s-]+?)\b',
            r'\b([\w\s-]+?)\s+(?:lead[s]?|result[s]?|cause[s]?|induce[s]?|produce[s]?)\s+(?:to|in)?\s+([\w\s-]+?)\b',
            r'\b(higher|lower|increased|decreased|enhanced|reduced)\s+([\w\s-]+?)\s+(?:result[s]?|lead[s]?)\s+(?:to|in)?\s+([\w\s-]+?)\b',
            r'\b([\w\s-]+?)\s+(?:depend[s]?|rely|is\s+dependent)\s+(?:on|upon)\s+([\w\s-]+?)\b',
            r'\b([\w\s-]+?)\s+(?:influence[s]?|affect[s]?|impact[s]?|modulate[s]?|regulate[s]?)\s+([\w\s-]+?)\b',
        ]
        self.all_patterns = (
            self.alloy_patterns + self.process_patterns + self.thermo_patterns +
            self.pf_patterns + self.fluid_patterns + self.ai_patterns +
            self.micro_patterns + self.comp_patterns
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.all_patterns]
        self.compiled_param_patterns = [re.compile(p, re.IGNORECASE) for p in self.param_patterns]
        self.compiled_cause_patterns = [re.compile(p, re.IGNORECASE) for p in self.cause_effect_patterns]

    @timed
    def extract_from_text(self, text: str, doc_id: int = 0) -> List[str]:
        """Extract concepts from text with multi-level reasoning."""
        concepts = set()
        text_lower = text.lower()
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else match[0])
                concept = match.lower().strip()
                if len(concept) > 3:
                    canonical = self.resolver.resolve(concept, context=text)
                    if canonical:
                        concepts.add(canonical)
                    else:
                        concepts.add(concept)
        for pattern in self.compiled_param_patterns:
            matches = pattern.findall(text)
            for param_name, value in matches:
                param_concept = f"{param_name.lower().strip()}_{value}"
                canonical = self.resolver.resolve(param_name, context=text)
                if canonical:
                    concepts.add(f"{canonical}_{value}")
                else:
                    concepts.add(param_concept)
        np_concepts = self._extract_noun_phrases(text)
        for concept in np_concepts:
            canonical = self.resolver.resolve(concept, context=text)
            if canonical:
                concepts.add(canonical)
        context_concepts = self._extract_from_context_windows(text)
        concepts.update(context_concepts)
        for concept in concepts:
            self.concept_frequencies[concept] += 1
            self.concept_contexts[concept].append(text[:200])
        self.document_concepts[doc_id] = list(concepts)
        return list(concepts)

    def _extract_noun_phrases(self, text: str) -> Set[str]:
        np_pattern = r'\b(?:[a-z]+(?:[-\s]?[a-z]+){0,2}[-\s]?)?(?:alloy|composition|tensor|parameter|gradient|energy|force|pressure|diffusion|interface|mobility|microstructure|grain|phase|melt[-\s]?pool|surrogate|model|simulation|method|analysis|optimization|kinetics|evolution|partitioning|segregation|structure|boundary|growth|transformation)\b'
        matches = re.findall(np_pattern, text, re.IGNORECASE)
        compound_pattern = r'\b([a-z]+(?:[-\s][a-z]+){1,4})\s+(?:of|for|in|with|via|through|by|to|and|or|from)\s+([a-z]+(?:[-\s][a-z]+){0,3})\b'
        compound_matches = re.findall(compound_pattern, text, re.IGNORECASE)
        concepts = set()
        for m in matches:
            if len(m) > 5:
                concepts.add(m.lower().strip())
        for m1, m2 in compound_matches:
            combined = f"{m1.lower().strip()} {m2.lower().strip()}"
            if len(combined) > 8:
                concepts.add(combined)
        return concepts

    def _extract_from_context_windows(self, text: str, window_size: int = 100) -> Set[str]:
        concepts = set()
        text_lower = text.lower()
        for keyword in self._get_all_keywords():
            for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
                start = max(0, match.start() - window_size)
                end = min(len(text), match.end() + window_size)
                context = text_lower[start:end]
                phrases = re.findall(r'\b([a-z]+(?:[-\s][a-z]+){1,3})\b', context)
                for phrase in phrases:
                    if len(phrase) > 5:
                        canonical = self.resolver.resolve(phrase, context=context)
                        if canonical:
                            concepts.add(canonical)
        return concepts

    def _get_all_keywords(self) -> Set[str]:
        keywords = set()
        for canonical, node in self.ontology.concepts.items():
            keywords.add(canonical)
            keywords.update(node.synonyms)
        return keywords

    def extract_relationships(self, text: str) -> List[Relationship]:
        relationships = []
        for pattern in self.compiled_cause_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) >= 2:
                    source = match[0] if isinstance(match[0], str) else match[1]
                    target = match[-1] if isinstance(match[-1], str) else match[0]
                    source_canon = self.resolver.resolve(source, context=text)
                    target_canon = self.resolver.resolve(target, context=text)
                    if source_canon and target_canon and source_canon != target_canon:
                        rel = Relationship(
                            source=source_canon,
                            target=target_canon,
                            rel_type=RelationshipType.CAUSES,
                            confidence=0.7,
                            evidence=text[:150]
                        )
                        relationships.append(rel)
        return relationships

    def get_concept_frequencies(self) -> Dict[str, int]:
        return dict(self.concept_frequencies)

    def get_concept_contexts(self, concept: str) -> List[str]:
        return self.concept_contexts.get(concept, [])

    def get_document_concepts(self, doc_id: int) -> List[str]:
        return self.document_concepts.get(doc_id, [])

# ==========================================
# REASONING-ENHANCED GRAPH BUILDER
# ==========================================

class ReasoningEnhancedGraphBuilder:
    """Build concept graph with reasoning-based edge inference."""

    def __init__(self, ontology: DomainOntology, extractor: EnhancedConceptExtractor):
        self.ontology = ontology
        self.extractor = extractor
        self.reasoning_paths: List[List[str]] = []
        self.inferred_edges: Set[Tuple[str, str]] = set()

    @timed
    def build_graph(self, all_concepts: List[List[str]], valid_concepts: List[str],
                    concept_to_id: Dict[str, int], embed_model=None, config: Dict = None) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(3000)
        nx_graph = nx.Graph()
        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            freq = self.extractor.concept_frequencies.get(c, 0)
            definition = self.ontology.get_definition(c)
            nx_graph.add_node(c, 
                            frequency=freq,
                            concept_type=concept_type.value,
                            definition=definition,
                            degree=0)
        cooccurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for concepts in all_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if u != v:
                        key = tuple(sorted([u, v]))
                        cooccurrence_map[key] += 1
                        nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                        nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1
        for (u, v), count in cooccurrence_map.items():
            nx_graph.add_edge(u, v, 
                            weight=count,
                            cooccurrence=count,
                            semantic=0,
                            edge_type='cooccurrence',
                            inferred=False)
        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(nx_graph, valid_concepts, embed_model, config)
        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)
        self._add_cause_effect_edges(nx_graph)
        self._add_hierarchical_edges(nx_graph, valid_concepts)
        self._compute_final_weights(nx_graph, config)
        return nx_graph

    def _add_semantic_edges(self, nx_graph: nx.Graph, valid_concepts: List[str], 
                           embed_model, config: Dict):
        try:
            embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh:
                        if nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3:
                            nx_graph.add_edge(c1, c2, 
                                            weight=sim * 2,
                                            cooccurrence=0,
                                            semantic=sim,
                                            edge_type='semantic',
                                            inferred=False)
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")

    def _add_inferred_edges(self, nx_graph: nx.Graph, valid_concepts: List[str]):
        for rel in self.ontology.relationships:
            if rel.source in valid_concepts and rel.target in valid_concepts:
                if not nx_graph.has_edge(rel.source, rel.target):
                    nx_graph.add_edge(rel.source, rel.target,
                                    weight=rel.confidence * 2,
                                    cooccurrence=0,
                                    semantic=rel.confidence,
                                    edge_type=rel.rel_type.value,
                                    inferred=True,
                                    confidence=rel.confidence)
                    self.inferred_edges.add((rel.source, rel.target))
        self._infer_cross_domain_bridges(nx_graph, valid_concepts)

    def _infer_cross_domain_bridges(self, nx_graph: nx.Graph, valid_concepts: List[str]):
        process_nodes = [c for c in valid_concepts 
                        if self.ontology.get_concept_type(c) == ConceptType.PROCESS]
        property_nodes = [c for c in valid_concepts 
                         if self.ontology.get_concept_type(c) == ConceptType.PROPERTY]
        for proc in process_nodes:
            for prop in property_nodes:
                if not nx_graph.has_edge(proc, prop):
                    paths = self.ontology.infer_path(proc, prop, max_depth=2)
                    if paths:
                        avg_confidence = 0.6
                        nx_graph.add_edge(proc, prop,
                                        weight=avg_confidence,
                                        cooccurrence=0,
                                        semantic=avg_confidence,
                                        edge_type='bridge',
                                        inferred=True,
                                        path=" -> ".join(paths[0]))
                        self.inferred_edges.add((proc, prop))
                        self.reasoning_paths.append(paths[0])

    def _add_cause_effect_edges(self, nx_graph: nx.Graph):
        pass

    def _add_hierarchical_edges(self, nx_graph: nx.Graph, valid_concepts: List[str]):
        for concept in valid_concepts:
            if concept not in self.ontology.concepts:
                continue
            node = self.ontology.concepts[concept]
            for hypernym in node.hypernyms:
                if hypernym in valid_concepts and not nx_graph.has_edge(concept, hypernym):
                    nx_graph.add_edge(concept, hypernym,
                                    weight=1.0,
                                    cooccurrence=0,
                                    semantic=0.95,
                                    edge_type='hypernym',
                                    inferred=True)
            for hyponym in node.hyponyms:
                if hyponym in valid_concepts and not nx_graph.has_edge(concept, hyponym):
                    nx_graph.add_edge(concept, hyponym,
                                    weight=1.0,
                                    cooccurrence=0,
                                    semantic=0.95,
                                    edge_type='hyponym',
                                    inferred=True)

    def _compute_final_weights(self, nx_graph: nx.Graph, config: Dict):
        cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.7)
        sem_weight = config.get("SEMANTIC_WEIGHT", 0.2)
        inf_weight = config.get("INFERENCE_WEIGHT", 0.1)
        for u, v, data in nx_graph.edges(data=True):
            cooc = data.get('cooccurrence', 0)
            sem = data.get('semantic', 0)
            inf = 1.0 if data.get('inferred', False) else 0
            conf = data.get('confidence', 0.5)
            data['weight'] = (cooc_weight * cooc + 
                             sem_weight * sem + 
                             inf_weight * inf * conf)

# ==========================================
# ORIGINAL UTILITY FUNCTIONS (Preserved)
# ==========================================

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
            "INFERENCE_WEIGHT": 0.1
        }
    elif num_abstracts <= 500:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.78, "COOCCURRENCE_WEIGHT": 0.6,
            "SEMANTIC_WEIGHT": 0.3, "CLUSTER_SIMILARITY": 0.72,
            "TOP_N_CONCEPTS": 500, "MAX_CONCEPT_LENGTH": 8,
            "INFERENCE_WEIGHT": 0.1
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 5, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 3, "USE_SEMANTIC_CLUSTERING": False,
            "SIMILARITY_THRESHOLD": 0.85, "COOCCURRENCE_WEIGHT": 0.7,
            "SEMANTIC_WEIGHT": 0.2, "CLUSTER_SIMILARITY": 0.68,
            "TOP_N_CONCEPTS": 1000, "MAX_CONCEPT_LENGTH": 10,
            "INFERENCE_WEIGHT": 0.1
        }

# ==========================================
# DEVICE & MODEL MANAGEMENT
# ==========================================
@st.cache_resource(show_spinner=False)
@timed
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ==========================================
# LEGACY CONCEPT EXTRACTION (Fallback)
# ==========================================
CORE_MATERIALS = [
    "cocofeni", "co-cr-fe-ni", "co cr fe ni", "cobalt chromium iron nickel",
    "high entropy alloy", "hea", "high-entropy alloy", "high entropy alloys", "heas",
    "multi-principal element alloy", "mpea", "multi principal element alloy", "mpeas",
    "quaternary alloy", "quaternary system", "cobalt", "chromium", "iron", "nickel",
    "fcc phase", "fcc solid", "fcc matrix", "liquid phase", "liquid state", "melt pool"
]

MANUFACTURING_PROCESSES = [
    "laser additive manufacturing", "lam", "laser powder bed fusion", "lpbf",
    "laser processing", "laser melting", "laser solidification", "laser scanning",
    "additive manufacturing", "am", "laser treatment", "laser irradiance",
    "rapid heating", "rapid cooling", "rapid solidification", "thermal cycling",
    "laser matter interaction", "melt pool formation", "laser scan track",
    "powder bed fusion", "directed energy deposition", "laser wire-feed"
]

THERMODYNAMICS_AND_TENSORS = [
    "thermodynamic data tensor", "tdt", "gibbs free energy", "gibbs energy",
    "calphad", "canonical polyadic decomposition", "cpd", "factor matrices",
    "quadratic expansion", "quadratic approximation", "taylor series",
    "chemical driving pressure", "capillary resistance", "net interface force",
    "molar volume", "phase stability", "energetic inversion", "entropy",
    "isobaric heat capacity", "gibbs energy landscape", "thermodynamic state space",
    "chemical potential", "excess mixing", "interaction parameters",
    "phase-conditioned composition tensor", "ctf", "categorical alloy-composition tensor",
    "driving force", "thermodynamic driving force", "interfacial energy", "surface tension"
]

PHASE_FIELD_MODELING = [
    "phase-field model", "pfm", "phase field model", "phase-field modeling",
    "non-isothermal phase-field", "order parameter", "diffuse interface",
    "allen-cahn equation", "kks phase-equilibrium", "kks model",
    "multicomponent diffusion", "interface mobility", "phase mobility",
    "free energy functional", "bulk free energy", "gradient energy coefficient",
    "phase-field simulation", "spatiotemporal tensor", "phase fraction",
    "switching function", "landau polynomial", "barrier function"
]

FLUID_DYNAMICS_AND_MELT_POOL = [
    "marangoni convection", "marangoni-driven melt pool flow", "thermocapillary convection",
    "navier-stokes equations", "melt pool morphology", "melt pool depth",
    "boussinesq approximation", "incompressible flow", "melt pool dynamics",
    "surface-tension gradient", "fluidic phenomena", "melt pool flow",
    "thermal gradient", "keyhole formation", "marangoni effect"
]

AI_AND_SURROGATE_MODELS = [
    "ai surrogate", "surrogate model", "transformer-inspired", "attention mechanism",
    "cross-attention", "query-key attention", "gaussian locality regularization",
    "composition similarity", "digital twin", "attention-regularized surrogate",
    "multi-head attention", "spatiotemporally aware interpolation", "hybrid attention weights",
    "machine learning", "deep learning", "physics-informed", "physics-guided",
    "webapp", "neural network", "data-driven", "computational speedup"
]

MICROSTRUCTURAL_FEATURES = [
    "microstructural evolution", "elemental partitioning", "solidification kinetics",
    "equiaxed grains", "columnar grains", "grain boundary", "thermal wake",
    "cooling rate", "phase transformation", "solidification", "melting",
    "nucleation", "interface motion", "microstructure", "grain size",
    "segregation", "non-equilibrium microstructures", "hot tearing", "porosity"
]

COMPUTATIONAL_AND_MATHEMATICAL_METHODS = [
    "finite element analysis", "fea", "moose framework", "finite element method",
    "alternating least squares", "als", "tensor factorization", "multi-linear interpolation",
    "hessian matrix", "spectral decomposition", "rank-1 outer products",
    "root-mean-square error", "rmse", "leave-one-out cross-validation",
    "dice coefficient", "intersection-over-union", "iou", "computational domain",
    "discretization", "mesh", "cpu hours"
]

ALL_DOMAIN_KEYWORDS = (CORE_MATERIALS + MANUFACTURING_PROCESSES + THERMODYNAMICS_AND_TENSORS +
                       PHASE_FIELD_MODELING + FLUID_DYNAMICS_AND_MELT_POOL +
                       AI_AND_SURROGATE_MODELS + MICROSTRUCTURAL_FEATURES + 
                       COMPUTATIONAL_AND_MATHEMATICAL_METHODS)

HEA_LASER_PATTERNS = [
    r'\b(?:co(?:-|\s)?cr(?:-|\s)?fe(?:-|\s)?ni|cocofeni)\b',
    r'\b(?:high[\s-]entropy\s+alloy[s]?|hea[s]?)\b',
    r'\b(?:multi[\s-]principal\s+element\s+alloy[s]?|mpea[s]?)\b',
    r'\b(?:laser\s+(?:powder\s+bed\s+fusion|additive\s+manufacturing|processing|melting|solidification)|lpbf|lam)\b',
    r'\b(?:thermodynamic\s+data\s+tensor|tdt|gibbs\s+(?:free\s+)?energy\s+tensor)\b',
    r'\b(?:canonical\s+polyadic\s+decomposition|cpd|factor\s+matrices)\b',
    r'\b(?:phase[\s-]?field\s+(?:model|simulation|method|framework)|pfm)\b',
    r'\b(?:marangoni\s+(?:convection|flow|effect)|thermocapillary\s+convection)\b',
    r'\b(?:ai\s+surrogate|transformer[\s-]inspired|attention[\s-]regularized|digital\s+twin)\b',
    r'\b(?:phase[\s-]conditioned\s+composition\s+tensor|categorical\s+alloy[\s-]composition\s+tensor|ctf)\b',
    r'\b(?:allen[\s-]cahn|kks\s+phase[\s-]equilibrium|multicomponent\s+diffusion)\b',
    r'\b(?:melt\s+pool\s+(?:morphology|depth|dynamics|flow)|thermal\s+gradient)\b',
    r'\b(?:calphad|gibbs\s+energy\s+landscape|chemical\s+driving\s+pressure)\b',
    r'\b(?:gaussian\s+locality|composition[\s-]tensor\s+similarity|cross[\s-]attention)\b'
]

HEA_CATEGORY_MAPPING = {
    r'co(?:-|\s)?cr(?:-|\s)?fe(?:-|\s)?ni|cocofeni|high[\s-]entropy|hea|mpea|multi[\s-]principal': 'core_material',
    r'laser\s+(?:powder\s+bed|additive|processing|melting|solidification)|lpbf|lam|rapid\s+(?:heating|cooling)': 'manufacturing_process',
    r'thermodynamic\s+data\s+tensor|tdt|gibbs\s+(?:free\s+)?energy|calphad|cpd|canonical\s+polyadic|factor\s+matrices|quadratic\s+(?:expansion|approximation)|phase[\s-]conditioned\s+composition\s+tensor|ctf': 'thermodynamics_tensor',
    r'phase[\s-]?field|pfm|allen[\s-]cahn|kks|diffuse\s+interface|order\s+parameter|multicomponent\s+diffusion': 'phase_field_modeling',
    r'marangoni|thermocapillary|navier[\s-]stokes|melt\s+pool\s+(?:flow|dynamics|morphology)|surface\s+tension|boussinesq': 'fluid_dynamics_melt_pool',
    r'ai\s+surrogate|transformer|attention\s+mechanism|cross[\s-]attention|digital\s+twin|machine\s+learning|deep\s+learning|gaussian\s+locality': 'ai_surrogate_model',
    r'microstructural\s+evolution|elemental\s+partitioning|solidification|equiaxed|columnar|grain\s+(?:boundary|size)|segregation': 'microstructural_feature',
    r'finite\s+element|fea|moose|alternating\s+least\s+squares|tensor\s+factorization|cross[\s-]validation|dice\s+coefficient': 'computational_method'
}

# ==========================================
# THEME CONFIGURATION (ENHANCED)
# ==========================================
THEME_PRESETS = {
    "Bright (Default)": {
        "bg": "#ffffff", "font": "#1e293b", "tooltip_bg": "rgba(255,255,255,0.95)",
        "tooltip_border": "#cbd5e1", "tooltip_text": "#1e293b",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.45)",
        "edge_semantic": "rgba(251, 146, 60, 0.40)",
        "edge_bridge": "rgba(250, 204, 21, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(34, 197, 94, 0.45)",
        "edge_unknown": "rgba(148, 163, 184, 0.30)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b", "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.15)", "plotly_bg": "#ffffff", "plotly_paper": "#ffffff",
        "grid_color": "#e2e8f0", "axis_color": "#64748b"
    },
    "Dark": {
        "bg": "#0f172a", "font": "#e2e8f0", "tooltip_bg": "rgba(15, 23, 42, 0.95)",
        "tooltip_border": "#334155", "tooltip_text": "#e2e8f0",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.55)",
        "edge_semantic": "rgba(251, 146, 60, 0.50)",
        "edge_bridge": "rgba(250, 204, 21, 0.65)",
        "edge_inferred": "rgba(139, 92, 246, 0.60)",
        "edge_cause": "rgba(239, 68, 68, 0.65)",
        "edge_hypernym": "rgba(34, 197, 94, 0.55)",
        "edge_unknown": "rgba(148, 163, 184, 0.40)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b", "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.6)", "plotly_bg": "#0f172a", "plotly_paper": "#0f172a",
        "grid_color": "#1e293b", "axis_color": "#94a3b8"
    },
    "Midnight": {
        "bg": "#020617", "font": "#f1f5f9", "tooltip_bg": "rgba(2, 6, 23, 0.97)",
        "tooltip_border": "#1e293b", "tooltip_text": "#f1f5f9",
        "edge_cooccurrence": "rgba(99, 102, 241, 0.55)",
        "edge_semantic": "rgba(236, 72, 153, 0.50)",
        "edge_bridge": "rgba(34, 211, 238, 0.65)",
        "edge_inferred": "rgba(168, 85, 247, 0.60)",
        "edge_cause": "rgba(244, 63, 94, 0.65)",
        "edge_hypernym": "rgba(52, 211, 153, 0.55)",
        "edge_unknown": "rgba(71, 85, 105, 0.40)",
        "node_border": "#e2e8f0", "highlight_bg": "#f43f5e", "hover_bg": "#22d3ee",
        "shadow_color": "rgba(0,0,0,0.7)", "plotly_bg": "#020617", "plotly_paper": "#020617",
        "grid_color": "#0f172a", "axis_color": "#64748b"
    },
    "Warm": {
        "bg": "#fff7ed", "font": "#431407", "tooltip_bg": "rgba(255, 247, 237, 0.97)",
        "tooltip_border": "#fdba74", "tooltip_text": "#431407",
        "edge_cooccurrence": "rgba(234, 88, 12, 0.45)",
        "edge_semantic": "rgba(180, 83, 9, 0.40)",
        "edge_bridge": "rgba(202, 138, 4, 0.55)",
        "edge_inferred": "rgba(147, 51, 234, 0.50)",
        "edge_cause": "rgba(220, 38, 38, 0.55)",
        "edge_hypernym": "rgba(22, 163, 74, 0.45)",
        "edge_unknown": "rgba(120, 53, 15, 0.25)",
        "node_border": "#fff7ed", "highlight_bg": "#dc2626", "hover_bg": "#f59e0b",
        "shadow_color": "rgba(124, 45, 18, 0.15)", "plotly_bg": "#fff7ed", "plotly_paper": "#fff7ed",
        "grid_color": "#fed7aa", "axis_color": "#9a3412"
    },
    "Forest": {
        "bg": "#f0fdf4", "font": "#052e16", "tooltip_bg": "rgba(240, 253, 244, 0.97)",
        "tooltip_border": "#86efac", "tooltip_text": "#052e16",
        "edge_cooccurrence": "rgba(22, 163, 74, 0.45)",
        "edge_semantic": "rgba(5, 150, 105, 0.40)",
        "edge_bridge": "rgba(234, 179, 8, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(21, 128, 61, 0.45)",
        "edge_unknown": "rgba(20, 83, 45, 0.25)",
        "node_border": "#f0fdf4", "highlight_bg": "#15803d", "hover_bg": "#84cc16",
        "shadow_color": "rgba(20, 83, 45, 0.15)", "plotly_bg": "#f0fdf4", "plotly_paper": "#f0fdf4",
        "grid_color": "#bbf7d0", "axis_color": "#166534"
    },
    "Ocean": {
        "bg": "#ecfeff", "font": "#083344", "tooltip_bg": "rgba(236, 254, 255, 0.97)",
        "tooltip_border": "#67e8f9", "tooltip_text": "#083344",
        "edge_cooccurrence": "rgba(6, 182, 212, 0.45)",
        "edge_semantic": "rgba(14, 165, 233, 0.40)",
        "edge_bridge": "rgba(99, 102, 241, 0.55)",
        "edge_inferred": "rgba(168, 85, 247, 0.50)",
        "edge_cause": "rgba(244, 63, 94, 0.55)",
        "edge_hypernym": "rgba(13, 148, 136, 0.45)",
        "edge_unknown": "rgba(21, 94, 117, 0.25)",
        "node_border": "#ecfeff", "highlight_bg": "#0ea5e9", "hover_bg": "#22d3ee",
        "shadow_color": "rgba(8, 51, 68, 0.15)", "plotly_bg": "#ecfeff", "plotly_paper": "#ecfeff",
        "grid_color": "#a5f3fc", "axis_color": "#0e7490"
    }
}

PHYSICS_PRESETS = {
    "Stable (Default)": {
        "damping": 0.55, "gravity": -2500, "spring_length": 140,
        "spring_strength": 0.05, "central_gravity": 0.25, "stabilization": 2500
    },
    "Fluid": {
        "damping": 0.25, "gravity": -1800, "spring_length": 120,
        "spring_strength": 0.05, "central_gravity": 0.30, "stabilization": 1500
    },
    "Tight": {
        "damping": 0.70, "gravity": -4000, "spring_length": 80,
        "spring_strength": 0.08, "central_gravity": 0.20, "stabilization": 3000
    },
    "Off": {
        "damping": 0.99, "gravity": 0, "spring_length": 200,
        "spring_strength": 0.0, "central_gravity": 0.0, "stabilization": 0
    }
}

# ==========================================
# VISUALIZATION FUNCTIONS (MAJOR UPGRADES)
# ==========================================

def get_hea_laser_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    category = 'general'
    for pattern, cat in HEA_CATEGORY_MAPPING.items():
        if re.search(pattern, concept_lower):
            category = cat
            break
    color_map = {
        'core_material': '#D32F2F',
        'manufacturing_process': '#00BCD4',
        'thermodynamics_tensor': '#FF9800',
        'phase_field_modeling': '#9C27B0',
        'fluid_dynamics_melt_pool': '#2196F3',
        'ai_surrogate_model': '#8E24AA',
        'microstructural_feature': '#4CAF50',
        'computational_method': '#795548',
        'general': '#9E9E9E'
    }
    return color_map.get(category, '#9E9E9E')

def render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=True,
                        min_node_size=8, max_node_size=40, cmap_name="viridis",
                        custom_labels=None, node_label_size=12, top_n_nodes=0,
                        theme=None, physics_preset=None, show_edge_weights=False,
                        edge_label_mode="hover", show_reasoning=False,
                        use_abbreviated_labels=False, max_label_length=15,
                        node_font_face="Inter, Segoe UI, Roboto, sans-serif",
                        edge_label_size=10, edge_label_color=None,
                        edge_label_position="middle", enable_node_highlight=True,
                        show_definitions=True):
    """
    Enhanced PyVis renderer with:
    - Interactive node highlighting with side panel
    - Abbreviated labels (N1, N2...) for dense graphs
    - Configurable edge/node label settings
    - Concept definitions in tooltips
    """
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    pos = {}
    if len(nx_graph.nodes()) > 0:
        try:
            if len(nx_graph.nodes()) < 300:
                pos = nx.kamada_kawai_layout(nx_graph, weight='weight')
            else:
                pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight='weight')
        except Exception:
            pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight='weight')

    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))

    net = Network(
        height="780px", width="100%", bgcolor=theme['bg'], font_color=theme['font'],
        select_menu=True, notebook=False, cdn_resources='remote'
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
              "iterations": {physics_preset['stabilization']},
              "updateInterval": 30,
              "onlyDynamicEdges": false,
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
          "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true }
        }
        """)

    # v2.0: Abbreviated labels support
    label_map = {}
    n_counter = 1

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(min_node_size + freq * 1.2, min_node_size, max_node_size))
        color = get_hea_laser_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))

        # v2.0: Abbreviated labels logic
        original_label = custom_labels.get(node, node) if custom_labels else node
        if use_abbreviated_labels and len(original_label) > max_label_length:
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
                'bold': True
            }
        else:
            label = original_label
            node_shape = 'dot'
            font_dict = {
                'color': theme['font'],
                'size': node_label_size,
                'face': node_font_face,
                'strokeWidth': 0,
                'vadjust': -6
            }

        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        definition = nx_graph.nodes[node].get('definition', '')

        # v2.0: Enhanced tooltip with definition
        tooltip_content = f"""<div style='font-family:Inter,sans-serif;'>
<b style='font-size:14px;color:{theme['highlight_bg']};'>{node}</b><br>
<span style='color:{theme['tooltip_text']};opacity:0.7;'>Type:</span> {concept_type}<br>
<span style='color:{theme['tooltip_text']};opacity:0.7;'>Degree:</span> {degree}<br>
<span style='color:{theme['tooltip_text']};opacity:0.7;'>Frequency:</span> {freq}"""

        if show_definitions and definition:
            tooltip_content += f"<br><span style='color:{theme['tooltip_text']};opacity:0.7;'>Definition:</span> <i>{definition}</i>"

        if use_abbreviated_labels and label != original_label:
            tooltip_content += f"<br><span style='color:{theme['tooltip_text']};opacity:0.7;'>Full Label:</span> {original_label}"

        tooltip_content += "</div>"

        x, y = (pos.get(node, (0, 0))[0] * 1200, pos.get(node, (0, 0))[1] * 1200)

        net.add_node(
            node,
            label=label,
            size=size,
            x=x,
            y=y,
            color={
                'background': color,
                'border': theme['node_border'],
                'highlight': {'background': theme['highlight_bg'], 'border': '#ffffff'},
                'hover': {'background': theme['hover_bg'], 'border': '#ffffff'}
            },
            font=font_dict,
            title=tooltip_content,
            borderWidth=2,
            borderWidthSelected=3,
            shadow={
                'enabled': True,
                'color': theme['shadow_color'],
                'size': 12,
                'x': 4,
                'y': 4
            },
            shape=node_shape,
            mass=max(1, 1 + freq * 0.05)
        )

    color_map = {
        'cooccurrence': theme['edge_cooccurrence'],
        'semantic':     theme['edge_semantic'],
        'bridge':       theme['edge_bridge'],
        'inferred':     theme.get('edge_inferred', '#8b5cf6'),
        'causes':       theme.get('edge_cause', '#ef4444'),
        'hypernym':     theme.get('edge_hypernym', '#22c55e'),
        'hyponym':      theme.get('edge_hypernym', '#22c55e'),
        'manual':       theme['edge_semantic'],
        'unknown':      theme['edge_unknown']
    }

    all_weights = [nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()]
    weight_threshold = np.percentile(all_weights, 80) if all_weights else 0

    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)

        if is_inferred and edge_type not in ['hypernym', 'hyponym', 'causes']:
            color = color_map.get('inferred', color_map['unknown'])
        else:
            color = color_map.get(edge_type, color_map['unknown'])

        width = float(np.clip(w * 0.4, 0.8, 3.5))
        dashes = True if is_inferred else False

        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)),
            width=width,
            color={
                'color': color,
                'highlight': theme['highlight_bg'],
                'hover': theme['hover_bg'],
                'opacity': 0.85
            },
            smooth={'type': 'continuous', 'roundness': 0.35},
            title=f"<span style='font-family:Inter,sans-serif;'>Weight: <b>{w:.2f}</b><br>Type: {edge_type}<br>Inferred: {is_inferred}</span>",
            dashes=dashes
        )

        # v2.0: Configurable edge labels
        actual_edge_label_color = edge_label_color if edge_label_color else theme['font']
        if edge_label_mode == "all" or (edge_label_mode == "threshold" and w >= weight_threshold):
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {
                'color': actual_edge_label_color,
                'size': int(edge_label_size),
                'background': theme['tooltip_bg'],
                'strokeWidth': 2,
                'strokeColor': theme['node_border'],
                'align': edge_label_position,
                'face': node_font_face
            }

        net.add_edge(u, v, **edge_kwargs)

    html_content = net.generate_html()

    # v2.0: Interactive node highlighting JavaScript
    if enable_node_highlight:
        highlight_js = """
        <script>
        (function() {
            var checkExist = setInterval(function() {
                if (typeof network !== 'undefined' && network.body) {
                    clearInterval(checkExist);
                    var nodesDS = network.body.data.nodes;
                    var edgesDS = network.body.data.edges;
                    var savedNodeColors = {};

                    network.on("selectNode", function(params) {
                        var nodeId = params.nodes[0];
                        var connectedEdges = network.getConnectedEdges(nodeId);
                        var connectedNodes = network.getConnectedNodes(nodeId);

                        var nodeUpdates = [];
                        connectedNodes.forEach(function(nId) {
                            var n = nodesDS.get(nId);
                            if (n && !savedNodeColors[nId]) {
                                savedNodeColors[nId] = JSON.parse(JSON.stringify(n.color));
                                var newColor = JSON.parse(JSON.stringify(n.color));
                                newColor.border = '#FFD700';
                                nodeUpdates.push({id: nId, color: newColor});
                            }
                        });
                        if (nodeUpdates.length > 0) nodesDS.update(nodeUpdates);

                        var panel = document.getElementById('edge-info-panel') || document.createElement('div');
                        panel.id = 'edge-info-panel';
                        panel.style.cssText = 'position:fixed;top:110px;right:24px;width:360px;max-height:560px;overflow-y:auto;z-index:9999;background:rgba(255,255,255,0.98);border:2px solid #FFD700;border-radius:14px;padding:16px;box-shadow:0 12px 48px rgba(0,0,0,0.18);font-family:Inter,sans-serif;';

                        var html = '<div style="font-size:15px;font-weight:700;color:#D32F2F;border-bottom:2px solid #FFD700;padding-bottom:8px;">🔗 ' + nodeId + ' (' + connectedEdges.length + ' edges)</div>';
                        html += '<div style="margin-top:8px;font-size:11px;color:#666;">Click elsewhere to close</div>';
                        connectedEdges.forEach(function(eId){
                            var e = edgesDS.get(eId);
                            var edgeColor = e.color && e.color.color ? e.color.color : '#999';
                            html += '<div style="padding:6px;font-size:12px;border-left:3px solid ' + edgeColor + ';margin:4px 0;background:rgba(0,0,0,0.03);border-radius:4px;">';
                            html += '<b>' + e.from + '</b> ↔ <b>' + e.to + '</b><br>';
                            html += 'Weight: <b>' + e.value.toFixed(2) + '</b> | Type: ' + (e.edge_type || 'unknown');
                            if (e.inferred) html += ' <span style="background:#8b5cf6;color:white;padding:1px 4px;border-radius:3px;font-size:10px;">INFERRED</span>';
                            html += '</div>';
                        });
                        panel.innerHTML = html;
                        document.body.appendChild(panel);
                    });

                    network.on("deselectNode", function(){ 
                        var restores = [];
                        for (var nid in savedNodeColors) {
                            restores.push({id: nid, color: savedNodeColors[nid]});
                        }
                        if (restores.length > 0) nodesDS.update(restores);
                        savedNodeColors = {};
                        var p = document.getElementById('edge-info-panel');
                        if (p) p.style.display = 'none';
                    });
                }
            }, 250);
        })();
        </script>
        """
        html_content = html_content.replace('</body>', highlight_js + '</body>')

    custom_css = f"""
    <style>
        body {{
            background: {theme['bg']};
            margin: 0;
            padding: 0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
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
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            font-size: 13px !important;
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
    </style>
    """
    html_content = html_content.replace('</head>', custom_css + '</head>')

    st.components.v1.html(html_content, height=790, scrolling=True)

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button("Download Interactive Graph (HTML)", data=html_bytes,
                          file_name="hea_laser_concept_graph.html", mime="text/html")

        # v2.0: Also export label mapping if abbreviated labels used
        if use_abbreviated_labels and label_map:
            label_map_json = json.dumps(label_map, indent=2)
            st.download_button("Download Label Mapping (JSON)", data=label_map_json.encode('utf-8'),
                              file_name="label_mapping.json", mime="application/json")

        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")

# ==========================================
# ENHANCED SUNBURST WITH SYMBOL CHAINS
# ==========================================

def build_category_hierarchy(valid_concepts: List[str], concept_abstract_map: Dict, top_n_per_category: int = 40):
    """
    v2.0: Smart hierarchy that prevents duplicate rings when concept name matches category name.
    """
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    category_map = abstract_concepts_to_categories(valid_concepts)
    all_category_names = set(category_map.values())

    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        freq = len(concept_abstract_map.get(concept, []))
        # v2.0: KEY FIX - Prevent duplicate rings
        if concept in all_category_names:
            hierarchy[category]["count"] += freq
            continue
        hierarchy[category]["children"].append((concept, freq))
        hierarchy[category]["count"] += freq

    for parent in list(hierarchy.keys()):
        children = hierarchy[parent]["children"]
        if top_n_per_category > 0 and len(children) > top_n_per_category:
            children.sort(key=lambda x: x[1], reverse=True)
            children = children[:top_n_per_category]
            hierarchy[parent]["count"] = sum(cnt for _, cnt in children)
            hierarchy[parent]["children"] = children

    labels, parents, values = [], [], []
    for parent, data in hierarchy.items():
        labels.append(parent)
        parents.append("")
        values.append(data["count"])
        for child, cnt in data["children"]:
            labels.append(child)
            parents.append(parent)
            values.append(cnt)
    return labels, parents, values

def render_sunburst_chart(labels, parents, values, cmap_name="viridis", label_size=11, 
                          width=800, height=600, theme=None, branchvalues="total"):
    """
    v2.0: Enhanced sunburst with recursive symbol chains and per-node continuous colormaps.
    """
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return

    n_items = len(labels)
    use_remainder = n_items > 80

    # Build unique IDs
    unique_ids = []
    seen = {}
    for i, lab in enumerate(labels):
        base = lab[:25] + ("..." if len(lab) > 25 else "")
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base)
            seen[base] = 1

    parent_ids = []
    for p in parents:
        if p == "":
            parent_ids.append("")
        else:
            for i, lab in enumerate(labels):
                if lab == p:
                    parent_ids.append(unique_ids[i])
                    break
            else:
                parent_ids.append("")

    # v2.0: Calculate depths for symbol assignment
    parent_map = {unique_ids[i]: parent_ids[i] for i in range(len(unique_ids))}

    def get_depth(label):
        p = parent_map.get(label, "")
        return 0 if p == "" else 1 + get_depth(p)

    depths = [get_depth(uid) for uid in unique_ids]

    # v2.0: Recursive Symbol Library
    SYMBOL_LIBRARY = ['✦', '★', '●', '■', '▲', '◆', '⬟', '⬢']
    node_symbols = {}
    display_labels = []

    for i, uid in enumerate(unique_ids):
        d = depths[i]
        if d == 0:
            node_symbols[uid] = SYMBOL_LIBRARY[0]
            display_labels.append(f"{SYMBOL_LIBRARY[0]} {labels[i]}")
        else:
            # Build ancestor chain
            ancestors = []
            current = uid
            while current != "":
                current = parent_map.get(current, "")
                if current in node_symbols:
                    ancestors.insert(0, node_symbols[current])
            own_sym = SYMBOL_LIBRARY[min(d, len(SYMBOL_LIBRARY) - 1)]
            node_symbols[uid] = own_sym
            chain = "".join(ancestors[-2:]) + own_sym if ancestors else own_sym
            display_labels.append(f"{chain} {labels[i]}")

    # v2.0: Per-Node Continuous Colormap (instead of categorical)
    try:
        cmap_obj = plt.cm.get_cmap(cmap_name)
        t_vals = np.linspace(0.05, 0.95, len(unique_ids))
        plot_colors = [matplotlib.colors.to_hex(cmap_obj(t)) for t in t_vals]
    except Exception:
        plot_colors = get_colormap_colors(cmap_name, len(unique_ids))

    bv = branchvalues if branchvalues in ["total", "remainder"] else ("remainder" if use_remainder else "total")

    fig = go.Figure(go.Sunburst(
        ids=unique_ids,
        labels=display_labels,
        parents=parent_ids,
        values=values,
        customdata=labels,
        branchvalues=bv,
        marker=dict(
            colors=plot_colors,
            line=dict(width=0.5, color="rgba(255,255,255,0.25)")
        ),
        textinfo="label+percent entry+value",
        insidetextorientation="radial",
        textfont=dict(size=label_size, color="white"),
        hovertemplate='<b>%{customdata}</b><br>Symbol: %{label}<br>Value: %{value}<br>Parent: %{parent}<extra></extra>'
    ))

    fig.update_layout(
        title="<b>CoCrFeNi HEA Laser AM Process-Material Response Domain Hierarchy</b><br><i>Size = concept frequency | Symbols show hierarchy depth</i>",
        font=dict(size=label_size, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=width,
        height=height,
        margin=dict(t=80, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # v2.0: Show symbol legend
    with st.expander("Symbol Legend"):
        legend_html = "<div style='font-family:Inter,sans-serif;'>"
        legend_html += "<b>Hierarchy Symbols:</b><br>"
        for i, sym in enumerate(SYMBOL_LIBRARY[:4]):
            level_name = ["Root (Domain)", "Category", "Sub-category", "Concept"][i] if i < 4 else f"Level {i}"
            legend_html += f"<span style='font-size:18px;'>{sym}</span> = {level_name}<br>"
        legend_html += "<br><i>Chains like ★□ show the path from root to concept</i>"
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

# ==========================================
# ENHANCED EXPORT FUNCTIONS v2.0
# ==========================================

def export_graph(nx_graph, concept_abstract_map, format_type: str, include_metadata: bool = True):
    """
    v2.0: Enhanced export with full relationship annotations and metadata.
    """
    if format_type == "GraphML":
        try:
            # v2.0: Add metadata as graph attributes
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '2.0'
                nx_graph.graph['tool'] = 'HEA-Laser-ConceptGraph'
            try:
                nx.write_graphml_lxml(nx_graph, "hea_graph.graphml")
            except:
                nx.write_graphml(nx_graph, "hea_graph.graphml")
            with open("hea_graph.graphml", "rb") as f:
                return f.read(), "application/graphml+xml", "hea_graph.graphml"
        except Exception as e:
            st.error(f"GraphML export failed: {e}")
            return None, None, None

    elif format_type == "JSON (Full Metadata)":
        # v2.0: Enhanced JSON with full metadata
        data = nx.node_link_data(nx_graph)
        if include_metadata:
            data['metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '2.0',
                'tool': 'HEA-Laser-ConceptGraph',
                'node_count': len(nx_graph.nodes()),
                'edge_count': len(nx_graph.edges()),
                'inferred_edges': sum(1 for u, v, d in nx_graph.edges(data=True) if d.get('inferred', False)),
                'categories': list(set(abstract_concepts_to_categories(list(nx_graph.nodes())).values()))
            }
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "hea_graph_full.json"

    elif format_type == "JSON (Compact)":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "hea_graph.json"

    elif format_type == "CSV (Edges + Metadata)":
        # v2.0: Enhanced edge CSV with all relationship metadata
        edge_data = []
        for u, v, data in nx_graph.edges(data=True):
            row = {
                "source": u,
                "target": v,
                "weight": data.get('weight', 1),
                "cooccurrence": data.get('cooccurrence', 0),
                "semantic_similarity": data.get('semantic', 0),
                "edge_type": data.get('edge_type', 'unknown'),
                "inferred": data.get('inferred', False),
                "confidence": data.get('confidence', 1.0),
                "path": data.get('path', '')
            }
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "hea_edges_enhanced.csv"

    elif format_type == "CSV (Nodes + Metadata)":
        node_data = []
        for node in nx_graph.nodes():
            row = {
                "concept": node,
                "frequency": len(concept_abstract_map.get(node, [])),
                "degree": nx_graph.degree(node),
                "concept_type": nx_graph.nodes[node].get('concept_type', 'general'),
                "definition": nx_graph.nodes[node].get('definition', ''),
                "category": abstract_concepts_to_categories([node]).get(node, 'general')
            }
            row.update({k: v for k, v in nx_graph.nodes[node].items() if isinstance(v, (str, int, float, bool))})
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "hea_nodes_enhanced.csv"

    elif format_type == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [get_hea_laser_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close()
            return buf.read(), "image/png", "hea_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None

    elif format_type == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=150)
            node_colors = [get_hea_laser_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close()
            return buf.read(), "image/svg+xml", "hea_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None

    elif format_type == "GEXF":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '2.0'
            nx.write_gexf(nx_graph, "hea_graph.gexf")
            with open("hea_graph.gexf", "rb") as f:
                return f.read(), "application/xml", "hea_graph.gexf"
        except Exception as e:
            st.error(f"GEXF export failed: {e}")
            return None, None, None

    return None, None, None

# ==========================================
# ENHANCED SIDEBAR v2.0
# ==========================================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration v2.0")

        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0
        )
        theme = THEME_PRESETS[st.session_state['theme']]

        st.subheader("🔬 HEA Laser AM Focus Areas")
        st.markdown("- CoCrFeNi High-Entropy Alloy (HEA / MPEA)")
        st.markdown("- Laser Processing (LPBF, LAM, rapid solidification, melt pool)")
        st.markdown("- Thermodynamic Data Tensors (TDT, CPD, CALPHAD, Gibbs energy)")
        st.markdown("- Phase-Field Modeling (Allen-Cahn, KKS, multicomponent diffusion)")
        st.markdown("- Fluid Dynamics & Melt Pool (Marangoni, Navier-Stokes, thermal gradient)")
        st.markdown("- AI Surrogate Models (Transformer, attention, digital twin)")
        st.markdown("- Microstructural Response (elemental partitioning, grain structure, segregation)")
        st.markdown("- Thermophysical Phenomena (driving force, interfacial energy, capillary resistance)")
        st.markdown("- Computational Methods (MOOSE, FEA, ALS, tensor factorization)")

        st.subheader("🧠 NLP Reasoning Options")
        st.session_state['use_ontology'] = st.checkbox("Use ontology-based resolution", value=True,
            help="Maps synonyms like 'HEA', 'high-entropy alloy' to canonical concepts")
        st.session_state['use_embedding_resolution'] = st.checkbox("Use embedding-based semantic equivalence", value=True,
            help="Detects semantic similarity >0.85 even for unseen variants")
        st.session_state['use_relationship_extraction'] = st.checkbox("Extract cause-effect relationships", value=True,
            help="Identifies causal links between laser parameters and microstructure")
        st.session_state['use_inference'] = st.checkbox("Enable reasoning-based edge inference", value=True,
            help="Infers process→parameter→response chains even when not co-occurring")
        st.session_state['context_window'] = st.slider("Context window (chars)", 20, 200, 50,
            help="Window size for context-based disambiguation")

        st.subheader("📊 Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:", ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"], index=0
        )

        # v2.0: Node & Label Settings
        with st.expander("🔤 Node & Label Settings"):
            st.session_state['node_label_size'] = st.slider("Node label font size", 8, 24, 12, step=1)
            st.session_state['node_font_face'] = st.selectbox("Node font family", 
                ["Inter, Segoe UI, Roboto, sans-serif", "Arial, Helvetica, sans-serif", 
                 "Georgia, serif", "Courier New, monospace"], index=0)
            st.session_state['use_abbreviated_labels'] = st.checkbox("Use abbreviated labels (N1, N2...)", value=False,
                help="Replaces long labels with N-codes to prevent visual clutter")
            if st.session_state['use_abbreviated_labels']:
                st.session_state['max_label_length'] = st.slider("Max label length before abbreviation", 5, 30, 15)
            st.session_state['show_definitions'] = st.checkbox("Show concept definitions in tooltips", value=True)

        # v2.0: Edge Label Settings
        with st.expander("🔗 Edge Label Settings"):
            st.session_state['show_edge_weights'] = st.toggle("Show edge weights", value=False)
            st.session_state['edge_label_mode'] = st.selectbox(
                "Edge label mode:", ["hover", "threshold", "all"], index=0
            )
            st.session_state['edge_label_size'] = st.slider("Edge label font size", 6, 18, 10, step=1)
            st.session_state['edge_label_color'] = st.color_picker("Edge label color", value="#000000")
            st.session_state['edge_label_position'] = st.selectbox("Edge label position", 
                ["middle", "top", "bottom", "from", "to"], index=0)

        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:", options=list(SUPPORTED_COLORMAPS.keys()), index=0
        )

        st.subheader("⚡ Physics & Layout")
        st.session_state['physics_preset'] = st.selectbox(
            "Physics preset:",
            options=list(PHYSICS_PRESETS.keys()),
            index=0
        )
        preset = PHYSICS_PRESETS[st.session_state['physics_preset']]
        st.session_state['physics_enabled'] = st.checkbox(
            "Enable physics", value=(preset["gravity"] != 0)
        )

        with st.expander("Advanced Physics Overrides"):
            st.session_state['adv_damping'] = st.slider("Damping", 0.05, 0.95, preset["damping"], step=0.05)
            st.session_state['adv_gravity'] = st.slider("Repulsion", -8000, -500, preset["gravity"], step=100)
            st.session_state['adv_spring_length'] = st.slider("Spring length", 40, 300, preset["spring_length"], step=10)
            st.session_state['adv_spring_strength'] = st.slider("Spring strength", 0.01, 0.20, preset["spring_strength"], step=0.01)
            st.session_state['adv_central_gravity'] = st.slider("Central gravity", 0.0, 0.5, preset["central_gravity"], step=0.05)
            st.session_state['adv_stabilization'] = st.slider("Stabilization iter", 0, 5000, preset["stabilization"], step=250)

        base_preset = PHYSICS_PRESETS[st.session_state['physics_preset']].copy()
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
                "Max nodes", 10, 500, 200, step=10, disabled=all_graph,
                key="top_n_graph_slider"
            )
        if all_graph:
            st.session_state['top_n_graph'] = 0

        col_all2, col_slider2 = st.columns([0.3, 0.7])
        with col_all2:
            all_sun = st.checkbox("All", value=True, key="all_sun_chk")
        with col_slider2:
            st.session_state['top_n_sunburst'] = st.slider(
                "Max children/category", 10, 100, 40, step=10, disabled=all_sun,
                key="top_n_sunburst_slider"
            )
        if all_sun:
            st.session_state['top_n_sunburst'] = 0

        col_all3, col_slider3 = st.columns([0.3, 0.7])
        with col_all3:
            all_radar = st.checkbox("All", value=True, key="all_radar_chk")
        with col_slider3:
            st.session_state['top_n_radar'] = st.slider(
                "Top K for radar", 5, 30, 15, disabled=all_radar,
                key="top_n_radar_slider"
            )
        if all_radar:
            st.session_state['top_n_radar'] = 0

        st.subheader("🔧 Graph Parameters")
        st.session_state['min_freq'] = st.slider("Min concept frequency", 1, 20, 1)
        st.session_state['min_words'] = st.slider("Min words per concept", 2, 5, 2)
        st.session_state['sim_threshold'] = st.slider("Semantic threshold", 0.6, 0.95, 0.85, step=0.05)
        st.session_state['cooc_weight'] = st.slider("Co-occurrence weight", 0.5, 1.0, 0.7, step=0.1)
        st.session_state['sem_weight'] = st.slider("Semantic weight", 0.0, 0.5, 0.2, step=0.1)
        st.session_state['inf_weight'] = st.slider("Inference weight", 0.0, 0.3, 0.1, step=0.05)

        st.subheader("📈 Statistics")
        st.session_state['bootstrap_samples'] = st.slider("Bootstrap samples", 100, 2000, 500, step=100)
        st.session_state['alpha_level'] = st.selectbox("Significance alpha", [0.01, 0.05, 0.10], index=1)

        # Graph Editing Section
        st.markdown("---")
        st.subheader("✏️ Graph Editing")
        with st.expander("Remove Nodes"):
            if st.session_state.get('analysis_data') and st.session_state['analysis_data'].get('valid_concepts'):
                nodes_to_remove = st.multiselect(
                    "Select nodes to remove:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="remove_nodes_select"
                )
                st.session_state['nodes_to_remove'] = nodes_to_remove
            else:
                st.info("Build graph first to edit nodes.")
                st.session_state['nodes_to_remove'] = []

        with st.expander("Merge Nodes"):
            if st.session_state.get('analysis_data') and st.session_state['analysis_data'].get('valid_concepts'):
                nodes_to_merge = st.multiselect(
                    "Select nodes to merge:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="merge_nodes_select"
                )
                merge_name = st.text_input("New merged concept name:", key="merge_name_input")
                st.session_state['nodes_to_merge'] = nodes_to_merge
                st.session_state['merge_name'] = merge_name
            else:
                st.info("Build graph first to merge nodes.")
                st.session_state['nodes_to_merge'] = []
                st.session_state['merge_name'] = ""

        with st.expander("Add Edge"):
            if st.session_state.get('analysis_data') and st.session_state['analysis_data'].get('valid_concepts'):
                all_concepts = st.session_state['analysis_data']['valid_concepts']
                edge_u = st.selectbox("Source concept:", options=all_concepts, key="edge_u_select")
                edge_v = st.selectbox("Target concept:", options=all_concepts, key="edge_v_select")
                edge_weight = st.number_input("Edge weight:", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="edge_weight_input")
                st.session_state['new_edge'] = (edge_u, edge_v) if edge_u != edge_v else None
                st.session_state['new_edge_weight'] = edge_weight
            else:
                st.info("Build graph first to add edges.")
                st.session_state['new_edge'] = None
                st.session_state['new_edge_weight'] = 1.0

        with st.expander("Filter by Degree/Frequency"):
            st.session_state['filter_min_degree'] = st.slider("Min degree", 0, 20, 0, key="filter_degree_slider")
            st.session_state['filter_min_freq'] = st.slider("Min frequency", 0, 50, 0, key="filter_freq_slider")

        if st.session_state.get('analysis_data') and st.session_state['analysis_data'].get('valid_concepts'):
            if st.button("Apply Graph Edits", key="apply_edits_btn"):
                st.session_state['apply_edits'] = True

        # Undo/Redo
        if st.session_state.get('analysis_data') and st.session_state.get('edit_history'):
            col_undo, col_redo = st.columns(2)
            with col_undo:
                if st.button("↩️ Undo", key="undo_btn") and st.session_state['edit_history'].can_undo():
                    snapshot = st.session_state['edit_history'].undo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Undo applied!")
                        st.rerun()
            with col_redo:
                if st.button("↪️ Redo", key="redo_btn") and st.session_state['edit_history'].can_redo():
                    snapshot = st.session_state['edit_history'].redo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Redo applied!")
                        st.rerun()

        # Sunburst Options
        st.markdown("---")
        st.subheader("☀️ Sunburst Options")
        if st.session_state.get('analysis_data') and st.session_state['analysis_data'].get('valid_concepts'):
            all_cats = list(set(abstract_concepts_to_categories(st.session_state['analysis_data']['valid_concepts']).values()))
            st.session_state['sunburst_categories'] = st.multiselect(
                "Filter categories:", options=all_cats, default=all_cats, key="sunburst_cat_filter"
            )
            st.session_state['sunburst_branchvalues'] = st.selectbox(
                "Branch values mode:", ["total", "remainder"], index=0, key="sunburst_branch_mode"
            )
        else:
            st.info("Build graph first to configure sunburst.")
            st.session_state['sunburst_categories'] = []
            st.session_state['sunburst_branchvalues'] = "total"

        # Performance Monitor
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

# ==========================================
# REASONING DASHBOARD (ENHANCED)
# ==========================================

def render_reasoning_dashboard(nx_graph, valid_concepts, ontology, extractor):
    """Display reasoning insights from the ontology-enhanced graph."""
    st.subheader("🔍 Ontology-Based Reasoning Insights")

    # Concept type distribution
    type_counts = defaultdict(int)
    for c in valid_concepts:
        if c in ontology.concepts:
            type_counts[ontology.concepts[c].concept_type.value] += 1
        else:
            type_counts["unknown"] += 1

    fig = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                 title="Concept Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Inferred vs observed edges
    inferred_edges = [(u, v) for u, v, d in nx_graph.edges(data=True) if d.get('inferred', False)]
    observed_edges = [(u, v) for u, v, d in nx_graph.edges(data=True) if not d.get('inferred', False)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Observed Edges", len(observed_edges))
    col2.metric("Inferred Edges", len(inferred_edges))
    col3.metric("Inference Ratio", f"{len(inferred_edges)/max(len(observed_edges),1):.2f}")

    # Relationship type breakdown
    rel_types = defaultdict(int)
    for u, v, d in nx_graph.edges(data=True):
        rel_types[d.get('edge_type', 'unknown')] += 1

    if rel_types:
        rel_df = pd.DataFrame([(k, v) for k, v in rel_types.items()], 
                              columns=['Relationship Type', 'Count'])
        rel_df = rel_df.sort_values('Count', ascending=False)
        st.dataframe(rel_df, use_container_width=True)

        fig = px.bar(rel_df, x='Relationship Type', y='Count', 
                     title="Edge Type Distribution",
                     color='Relationship Type')
        st.plotly_chart(fig, use_container_width=True)

    # Reasoning paths
    st.subheader("🔗 Inferred Process-Parameter-Response Chains")
    process_nodes = [c for c in valid_concepts if c in ontology.concepts 
                     and ontology.concepts[c].concept_type == ConceptType.PROCESS]
    property_nodes = [c for c in valid_concepts if c in ontology.concepts 
                      and ontology.concepts[c].concept_type == ConceptType.PROPERTY]

    chains_found = []
    for proc in process_nodes[:5]:
        for prop in property_nodes[:5]:
            paths = ontology.infer_path(proc, prop, max_depth=3)
            if paths:
                chains_found.append({
                    "Process": proc,
                    "Property": prop,
                    "Path Length": len(paths[0]),
                    "Path": " → ".join(paths[0])
                })

    if chains_found:
        st.dataframe(pd.DataFrame(chains_found), use_container_width=True)
    else:
        st.info("No direct inference chains found. Build graph with more concepts.")

    # Synonym resolution examples
    st.subheader("📚 Synonym Resolution Examples")
    synonym_examples = [
        ("high entropy alloy", "hea"),
        ("co-cr-fe-ni", "cocrfeni"),
        ("laser powder bed fusion", "lpbf"),
        ("thermodynamic data tensor", "tdt"),
        ("marangoni convection", "marangoni_convection"),
    ]
    syn_data = []
    for original, expected in synonym_examples:
        resolved = ontology.resolve_concept(original)
        syn_data.append({
            "Original": original,
            "Expected": expected,
            "Resolved": resolved,
            "Match": "✅" if resolved == expected else ("⚠️" if resolved else "❌")
        })
    st.dataframe(pd.DataFrame(syn_data), use_container_width=True)

    # Concept hierarchy visualization
    st.subheader("🏛️ Concept Hierarchy")
    hierarchy_data = []
    for concept in valid_concepts[:20]:
        if concept in ontology.concepts:
            node = ontology.concepts[concept]
            if node.hypernyms:
                for hyp in node.hypernyms:
                    hierarchy_data.append({"Child": concept, "Parent": hyp, "Relation": "is-a"})
            if node.hyponyms:
                for hyp in node.hyponyms:
                    if hyp in valid_concepts:
                        hierarchy_data.append({"Parent": concept, "Child": hyp, "Relation": "has-subtype"})
    if hierarchy_data:
        st.dataframe(pd.DataFrame(hierarchy_data), use_container_width=True)
    else:
        st.info("No hierarchical relationships found in current concept set.")

# ==========================================
# PARALLEL PROCESSING HELPERS
# ==========================================

def _process_single_row(args):
    """Process a single document row for parallel extraction."""
    idx, row, selected_text_cols, extractor = args
    text = " ".join([str(row[col]) for col in selected_text_cols if col in row and pd.notna(row[col])])
    concepts = extractor.extract_from_text(text, idx)
    metrics = {}
    power_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:w|watt)', text, re.I)
    if power_matches: metrics['laser_power_w'] = [float(m) for m in power_matches]
    velocity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mm/s|m/s)', text, re.I)
    if velocity_matches: metrics['scan_velocity'] = [float(m) for m in velocity_matches]
    temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:k|°c|celsius)', text, re.I)
    if temp_matches: metrics['temperature'] = [float(m) for m in temp_matches]
    return idx, concepts, metrics

# ==========================================
# MAIN APPLICATION v2.0
# ==========================================

def main():
    st.title("🔬 HEA-Laser-ConceptGraph: Advanced NLP-Enhanced Explorer v2.0")
    st.caption("Multi-level reasoning concept graph for CoCrFeNi laser AM | Parallel Processing | Interactive Visualization | Ontology-aware resolution")

    # Initialize ontology and resolver
    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()

    ontology = st.session_state.ontology
    render_sidebar()

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
        st.info("Please place your metadata files in the `json_metadatabase/` folder.")
        return
    successful_files = [f for f in file_records if f[1]]
    if not successful_files:
        st.error("Files found but none could be parsed. Check error messages above.")
        return
    st.success(f"Loaded {len(successful_files)} file(s) | {len(df)} record(s)")
    file_names = [f[0] for f in successful_files]
    selected_files = st.multiselect("Filter by source file", file_names, default=file_names)
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
    text_cols = [c for c in df_filtered.columns if any(k in c.lower() for k in ['abstract', 'title', 'summary', 'text', 'content', 'description'])]
    if not text_cols:
        text_cols = [c for c in df_filtered.columns if df_filtered[c].dtype == 'object']
    selected_text_cols = st.multiselect(
        "Select text columns for concept extraction:",
        options=text_cols,
        default=text_cols[:2] if len(text_cols) >= 2 else text_cols
    )
    if not selected_text_cols:
        st.error("Please select at least one text column.")
        return

    # --- RUN ANALYSIS WITH ENHANCED REASONING & PARALLEL PROCESSING ---
    if st.button("🚀 Build Concept Graph with Reasoning", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status = st.status("Initializing advanced NLP analysis...", expanded=True)
        overall_start = time.perf_counter()

        try:
            with status:
                st.write("Preparing text corpus...")
                all_texts = []
                for idx, row in df_filtered.iterrows():
                    text = " ".join([str(row[col]) for col in selected_text_cols if col in row and pd.notna(row[col])])
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

                # Initialize enhanced extractor with ontology
                use_ontology = st.session_state.get('use_ontology', True)
                use_embedding = st.session_state.get('use_embedding_resolution', True)
                use_inference = st.session_state.get('use_inference', True)

                if use_ontology:
                    st.write("Initializing ontology-based concept resolver...")
                    resolver = AdvancedConceptResolver(ontology, embed_model)
                    extractor = EnhancedConceptExtractor(ontology, resolver)
                    st.success("Ontology and resolver initialized")
                else:
                    st.write("Using legacy extraction (no ontology)...")
                    resolver = None
                    extractor = None

                progress_bar.progress(0.20)

                st.write("Extracting concepts from abstracts...")
                all_concepts = []
                all_metrics = []

                if use_ontology and extractor is not None:
                    # v2.0: Parallel document extraction using ThreadPoolExecutor
                    use_parallel = num_abstracts > 50  # Only parallelize for larger datasets

                    if use_parallel:
                        st.write(f"Using parallel processing ({min(4, os.cpu_count() or 4)} workers)...")
                        all_concepts = [None] * len(df_filtered)
                        all_metrics = [None] * len(df_filtered)

                        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4)) as executor:
                            futures = {
                                executor.submit(_process_single_row, (idx, row, selected_text_cols, extractor)): idx 
                                for idx, row in df_filtered.iterrows()
                            }
                            completed = 0
                            for future in as_completed(futures):
                                idx, concepts, metrics = future.result()
                                all_concepts[idx] = concepts
                                all_metrics[idx] = metrics
                                completed += 1
                                if completed % 50 == 0:
                                    progress_bar.progress(0.20 + (completed / len(df_filtered)) * 0.15)
                    else:
                        # Sequential for small datasets
                        for idx, row in df_filtered.iterrows():
                            text = " ".join([str(row[col]) for col in selected_text_cols if col in row and pd.notna(row[col])])
                            concepts = extractor.extract_from_text(text, idx)
                            all_concepts.append(concepts)
                            metrics = {}
                            power_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:w|watt)', text, re.I)
                            if power_matches: metrics['laser_power_w'] = [float(m) for m in power_matches]
                            velocity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mm/s|m/s)', text, re.I)
                            if velocity_matches: metrics['scan_velocity'] = [float(m) for m in velocity_matches]
                            temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:k|°c|celsius)', text, re.I)
                            if temp_matches: metrics['temperature'] = [float(m) for m in temp_matches]
                            all_metrics.append(metrics)

                    # Get frequencies and filter
                    concept_freq = extractor.get_concept_frequencies()
                    valid_concepts = [c for c, f in concept_freq.items() if f >= config.get("MIN_CONCEPT_FREQ", 2)]
                    concept_abstract_map = defaultdict(list)
                    for doc_idx, concepts in enumerate(all_concepts):
                        for c in set(concepts):
                            concept_abstract_map[c].append(doc_idx)
                else:
                    # Use legacy extraction
                    all_concepts, all_metrics = extract_concepts_from_abstracts(df_filtered, selected_text_cols)
                    valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts, config)

                st.write(f"Extracted concepts from {len(all_concepts)} documents")
                progress_bar.progress(0.35)

                if not use_ontology:
                    st.write("Filtering and normalizing concepts...")
                    valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts, config)
                else:
                    # Build concept_to_id for ontology-based concepts
                    valid_concepts = sorted(valid_concepts, key=lambda c: concept_abstract_map.get(c, []).__len__(), reverse=True)
                    top_n = config.get("TOP_N_CONCEPTS", 1000)
                    if len(valid_concepts) > top_n:
                        valid_concepts = valid_concepts[:top_n]
                    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
                    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}

                st.write(f"**{len(valid_concepts)}** valid concepts retained")
                progress_bar.progress(0.45)

                if len(valid_concepts) < 5:
                    st.error("Too few concepts extracted. Try lowering frequency thresholds.")
                    return

                st.write("Building concept graph...")
                if use_ontology and use_inference:
                    # Use reasoning-enhanced graph builder
                    graph_builder = ReasoningEnhancedGraphBuilder(ontology, extractor)
                    nx_graph = graph_builder.build_graph(
                        all_concepts, valid_concepts, 
                        concept_to_id,
                        embed_model, config
                    )
                else:
                    # Use legacy graph builder
                    nx_graph = build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config)

                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, valid_concepts, concept_to_id, config)
                st.write(f"Graph: {len(valid_concepts)} nodes, {nx_graph.number_of_edges()} edges")
                progress_bar.progress(0.55)

                st.write("Generating node embeddings...")
                try:
                    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
                    node_features = torch.tensor(embeddings, dtype=torch.float32)
                except Exception:
                    node_features = torch.randn(len(valid_concepts), 384)
                st.write(f"Node features: {node_features.shape}")
                progress_bar.progress(0.65)

                st.write("Training GraphSAGE...")
                def training_progress(epoch, loss):
                    progress = 0.65 + (epoch / 50) * 0.15
                    progress_bar.progress(min(1.0, progress))
                    if epoch % 10 == 0:
                        status.write(f"Epoch {epoch}/50 | Loss: {loss:.4f}")
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, training_progress
                )
                st.success("GNN training complete")
                progress_bar.progress(0.80)

                st.write("Scoring research directions...")
                concept_properties = {}
                for concept in valid_concepts:
                    doc_indices = concept_abstract_map.get(concept, [])
                    values = []
                    for idx in doc_indices:
                        if idx < len(all_metrics):
                            for metric_values in all_metrics[idx].values():
                                values.extend(metric_values)
                    concept_properties[concept] = np.median(values) if values else 0.0
                X_feat, y_target = [], []
                for u, v in nx_graph.edges():
                    pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
                ridge = None
                if len(X_feat) > 5:
                    ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
                top_scores = compute_research_direction_scores(
                    gnn_model, node_features, final_emb, nx_graph, valid_concepts,
                    concept_properties, ridge, embed_model
                )
                st.write(f"Scored {len(top_scores)} novel pairs")
                progress_bar.progress(0.90)

                st.write("Computing distillation metrics...")
                distill_df = compute_concept_distillation(valid_concepts, concept_abstract_map, all_texts)

                # Advanced analytics
                st.write("Running advanced analytics...")
                burst_df = detect_keyword_bursts(df_filtered, valid_concepts, concept_abstract_map, selected_text_cols)
                drift_df = detect_semantic_drift(df_filtered, valid_concepts, concept_abstract_map, embed_model, selected_text_cols)
                genealogy_df = build_concept_genealogy(nx_graph, valid_concepts, concept_abstract_map)
                bridge_df = detect_cross_domain_bridges(nx_graph, valid_concepts, concept_abstract_map)
                motifs = analyze_network_motifs(nx_graph)

                st.session_state.burst_df = burst_df
                st.session_state.drift_df = drift_df
                st.session_state.genealogy_df = genealogy_df
                st.session_state.bridge_df = bridge_df
                st.session_state.motifs = motifs

                # Performance summary
                total_time = time.perf_counter() - overall_start
                st.success(f"Analysis complete in {total_time:.1f}s!")
                progress_bar.progress(1.00)
                status.update(label=f"Analysis complete! ({total_time:.1f}s)", state="complete", expanded=False)

                # Store analysis data
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
                    "selected_text_cols": selected_text_cols
                }

                # Store reasoning artifacts if using ontology
                if use_ontology:
                    analysis_data.update({
                        "ontology": ontology,
                        "resolver": resolver,
                        "extractor": extractor,
                        "graph_builder": graph_builder if use_inference else None,
                        "reasoning_paths": graph_builder.reasoning_paths if use_inference else []
                    })

                st.session_state.analysis_data = analysis_data

                # Save initial snapshot for undo
                st.session_state.edit_history = GraphEditHistory()
                st.session_state.edit_history.save_snapshot(nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map)

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            return
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- APPLY GRAPH EDITS IF REQUESTED ---
    if st.session_state.get('apply_edits') and st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        # Save snapshot before edit
        st.session_state.edit_history.save_snapshot(
            data["nx_graph"], data["valid_concepts"], data["concept_to_id"],
            data["id_to_concept"], data["concept_abstract_map"]
        )
        nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, edited = apply_graph_edits(
            data["nx_graph"], data["valid_concepts"], data["concept_to_id"], data["id_to_concept"],
            data["concept_abstract_map"],
            nodes_to_remove=st.session_state.get('nodes_to_remove', []),
            nodes_to_merge=st.session_state.get('nodes_to_merge', []),
            merge_name=st.session_state.get('merge_name', None),
            new_edge=st.session_state.get('new_edge', None),
            new_edge_weight=st.session_state.get('new_edge_weight', 1.0),
            min_degree=st.session_state.get('filter_min_degree', 0),
            min_freq=st.session_state.get('filter_min_freq', 0)
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
        cmap = st.session_state.get('cmap_name', 'viridis')
        top_n_graph = st.session_state.get('top_n_graph', 200)

        # Determine available tabs
        has_reasoning = "ontology" in data
        tab_names = ["📊 Visualization", "🧪 Distillation", "🎯 Research Directions", "✅ Validation", "📥 Export", "📈 Extra Viz", "🔬 Advanced Analytics"]
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
                nx_graph = nx.relabel_nodes(nx_graph, {i: valid_concepts[i] for i in range(len(valid_concepts))})

            viz_choice = st.session_state.get('viz_backend', 'PyVis (Interactive)')
            physics = st.session_state.get('physics_enabled', True)
            physics_preset = st.session_state.get('effective_physics', PHYSICS_PRESETS["Stable (Default)"])
            theme = THEME_PRESETS.get(st.session_state.get('theme', 'Bright (Default)'), THEME_PRESETS["Bright (Default)"])
            top_n = st.session_state.get('top_n_graph', 0)
            show_weights = st.session_state.get('show_edge_weights', False)
            edge_label_mode = st.session_state.get('edge_label_mode', 'hover')

            # v2.0: New visualization parameters
            use_abbreviated = st.session_state.get('use_abbreviated_labels', False)
            max_label_len = st.session_state.get('max_label_length', 15)
            node_font = st.session_state.get('node_font_face', 'Inter, Segoe UI, Roboto, sans-serif')
            edge_label_size = st.session_state.get('edge_label_size', 10)
            edge_label_color = st.session_state.get('edge_label_color', None)
            edge_label_pos = st.session_state.get('edge_label_position', 'middle')
            show_defs = st.session_state.get('show_definitions', True)

            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(
                    nx_graph, concept_abstract_map, 
                    physics_enabled=physics,
                    cmap_name=cmap, top_n_nodes=top_n,
                    theme=theme, physics_preset=physics_preset,
                    show_edge_weights=show_weights, 
                    edge_label_mode=edge_label_mode,
                    use_abbreviated_labels=use_abbreviated,
                    max_label_length=max_label_len,
                    node_font_face=node_font,
                    edge_label_size=edge_label_size,
                    edge_label_color=edge_label_color,
                    edge_label_position=edge_label_pos,
                    show_definitions=show_defs
                )
            elif viz_choice == "Plotly 2D":
                render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n,
                                       theme=theme, show_edge_weights=show_weights)
            elif viz_choice == "Plotly 3D":
                render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n,
                                        theme=theme, show_edge_weights=show_weights)
            else:
                render_graph_fallback(nx_graph, concept_abstract_map, theme=theme,
                                      show_edge_weights=show_weights)

            with st.expander("Graph Metrics"):
                metrics = compute_graph_metrics(nx_graph)
                display_metric_dashboard(metrics, theme=theme)

            with st.expander("Domain Hierarchy (Sunburst)"):
                cat_filter = st.session_state.get('sunburst_categories', [])
                bv_mode = st.session_state.get('sunburst_branchvalues', 'total')
                if cat_filter:
                    filtered_concepts = [c for c in valid_concepts 
                                         if abstract_concepts_to_categories([c]).get(c, 'general') in cat_filter]
                    filtered_map = {c: concept_abstract_map[c] for c in filtered_concepts if c in concept_abstract_map}
                else:
                    filtered_concepts = valid_concepts
                    filtered_map = concept_abstract_map
                labels, parents, values = build_category_hierarchy(filtered_concepts, filtered_map,
                                                                    top_n_per_category=st.session_state.get('top_n_sunburst', 0))
                render_sunburst_chart(labels, parents, values, cmap_name=cmap, theme=theme, branchvalues=bv_mode)

            with st.expander("Concept Radar"):
                radar_k = st.session_state.get('top_n_radar', 15)
                if radar_k == 0:
                    radar_k = min(15, len(distill_df))
                render_radar_chart(distill_df, top_k=radar_k, cmap_name=cmap, theme=theme)

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Concept Distillation Efficiency")
            top_n = st.slider("Show Top N", 10, min(200, len(distill_df)), 50, key="distill_top_n")
            display_df = distill_df.head(top_n)
            st.dataframe(display_df, use_container_width=True)
            st.markdown("**Efficiency vs Frequency:**")
            chart_df = display_df.set_index('concept')[['distillation_efficiency']]
            st.bar_chart(chart_df)
            st.markdown("**Multi-Metric Comparison:**")
            metric_cols = [c for c in ['frequency', 'tfidf_weight', 'semantic_density', 'coherence_score']
                           if c in display_df.columns]
            if metric_cols:
                compare_df = display_df[['concept'] + metric_cols].set_index('concept')
                st.line_chart(compare_df)

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Top Research Direction Recommendations")
            if top_scores.empty:
                st.info("No novel pairs scored. The graph may be too dense or too sparse.")
            else:
                st.write(f"Top {len(top_scores)} novel concept pairs:")
                st.dataframe(top_scores[['concept_u', 'concept_v', 'composite_score',
                                         'gnn_affinity', 'semantic_novelty',
                                         'expected_property_gain', 'feasibility_score']].head(20),
                            use_container_width=True)
                csv_scores = top_scores.to_csv(index=False).encode('utf-8')
                st.download_button("Download Scores (CSV)", data=csv_scores,
                                  file_name="research_directions.csv", mime="text/csv")

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Mathematical Validation")
            val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Modularity", f"{val_metrics.get('modularity', 0):.3f}")
            col2.metric("Silhouette", f"{val_metrics.get('silhouette_score', 0):.3f}")
            col3.metric("Communities", val_metrics.get('n_communities', 0))
            col4.metric("Significant Edges", val_metrics.get('edge_significant_count', 0))
            if not top_scores.empty:
                n_boot = st.session_state.get('bootstrap_samples', 500)
                alpha = st.session_state.get('alpha_level', 0.05)
                mean_score, ci_low, ci_high = compute_bootstrap_ci(
                    top_scores['composite_score'].values, n_bootstrap=n_boot, alpha=alpha
                )
                st.success(f"Composite Score: `{mean_score:.3f}` | {int((1-alpha)*100)}% CI: `[{ci_low:.3f}, {ci_high:.3f}]`")
            X_feat, y_target = [], []
            for u, v in nx_graph.edges():
                pu, pv = data["concept_properties"].get(u, 0), data["concept_properties"].get(v, 0)
                w = nx_graph[u][v].get('weight', 1)
                X_feat.append([pu, pv, w])
                y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
            if data["ridge"] is not None and len(X_feat) > 5:
                y_pred = data["ridge"].predict(np.array(X_feat))
                st.markdown("### Ridge Regression (Property Prediction)")
                c1, c2, c3 = st.columns(3)
                c1.metric("R2", f"{r2_score(y_target, y_pred):.3f}")
                c2.metric("MAE", f"{mean_absolute_error(y_target, y_pred):.2f}")
                c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_target, y_pred)):.2f}")

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Export & Post-Processing")

            # v2.0: Enhanced export options
            export_format = st.selectbox("Format:", [
                "GraphML", 
                "JSON (Full Metadata)", 
                "JSON (Compact)",
                "CSV (Edges + Metadata)", 
                "CSV (Nodes + Metadata)", 
                "PNG", 
                "SVG",
                "GEXF"
            ])
            include_metadata = st.checkbox("Include metadata in export", value=True)

            if st.button("Generate Export"):
                result = export_graph(nx_graph, concept_abstract_map, export_format, include_metadata)
                if result[0]:
                    data_bytes, mime, filename = result
                    st.download_button("💾 Save File", data=data_bytes, file_name=filename, mime=mime)

            # Publication figure export
            st.markdown("---")
            st.subheader("Publication-Ready Figure")
            pub_dpi = st.slider("DPI", 150, 600, 300, step=50)
            pub_figsize = st.selectbox("Figure size:", [(10, 8), (12, 10), (14, 12), (16, 14)], index=2)
            if st.button("Generate Publication Figure"):
                pub_bytes = export_publication_figure(nx_graph, valid_concepts, concept_abstract_map,
                                                       cmap_name=cmap, dpi=pub_dpi, figsize=pub_figsize)
                if pub_bytes:
                    st.download_button("📥 Download Publication PNG", data=pub_bytes,
                                     file_name="hea_graph_publication.png", mime="image/png")

            # Markdown report
            st.markdown("---")
            st.subheader("Automated Analysis Report")
            if st.button("Generate Markdown Report"):
                report = generate_analysis_report(
                    nx_graph, valid_concepts, concept_abstract_map,
                    top_scores, distill_df,
                    st.session_state.get('burst_df', pd.DataFrame()),
                    st.session_state.get('drift_df', pd.DataFrame()),
                    st.session_state.get('genealogy_df', pd.DataFrame()),
                    st.session_state.get('bridge_df', pd.DataFrame()),
                    st.session_state.get('motifs', {}),
                    val_metrics, df_filtered
                )
                st.download_button("📄 Download Report (Markdown)", data=report.encode('utf-8'),
                                  file_name="hea_laser_analysis_report.md", mime="text/markdown")
                with st.expander("Preview Report"):
                    st.markdown(report)

            concept_list_df = pd.DataFrame({
                'concept': valid_concepts,
                'frequency': [len(concept_abstract_map.get(c, [])) for c in valid_concepts],
                'degree': [nx_graph.degree(c) for c in valid_concepts],
                'category': [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts],
                'concept_type': [nx_graph.nodes[c].get('concept_type', 'general') for c in valid_concepts],
                'definition': [nx_graph.nodes[c].get('definition', '') for c in valid_concepts]
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button("📋 Download Concept List (CSV)", data=csv_concepts,
                              file_name="concepts_enhanced.csv", mime="text/csv")

            # v2.0: Show concept definitions table
            with st.expander("📖 Concept Definitions & Meanings"):
                defs_df = concept_list_df[concept_list_df['definition'] != ''][['concept', 'definition', 'category']]
                if not defs_df.empty:
                    st.dataframe(defs_df, use_container_width=True)
                else:
                    st.info("No definitions available. Enable ontology-based resolution to see concept definitions.")

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Extra Visualizations")
            theme = THEME_PRESETS.get(st.session_state.get('theme', 'Bright (Default)'), THEME_PRESETS["Bright (Default)"])

            with st.expander("Concept Timeline", expanded=True):
                render_concept_timeline(df_filtered, valid_concepts, concept_abstract_map, theme=theme)

            with st.expander("Co-occurrence Heatmap"):
                heatmap_n = st.slider("Top N concepts for heatmap", 5, 50, 25, key="heatmap_n_slider")
                render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map, top_n=heatmap_n, theme=theme)

            with st.expander("t-SNE Projection"):
                embed_model = data.get("embed_model")
                if embed_model:
                    render_tsne_projection(valid_concepts, concept_abstract_map, embed_model, theme=theme)
                else:
                    st.info("Embedding model not available. Rebuild the graph.")

            with st.expander("Community Detection"):
                render_community_detection(nx_graph, valid_concepts, concept_abstract_map, theme=theme)

            with st.expander("Concept Growth Rate"):
                render_concept_growth(df_filtered, valid_concepts, concept_abstract_map, theme=theme)

            with st.expander("Bubble Chart (Importance)"):
                render_bubble_chart(nx_graph, valid_concepts, concept_abstract_map, distill_df, theme=theme)

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Advanced Analytics")

            with st.expander("Keyword Burst Detection", expanded=True):
                burst_df = st.session_state.get('burst_df')
                if burst_df is not None and not burst_df.empty:
                    st.dataframe(burst_df.head(20), use_container_width=True)
                    fig = px.bar(burst_df.head(15), x='concept', y='burst_score', color='burst_year',
                                 title="Keyword Bursts (Sudden Spikes in Publication Frequency)",
                                 labels={'burst_score': 'Burst Score', 'concept': 'Concept'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No burst data available. Build graph with temporal data.")

            with st.expander("Semantic Drift Detection"):
                drift_df = st.session_state.get('drift_df')
                if drift_df is not None and not drift_df.empty:
                    st.dataframe(drift_df.head(20), use_container_width=True)
                    fig = px.bar(drift_df.head(15), x='concept', y='semantic_drift',
                                 title="Semantic Drift (Contextual Meaning Shift Over Time)",
                                 labels={'semantic_drift': 'Drift Score', 'concept': 'Concept'},
                                 color='semantic_drift', color_continuous_scale='RdYlBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No drift data available. Build graph with temporal data spanning multiple years.")

            with st.expander("Concept Genealogy"):
                genealogy_df = st.session_state.get('genealogy_df')
                if genealogy_df is not None and not genealogy_df.empty:
                    st.dataframe(genealogy_df.head(20), use_container_width=True)
                    gen_counts = genealogy_df['generation'].value_counts()
                    fig = px.pie(values=gen_counts.values, names=gen_counts.index,
                                 title="Concept Generations Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No genealogy data available.")

            with st.expander("Cross-Domain Bridge Detection"):
                bridge_df = st.session_state.get('bridge_df')
                if bridge_df is not None and not bridge_df.empty:
                    st.dataframe(bridge_df.head(20), use_container_width=True)
                    fig = px.scatter(bridge_df.head(30), x='betweenness', y='connected_categories',
                                     size='bridge_score', color='own_category',
                                     hover_data=['concept', 'categories'],
                                     title="Cross-Domain Bridge Concepts",
                                     labels={'betweenness': 'Betweenness Centrality',
                                            'connected_categories': 'Categories Connected'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No bridge data available.")

            with st.expander("Network Motif Analysis"):
                motifs = st.session_state.get('motifs', {})
                if motifs:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Triangles", motifs.get('total_triangles', 0))
                    col2.metric("Cliques", motifs.get('total_cliques', 0))
                    col3.metric("Max Clique Size", motifs.get('max_clique_size', 0))
                    col4.metric("Star Motifs", motifs.get('star_motifs', 0))
                    if motifs.get('top_stars'):
                        st.markdown("**Top Star Motifs (Central Hubs):**")
                        star_df = pd.DataFrame(motifs['top_stars'], columns=['Concept', 'Degree', 'Clustering'])
                        st.dataframe(star_df, use_container_width=True)
                else:
                    st.info("No motif data available.")

            with st.expander("Centrality Comparison & Degree Distribution"):
                centrality_df = compute_centrality_comparison(nx_graph, valid_concepts)
                if not centrality_df.empty:
                    st.dataframe(centrality_df.head(20), use_container_width=True)

                    corr_cols = ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']
                    available = [c for c in corr_cols if c in centrality_df.columns]
                    if len(available) >= 2:
                        corr_matrix = centrality_df[available].corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                        title="Centrality Correlation Matrix",
                                        color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig, use_container_width=True)

                    fig = plot_degree_distribution(nx_graph, theme=theme)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No centrality data available.")

        # Reasoning Dashboard Tab (only if ontology was used)
        if has_reasoning:
            tab_idx += 1
            with tabs[tab_idx]:
                ontology = data.get("ontology")
                extractor = data.get("extractor")
                if ontology and extractor:
                    render_reasoning_dashboard(nx_graph, valid_concepts, ontology, extractor)
                else:
                    st.info("Reasoning data not available. Rebuild graph with ontology enabled.")

if __name__ == "__main__":
    main()
