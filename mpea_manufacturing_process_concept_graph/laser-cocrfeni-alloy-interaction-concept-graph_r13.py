#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HEA-Laser-ConceptGraph v5.1.2 (Faithful AgNPs Architecture Port)
==============================================================
This is a TRUE architectural port of the AgNP-Sustainability-ConceptGraph
codebase, preserving every memory-safe pattern, visualization pattern,
and session-state management pattern from the working AgNPs code.

v5.1.2 FIXES (Batch Processing Crash):
- CRITICAL: Fixed UnboundLocalError in ConceptExtractor.extract_from_text —
  `self.concept_contexts[concept].append(...)` referenced loop variable
  `concept` OUTSIDE the `for concept in concepts:` loop, crashing whenever
  a document yielded zero extracted concepts. Line moved inside the loop.
- CRITICAL: Fixed TypeError in IncrementalGraphBuilder.process_batch —
  `self.extractor.concept_frequencies()` called the defaultdict ATTRIBUTE
  as a function. Replaced with the getter `get_concept_frequencies()`,
  matching the main non-batch pipeline.

v5.1.1 FIXES (Batch Processing Silent Failure):
- CRITICAL: Builder now initializes in BOTH ontology and non-ontology modes
- CRITICAL: Reinitializes if session_state contains stale/corrupted builder (None)
- Ontology initialization wrapped in explicit try/except with error persistence
- Error state stored in session_state (survives st.rerun)
- Defensive checks for empty graphs / insufficient concepts before GNN training
- GNN training wrapped with graceful fallback for small/edge-case graphs
- Sidebar shows error-aware status (not misleading "Batch 0/5")
- Default batch size reduced to 500 for stability on limited RAM
- Memory cleanup in finally block ensures no leaks on crash
- Troubleshooting tips displayed directly in UI on failure

DOMAIN: CoCrFeNi HEA Laser Additive Manufacturing
- Materials: CoCrFeNi, HEA, MPEA, CCA, FCC/BCC phases
- Processes: LPBF, LAM, DED, rapid solidification, melt pool dynamics
- Thermodynamics: TDT, CPD, CALPHAD, Gibbs energy, CTF
- Phase-field: Allen-Cahn, KKS, multicomponent diffusion
- Fluid dynamics: Marangoni, Navier-Stokes, Boussinesq
- AI surrogates: Transformer, attention, digital twin
- Microstructure: elemental partitioning, grains, segregation
- Computational: FEA, MOOSE, ALS, tensor factorization

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run:
    streamlit run hea_laser_concept_graph_v5_1_2_batch.py

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
import math
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
    page_title="HEA-Laser-ConceptGraph v5.1.1.1: Faithful AgNPs Architecture Port",
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
# ENHANCED ONTOLOGY & NLP REASONING SYSTEM (HEA-LASER DOMAIN)
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
    DEGRADES = "degrades"
    REDUCES = "reduces"


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
    """Comprehensive ontology for CoCrFeNi HEA laser AM domain."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self._build_ontology()

    def _build_ontology(self) -> None:
        # === MATERIALS ===
        self._add_concept(
            "cocrfeni", ConceptType.MATERIAL,
            synonyms={
                "co-cr-fe-ni", "co cr fe ni", "cobalt chromium iron nickel",
                "cocofeni", "cocrfeni alloy", "cocrfeni hea",
                "co-cr-fe-ni alloy", "co cr fe ni alloy",
            },
            definition="Quaternary high-entropy alloy system",
        )
        self._add_concept(
            "hea", ConceptType.MATERIAL,
            synonyms={
                "high entropy alloy", "high-entropy alloy",
                "high entropy alloys", "heas",
                "multi-principal element alloy", "mpea",
                "multi principal element alloy", "mpeas",
                "quaternary alloy", "quaternary system",
                "complex concentrated alloy", "cca",
            },
            hypernyms={"alloy"},
            definition="High-entropy alloy class",
        )
        self._add_concept(
            "alloy", ConceptType.MATERIAL,
            synonyms={
                "alloys", "metallic alloy", "multi-component alloy",
            },
            hyponyms={"hea", "cocrfeni", "steel", "superalloy"},
        )
        self._add_concept(
            "fcc_phase", ConceptType.MICROSTRUCTURE,
            synonyms={
                "fcc solid", "fcc matrix", "face-centered cubic",
                "face centered cubic", "gamma phase", "γ phase",
                "austenitic phase",
            },
            definition="Face-centered cubic crystal structure",
        )
        self._add_concept(
            "bcc_phase", ConceptType.MICROSTRUCTURE,
            synonyms={
                "bcc solid", "body-centered cubic", "body centered cubic",
                "alpha phase", "α phase",
            },
            definition="Body-centered cubic crystal structure",
        )
        self._add_concept(
            "liquid_phase", ConceptType.MATERIAL,
            synonyms={
                "liquid state", "melt pool", "molten pool", "molten metal",
                "liquid metal", "melt zone", "fusion zone", "melted region",
            },
            definition="Liquid/molten state during laser processing",
        )

        # === PROCESSES ===
        self._add_concept(
            "lpbf", ConceptType.PROCESS,
            synonyms={
                "laser powder bed fusion", "selective laser melting", "slm",
                "laser powder-bed fusion", "laser powder bed fusion process",
                "laser powder-bed fusion process", "lpbf process",
            },
            hypernyms={"additive_manufacturing"},
            definition="Laser Powder Bed Fusion additive manufacturing",
        )
        self._add_concept(
            "lam", ConceptType.PROCESS,
            synonyms={
                "laser additive manufacturing", "laser based additive manufacturing",
                "laser-based additive manufacturing", "laser am", "laser based am",
            },
            hypernyms={"additive_manufacturing"},
            definition="Laser-based additive manufacturing",
        )
        self._add_concept(
            "additive_manufacturing", ConceptType.PROCESS,
            synonyms={
                "am", "3d printing", "three-dimensional printing",
                "layer manufacturing", "layered manufacturing", "rapid prototyping",
            },
            hyponyms={
                "lpbf", "lam", "directed_energy_deposition", "electron_beam_melting",
            },
            definition="Additive manufacturing processes",
        )
        self._add_concept(
            "directed_energy_deposition", ConceptType.PROCESS,
            synonyms={
                "ded", "laser engineered net shaping", "lens",
                "direct laser deposition", "dld",
            },
            hypernyms={"additive_manufacturing"},
        )
        self._add_concept(
            "laser_processing", ConceptType.PROCESS,
            synonyms={
                "laser melting", "laser solidification", "laser scanning",
                "laser treatment", "laser irradiance", "laser matter interaction",
                "laser material interaction", "laser irradiation",
                "laser beam processing",
            },
            definition="General laser material processing",
        )
        self._add_concept(
            "rapid_solidification", ConceptType.PROCESS,
            synonyms={
                "rapid cooling", "rapid heating", "thermal cycling",
                "fast solidification", "high cooling rate", "ultrafast cooling",
                "directional solidification",
            },
            definition="Non-equilibrium solidification conditions",
        )
        self._add_concept(
            "melt_pool_dynamics", ConceptType.PROCESS,
            synonyms={
                "melt pool formation", "melt pool flow", "melt pool morphology",
                "melt pool depth", "melt pool width", "melt pool shape",
                "melt pool geometry", "melt pool behavior", "melt pool evolution",
                "melt pool stability",
            },
            definition="Dynamics of melt pool during laser processing",
        )

        # === THERMODYNAMICS & TENSORS ===
        self._add_concept(
            "tdt", ConceptType.MODEL,
            synonyms={
                "thermodynamic data tensor", "thermodynamic tensor",
                "gibbs free energy tensor", "gibbs energy tensor",
                "thermodynamic state tensor",
            },
            definition="Thermodynamic Data Tensor for alloy systems",
        )
        self._add_concept(
            "cpd", ConceptType.METHOD,
            synonyms={
                "canonical polyadic decomposition", "cp decomposition",
                "parafac", "tensor decomposition", "factor matrices",
                "rank decomposition",
            },
            definition="Canonical Polyadic Decomposition for tensor factorization",
        )
        self._add_concept(
            "gibbs_energy", ConceptType.PROPERTY,
            synonyms={
                "gibbs free energy", "free energy", "thermodynamic potential",
                "chemical potential", "gibbs energy landscape",
                "free energy landscape", "thermodynamic driving force",
            },
            definition="Gibbs free energy and related thermodynamic potentials",
        )
        self._add_concept(
            "calphad", ConceptType.METHOD,
            synonyms={
                "calculation of phase diagrams", "thermodynamic calculation",
                "phase diagram calculation",
            },
            definition="CALPHAD methodology for thermodynamic calculations",
        )
        self._add_concept(
            "ctf", ConceptType.MODEL,
            synonyms={
                "phase-conditioned composition tensor",
                "categorical alloy-composition tensor", "composition tensor",
                "alloy composition tensor", "categorical composition tensor",
            },
            definition="Phase-conditioned composition tensor framework",
        )
        self._add_concept(
            "quadratic_expansion", ConceptType.METHOD,
            synonyms={
                "quadratic approximation", "taylor series expansion",
                "second order expansion", "polynomial expansion",
                "series expansion",
            },
            definition="Quadratic expansion methods for thermodynamic properties",
        )

        # === PHASE-FIELD MODELING ===
        self._add_concept(
            "phase_field_model", ConceptType.MODEL,
            synonyms={
                "phase-field model", "phase field model", "phase-field modeling",
                "phase field modeling", "phase field simulation",
                "phase-field simulation", "pfm", "phase field method",
                "diffuse interface model", "diffuse interface method",
            },
            definition="Phase-field modeling framework",
        )
        self._add_concept(
            "allen_cahn", ConceptType.MODEL,
            synonyms={
                "allen-cahn equation", "allen cahn",
                "non-conserved order parameter", "order parameter evolution",
                "phase order parameter",
            },
            definition="Allen-Cahn equation for non-conserved order parameters",
        )
        self._add_concept(
            "kks_model", ConceptType.MODEL,
            synonyms={
                "kks phase-equilibrium", "kks phase equilibrium",
                "kim-kim-suzuki", "kim kim suzuki", "kks",
                "phase equilibrium model",
            },
            definition="Kim-Kim-Suzuki phase-field model",
        )
        self._add_concept(
            "multicomponent_diffusion", ConceptType.PHENOMENON,
            synonyms={
                "multi-component diffusion", "multicomponent diffusion",
                "interdiffusion", "cross diffusion", "diffusion matrix",
                "diffusion tensor", "atomic mobility",
            },
            definition="Diffusion in multicomponent alloy systems",
        )
        self._add_concept(
            "interface_mobility", ConceptType.PROPERTY,
            synonyms={
                "phase mobility", "interface kinetic coefficient",
                "interface velocity", "interface migration",
                "boundary mobility", "grain boundary mobility",
            },
            definition="Kinetic mobility of phase interfaces",
        )

        # === FLUID DYNAMICS & MELT POOL ===
        self._add_concept(
            "marangoni_convection", ConceptType.PHENOMENON,
            synonyms={
                "marangoni-driven flow", "marangoni driven flow",
                "marangoni effect", "thermocapillary convection",
                "thermocapillary flow", "surface tension gradient flow",
                "marangoni force", "marangoni stress", "marangoni number",
            },
            definition="Marangoni convection driven by surface tension gradients",
        )
        self._add_concept(
            "navier_stokes", ConceptType.METHOD,
            synonyms={
                "navier-stokes equations", "navier stokes", "n-s equations",
                "fluid flow equations", "momentum equations",
                "incompressible navier-stokes",
            },
            definition="Navier-Stokes equations for fluid dynamics",
        )
        self._add_concept(
            "surface_tension", ConceptType.PROPERTY,
            synonyms={
                "surface tension gradient", "interfacial tension",
                "capillary force", "capillary pressure", "surface energy",
                "liquid surface tension",
            },
            definition="Surface tension and related capillary phenomena",
        )
        self._add_concept(
            "boussinesq_approximation", ConceptType.METHOD,
            synonyms={
                "boussinesq", "thermal buoyancy", "buoyancy-driven flow",
                "natural convection", "thermal convection",
                "gravitational convection",
            },
            definition="Boussinesq approximation for buoyancy effects",
        )
        self._add_concept(
            "thermal_gradient", ConceptType.PARAMETER,
            synonyms={
                "temperature gradient", "thermal gradient",
                "temperature difference", "thermal profile",
                "temperature profile", "thermal field", "temperature field",
            },
            definition="Spatial temperature gradient in the melt pool",
        )
        self._add_concept(
            "keyhole", ConceptType.PHENOMENON,
            synonyms={
                "keyhole formation", "keyhole mode", "keyhole porosity",
                "keyhole collapse", "keyhole instability", "vapor cavity",
                "vapor depression",
            },
            definition="Keyhole mode in laser processing",
        )

        # === AI SURROGATE MODELS ===
        self._add_concept(
            "ai_surrogate", ConceptType.MODEL,
            synonyms={
                "surrogate model", "surrogate modeling", "ai surrogate model",
                "machine learning surrogate", "data-driven surrogate",
                "reduced order model",
            },
            definition="AI-based surrogate models for process simulation",
        )
        self._add_concept(
            "transformer", ConceptType.MODEL,
            synonyms={
                "transformer architecture", "transformer model",
                "attention-based model", "self-attention",
                "transformer network", "transformer-inspired",
            },
            definition="Transformer neural network architecture",
        )
        self._add_concept(
            "attention_mechanism", ConceptType.METHOD,
            synonyms={
                "attention", "cross-attention", "cross attention",
                "multi-head attention", "self attention",
                "query-key attention", "query key attention", "qk attention",
            },
            definition="Attention mechanism in neural networks",
        )
        self._add_concept(
            "digital_twin", ConceptType.MODEL,
            synonyms={
                "digital twin model", "virtual twin", "process twin",
                "manufacturing twin",
            },
            definition="Digital twin for process monitoring and control",
        )
        self._add_concept(
            "gaussian_locality", ConceptType.METHOD,
            synonyms={
                "gaussian locality regularization", "gaussian regularization",
                "locality constraint", "spatial locality",
                "composition similarity", "composition-based locality",
            },
            definition="Gaussian locality regularization for composition space",
        )
        self._add_concept(
            "machine_learning", ConceptType.METHOD,
            synonyms={
                "ml", "machine learning", "statistical learning",
                "supervised learning", "unsupervised learning",
                "deep learning", "neural network", "artificial neural network",
                "ann", "data-driven", "data driven", "computational intelligence",
            },
            definition="Machine learning methods",
        )

        # === MICROSTRUCTURAL FEATURES ===
        self._add_concept(
            "microstructural_evolution", ConceptType.PHENOMENON,
            synonyms={
                "microstructure evolution", "microstructure development",
                "microstructural development", "grain evolution",
                "structure evolution", "microstructure formation",
            },
            definition="Evolution of microstructure during processing",
        )
        self._add_concept(
            "elemental_partitioning", ConceptType.PHENOMENON,
            synonyms={
                "segregation", "solute partitioning", "elemental segregation",
                "compositional partitioning", "microsegregation",
                "macrosegregation", "partition coefficient",
                "distribution coefficient",
            },
            definition="Elemental partitioning and segregation during solidification",
        )
        self._add_concept(
            "solidification_kinetics", ConceptType.PHENOMENON,
            synonyms={
                "solidification", "freezing", "crystallization",
                "nucleation and growth", "dendritic growth", "cellular growth",
                "planar growth", "columnar growth",
            },
            definition="Solidification kinetics and growth mechanisms",
        )
        self._add_concept(
            "equiaxed_grains", ConceptType.MICROSTRUCTURE,
            synonyms={
                "equiaxed", "equiaxed grain", "equiaxed structure",
                "randomly oriented grains",
            },
            definition="Equiaxed grain morphology",
        )
        self._add_concept(
            "columnar_grains", ConceptType.MICROSTRUCTURE,
            synonyms={
                "columnar", "columnar grain", "columnar structure",
                "directional grains", "elongated grains", "dendritic grains",
            },
            definition="Columnar grain morphology",
        )
        self._add_concept(
            "grain_boundary", ConceptType.MICROSTRUCTURE,
            synonyms={
                "grain boundaries", "boundary", "interface",
                "interphase boundary", "solid-liquid interface",
                "s-l interface", "phase boundary",
            },
            definition="Grain and phase boundaries",
        )
        self._add_concept(
            "cooling_rate", ConceptType.PARAMETER,
            synonyms={
                "cooling rate", "solidification rate", "freezing rate",
                "thermal cooling rate", "quenching rate", "cooling speed",
                "thermal history",
            },
            definition="Cooling rate during solidification",
        )
        self._add_concept(
            "thermal_wake", ConceptType.PHENOMENON,
            synonyms={
                "thermal trail", "heat affected zone", "haz",
                "thermal affected zone", "heat affected region",
                "thermal history zone",
            },
            definition="Thermal wake/heat affected zone behind laser scan",
        )
        self._add_concept(
            "porosity", ConceptType.MICROSTRUCTURE,
            synonyms={
                "pores", "voids", "gas porosity", "lack of fusion porosity",
                "keyhole porosity", "microporosity", "porosity defect",
            },
            definition="Porosity defects in additively manufactured parts",
        )
        self._add_concept(
            "hot_tearing", ConceptType.PHENOMENON,
            synonyms={
                "hot cracking", "solidification cracking", "cracking",
                "crack formation", "solidification crack", "thermal cracking",
            },
            definition="Hot tearing/cracking during solidification",
        )

        # === COMPUTATIONAL METHODS ===
        self._add_concept(
            "fea", ConceptType.METHOD,
            synonyms={
                "finite element analysis", "finite element method", "fem",
                "finite element", "finite element simulation",
                "finite element modeling",
            },
            definition="Finite Element Analysis",
        )
        self._add_concept(
            "moose", ConceptType.METHOD,
            synonyms={
                "moose framework",
                "multiphysics object-oriented simulation environment",
                "moose multiphysics", "moose platform",
            },
            definition="MOOSE multiphysics framework",
        )
        self._add_concept(
            "als", ConceptType.METHOD,
            synonyms={
                "alternating least squares", "als algorithm",
                "tensor factorization algorithm", "cp-als", "parafac-als",
            },
            definition="Alternating Least Squares for tensor decomposition",
        )
        self._add_concept(
            "tensor_factorization", ConceptType.METHOD,
            synonyms={
                "tensor decomposition", "multiway analysis",
                "multilinear decomposition", "tensor rank decomposition",
                "higher-order svd", "hosvd", "tucker decomposition",
            },
            definition="Tensor factorization methods",
        )

        # === PARAMETERS ===
        self._add_concept(
            "laser_power", ConceptType.PARAMETER,
            synonyms={
                "laser power", "beam power", "laser beam power",
                "power density", "laser intensity", "beam intensity",
                "laser wattage",
            },
            definition="Laser power parameter",
        )
        self._add_concept(
            "scan_velocity", ConceptType.PARAMETER,
            synonyms={
                "scan speed", "scanning speed", "scanning velocity",
                "laser scan speed", "laser scanning velocity",
                "beam velocity", "scan rate",
            },
            definition="Laser scan velocity",
        )
        self._add_concept(
            "laser_temperature", ConceptType.PARAMETER,
            synonyms={
                "temperature", "melt temperature", "pool temperature",
                "peak temperature", "maximum temperature",
                "superheat temperature", "liquidus temperature",
            },
            definition="Temperature during laser processing",
        )

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
            self.relationships.append(
                Relationship(source, target, rel_type, confidence)
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
    ) -> None:
        self.ontology = ontology
        self.embed_model = embed_model
        self.resolution_cache: Dict[str, str] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.85
        self.ontology_concepts_list: Optional[List[str]] = None
        self.ontology_embedding_matrix: Optional[np.ndarray] = None
        self._precompute_ontology_embeddings()

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
# ENHANCED CONCEPT EXTRACTOR
# ============================================================================
class EnhancedConceptExtractor:
    def __init__(
        self, ontology: DomainOntology, resolver: AdvancedConceptResolver
    ) -> None:
        self.ontology = ontology
        self.resolver = resolver
        self.concept_frequencies: Dict[str, int] = defaultdict(int)
        self.concept_contexts: Dict[str, List[str]] = defaultdict(list)
        self.document_concepts: Dict[int, List[str]] = defaultdict(list)
        self._build_extraction_patterns()
        self._np_regex = re.compile(
            r'\b(?:[a-z]+(?:[-\s]?[a-z]+){0,2}[-\s]?)?'
            r'(?:alloy|composition|tensor|parameter|gradient|energy|force'
            r'|pressure|diffusion|interface|mobility|microstructure|grain'
            r'|phase|melt[-\s]?pool|surrogate|model|simulation|method'
            r'|analysis|optimization|kinetics|evolution|partitioning'
            r'|segregation|structure|boundary|growth|transformation)\b',
            re.IGNORECASE,
        )
        self._compound_regex = re.compile(
            r'\b([a-z]+(?:[-\s][a-z]+){1,4})\s+'
            r'(?:of|for|in|with|via|through|by|to|and|or|from)\s+'
            r'([a-z]+(?:[-\s][a-z]+){0,3})\b',
            re.IGNORECASE,
        )
        self._phrase_regex = re.compile(
            r'\b([a-z]+(?:[-\s][a-z]+){1,3})\b',
            re.IGNORECASE,
        )
        all_keywords = self._get_all_keywords()
        if all_keywords:
            sorted_keywords = sorted(all_keywords, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(
                re.escape(k) for k in sorted_keywords
            ) + r')\b'
            self._keyword_regex = re.compile(pattern, re.IGNORECASE)
        else:
            self._keyword_regex = None

    def _build_extraction_patterns(self) -> None:
        self.alloy_patterns = [
            r'\bco(?:-|\s+)cr(?:-|\s+)fe(?:-|\s+)ni\b',
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
            # v5.1.2 FIX: moved inside the loop — when `concepts` is empty the
            # loop never binds `concept`, causing UnboundLocalError here.
            self.concept_contexts[concept].append(text[:200])
        self.document_concepts[doc_id] = list(concepts)
        return list(concepts)

    def _extract_noun_phrases(self, text: str) -> Set[str]:
        matches = self._np_regex.findall(text)
        compound_matches = self._compound_regex.findall(text)
        concepts: Set[str] = set()
        for m in matches:
            if 5 < len(m) < 40:
                concepts.add(m.lower().strip())
        for m1, m2 in compound_matches:
            combined = f"{m1.lower().strip()} {m2.lower().strip()}"
            if 8 < len(combined) < 40:
                concepts.add(combined)
        return concepts

    def _extract_from_context_windows(
        self, text: str, window_size: int = 100
    ) -> Set[str]:
        if not self._keyword_regex:
            return set()
        candidate_phrases: Set[str] = set()
        text_lower = text.lower()
        for match in self._keyword_regex.finditer(text_lower):
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            context = text_lower[start:end]
            phrases = self._phrase_regex.findall(context)
            for phrase in phrases:
                if 5 <= len(phrase) <= 40:
                    candidate_phrases.add(phrase)
        if candidate_phrases:
            resolved = self.resolver.resolve_batch(list(candidate_phrases))
            return set(v for v in resolved.values() if v is not None)
        return set()

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
                    source_canon = self.resolver.resolve(source, context=text)
                    target_canon = self.resolver.resolve(target, context=text)
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
            nx_graph.nodes[u]['frequency'] = (
                nx_graph.nodes[u].get('frequency', 0) + 1
            )
            nx_graph.nodes[v]['frequency'] = (
                nx_graph.nodes[v].get('frequency', 0) + 1
            )

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
# DOMAIN KEYWORDS & PATTERNS (HEA-LASER)
# ============================================================================
CORE_MATERIALS = [
    "cocofeni", "co-cr-fe-ni", "co cr fe ni", "cobalt chromium iron nickel",
    "high entropy alloy", "hea", "high-entropy alloy", "high entropy alloys", "heas",
    "multi-principal element alloy", "mpea", "multi principal element alloy", "mpeas",
    "quaternary alloy", "quaternary system", "cobalt", "chromium", "iron", "nickel",
    "fcc phase", "fcc solid", "fcc matrix", "liquid phase", "liquid state", "melt pool",
]

MANUFACTURING_PROCESSES = [
    "laser additive manufacturing", "lam", "laser powder bed fusion", "lpbf",
    "laser processing", "laser melting", "laser solidification", "laser scanning",
    "additive manufacturing", "am", "laser treatment", "laser irradiance",
    "rapid heating", "rapid cooling", "rapid solidification", "thermal cycling",
    "laser matter interaction", "melt pool formation", "laser scan track",
    "powder bed fusion", "directed energy deposition", "laser wire-feed",
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
    "driving force", "thermodynamic driving force", "interfacial energy", "surface tension",
]

PHASE_FIELD_MODELING = [
    "phase-field model", "pfm", "phase field model", "phase-field modeling",
    "non-isothermal phase-field", "order parameter", "diffuse interface",
    "allen-cahn equation", "kks phase-equilibrium", "kks model",
    "multicomponent diffusion", "interface mobility", "phase mobility",
    "free energy functional", "bulk free energy", "gradient energy coefficient",
    "phase-field simulation", "spatiotemporal tensor", "phase fraction",
    "switching function", "landau polynomial", "barrier function",
]

FLUID_DYNAMICS_AND_MELT_POOL = [
    "marangoni convection", "marangoni-driven melt pool flow", "thermocapillary convection",
    "navier-stokes equations", "melt pool morphology", "melt pool depth",
    "boussinesq approximation", "incompressible flow", "melt pool dynamics",
    "surface-tension gradient", "fluidic phenomena", "melt pool flow",
    "thermal gradient", "keyhole formation", "marangoni effect",
]

AI_AND_SURROGATE_MODELS = [
    "ai surrogate", "surrogate model", "transformer-inspired", "attention mechanism",
    "cross-attention", "query-key attention", "gaussian locality regularization",
    "composition similarity", "digital twin", "attention-regularized surrogate",
    "multi-head attention", "spatiotemporally aware interpolation", "hybrid attention weights",
    "machine learning", "deep learning", "physics-informed", "physics-guided",
    "webapp", "neural network", "data-driven", "computational speedup",
]

MICROSTRUCTURAL_FEATURES = [
    "microstructural evolution", "elemental partitioning", "solidification kinetics",
    "equiaxed grains", "columnar grains", "grain boundary", "thermal wake",
    "cooling rate", "phase transformation", "solidification", "melting",
    "nucleation", "interface motion", "microstructure", "grain size",
    "segregation", "non-equilibrium microstructures", "hot tearing", "porosity",
]

COMPUTATIONAL_AND_MATHEMATICAL_METHODS = [
    "finite element analysis", "fea", "moose framework", "finite element method",
    "alternating least squares", "als", "tensor factorization", "multi-linear interpolation",
    "hessian matrix", "spectral decomposition", "rank-1 outer products",
    "root-mean-square error", "rmse", "leave-one-out cross-validation",
    "dice coefficient", "intersection-over-union", "iou", "computational domain",
    "discretization", "mesh", "cpu hours",
]

ALL_DOMAIN_KEYWORDS = (
    CORE_MATERIALS + MANUFACTURING_PROCESSES + THERMODYNAMICS_AND_TENSORS
    + PHASE_FIELD_MODELING + FLUID_DYNAMICS_AND_MELT_POOL
    + AI_AND_SURROGATE_MODELS + MICROSTRUCTURAL_FEATURES
    + COMPUTATIONAL_AND_MATHEMATICAL_METHODS
)

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
    r'\b(?:gaussian\s+locality|composition[\s-]tensor\s+similarity|cross[\s-]attention)\b',
]

HEA_CATEGORY_MAPPING = {
    r'co(?:-|\s)?cr(?:-|\s)?fe(?:-|\s)?ni|cocofeni|high[\s-]entropy|hea|mpea|multi[\s-]principal': 'core_material',
    r'laser\s+(?:powder\s+bed|additive|processing|melting|solidification)|lpbf|lam|rapid\s+(?:heating|cooling)': 'manufacturing_process',
    r'thermodynamic\s+data\s+tensor|tdt|gibbs\s+(?:free\s+)?energy|calphad|cpd|canonical\s+polyadic|factor\s+matrices|quadratic\s+(?:expansion|approximation)|phase[\s-]conditioned\s+composition\s+tensor|ctf': 'thermodynamics_tensor',
    r'phase[\s-]?field|pfm|allen[\s-]cahn|kks|diffuse\s+interface|order\s+parameter|multicomponent\s+diffusion': 'phase_field_modeling',
    r'marangoni|thermocapillary|navier[\s-]stokes|melt\s+pool\s+(?:flow|dynamics|morphology)|surface\s+tension|boussinesq': 'fluid_dynamics_melt_pool',
    r'ai\s+surrogate|transformer|attention\s+mechanism|cross[\s-]attention|digital\s+twin|machine\s+learning|deep\s+learning|gaussian\s+locality': 'ai_surrogate_model',
    r'microstructural\s+evolution|elemental\s+partitioning|solidification|equiaxed|columnar|grain\s+(?:boundary|size)|segregation': 'microstructural_feature',
    r'finite\s+element|fea|moose|alternating\s+least\s+squares|tensor\s+factorization|cross[\s-]validation|dice\s+coefficient': 'computational_method',
}


# ============================================================================
# CONCEPT FILTERING & NORMALIZATION (HEA-LASER)
# ============================================================================
def is_valid_hea_laser_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw.lower() in concept_lower for kw in ALL_DOMAIN_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in HEA_LASER_PATTERNS)
    generic = {
        'study', 'analysis', 'effect', 'role', 'investigation', 'research',
        'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
        'new', 'recent', 'various', 'different', 'significant', 'important',
        'report', 'demonstrate', 'show', 'result', 'data', 'find', 'present',
        'propose', 'develop', 'investigate', 'discuss', 'conclude',
    }
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic


def normalize_hea_laser_term(concept: str) -> str:
    concept = concept.lower().strip()
    concept = re.sub(r'\bco(?:-|\s)?cr(?:-|\s)?fe(?:-|\s)?ni\b', 'cocrfeni', concept)
    concept = re.sub(r'\bcocofeni\b', 'cocrfeni', concept)
    concept = re.sub(r'\bcobalt\s+chromium\s+iron\s+nickel\b', 'cocrfeni', concept)
    concept = re.sub(r'\blaser\s+powder\s+bed\s+fusion\b', 'lpbf', concept)
    concept = re.sub(r'\blaser\s+additive\s+manufacturing\b', 'lam', concept)
    concept = re.sub(r'\badditive\s+manufacturing\b', 'am', concept)
    concept = re.sub(r'\bthermodynamic\s+data\s+tensor\b', 'tdt', concept)
    concept = re.sub(r'\bcanonical\s+polyadic\s+decomposition\b', 'cpd', concept)
    concept = re.sub(r'\bphase[\s-]conditioned\s+composition\s+tensor\b', 'ctf', concept)
    concept = re.sub(r'\bgibbs\s+free\s+energy\b', 'gibbs energy', concept)
    concept = re.sub(r'\bphase[\s-]field\s+model\b', 'phase-field model', concept)
    concept = re.sub(r'\bphase[\s-]field\s+simulation\b', 'phase-field simulation', concept)
    concept = re.sub(r'\ballen[\s-]cahn\b', 'allen-cahn', concept)
    concept = re.sub(r'\bkks\s+phase[\s-]equilibrium\b', 'kks', concept)
    concept = re.sub(r'\bmarangoni\s+convection\b', 'marangoni convection', concept)
    concept = re.sub(r'\bnavier[\s-]stokes\b', 'navier-stokes', concept)
    concept = re.sub(r'\bmelt\s+pool\s+(?:morphology|dynamics|flow)\b', 'melt pool', concept)
    concept = re.sub(r'\bai\s+surrogate\b', 'ai surrogate', concept)
    concept = re.sub(r'\bdigital\s+twin\b', 'digital twin', concept)
    concept = re.sub(r'\bmachine\s+learning\b', 'machine learning', concept)
    concept = re.sub(r'\bdeep\s+learning\b', 'deep learning', concept)
    concept = re.sub(r'\bequiaxed\s+grains\b', 'equiaxed grains', concept)
    concept = re.sub(r'\bcolumnar\s+grains\b', 'columnar grains', concept)
    concept = re.sub(r'\bgrain\s+boundary\b', 'grain boundary', concept)
    concept = re.sub(r'\bfinite\s+element\s+analysis\b', 'fea', concept)
    concept = re.sub(r'\balternating\s+least\s+squares\b', 'als', concept)
    return concept


def extract_concepts_from_text(text: str) -> List[str]:
    concepts: Set[str] = set()
    text_lower = text.lower()
    for pattern in HEA_LASER_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    noun_pattern = (
        r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){1,3}'
        r'(?:alloy|composition|tensor|parameter|gradient|energy|force'
        r'|pressure|diffusion|interface|mobility|microstructure|grain'
        r'|phase|melt\s+pool|surrogate|model|simulation|method'
        r'|analysis|optimization)\b'
    )
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_hea_laser_concept(concept):
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
                if is_valid_hea_laser_concept(concept):
                    concepts.add(concept)
    param_pattern = (
        r'\b([a-z\s]+(?:temperature|velocity|power|density|gradient|energy'
        r'|pressure|force|viscosity|conductivity|capacity|tension))\s+'
        r'(?:of|is|=|:)?\s*(\d+(?:\.\d+)?\s*'
        r'(?:k|mpa|gpa|w|m/s|mm/s|j|pa|nm|µm|um|k/w|w/mk|j/kgk|pa·s|n/m))\b'
    )
    matches = re.findall(param_pattern, text, re.I)
    for param, value in matches:
        concept = f"{param.lower().strip()} {value.lower().strip()}"
        if is_valid_hea_laser_concept(concept):
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
        normalized = [normalize_hea_laser_term(c) for c in concepts]
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
            if c not in seen_in_doc and is_valid_hea_laser_concept(c):
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
        for pattern, category in HEA_CATEGORY_MAPPING.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            if any(re.search(p, concept, re.I) for p in [
                r'\bcocrfeni', r'\bhea', r'\bmpea', r'\bhigh[\s-]entropy'
            ]):
                concept_to_abstract[concept] = 'core_material'
            elif any(re.search(p, concept, re.I) for p in [
                r'\blaser', r'\blpbf', r'\blam', r'\bmelt\s+pool',
                r'\brapid\s+(?:heating|cooling)'
            ]):
                concept_to_abstract[concept] = 'manufacturing_process'
            elif any(re.search(p, concept, re.I) for p in [
                r'\btdt', r'\bcpd', r'\bcalphad', r'\bgibbs',
                r'\bthermodynamic', r'\bctf'
            ]):
                concept_to_abstract[concept] = 'thermodynamics_tensor'
            elif any(re.search(p, concept, re.I) for p in [
                r'\bphase[\s-]field', r'\ballen[\s-]cahn', r'\bkks',
                r'\bdiffuse\s+interface'
            ]):
                concept_to_abstract[concept] = 'phase_field_modeling'
            elif any(re.search(p, concept, re.I) for p in [
                r'\bmarangoni', r'\bnavier[\s-]stokes', r'\bfluid',
                r'\bsurface\s+tension'
            ]):
                concept_to_abstract[concept] = 'fluid_dynamics_melt_pool'
            elif any(re.search(p, concept, re.I) for p in [
                r'\bai\s+surrogate', r'\btransformer', r'\battention',
                r'\bdigital\s+twin', r'\bmachine\s+learning', r'\bdeep\s+learning'
            ]):
                concept_to_abstract[concept] = 'ai_surrogate_model'
            elif any(re.search(p, concept, re.I) for p in [
                r'\bmicrostructural', r'\bsolidification', r'\bgrain',
                r'\bsegregation', r'\bpartitioning'
            ]):
                concept_to_abstract[concept] = 'microstructural_feature'
            elif any(re.search(p, concept, re.I) for p in [
                r'\bfea', r'\bmoose', r'\bfinite\s+element',
                r'\btensor\s+factorization'
            ]):
                concept_to_abstract[concept] = 'computational_method'
            else:
                concept_to_abstract[concept] = 'general'
    return concept_to_abstract


# ============================================================================
# CONCEPT DISTILLATION
# ============================================================================
def compute_concept_distillation(
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    all_texts: List[str],
) -> pd.DataFrame:
    distill_data: List[Dict[str, Any]] = []
    doc_corpus: List[str] = []
    for c in valid_concepts:
        doc_text = " ".join([
            all_texts[i] for i in concept_abstract_map.get(c, [])
            if i < len(all_texts)
        ])
        doc_corpus.append(doc_text)
    tfidf = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        stop_words='english', max_features=5000,
    )
    try:
        tfidf_matrix = tfidf.fit_transform(doc_corpus)
        tfidf_scores = tfidf_matrix.max(axis=1).A1
    except Exception:
        tfidf_scores = np.ones(len(valid_concepts))
    embed_model = load_embedding_model()
    for i, c in enumerate(valid_concepts):
        freq = len(concept_abstract_map.get(c, []))
        semantic_density = float(tfidf_scores[i])
        coherence = 0.0
        if freq > 1 and doc_corpus[i].strip():
            try:
                words = doc_corpus[i].split()[:50]
                with torch.no_grad():
                    concept_embeddings = embed_model.encode(
                        words, show_progress_bar=False,
                        batch_size=32, convert_to_numpy=True,
                    )
                if len(concept_embeddings) > 1:
                    sim_matrix = cosine_similarity(concept_embeddings)
                    coherence = float(np.mean(
                        sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                    ))
                del concept_embeddings
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
) -> Tuple[List[Tuple], List[Tuple]]:
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs: List[Tuple[int, int]] = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    target_negs = min(len(pos_pairs) * 3 if pos_pairs else 30, 5000)
    attempts = 0
    max_attempts = 50000
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
    nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    if nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    try:
        pagerank = nx.pagerank(nx_graph, weight='weight')
    except Exception:
        pagerank = {n: 1.0 for n in nx_graph.nodes()}
    try:
        betweenness = nx.betweenness_centrality(nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in nx_graph.nodes()}
    genealogy_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in nx_graph:
            continue
        pr = pagerank.get(concept, 0)
        bc = betweenness.get(concept, 0)
        freq = len(concept_abstract_map.get(concept, []))
        degree = nx_graph.degree(concept)
        if (
            pr > np.percentile(list(pagerank.values()), 75)
            and degree > np.percentile(
                [nx_graph.degree(n) for n in nx_graph.nodes()], 75
            )
        ):
            generation = "Foundational (Parent)"
        elif (
            pr < np.percentile(list(pagerank.values()), 25)
            and degree < np.percentile(
                [nx_graph.degree(n) for n in nx_graph.nodes()], 25
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
    nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    if nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    category_map = abstract_concepts_to_categories(valid_concepts)
    try:
        betweenness = nx.betweenness_centrality(nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in nx_graph.nodes()}
    bridge_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in nx_graph:
            continue
        neighbors = list(nx_graph.neighbors(concept))
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
def analyze_network_motifs(nx_graph: nx.Graph) -> Dict[str, Any]:
    if nx_graph.number_of_nodes() < 3:
        return {}
    motifs: Dict[str, Any] = {}
    try:
        triangles = nx.triangles(nx_graph)
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
        cliques = list(nx.find_cliques(nx_graph))
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
        clustering = nx.clustering(nx_graph)
        stars: List[Tuple[str, int, float]] = []
        for node in nx_graph.nodes():
            deg = nx_graph.degree(node)
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
    filename="hea_graph_pub.png",
) -> bytes:
    try:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5, iterations=200)
        plt.figure(figsize=figsize, dpi=dpi)
        node_colors = [get_hea_laser_category_color(n) for n in nx_graph.nodes()]
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
            "CoCrFeNi HEA Laser AM Concept Graph",
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
    report.append("# CoCrFeNi HEA Laser AM Concept Graph Analysis Report")
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
    report.append("*Report generated by HEA-Laser-ConceptGraph v5.0*")
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
def get_hea_laser_category_color(
    concept: str, cmap_colors: Optional[List[str]] = None
) -> str:
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
        'general': '#9E9E9E',
    }
    return color_map.get(category, '#9E9E9E')


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
) -> None:
    """
    Faithful AgNPs-pattern PyVis renderer:
    - tempfile-based HTML generation (robust)
    - Glassmorphism UI with edge info panel
    - Label mode switching (short/full)
    - Robust tooltip parsing
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

    pos: Dict[str, Tuple[float, float]] = {}
    if len(nx_graph.nodes()) > 0:
        try:
            if len(nx_graph.nodes()) < 300:
                pos = nx.kamada_kawai_layout(nx_graph, weight='weight')
            else:
                pos = nx.spring_layout(
                    nx_graph, k=2.5, iterations=200, seed=42, weight='weight'
                )
        except Exception:
            pos = nx.spring_layout(
                nx_graph, k=2.5, iterations=200, seed=42, weight='weight'
            )

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
    "interaction": {
        "hover": true, "dragNodes": true,
        "dragView": true, "zoomView": true
    }
}
""")

    label_map: Dict[str, str] = {}
    n_counter = 1
    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(
            min_node_size + freq * 1.2, min_node_size, max_node_size
        ))
        color = get_hea_laser_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        original_label = (
            custom_labels.get(node, node) if custom_labels else node
        )
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
            label = original_label
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

        x, y = (
            pos.get(node, (0, 0))[0] * 1200,
            pos.get(node, (0, 0))[1] * 1200,
        )
        net.add_node(
            node, label=label, size=size, x=x, y=y,
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
            shape=node_shape, mass=max(1, 1 + freq * 0.05),
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
        'unknown':      theme['edge_unknown'],
    }
    all_weights = [
        nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()
    ]
    weight_threshold = (
        np.percentile(all_weights, 80) if all_weights else 0
    )
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
        actual_edge_label_color = (
            edge_label_color if edge_label_color else theme['font']
        )
        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)),
            width=width,
            color={
                'color': color,
                'highlight': theme['highlight_bg'],
                'hover': theme['hover_bg'],
                'opacity': 0.85,
            },
            smooth={'type': 'continuous', 'roundness': 0.35},
            title=(
                f"<span style='font-family:{node_font_face};'>"
                f"Weight: <b>{w:.2f}</b><br>"
                f"Type: {edge_type}<br>"
                f"Inferred: {is_inferred}</span>"
            ),
            dashes=dashes,
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

    # ✅ AgNPs pattern: tempfile-based HTML generation (robust)
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

    # ✅ AgNPs pattern: Glassmorphism JS with label-mode switching
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
                    var firstLine = txt.split('\\n')[0];
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
                    var defMatch = tooltipText.match(/Definition:\\s*(.+)/i);
                    if (defMatch && defMatch[1]) { nodeDefinition = defMatch[1].trim(); }
                    var typeMatch = tooltipText.match(/Type:\\s*(\\w+)/i);
                    if (typeMatch && typeMatch[1]) { nodeType = typeMatch[1].trim(); }
                    var freqMatch = tooltipText.match(/Frequency:\\s*(\\d+)/i);
                    if (freqMatch && freqMatch[1]) { nodeFreq = freqMatch[1].trim(); }
                    var degMatch = tooltipText.match(/Degree:\\s*(\\d+)/i);
                    if (degMatch && degMatch[1]) { nodeDegree = degMatch[1].trim(); }
                    var nameMatch = tooltipText.match(/^([^\\n]+)/);
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
                html += '<button id="btn-short" onclick="window._heaSetLabelMode(\\\'short\\\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:#D32F2F;color:white;">Short</button>';
                html += '<button id="btn-full" onclick="window._heaSetLabelMode(\\\'full\\\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:transparent;color:#64748b;">Full</button>';
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
                        var m = _txt.match(/Type:\\s*(\\w+)/); if (m) edgeType = m[1];
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
border-left:4px solid {theme.get('highlight_bg', '#ff6b6b')}; margin-bottom:6px;'>
<b style='color:{theme.get('highlight_bg', '#ff6b6b')}; font-size:14px;'>{short}</b>:
<span style='font-size:13px; color:{theme.get('font', '#1e293b')};'>{full}</span>
</div>""",
                    unsafe_allow_html=True,
                )

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button(
            "Download Interactive Graph (HTML)",
            data=html_bytes,
            file_name="hea_laser_concept_graph.html",
            mime="text/html",
        )
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")


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
def build_category_hierarchy(
    valid_concepts: List[str],
    concept_abstract_map: Dict,
    top_n_per_category: int = 40,
) -> Tuple[List, List, List]:
    """
    Faithful AgNPs pattern: 2-level hierarchy with DUPLICATE PREVENTION.
    - Root (center): "All Concepts"
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

    root_label = "All Concepts"
    total = sum(h["count"] for h in hierarchy.values())
    labels.append(root_label)
    parents.append("")
    values.append(total)

    for category, data in hierarchy.items():
        children = data["children"]
        children.sort(key=lambda x: x[1], reverse=True)
        if top_n_per_category > 0 and len(children) > top_n_per_category:
            children = children[:top_n_per_category]
        cat_child_sum = sum(freq for _, freq in children)
        labels.append(category)
        parents.append(root_label)
        values.append(cat_child_sum if cat_child_sum > 0 else data["count"])
        for concept, freq in children:
            # ★ SAFETY: Never add a concept that duplicates any category name
            if concept in all_category_names:
                continue
            labels.append(concept)
            parents.append(category)
            values.append(max(freq, 1))

    return labels, parents, values


def render_sunburst_chart(
    labels, parents, values, cmap_name="viridis",
    label_size=20, width=900, height=700,
    theme=None, branchvalues="total",
    show_labels=True, show_values=False,
    hover_info="all", color_continuous_scale=None,
    font_family="Arial, sans-serif",
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
border-left:4px solid {entry['color']}; margin-bottom:6px;'>
<span style='font-size:18px; color:{entry['color']}; margin-right:6px;'>{entry['symbol']}</span>
<span style='font-size:12px; color:#333; font-weight:500;'>{entry['label']}</span>
<span style='font-size:10px; color:#666; float:right;'>({entry['value']})</span>
</div>""",
                        unsafe_allow_html=True,
                    )


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
                nx_graph.graph['version'] = '5.0'
                nx_graph.graph['tool'] = 'HEA-Laser-ConceptGraph'
            try:
                nx.write_graphml_lxml(nx_graph, "hea_graph.graphml")
            except Exception:
                nx.write_graphml(nx_graph, "hea_graph.graphml")
            with open("hea_graph.graphml", "rb") as f:
                return f.read(), "application/graphml+xml", "hea_graph.graphml"
        except Exception as e:
            st.error(f"GraphML export failed: {e}")
            return None, None, None
    elif export_format == "JSON (Full Metadata)":
        data = nx.node_link_data(nx_graph)
        if include_metadata:
            data['metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '5.0',
                'tool': 'HEA-Laser-ConceptGraph',
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
        return json_str.encode('utf-8'), "application/json", "hea_graph_full.json"
    elif export_format == "JSON (Compact)":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "hea_graph.json"
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
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "hea_edges_enhanced.csv"
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
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "hea_nodes_enhanced.csv"
    elif export_format == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [
                get_hea_laser_category_color(n) for n in nx_graph.nodes()
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
            return buf.read(), "image/png", "hea_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif export_format == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=150)
            node_colors = [
                get_hea_laser_category_color(n) for n in nx_graph.nodes()
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
            return buf.read(), "image/svg+xml", "hea_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    elif export_format == "GEXF":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '5.0'
            nx.write_gexf(nx_graph, "hea_graph.gexf")
            with open("hea_graph.gexf", "rb") as f:
                return f.read(), "application/xml", "hea_graph.gexf"
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
        ("high entropy alloy", "hea"),
        ("co-cr-fe-ni", "cocrfeni"),
        ("laser powder bed fusion", "lpbf"),
        ("thermodynamic data tensor", "tdt"),
        ("marangoni convection", "marangoni_convection"),
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
# BATCH PROCESSING UTILITIES
# ============================================================================
def split_into_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Split DataFrame into batches for incremental processing."""
    total_batches = math.ceil(len(df) / batch_size)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        yield df.iloc[start_idx:end_idx].copy()


def merge_graphs(existing_graph: nx.Graph, new_graph: nx.Graph) -> nx.Graph:
    """
    Merge two graphs by:
    - Adding new nodes from new_graph
    - Updating edge weights (sum co-occurrences)
    - Preserving existing node attributes
    """
    merged = existing_graph.copy()

    # Add new nodes
    for node, data in new_graph.nodes(data=True):
        if node not in merged:
            merged.add_node(node, **data)
        else:
            # Update frequency (sum)
            merged.nodes[node]['frequency'] = (
                merged.nodes[node].get('frequency', 0) +
                data.get('frequency', 0)
            )

    # Add/update edges
    for u, v, data in new_graph.edges(data=True):
        if merged.has_edge(u, v):
            # Sum co-occurrence weights
            merged[u][v]['cooccurrence'] = (
                merged[u][v].get('cooccurrence', 0) +
                data.get('cooccurrence', 0)
            )
            # Recalculate weight (will be recomputed later)
            merged[u][v]['weight'] = merged[u][v]['cooccurrence']
        else:
            merged.add_edge(u, v, **data)

    return merged


# ============================================================================
# INCREMENTAL GRAPH BUILDER
# ============================================================================
class IncrementalGraphBuilder:
    """Build concept graph incrementally from document batches."""

    def __init__(self, ontology: DomainOntology, extractor: EnhancedConceptExtractor):
        self.ontology = ontology
        self.extractor = extractor
        self.existing_graph: Optional[nx.Graph] = None
        self.processed_doc_count = 0
        self.batch_history: List[Dict] = []

    def process_batch(
        self,
        batch_df: pd.DataFrame,
        batch_number: int,
        selected_text_cols: List[str],
        existing_graph: Optional[nx.Graph] = None,
        embed_model=None,
        config: Dict = None,
    ) -> Tuple[nx.Graph, Dict[str, List[int]], List[Dict]]:
        """
        Process a single batch and merge with existing graph.

        Returns:
            - Updated graph
            - Updated concept_abstract_map
            - Batch metrics
        """
        if config is None:
            config = get_adaptive_config(len(batch_df))

        # Extract concepts from this batch
        all_concepts = []
        all_metrics = []

        for idx, row in batch_df.iterrows():
            text = " ".join([
                str(row[col]) for col in selected_text_cols
                if col in row and pd.notna(row[col])
            ])
            concepts = self.extractor.extract_from_text(text, idx)
            all_concepts.append(concepts)

            # Extract metrics
            metrics = {}
            power_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:w|watt)', text, re.I)
            if power_matches:
                metrics['laser_power_w'] = [float(m) for m in power_matches]
            all_metrics.append(metrics)

        # Build concept frequencies for this batch
        # v5.1.2 FIX: concept_frequencies is a defaultdict attribute, not a
        # method — calling it with () raises TypeError. Use the getter,
        # matching the main non-batch pipeline.
        concept_freq = self.extractor.get_concept_frequencies()
        valid_concepts = [
            c for c, f in concept_freq.items()
            if f >= config.get("MIN_CONCEPT_FREQ", 2)
        ]

        # Build concept-to-doc mapping for this batch
        batch_concept_map: Dict[str, List[int]] = defaultdict(list)
        for doc_idx, concepts in enumerate(all_concepts):
            for c in set(concepts):
                batch_concept_map[c].append(doc_idx)

        # Build graph for this batch
        concept_to_id = {c: i for i, c in enumerate(valid_concepts)}

        if existing_graph is None:
            # First batch: build from scratch
            nx_graph = self._build_graph_from_scratch(
                all_concepts, valid_concepts, concept_to_id, embed_model, config
            )
        else:
            # Incremental: build batch graph and merge
            batch_graph = self._build_graph_from_scratch(
                all_concepts, valid_concepts, concept_to_id, embed_model, config
            )
            nx_graph = merge_graphs(existing_graph, batch_graph)

        # Track batch metadata
        self.batch_history.append({
            'batch_number': batch_number,
            'docs_processed': len(batch_df),
            'concepts_found': len(valid_concepts),
            'edges_added': nx_graph.number_of_edges() - (existing_graph.number_of_edges() if existing_graph else 0),
            'timestamp': datetime.now().isoformat()
        })

        self.processed_doc_count += len(batch_df)

        return nx_graph, batch_concept_map, all_metrics

    def _build_graph_from_scratch(
        self,
        all_concepts: List[List[str]],
        valid_concepts: List[str],
        concept_to_id: Dict[str, int],
        embed_model,
        config: Dict,
    ) -> nx.Graph:
        """Build graph from a single batch (no merging)."""
        nx_graph = nx.Graph()

        # Add nodes
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

        # Build co-occurrence edges
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

        # Add semantic edges
        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(nx_graph, valid_concepts, embed_model, config)

        # Add inferred edges
        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)

        return nx_graph

    def _add_semantic_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str],
        embed_model, config: Dict,
    ) -> None:
        """Add semantic similarity edges."""
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
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")

    def _add_inferred_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        """Add ontology-inferred edges."""
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


# ============================================================================
# BATCH PROCESSING UI CONTROLS (Add to sidebar)
# ============================================================================
def render_batch_processing_controls():
    """Render batch processing controls in sidebar."""
    st.markdown("---")
    st.subheader("📦 Batch Processing Mode")

    st.session_state['batch_mode'] = st.checkbox(
        "Enable batch processing",
        value=False,
        help="Process documents in batches to reduce memory usage"
    )

    if st.session_state.get('batch_mode', False):
        st.session_state['batch_size'] = st.slider(
            "Batch size (documents)",
            min_value=100,
            max_value=2000,
            value=500,  # Default reduced for stability
            step=100,
            help="Number of documents to process per batch. Lower = more stable on limited RAM."
        )

        # Show batch status with error awareness
        if st.session_state.get('batch_status'):
            status = st.session_state['batch_status']

            # Check if there's an error state
            has_error = st.session_state.get('batch_error') is not None

            if has_error:
                st.error(
                    f"**Failed at**: Batch {status['current_batch']}/{status['total_batches']} | "
                    f"Docs: {status['processed_docs']}/{status['total_docs']}"
                )
                st.caption("See main panel for error details and troubleshooting.")
            else:
                st.info(
                    f"**Progress**: Batch {status['current_batch']}/{status['total_batches']} | "
                    f"Docs: {status['processed_docs']}/{status['total_docs']}"
                )

            # Only show "Process Next" if no error and batches remain
            if not has_error and status['current_batch'] < status['total_batches']:
                if st.button("🔄 Process Next Batch", type="primary", key="proc_next"):
                    st.session_state['process_next_batch'] = True
                    st.rerun()
            elif not has_error and status['current_batch'] >= status['total_batches']:
                st.success("✅ All batches complete!")

            # Always show reset
            if st.button("🗑️ Reset All Batches", key="reset_batches"):
                st.session_state['batch_status'] = None
                st.session_state['analysis_data'] = None
                st.session_state['process_next_batch'] = False
                st.session_state['batch_error'] = None
                st.session_state['batch_traceback'] = None
                st.session_state['incremental_builder'] = None
                st.success("Batch processing reset!")
                st.rerun()
        else:
            st.info("Start by clicking '🚀 Build Concept Graph with Reasoning' below")


# ============================================================================
# BATCH ANALYSIS FUNCTION (ROBUST — Fixed Silent Failure Bug)
# ============================================================================
def run_batch_analysis(
    df_filtered: pd.DataFrame,
    selected_text_cols: List[str],
    ontology: DomainOntology,
    embed_model,
    config: Dict,
):
    """Run analysis in batches with incremental graph updates.

    FIXES applied (v5.1):
    - Ontology init wrapped in explicit try/except with session_state error persistence
    - Error state stored in session_state so it survives st.rerun()
    - Defensive checks for empty graphs / insufficient concepts before GNN training
    - GNN training wrapped with fallback for small graphs
    - Progress tracking stored in session_state for sidebar visibility
    - Memory cleanup in finally block ensures no leaks on crash
    """

    batch_size = st.session_state.get('batch_size', 1000)
    batches = list(split_into_batches(df_filtered, batch_size))
    total_batches = len(batches)

    # Initialize or load existing state
    if 'batch_status' not in st.session_state or st.session_state['batch_status'] is None:
        st.session_state['batch_status'] = {
            'current_batch': 0,
            'total_batches': total_batches,
            'processed_docs': 0,
            'total_docs': len(df_filtered),
            'existing_graph': None,
            'concept_abstract_map': defaultdict(list),
            'all_metrics': [],
        }
        # Clear any stale error state on fresh start
        st.session_state['batch_error'] = None
        st.session_state['batch_traceback'] = None

    status = st.session_state['batch_status']

    # Check if we should process next batch
    if not st.session_state.get('process_next_batch', False):
        return

    st.session_state['process_next_batch'] = False

    # Get next batch
    batch_num = status['current_batch']
    if batch_num >= total_batches:
        st.success("✅ All batches processed!")
        return

    batch_df = batches[batch_num]

    # Progress UI
    progress_bar = st.progress(0.0)
    status_container = st.status(f"Processing batch {batch_num + 1}/{total_batches}...", expanded=True)

    # Local error flag - only commit to session_state on actual failure
    local_error = None
    local_traceback = None

    try:
        with status_container:
            # =====================================================================
            # STEP 1: Initialize incremental builder (with explicit error handling)
            # =====================================================================
            st.write(f"📄 Loading batch {batch_num + 1} ({len(batch_df)} documents)...")
            progress_bar.progress(0.05)

            # Ensure builder exists and is valid (reinit if corrupted)
            if 'incremental_builder' not in st.session_state or st.session_state.get('incremental_builder') is None:
                use_ontology = st.session_state.get('use_ontology', True)
                st.write("🧠 Initializing batch processor (one-time setup)...")
                try:
                    if use_ontology:
                        # Full ontology mode
                        resolver = AdvancedConceptResolver(ontology, embed_model)
                        extractor = EnhancedConceptExtractor(ontology, resolver)
                        st.session_state['incremental_builder'] = IncrementalGraphBuilder(ontology, extractor)
                        st.session_state['resolver'] = resolver
                        st.session_state['extractor'] = extractor
                        st.write("✅ Ontology resolver initialized")
                    else:
                        # Non-ontology mode: create minimal builder with fresh ontology
                        # (DomainOntology is lightweight; we reuse the existing one)
                        resolver = AdvancedConceptResolver(ontology, embed_model)
                        extractor = EnhancedConceptExtractor(ontology, resolver)
                        st.session_state['incremental_builder'] = IncrementalGraphBuilder(ontology, extractor)
                        st.session_state['resolver'] = resolver
                        st.session_state['extractor'] = extractor
                        st.write("✅ Batch processor initialized (non-ontology mode)")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as init_err:
                    local_error = f"Batch processor initialization failed: {init_err}"
                    local_traceback = traceback.format_exc()
                    st.error(f"❌ {local_error}")
                    st.info("💡 Try reducing batch size or checking available memory.")
                    raise  # Re-raise to hit outer except

            builder = st.session_state['incremental_builder']

            # Defensive: ensure builder is actually initialized
            if builder is None:
                st.error("❌ Builder not initialized. This can happen if:")
                st.markdown("""
                1. **Session state was corrupted** → Click '🗑️ Reset All Batches' to start fresh
                2. **Out of memory during init** → Reduce batch size to 250-500 in sidebar
                3. **Embedding model failed to load** → Check internet connection for model download
                """)
                st.info("Click '🗑️ Reset All Batches' in the sidebar to reinitialize.")
                raise RuntimeError("IncrementalGraphBuilder not initialized")

            progress_bar.progress(0.10)

            # =====================================================================
            # STEP 2: Extract concepts from batch
            # =====================================================================
            st.write("🔍 Extracting concepts from documents...")

            new_graph, batch_concept_map, batch_metrics = builder.process_batch(
                batch_df=batch_df,
                batch_number=batch_num + 1,
                selected_text_cols=selected_text_cols,
                existing_graph=status['existing_graph'],
                embed_model=embed_model,
                config=config,
            )

            # Validate: did we get any concepts?
            if new_graph.number_of_nodes() == 0:
                local_error = f"Batch {batch_num + 1} produced zero concepts. Check your text columns and domain filters."
                st.error(f"❌ {local_error}")
                raise ValueError(local_error)

            progress_bar.progress(0.35)
            st.write(f"✅ Found {new_graph.number_of_nodes()} concepts, {new_graph.number_of_edges()} edges")

            # =====================================================================
            # STEP 3: Merge with existing graph state
            # =====================================================================
            st.write("🔗 Merging with existing graph...")

            status['existing_graph'] = new_graph
            status['current_batch'] = batch_num + 1
            status['processed_docs'] += len(batch_df)

            # Merge concept maps (with offset for global doc indices)
            offset = status['processed_docs'] - len(batch_df)
            for concept, doc_indices in batch_concept_map.items():
                status['concept_abstract_map'][concept].extend(
                    [idx + offset for idx in doc_indices]
                )

            status['all_metrics'].extend(batch_metrics)
            progress_bar.progress(0.50)

            # =====================================================================
            # STEP 4: Prepare for GNN (with defensive checks)
            # =====================================================================
            st.write("🧠 Computing embeddings & training GNN...")

            valid_concepts = list(new_graph.nodes())
            concept_to_id = {c: i for i, c in enumerate(valid_concepts)}

            # Defensive: need at least 2 concepts for GNN
            if len(valid_concepts) < 2:
                local_error = f"Only {len(valid_concepts)} concept(s) found. Need at least 2 for GNN training."
                st.warning(f"⚠️ {local_error}")
                st.info("Skipping GNN training. Graph will still be available for visualization.")
                # Create dummy embeddings for compatibility
                with torch.no_grad():
                    embeddings = embed_model.encode(
                        valid_concepts if valid_concepts else ["placeholder"],
                        show_progress_bar=False,
                        batch_size=64,
                        convert_to_numpy=True,
                    )
                if len(valid_concepts) < 2:
                    # Pad to at least 2
                    embeddings = np.vstack([embeddings, embeddings[-1:] if len(embeddings) > 0 else np.zeros((1, embeddings.shape[1]))])
                    valid_concepts = valid_concepts + ["_placeholder_"]
                    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
                node_features = torch.tensor(embeddings[:len(valid_concepts)], dtype=torch.float32)
            else:
                with torch.no_grad():
                    embeddings = embed_model.encode(
                        valid_concepts,
                        show_progress_bar=False,
                        batch_size=64,
                        convert_to_numpy=True,
                    )
                node_features = torch.tensor(embeddings, dtype=torch.float32)

            progress_bar.progress(0.65)

            # =====================================================================
            # STEP 5: Train GNN (with fallback for edge cases)
            # =====================================================================
            try:
                pos_pairs, neg_pairs = sample_edges_for_training(
                    new_graph, valid_concepts, concept_to_id, config
                )

                # Defensive: if no positive pairs, create minimal ones
                if not pos_pairs and len(valid_concepts) >= 2:
                    st.info("⚠️ No positive edge pairs found. Creating minimal pairs for GNN.")
                    pos_pairs = [(0, 1)]
                    neg_pairs = []

                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, new_graph, concept_to_id, pos_pairs, neg_pairs,
                    epochs=30  # Fewer epochs for incremental updates
                )
                st.write("✅ GNN training complete")
            except Exception as gnn_err:
                st.warning(f"⚠️ GNN training skipped: {gnn_err}")
                st.info("Graph structure is still available for visualization and analysis.")
                # Create dummy model/embedding for downstream compatibility
                gnn_model = None
                final_emb = node_features  # Use raw embeddings as fallback
                adj_indices = torch.zeros((2, 0), dtype=torch.long)
                adj_values = torch.zeros(0, dtype=torch.float32)

            progress_bar.progress(0.85)

            # =====================================================================
            # STEP 6: Store analysis data
            # =====================================================================
            analysis_data = {
                "valid_concepts": valid_concepts,
                "concept_to_id": concept_to_id,
                "id_to_concept": {i: c for i, c in enumerate(valid_concepts)},
                "concept_abstract_map": dict(status['concept_abstract_map']),
                "nx_graph": new_graph,
                "gnn_model": gnn_model,
                "final_emb": final_emb,
                "embed_model": embed_model,
                "all_metrics": status['all_metrics'],
                "config": config,
                "df_filtered": df_filtered,
                "selected_text_cols": selected_text_cols,
                "ontology": ontology,
                "resolver": st.session_state.get('resolver'),
                "extractor": st.session_state.get('extractor'),
            }

            st.session_state.analysis_data = analysis_data
            progress_bar.progress(1.0)

            status_container.update(
                label=f"✅ Batch {batch_num + 1}/{total_batches} complete! ({new_graph.number_of_nodes()} nodes, {new_graph.number_of_edges()} edges)",
                state="complete",
                expanded=False
            )

            # Success message
            st.success(
                f"✅ Batch {batch_num + 1}/{total_batches} processed successfully! "
                f"Graph: {new_graph.number_of_nodes()} nodes, {new_graph.number_of_edges()} edges"
            )

            remaining = total_batches - batch_num - 1
            if remaining > 0:
                st.info(
                    f"📦 {remaining} batch{'es' if remaining > 1 else ''} remaining. "
                    f"Click '🔄 Process Next Batch' in the sidebar to continue."
                )
            else:
                st.balloons()
                st.success("🎉 All batches complete! Explore the results in the tabs below.")

    except Exception as e:
        # Capture error state persistently
        if local_error is None:
            local_error = str(e)
            local_traceback = traceback.format_exc()

        st.session_state['batch_error'] = local_error
        st.session_state['batch_traceback'] = local_traceback

        st.error(f"❌ Batch processing failed: {local_error}")
        with st.expander("🔍 Full Traceback", expanded=True):
            st.code(local_traceback)

        st.info("""
        💡 **Troubleshooting tips:**
        - Reduce **Batch size** in sidebar (try 500 instead of 1000)
        - Disable **ontology-based resolution** if memory is limited
        - Check that your JSON/CSV files have valid text in the selected columns
        - Clear cache via sidebar and try again
        - Click **🗑️ Reset All Batches** to start fresh
        """)

        # Update status container to show failure
        try:
            status_container.update(
                label=f"❌ Batch {batch_num + 1} failed: {str(local_error)[:60]}",
                state="error",
                expanded=False
            )
        except Exception:
            pass  # Status container might not exist

    finally:
        # Always clean up memory, even on failure
        try:
            del node_features, embeddings
        except NameError:
            pass
        try:
            del pos_pairs, neg_pairs
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# SIDEBAR (AgNPs Pattern — Full Sunburst Customization)
# ============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration v5.0")
        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0,
        )
        theme = THEME_PRESETS[st.session_state['theme']]
        st.subheader("🔬 HEA Laser AM Focus Areas")
        st.markdown("- **Core Materials**: CoCrFeNi HEA/MPEA/CCA, FCC/BCC phases")
        st.markdown("- **Processes**: LPBF, LAM, DED, rapid solidification, melt pool")
        st.markdown("- **Thermodynamics**: TDT, CPD, CALPHAD, Gibbs energy, CTF")
        st.markdown("- **Phase-Field**: Allen-Cahn, KKS, multicomponent diffusion")
        st.markdown("- **Fluid Dynamics**: Marangoni, Navier-Stokes, Boussinesq")
        st.markdown("- **AI Surrogates**: Transformer, attention, digital twin")
        st.markdown("- **Microstructure**: elemental partitioning, grains, segregation")
        st.markdown("- **Computational**: MOOSE, FEA, ALS, tensor factorization")
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

        # Add batch processing controls
        render_batch_processing_controls()

        # Show persistent error state in sidebar if present
        if st.session_state.get('batch_error'):
            st.markdown("---")
            st.error("⚠️ Last batch failed")
            st.caption(f"Error: {st.session_state['batch_error'][:100]}...")
            if st.button("📋 Copy Error to Clipboard", key="copy_err"):
                st.code(st.session_state.get('batch_traceback', 'No traceback'))
            if st.button("🗑️ Clear Error & Reset", key="clear_err"):
                st.session_state['batch_error'] = None
                st.session_state['batch_traceback'] = None
                st.session_state['batch_status'] = None
                st.session_state['analysis_data'] = None
                st.session_state['process_next_batch'] = False
                st.success("Error cleared! Click '🗑️ Reset All Batches' to fully restart.")
                st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main() -> None:
    st.title(
        "🔬 HEA-Laser-ConceptGraph v5.1.1.1: Faithful AgNPs Architecture Port"
    )
    st.caption(
        "Multi-level reasoning concept graph for CoCrFeNi laser AM | "
        "Faithful AgNPs Architecture v5.1.1 | Memory-Safe | "
        "Interactive Visualization | Ontology-aware resolution | "
        "Robust Batch Processing"
    )

    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology

    render_sidebar()

    # ✅ AgNPs pattern: Initialize ALL session_state keys
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
    if "batch_status" not in st.session_state:
        st.session_state.batch_status = None
    if "process_next_batch" not in st.session_state:
        st.session_state.process_next_batch = False
    if "incremental_builder" not in st.session_state:
        st.session_state.incremental_builder = None
    if "batch_error" not in st.session_state:
        st.session_state.batch_error = None
    if "batch_traceback" not in st.session_state:
        st.session_state.batch_traceback = None

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
    if st.button(
        "🚀 Build Concept Graph with Reasoning",
        type="primary", use_container_width=True,
    ):
        if st.session_state.get('batch_mode', False):
            # BATCH MODE
            st.info("📦 Batch processing mode enabled")

            # Initialize first batch
            if st.session_state.get('batch_status') is None:
                st.session_state['process_next_batch'] = True
                # Clear any previous error state
                st.session_state['batch_error'] = None

            # Run batch analysis
            run_batch_analysis(
                df_filtered=df_filtered,
                selected_text_cols=selected_text_cols,
                ontology=st.session_state.ontology,
                embed_model=load_embedding_model(),
                config=get_adaptive_config(len(df_filtered)),
            )

            # Only rerun if no error occurred and batch is still processing
            # If there was an error, keep the error message visible
            if st.session_state.get('batch_error') is None:
                if st.session_state.get('batch_status') and st.session_state['batch_status']['current_batch'] < st.session_state['batch_status']['total_batches']:
                    # Still have batches to process, but don't auto-rerun
                    # Let user click "Process Next Batch"
                    pass
                elif st.session_state.get('batch_status') and st.session_state['batch_status']['current_batch'] >= st.session_state['batch_status']['total_batches']:
                    # All done - analysis_data should be set
                    pass
            else:
                # Error occurred - show it prominently and DON'T rerun
                st.error(f"❌ Batch processing failed: {st.session_state['batch_error']}")
                with st.expander("Error Details", expanded=True):
                    st.code(st.session_state.get('batch_traceback', 'No traceback available'))
                st.info("Click '🗑️ Reset All Batches' in the sidebar to start over, or check the error details above.")
        else:
            # FULL MODE (original single-pass analysis)
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

                    # ✅ AgNPs pattern: None-safety after threads
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

                    # ✅ AgNPs pattern: Cache analytics in session_state
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
                    file_name="hea_research_directions.csv", mime="text/csv",
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
                        file_name="hea_graph_publication.png",
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
                    file_name="hea_laser_analysis_report.md",
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
                file_name="hea_concepts_enhanced.csv", mime="text/csv",
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
