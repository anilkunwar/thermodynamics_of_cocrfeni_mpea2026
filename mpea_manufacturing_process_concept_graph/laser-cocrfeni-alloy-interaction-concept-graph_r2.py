#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HEA-LaserConceptGraph: Concept Graph Builder for CoCrFeNi HEA & Laser AM
========================================================================
Large-corpus concept graph extraction (3000+ abstracts) from JSON/BibTeX/CSV metadata.
Tailored for CoCrFeNi high-entropy alloys, laser powder bed fusion, thermodynamic
data tensors (TDT), canonical polyadic decomposition (CPD), phase-field modeling,
Marangoni fluid dynamics, and AI surrogate models with cross-attention mechanisms.

Features:
- Robust JSON/JSONL/CSV/BibTeX loading with BOM handling and error recovery
- HEA/Laser-AM optimized concept extraction (TF-IDF, semantic clustering, PageRank)
- Domain filtering: CoCrFeNi, LPBF, TDT/CPD, phase-field, Marangoni, AI surrogates
- Interactive PyVis/Plotly 2D/3D visualizations with 50+ colormaps
- Statistical validation: modularity, silhouette, centrality, bootstrap CIs
- GNN-powered (GraphSAGE) research direction scoring with PyTorch
- Export: GraphML, JSON, CSV, HTML, SVG, PNG, Markdown Report
- Streamlit UI with session persistence and crash prevention
- Interactive graph editing (remove/merge/rename nodes, add edges, filter) with Undo/Redo
- Enhanced sunburst with category filter and branch value modes
- Concept timeline, co-occurrence heatmap, t-SNE, community detection,
  concept growth scores, bubble chart
- NEW: Keyword burst detection, semantic drift detection, concept genealogy
- NEW: Cross-domain bridge detection, network motif analysis, centrality comparison
- NEW: Automated Markdown report generation, publication-ready figure exports
- NEW: BibTeX ingestion support

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run: streamlit run hea_laser_concept_graph.py

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
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path

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
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="HEA-LaserConceptGraph: CoCrFeNi HEA & Laser AM Explorer",
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
    sanitized = re.sub(r',(\s*[}\]])', r'', sanitized)
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
                st.code(f"Hex preview (first {len(raw_bytes)} bytes):
{formatted}", language="text")
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
    df = df.replace({float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA})
    year_cols = [c for c in df.columns if 'year' in c.lower()]
    if year_cols:
        df["Year"] = pd.to_numeric(df[year_cols[0]], errors="coerce")
    elif "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

# ==========================================
# CoCrFeNi HEA & LASER AM DOMAIN CONFIGURATION
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
    r'(?:co(?:-|\s)?cr(?:-|\s)?fe(?:-|\s)?ni|cocofeni)',
    r'(?:high[\s-]entropy\s+alloy[s]?|hea[s]?)',
    r'(?:multi[\s-]principal\s+element\s+alloy[s]?|mpea[s]?)',
    r'(?:laser\s+(?:powder\s+bed\s+fusion|additive\s+manufacturing|processing|melting|solidification)|lpbf|lam)',
    r'(?:thermodynamic\s+data\s+tensor|tdt|gibbs\s+(?:free\s+)?energy\s+tensor)',
    r'(?:canonical\s+polyadic\s+decomposition|cpd|factor\s+matrices)',
    r'(?:phase[\s-]?field\s+(?:model|simulation|method|framework)|pfm)',
    r'(?:marangoni\s+(?:convection|flow|effect)|thermocapillary\s+convection)',
    r'(?:ai\s+surrogate|transformer[\s-]inspired|attention[\s-]regularized|digital\s+twin)',
    r'(?:phase[\s-]conditioned\s+composition\s+tensor|categorical\s+alloy[\s-]composition\s+tensor|ctf)',
    r'(?:allen[\s-]cahn|kks\s+phase[\s-]equilibrium|multicomponent\s+diffusion)',
    r'(?:melt\s+pool\s+(?:morphology|depth|dynamics|flow)|thermal\s+gradient)',
    r'(?:calphad|gibbs\s+energy\s+landscape|chemical\s+driving\s+pressure)',
    r'(?:gaussian\s+locality|composition[\s-]tensor\s+similarity|cross[\s-]attention)'
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
# UTILITY FUNCTIONS
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
            "TOP_N_CONCEPTS": 200, "MAX_CONCEPT_LENGTH": 6
        }
    elif num_abstracts <= 500:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.78, "COOCCURRENCE_WEIGHT": 0.7,
            "SEMANTIC_WEIGHT": 0.3, "CLUSTER_SIMILARITY": 0.72,
            "TOP_N_CONCEPTS": 500, "MAX_CONCEPT_LENGTH": 8
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 5, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 3, "USE_SEMANTIC_CLUSTERING": False,
            "SIMILARITY_THRESHOLD": 0.85, "COOCCURRENCE_WEIGHT": 0.9,
            "SEMANTIC_WEIGHT": 0.1, "CLUSTER_SIMILARITY": 0.68,
            "TOP_N_CONCEPTS": 1000, "MAX_CONCEPT_LENGTH": 10
        }

# ==========================================
# DEVICE & MODEL MANAGEMENT
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ==========================================
# CONCEPT EXTRACTION & NORMALIZATION
# ==========================================
def is_valid_nanomaterials_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw.lower() in concept_lower for kw in ALL_DOMAIN_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in HEA_LASER_PATTERNS)
    generic = {'study', 'analysis', 'effect', 'role', 'investigation', 'research',
               'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
               'new', 'recent', 'various', 'different', 'significant', 'important',
               'report', 'demonstrate', 'show', 'result', 'data', 'find', 'present',
               'propose', 'develop', 'investigate', 'discuss', 'conclude'}
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic

def normalize_nanomaterials_term(concept: str) -> str:
    concept = concept.lower().strip()
    # Core material normalizations
    concept = re.sub(r'co(?:-|\s)?cr(?:-|\s)?fe(?:-|\s)?ni', 'cocofeni', concept)
    concept = re.sub(r'cobalt\s+chromium\s+iron\s+nickel', 'cocofeni', concept)
    concept = re.sub(r'high[\s-]entropy\s+alloy[s]?', 'hea', concept)
    concept = re.sub(r'multi[\s-]principal\s+element\s+alloy[s]?', 'mpea', concept)
    # Manufacturing
    concept = re.sub(r'laser\s+powder\s+bed\s+fusion', 'lpbf', concept)
    concept = re.sub(r'laser\s+additive\s+manufacturing', 'lam', concept)
    concept = re.sub(r'additive\s+manufacturing', 'am', concept)
    # Thermodynamics & Tensors
    concept = re.sub(r'thermodynamic\s+data\s+tensor', 'tdt', concept)
    concept = re.sub(r'canonical\s+polyadic\s+decomposition', 'cpd', concept)
    concept = re.sub(r'phase[\s-]conditioned\s+composition\s+tensor', 'ctf', concept)
    concept = re.sub(r'gibbs\s+(?:free\s+)?energy', 'gibbs energy', concept)
    concept = re.sub(r'chemical\s+driving\s+pressure', 'chemical driving pressure', concept)
    # Phase-field
    concept = re.sub(r'phase[\s-]?field\s+model(?:ing)?', 'phase-field model', concept)
    concept = re.sub(r'allen[\s-]cahn\s+equation', 'allen-cahn', concept)
    concept = re.sub(r'kks\s+phase[\s-]equilibrium', 'kks model', concept)
    # Fluid dynamics
    concept = re.sub(r'marangoni\s+convection', 'marangoni convection', concept)
    concept = re.sub(r'thermocapillary\s+convection', 'thermocapillary convection', concept)
    concept = re.sub(r'navier[\s-]stokes\s+equations', 'navier-stokes', concept)
    # AI & Surrogates
    concept = re.sub(r'ai\s+surrogate', 'ai surrogate', concept)
    concept = re.sub(r'transformer[\s-]inspired', 'transformer-inspired', concept)
    concept = re.sub(r'attention[\s-]regularized', 'attention-regularized', concept)
    concept = re.sub(r'digital\s+twin', 'digital twin', concept)
    concept = re.sub(r'cross[\s-]attention', 'cross-attention', concept)
    concept = re.sub(r'gaussian\s+locality\s+regularization', 'gaussian locality', concept)
    # Computational
    concept = re.sub(r'finite\s+element\s+analysis', 'fea', concept)
    concept = re.sub(r'moose\s+framework', 'moose', concept)
    concept = re.sub(r'alternating\s+least\s+squares', 'als', concept)
    concept = re.sub(r'tensor\s+factorization', 'tensor factorization', concept)
    concept = re.sub(r'root[\s-]mean[\s-]square\s+error', 'rmse', concept)
    concept = re.sub(r'dice\s+coefficient', 'dice coefficient', concept)
    concept = re.sub(r'intersection[\s-]over[\s-]union', 'iou', concept)
    # Microstructure
    concept = re.sub(r'microstructural\s+evolution', 'microstructural evolution', concept)
    concept = re.sub(r'elemental\s+partitioning', 'elemental partitioning', concept)
    concept = re.sub(r'solidification\s+kinetics', 'solidification kinetics', concept)
    return concept

def extract_concepts_from_text(text: str) -> List[str]:
    concepts = set()
    text_lower = text.lower()
    for pattern in HEA_LASER_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    # HEA-specific noun pattern for multi-word technical terms
    noun_pattern = r'(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,4}(?:alloy|phase|tensor|decomposition|model|simulation|framework|method|analysis|surrogate|network|learning|diffusion|convection|flow|pool|grain|boundary|interface|parameter|energy|potential|force|pressure|gradient|coefficient|function|polynomial|mechanism|process|kinetics|evolution|partitioning|segregation|nucleation|growth|morphology|structure|property|performance|design|optimization)'
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_nanomaterials_concept(concept):
            concepts.add(concept)
    # Contextual phrases with domain keywords
    for keyword in ALL_DOMAIN_KEYWORDS:
        for match in re.finditer(r'' + re.escape(keyword) + r'', text_lower):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text_lower[start:end]
            context_phrases = re.findall(r'([a-z]+(?:\s+[a-z]+){1,3})\s+(?:of|for|in|with|using|via|through|by|to|and|or)\s+' + re.escape(keyword) + r'', context)
            for phrase in context_phrases:
                concept = f"{phrase.strip()} {keyword}"
                if is_valid_nanomaterials_concept(concept):
                    concepts.add(concept)
    # Property-value patterns for HEA/Laser AM
    material_prop_pattern = r'([A-Z][a-z]+(?:\d+(?:\.\d+)?)?(?:[\s\-][A-Z][a-z]?\d*)+)\s+(?:with|having|exhibiting|showing|demonstrating|achieving|reaching|delivering|providing|offering)\s+(?:a\s+)?([\d\.]+\s*(?:gpa|mpa|hv|nm|um|µm|angstrom|a|m/s|k|°c|pct|pct\.|percent|hours|cpu\s*hours))'
    matches = re.findall(material_prop_pattern, text, re.I)
    for material, value in matches:
        concept = f"{material.lower()} {value.lower()}"
        if is_valid_nanomaterials_concept(concept):
            concepts.add(concept)
    return list(concepts)

def extract_concepts_from_abstracts(df: pd.DataFrame, text_columns: List[str]) -> Tuple[List[List[str]], List[Dict]]:
    all_concepts = []
    all_metrics = []
    for idx, row in df.iterrows():
        combined_text = ""
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])
        metrics = {}
        # HEA-specific metric extraction
        temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:k|°c|kelvin)', combined_text, re.I)
        if temp_matches: metrics['temperature_k'] = [float(m) for m in temp_matches]
        velocity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:m/s|mm/s)', combined_text, re.I)
        if velocity_matches: metrics['scan_velocity'] = [float(m) for m in velocity_matches]
        power_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:w|watt)', combined_text, re.I)
        if power_matches: metrics['laser_power_w'] = [float(m) for m in power_matches]
        size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm|um|µm)', combined_text, re.I)
        if size_matches: metrics['size_nm_um'] = [float(m) for m in size_matches]
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:j/m|j/mm)', combined_text, re.I)
        if energy_matches: metrics['energy_density'] = [float(m) for m in energy_matches]
        rmse_matches = re.findall(r'rmse\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)', combined_text, re.I)
        if rmse_matches: metrics['rmse_error'] = [float(m) for m in rmse_matches]
        dice_matches = re.findall(r'dice\s*(?:coefficient|score)?\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)', combined_text, re.I)
        if dice_matches: metrics['dice_score'] = [float(m) for m in dice_matches]
        all_metrics.append(metrics)
        concepts = extract_concepts_from_text(combined_text)
        normalized = [normalize_nanomaterials_term(c) for c in concepts]
        all_concepts.append(normalized)
    return all_concepts, all_metrics

def cluster_similar_concepts(valid_concepts: List[str], embed_model, similarity_threshold: float = 0.75):
    if len(valid_concepts) < 5:
        return valid_concepts, {c: c for c in valid_concepts}
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1 - similarity_threshold,
            linkage='average', metric='cosine'
        ).fit(embeddings)
        cluster_members = defaultdict(list)
        concept_to_cluster = {}
        for idx, label in enumerate(clustering.labels_):
            concept = valid_concepts[idx]
            cluster_members[label].append(concept)
            concept_to_cluster[concept] = label
        cluster_representatives = {}
        for label, members in cluster_members.items():
            def score(m):
                domain_hits = sum(1 for kw in ALL_DOMAIN_KEYWORDS if kw.lower() in m.lower())
                return (domain_hits, -len(m))
            representative = max(members, key=score)
            cluster_representatives[label] = representative
        final_mapping = {c: cluster_representatives[label] for c, label in concept_to_cluster.items()}
        return list(cluster_representatives.values()), final_mapping
    except Exception as e:
        return valid_concepts, {c: c for c in valid_concepts}

def normalize_and_filter_concepts(all_concepts: List[List[str]], config: Dict) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_nanomaterials_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    min_freq = config.get("MIN_CONCEPT_FREQ", 5)
    min_words = config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    max_words = config.get("MAX_CONCEPT_LENGTH", 10)
    valid_concepts = [c for c, cnt in concept_counts.items()
                      if cnt >= min_freq and min_words <= len(c.split()) <= max_words]
    if config.get("USE_SEMANTIC_CLUSTERING", False) and len(valid_concepts) > 50:
        try:
            embed_model = load_embedding_model()
            valid_concepts, concept_to_cluster = cluster_similar_concepts(
                valid_concepts, embed_model,
                similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.72)
            )
            new_abstract_map = defaultdict(list)
            for orig_concept, docs in concept_abstract_map.items():
                clustered = concept_to_cluster.get(orig_concept, orig_concept)
                if clustered in valid_concepts:
                    new_abstract_map[clustered].extend(docs)
            concept_abstract_map = new_abstract_map
        except Exception as e:
            st.warning(f"Semantic clustering skipped: {e}")
    valid_concepts = sorted(valid_concepts, key=lambda c: concept_counts[c], reverse=True)
    top_n = config.get("TOP_N_CONCEPTS", 1000)
    if len(valid_concepts) > top_n:
        valid_concepts = valid_concepts[:top_n]
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

def abstract_concepts_to_categories(concepts: List[str]) -> Dict[str, str]:
    concept_to_abstract = {}
    for concept in concepts:
        matched = False
        for pattern, category in HEA_CATEGORY_MAPPING.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            if any(re.search(p, concept, re.I) for p in [r'cocofeni', r'co cr fe ni', r'hea', r'mpea', r'high[\s-]entropy']):
                concept_to_abstract[concept] = 'core_material'
            elif any(re.search(p, concept, re.I) for p in [r'laser', r'lpbf', r'lam', r'am', r'rapid\s+(?:heating|cooling)']):
                concept_to_abstract[concept] = 'manufacturing_process'
            elif any(re.search(p, concept, re.I) for p in [r'tdt', r'cpd', r'calphad', r'gibbs', r'ctf', r'thermodynamic']):
                concept_to_abstract[concept] = 'thermodynamics_tensor'
            elif any(re.search(p, concept, re.I) for p in [r'tem|hrtem|stem|ebsd|xrd|apt|synchrotron|sem|in\s*situ']):
                concept_to_abstract[concept] = 'microscopy_characterization'
            elif any(re.search(p, concept, re.I) for p in [r'phase[\s-]?field|pfm|allen[\s-]cahn|kks|diffuse\s+interface']):
                concept_to_abstract[concept] = 'phase_field_modeling'
            elif any(re.search(p, concept, re.I) for p in [r'marangoni|navier[\s-]stokes|thermocapillary|melt\s+pool']):
                concept_to_abstract[concept] = 'fluid_dynamics_melt_pool'
            elif any(re.search(p, concept, re.I) for p in [r'ai\s+surrogate|transformer|attention|digital\s+twin|machine\s+learning|deep\s+learning|neural\s+network']):
                concept_to_abstract[concept] = 'ai_surrogate_model'
            elif any(re.search(p, concept, re.I) for p in [r'microstructural|elemental\s+partitioning|solidification|equiaxed|columnar|grain\s+boundary|segregation']):
                concept_to_abstract[concept] = 'microstructural_feature'
            elif any(re.search(p, concept, re.I) for p in [r'fea|moose|als|tensor\s+factorization|cross[\s-]validation|dice|iou']):
                concept_to_abstract[concept] = 'computational_method'
            else:
                concept_to_abstract[concept] = 'general'
    return concept_to_abstract

# ==========================================
# CONCEPT DISTILLATION
# ==========================================
def compute_concept_distillation(valid_concepts: List[str], concept_abstract_map: Dict[str, List[int]],
                                  all_texts: List[str]) -> pd.DataFrame:
    distill_data = []
    doc_corpus = []
    for c in valid_concepts:
        doc_text = " ".join([all_texts[i] for i in concept_abstract_map.get(c, []) if i < len(all_texts)])
        doc_corpus.append(doc_text)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', max_features=5000)
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
                concept_embeddings = embed_model.encode(words, show_progress_bar=False, batch_size=32)
                if len(concept_embeddings) > 1:
                    sim_matrix = cosine_similarity(concept_embeddings)
                    coherence = float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
            except Exception:
                coherence = 0.0
        distill_data.append({
            "concept": c, "frequency": freq, "tfidf_weight": semantic_density,
            "semantic_density": semantic_density, "coherence_score": float(coherence),
            "distillation_efficiency": float(semantic_density * np.log1p(freq) * (0.5 + 0.5 * coherence))
        })
    return pd.DataFrame(distill_data).sort_values("distillation_efficiency", ascending=False)

# ==========================================
# GRAPH CONSTRUCTION
# ==========================================
def build_hybrid_graph(all_concepts: List[List[str]], valid_concepts: List[str],
                        concept_to_id: Dict[str, int], embed_model=None, config: Dict = None) -> nx.Graph:
    if config is None:
        config = get_adaptive_config(3000)
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c, frequency=0)
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1, cooccurrence=1, semantic=0, edge_type='cooccurrence')
                nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1
    if embed_model and len(valid_concepts) >= 10:
        try:
            embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh and (nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3):
                        nx_graph.add_edge(c1, c2, weight=sim * 2, cooccurrence=0,
                                         semantic=sim, edge_type='semantic')
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.9)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.1)
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    return nx_graph

def sample_edges_for_training(nx_graph: nx.Graph, valid_concepts: List[str],
                               concept_to_id: Dict[str, int], config: Dict = None) -> Tuple[List[Tuple], List[Tuple]]:
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
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
            neg_pairs.append((u_idx, v_idx))
        elif dist == 999 and np.random.rand() < 0.1:
            neg_pairs.append((u_idx, v_idx))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append((u_idx, v_idx))
    return pos_pairs, neg_pairs

# ==========================================
# GNN MODEL
# ==========================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
    def forward(self, adj_indices, adj_values, num_nodes, h, pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs,
              progress_callback=None, epochs: int = 50, lr: float = 1e-3):
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
    target_device = node_features.device if node_features.numel() > 0 else torch.device('cpu')
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=target_device)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=target_device)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=128).to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if len(neg_pairs) == 0:
            pos_out, _, _ = model(adj_indices, adj_values, num_nodes, node_features,
                                 pos_u, pos_v, pos_u[:1], pos_v[:1])
            loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
        else:
            pos_out, neg_out, _ = model(adj_indices, adj_values, num_nodes, node_features,
                                         pos_u, pos_v, neg_u, neg_v)
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            loss = 0.5 * (pos_loss + neg_loss)
        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(adj_indices, adj_values, num_nodes, node_features,
                                       pos_u[:1], pos_v[:1], neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1],
                                       neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1])
    return model, final_embeddings.cpu(), adj_indices.cpu(), adj_values.cpu()

# ==========================================
# RESEARCH DIRECTION SCORING
# ==========================================
def compute_research_direction_scores(model, node_features, final_emb, nx_graph,
                                       valid_concepts, concept_properties, ridge,
                                       embed_model, n_samples: int = 5000) -> pd.DataFrame:
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 5))
    v_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 5))
    candidate_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx:
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            continue
        candidate_pairs.append((u_idx, v_idx, u_c, v_c))
    if not candidate_pairs:
        return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pair_features = torch.cat([final_emb[u_tensor], final_emb[v_tensor]], dim=1)
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).numpy()
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
    cos_sims = np.sum(emb_np[u_tensor.numpy()] * emb_np[v_tensor.numpy()], axis=1)
    results = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
            except:
                expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = np.clip((expected_improvement - 50) / 200, 0, 1) if expected_improvement > 0 else 0
        D_uv = (alpha['gnn'] * gnn_scores[i] + alpha['novelty'] * semantic_novelty +
                alpha['gain'] * norm_gain + alpha['feas'] * (1.0 - feasibility))
        results.append({
            'concept_u': u_c, 'concept_v': v_c, 'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty), 'expected_property_gain': expected_improvement,
            'feasibility_score': float(feasibility), 'composite_score': float(D_uv)
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(min(100, len(df)))

# ==========================================
# MATHEMATICAL VALIDATION
# ==========================================
def validate_graph_metrics(nx_graph: nx.Graph, valid_concepts: List[str]) -> Dict[str, Any]:
    metrics = {}
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
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        if len(valid_concepts) >= 3:
            labels = np.zeros(len(valid_concepts))
            for i, c in enumerate(valid_concepts):
                for idx, comm in enumerate(partition if 'partition' in locals() else [[]]):
                    if c in comm:
                        labels[i] = idx
                        break
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        else:
            metrics["silhouette_score"] = 0.0
    except Exception:
        metrics["silhouette_score"] = 0.0
    weights = [d.get('weight', 1) for _, _, d in nx_graph.edges(data=True)]
    if len(weights) > 10:
        p_values = []
        for w in weights[:50]:
            permuted = np.random.permutation(weights)
            p_values.append(np.sum(permuted >= w) / len(weights))
        metrics["edge_significance_p_mean"] = float(np.mean(p_values))
        metrics["edge_significant_count"] = int(sum(1 for p in p_values if p < 0.05))
    else:
        metrics["edge_significance_p_mean"] = 1.0
        metrics["edge_significant_count"] = 0
    try:
        metrics["avg_betweenness"] = np.mean(list(nx.betweenness_centrality(nx_graph).values()))
        metrics["avg_closeness"] = np.mean(list(nx.closeness_centrality(nx_graph).values()))
    except Exception:
        pass
    return metrics

@st.cache_data(ttl=3600)
def compute_bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05):
    if len(scores) < 2:
        return float(np.mean(scores)), 0.0, 0.0
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(ci_low), float(ci_high)

# ==========================================
# ADVANCED ANALYTICS: KEYWORD BURST DETECTION
# ==========================================
def detect_keyword_bursts(df_filtered: pd.DataFrame, valid_concepts: List[str], 
                          concept_abstract_map: Dict[str, List[int]], 
                          text_columns: List[str], burst_threshold: float = 2.0) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame()
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 3:
        return pd.DataFrame()
    year_range = sorted(years.unique())
    burst_data = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 5:
            continue
        concept_years = []
        for idx in doc_indices:
            if idx < len(df_filtered) and pd.notna(df_filtered.iloc[idx].get("Year")):
                concept_years.append(int(df_filtered.iloc[idx]["Year"]))
        if len(concept_years) < 3:
            continue
        year_counts = Counter(concept_years)
        counts = [year_counts.get(y, 0) for y in year_range]
        if len(counts) < 3:
            continue
        window = max(2, len(counts) // 5)
        moving_avg = pd.Series(counts).rolling(window=window, min_periods=1).mean()
        burst_scores = []
        for i in range(window, len(counts)):
            if moving_avg.iloc[i-1] > 0:
                ratio = counts[i] / max(moving_avg.iloc[i-1], 0.1)
                burst_scores.append(ratio)
        if burst_scores:
            max_burst = max(burst_scores)
            burst_year = year_range[window + burst_scores.index(max_burst)]
            if max_burst >= burst_threshold:
                burst_data.append({
                    "concept": concept,
                    "burst_score": round(max_burst, 2),
                    "burst_year": burst_year,
                    "total_mentions": len(concept_years),
                    "year_range": f"{min(concept_years)}-{max(concept_years)}"
                })
    return pd.DataFrame(burst_data).sort_values("burst_score", ascending=False)

# ==========================================
# ADVANCED ANALYTICS: SEMANTIC DRIFT DETECTION
# ==========================================
def detect_semantic_drift(df_filtered: pd.DataFrame, valid_concepts: List[str],
                          concept_abstract_map: Dict[str, List[int]],
                          embed_model, text_columns: List[str],
                          early_fraction: float = 0.3, late_fraction: float = 0.3) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame()
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 4:
        return pd.DataFrame()
    sorted_years = sorted(years.unique())
    n_years = len(sorted_years)
    early_cutoff = sorted_years[int(n_years * early_fraction)]
    late_cutoff = sorted_years[int(n_years * (1 - late_fraction))]
    drift_data = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 10:
            continue
        early_texts = []
        late_texts = []
        for idx in doc_indices:
            if idx >= len(df_filtered):
                continue
            row = df_filtered.iloc[idx]
            year = row.get("Year")
            if pd.isna(year):
                continue
            year = int(year)
            text = " ".join([str(row.get(col, "")) for col in text_columns if pd.notna(row.get(col))])
            if year <= early_cutoff:
                early_texts.append(text)
            elif year >= late_cutoff:
                late_texts.append(text)
        if len(early_texts) < 3 or len(late_texts) < 3:
            continue
        try:
            early_emb = embed_model.encode(early_texts, show_progress_bar=False, batch_size=32)
            late_emb = embed_model.encode(late_texts, show_progress_bar=False, batch_size=32)
            early_centroid = np.mean(early_emb, axis=0)
            late_centroid = np.mean(late_emb, axis=0)
            drift = 1.0 - cosine_similarity([early_centroid], [late_centroid])[0][0]
            drift_data.append({
                "concept": concept,
                "semantic_drift": round(float(drift), 4),
                "early_papers": len(early_texts),
                "late_papers": len(late_texts),
                "early_period": f"{sorted_years[0]}-{early_cutoff}",
                "late_period": f"{late_cutoff}-{sorted_years[-1]}"
            })
        except Exception:
            continue
    return pd.DataFrame(drift_data).sort_values("semantic_drift", ascending=False)

# ==========================================
# ADVANCED ANALYTICS: CONCEPT GENEALOGY
# ==========================================
def build_concept_genealogy(nx_graph: nx.Graph, valid_concepts: List[str],
                            concept_abstract_map: Dict[str, List[int]]) -> pd.DataFrame:
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
    genealogy_data = []
    for concept in valid_concepts:
        if concept not in nx_graph:
            continue
        pr = pagerank.get(concept, 0)
        bc = betweenness.get(concept, 0)
        freq = len(concept_abstract_map.get(concept, []))
        degree = nx_graph.degree(concept)
        if pr > np.percentile(list(pagerank.values()), 75) and degree > np.percentile([nx_graph.degree(n) for n in nx_graph.nodes()], 75):
            generation = "Foundational (Parent)"
        elif pr < np.percentile(list(pagerank.values()), 25) and degree < np.percentile([nx_graph.degree(n) for n in nx_graph.nodes()], 25):
            generation = "Emerging (Child)"
        else:
            generation = "Intermediate"
        genealogy_data.append({
            "concept": concept,
            "pagerank": round(pr, 5),
            "betweenness": round(bc, 5),
            "frequency": freq,
            "degree": degree,
            "generation": generation
        })
    return pd.DataFrame(genealogy_data).sort_values("pagerank", ascending=False)

# ==========================================
# ADVANCED ANALYTICS: CROSS-DOMAIN BRIDGE DETECTION
# ==========================================
def detect_cross_domain_bridges(nx_graph: nx.Graph, valid_concepts: List[str],
                                concept_abstract_map: Dict[str, List[int]]) -> pd.DataFrame:
    if nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    category_map = abstract_concepts_to_categories(valid_concepts)
    try:
        betweenness = nx.betweenness_centrality(nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in nx_graph.nodes()}
    bridge_data = []
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
            "own_category": own_cat
        })
    return pd.DataFrame(bridge_data).sort_values("bridge_score", ascending=False)

# ==========================================
# ADVANCED ANALYTICS: NETWORK MOTIF ANALYSIS
# ==========================================
def analyze_network_motifs(nx_graph: nx.Graph) -> Dict[str, Any]:
    if nx_graph.number_of_nodes() < 3:
        return {}
    motifs = {}
    try:
        triangles = nx.triangles(nx_graph)
        motifs["total_triangles"] = sum(triangles.values()) // 3
        motifs["avg_triangles_per_node"] = round(np.mean(list(triangles.values())), 2)
        motifs["nodes_in_triangles"] = sum(1 for v in triangles.values() if v > 0)
    except Exception:
        motifs["total_triangles"] = 0
    try:
        cliques = list(nx.find_cliques(nx_graph))
        clique_sizes = [len(c) for c in cliques]
        motifs["total_cliques"] = len(cliques)
        motifs["max_clique_size"] = max(clique_sizes) if clique_sizes else 0
        motifs["avg_clique_size"] = round(np.mean(clique_sizes), 2) if clique_sizes else 0
        motifs["4cliques"] = sum(1 for c in clique_sizes if c >= 4)
    except Exception:
        motifs["total_cliques"] = 0
    try:
        clustering = nx.clustering(nx_graph)
        stars = []
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

# ==========================================
# ADVANCED ANALYTICS: CENTRALITY COMPARISON & DEGREE DISTRIBUTION
# ==========================================
def compute_centrality_comparison(nx_graph: nx.Graph, valid_concepts: List[str]) -> pd.DataFrame:
    if nx_graph.number_of_nodes() < 3:
        return pd.DataFrame()
    centrality_data = []
    try:
        degree_c = dict(nx_graph.degree())
        betweenness_c = nx.betweenness_centrality(nx_graph, weight='weight')
        closeness_c = nx.closeness_centrality(nx_graph)
        eigenvector_c = nx.eigenvector_centrality(nx_graph, weight='weight', max_iter=1000)
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
                "pagerank": round(pagerank_c.get(concept, 0), 5)
            })
    except Exception as e:
        st.warning(f"Centrality computation error: {e}")
    return pd.DataFrame(centrality_data)

def plot_degree_distribution(nx_graph: nx.Graph, theme: Dict = None) -> go.Figure:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    degrees = [d for n, d in nx_graph.degree()]
    if len(degrees) < 3:
        return go.Figure()
    degree_counts = Counter(degrees)
    x = sorted(degree_counts.keys())
    y = [degree_counts[k] for k in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Degree Distribution',
                              marker=dict(size=10, color=theme.get('highlight_bg', '#ff6b6b'))))
    fig.update_layout(
        title="Degree Distribution (Log-Log)",
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Degree (k)", yaxis_title="Frequency P(k)",
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000")
    )
    return fig

# ==========================================
# PUBLICATION-READY EXPORTS
# ==========================================
def export_publication_figure(nx_graph, valid_concepts, concept_abstract_map, 
                              cmap_name="viridis", dpi=300, figsize=(14, 12),
                              filename="hea_laser_graph_pub.png") -> bytes:
    """Generate a publication-quality network figure using Matplotlib."""
    try:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5, iterations=200)
        plt.figure(figsize=figsize, dpi=dpi)
        node_colors = [get_nanomaterials_category_color(n) for n in nx_graph.nodes()]
        node_sizes = [max(100, min(800, len(concept_abstract_map.get(n, [])) * 20 + 50)) for n in nx_graph.nodes()]
        nx.draw(nx_graph, pos, 
                with_labels=True, 
                node_color=node_colors, 
                edge_color='lightgray',
                node_size=node_sizes, 
                font_size=6, 
                font_weight='bold', 
                edgecolors='white', 
                linewidths=1.5,
                width=0.5,
                alpha=0.9)
        plt.title("CoCrFeNi HEA & Laser AM Concept Graph", fontsize=14, fontweight='bold', pad=20)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close()
        return buf.read()
    except Exception as e:
        st.error(f"Publication figure export failed: {e}")
        return b''

def generate_analysis_report(nx_graph, valid_concepts, concept_abstract_map, 
                             top_scores, distill_df, burst_df, drift_df, 
                             genealogy_df, bridge_df, motifs, val_metrics,
                             df_filtered) -> str:
    """Generate a comprehensive Markdown analysis report."""
    report = []
    report.append("# CoCrFeNi HEA & Laser AM Concept Graph Analysis Report")
    report.append(f"
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
")

    report.append("## 1. Dataset Overview")
    report.append(f"- **Total Records**: {len(df_filtered)}")
    if 'Year' in df_filtered.columns:
        years = df_filtered['Year'].dropna()
        report.append(f"- **Year Range**: {int(years.min())} - {int(years.max())}")
    report.append(f"- **Total Concepts**: {len(valid_concepts)}")
    report.append(f"- **Total Edges**: {nx_graph.number_of_edges()}")
    report.append(f"- **Graph Density**: {nx.density(nx_graph):.4f}")
    report.append("")

    report.append("## 2. Top Concepts by Frequency")
    top_concepts = sorted(valid_concepts, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)[:20]
    for i, c in enumerate(top_concepts, 1):
        freq = len(concept_abstract_map.get(c, []))
        deg = nx_graph.degree(c)
        report.append(f"{i}. **{c}** - Freq: {freq}, Degree: {deg}")
    report.append("")

    report.append("## 3. Concept Distillation Efficiency (Top 15)")
    if not distill_df.empty:
        for _, row in distill_df.head(15).iterrows():
            report.append(f"- **{row['concept']}**: Efficiency={row['distillation_efficiency']:.3f}, "
                         f"Freq={row['frequency']}, Coherence={row['coherence_score']:.3f}")
    report.append("")

    report.append("## 4. Research Direction Recommendations (Top 10)")
    if not top_scores.empty:
        for i, (_, row) in enumerate(top_scores.head(10).iterrows(), 1):
            report.append(f"{i}. **{row['concept_u']}** + **{row['concept_v']}** - "
                         f"Composite Score: {row['composite_score']:.3f}")
    report.append("")

    report.append("## 5. Keyword Burst Detection")
    if not burst_df.empty:
        for _, row in burst_df.head(10).iterrows():
            report.append(f"- **{row['concept']}**: Burst Score={row['burst_score']:.2f} "
                         f"(Year {row['burst_year']})")
    else:
        report.append("No significant keyword bursts detected.")
    report.append("")

    report.append("## 6. Semantic Drift Detection")
    if not drift_df.empty:
        for _, row in drift_df.head(10).iterrows():
            report.append(f"- **{row['concept']}**: Drift={row['semantic_drift']:.4f} "
                         f"({row['early_period']} -> {row['late_period']})")
    else:
        report.append("No significant semantic drift detected.")
    report.append("")

    report.append("## 7. Cross-Domain Bridge Concepts")
    if not bridge_df.empty:
        for _, row in bridge_df.head(10).iterrows():
            report.append(f"- **{row['concept']}**: Bridge Score={row['bridge_score']:.4f}, "
                         f"Connects {row['connected_categories']} categories")
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
    report.append(f"- Silhouette Score: {val_metrics.get('silhouette_score', 0):.3f}")
    report.append(f"- Number of Communities: {val_metrics.get('n_communities', 0)}")
    report.append(f"- Avg Betweenness: {val_metrics.get('avg_betweenness', 0):.3f}")
    report.append("")

    report.append("---")
    report.append("*Report generated by HEA-LaserConceptGraph*")
    return "
".join(report)

# ==========================================
# GRAPH EDIT HISTORY (UNDO/REDO)
# ==========================================
class GraphEditHistory:
    def __init__(self, max_history=20):
        self.history = deque(maxlen=max_history)
        self.redo_stack = deque(maxlen=max_history)
        self._snapshot_counter = 0

    def save_snapshot(self, nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map):
        import copy
        snapshot = {
            'id': self._snapshot_counter,
            'nx_graph': nx_graph.copy(),
            'valid_concepts': list(valid_concepts),
            'concept_to_id': dict(concept_to_id),
            'id_to_concept': dict(id_to_concept),
            'concept_abstract_map': {k: list(v) for k, v in concept_abstract_map.items()},
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(snapshot)
        self._snapshot_counter += 1
        self.redo_stack.clear()
        return snapshot['id']

    def undo(self):
        if len(self.history) < 2:
            return None
        current = self.history.pop()
        self.redo_stack.append(current)
        previous = self.history[-1]
        return previous

    def redo(self):
        if not self.redo_stack:
            return None
        snapshot = self.redo_stack.pop()
        self.history.append(snapshot)
        return snapshot

    def can_undo(self):
        return len(self.history) >= 2

    def can_redo(self):
        return len(self.redo_stack) > 0

    def get_history_summary(self):
        return [f"Snapshot {s['id']} @ {s['timestamp']}" for s in self.history]

# ==========================================
# VISUALIZATION FUNCTIONS (ENHANCED)
# ==========================================
def get_nanomaterials_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    # Core materials - deep red
    if any(c in concept_lower for c in ['cocofeni', 'co cr fe ni', 'hea', 'mpea', 'high-entropy', 'quaternary', 'cobalt', 'chromium', 'iron', 'nickel', 'fcc phase']):
        return "#D32F2F"
    # Manufacturing processes - cyan
    elif any(c in concept_lower for c in ['lpbf', 'lam', 'laser', 'additive manufacturing', 'rapid heating', 'rapid cooling', 'powder bed', 'directed energy']):
        return "#00BCD4"
    # Thermodynamics & tensors - purple
    elif any(c in concept_lower for c in ['tdt', 'cpd', 'ctf', 'gibbs energy', 'calphad', 'thermodynamic', 'chemical driving', 'factor matrices', 'quadratic expansion', 'canonical polyadic']):
        return "#9C27B0"
    # Phase-field modeling - green
    elif any(c in concept_lower for c in ['phase-field', 'pfm', 'allen-cahn', 'kks model', 'diffuse interface', 'order parameter', 'multicomponent diffusion', 'free energy functional']):
        return "#4CAF50"
    # Fluid dynamics & melt pool - blue
    elif any(c in concept_lower for c in ['marangoni', 'navier-stokes', 'thermocapillary', 'melt pool', 'surface tension', 'boussinesq', 'incompressible flow']):
        return "#3F51B5"
    # AI & surrogate models - magenta
    elif any(c in concept_lower for c in ['ai surrogate', 'transformer', 'attention', 'digital twin', 'machine learning', 'deep learning', 'neural network', 'gaussian locality']):
        return "#8E24AA"
    # Microstructural features - orange
    elif any(c in concept_lower for c in ['microstructural evolution', 'elemental partitioning', 'solidification', 'equiaxed', 'columnar', 'grain boundary', 'segregation', 'nucleation']):
        return "#FF9800"
    # Computational methods - teal
    elif any(c in concept_lower for c in ['fea', 'moose', 'als', 'tensor factorization', 'cross-validation', 'dice coefficient', 'iou', 'rmse', 'finite element']):
        return "#009688"
    # Characterization (general fallback) - indigo
    elif any(c in concept_lower for c in ['tem', 'hrtem', 'stem', 'ebsd', 'xrd', 'apt', 'synchrotron', 'sem', 'in situ']):
        return "#3F51B5"
    else:
        return "#9E9E9E"

def render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=True,
                        min_node_size=8, max_node_size=40, cmap_name="viridis",
                        custom_labels=None, node_label_size=12, top_n_nodes=0,
                        theme=None, physics_preset=None, show_edge_weights=False,
                        edge_label_mode="hover"):
    """
    Enhanced PyVis renderer with improved edge label readability.
    edge_label_mode: "hover" (default), "threshold", "all"
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

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(min_node_size + freq * 1.2, min_node_size, max_node_size))
        color = get_nanomaterials_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        label = custom_labels.get(node, node) if custom_labels else node

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
            font={
                'color': theme['font'],
                'size': node_label_size,
                'face': 'Inter, Segoe UI, Roboto, sans-serif',
                'strokeWidth': 0,
                'vadjust': -6
            },
            title=(
                f"<div style='font-family:Inter,sans-serif;'>"
                f"<b style='font-size:14px;color:{theme['highlight_bg']};'>{node}</b><br>"
                f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>Degree:</span> {degree}<br>"
                f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>Frequency:</span> {freq}"
                f"</div>"
            ),
            borderWidth=2,
            borderWidthSelected=3,
            shadow={
                'enabled': True,
                'color': theme['shadow_color'],
                'size': 12,
                'x': 4,
                'y': 4
            },
            shape='dot',
            mass=max(1, 1 + freq * 0.05)
        )

    color_map = {
        'cooccurrence': theme['edge_cooccurrence'],
        'semantic':     theme['edge_semantic'],
        'bridge':       theme['edge_bridge'],
        'manual':       theme['edge_semantic'],
        'unknown':      theme['edge_unknown']
    }

    all_weights = [nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()]
    weight_threshold = np.percentile(all_weights, 80) if all_weights else 0

    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        color = color_map.get(edge_type, color_map['unknown'])
        width = float(np.clip(w * 0.4, 0.8, 3.5))

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
            title=f"<span style='font-family:Inter,sans-serif;'>Weight: <b>{w:.2f}</b><br>Type: {edge_type}</span>"
        )

        if edge_label_mode == "all":
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {
                'color': theme['font'],
                'size': 9,
                'background': theme['tooltip_bg'],
                'strokeWidth': 2,
                'strokeColor': theme['node_border']
            }
        elif edge_label_mode == "threshold" and w >= weight_threshold:
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {
                'color': theme['font'],
                'size': 9,
                'background': theme['tooltip_bg'],
                'strokeWidth': 2,
                'strokeColor': theme['node_border']
            }

        net.add_edge(u, v, **edge_kwargs)

    html_content = net.generate_html()

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
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")

def render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name="viridis",
                            custom_labels=None, top_n_nodes=0, node_label_size=10,
                            theme=None, show_edge_weights=False):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(nx_graph, k=1.5, iterations=50, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x, edge_y, edge_hover = [], [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        edge_hover.extend([f"<b>{u} + {v}</b><br>Weight: {w:.2f}<br>Type: {edge_type}"] * 2 + [None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1, color=theme['edge_unknown']),
                            hoverinfo='text', hovertext=edge_hover, name='Connections')
    node_x, node_y, node_text, node_size, node_color, node_labels = [], [], [], [], [], []
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}")
        node_size.append(max(8, min(35, deg * 2.5 + 10)))
        node_color.append(cmap_colors[i])
        node_labels.append(custom_labels.get(node, node) if custom_labels else node)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            marker=dict(size=node_size, color=node_color,
                                       line=dict(width=2, color=theme['node_border'])),
                            text=node_labels, textposition="bottom center",
                            textfont=dict(size=node_label_size, color=theme['font']),
                            hovertext=node_text, hoverinfo='text', name='Concepts')
    fig_data = [edge_trace, node_trace]

    if show_edge_weights:
        for u, v in nx_graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = nx_graph[u][v].get('weight', 1)
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            fig_data.append(go.Scatter(
                x=[mid_x], y=[mid_y],
                mode='text',
                text=[f"{w:.1f}"],
                textfont=dict(size=8, color=theme['font']),
                hoverinfo='skip',
                showlegend=False
            ))

    fig = go.Figure(data=fig_data,
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0),
                                     plot_bgcolor=theme['plotly_bg'], paper_bgcolor=theme['plotly_paper'],
                                     font=dict(color=theme['font']),
                                     xaxis=dict(showgrid=True, gridcolor=theme['grid_color'],
                                                zeroline=False, showticklabels=False, linecolor=theme['axis_color']),
                                     yaxis=dict(showgrid=True, gridcolor=theme['grid_color'],
                                                zeroline=False, showticklabels=False, linecolor=theme['axis_color'])))
    st.plotly_chart(fig, use_container_width=True)

def render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name="viridis", top_n_nodes=0,
                            theme=None, show_edge_weights=False):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(nx_graph.nodes()) < 3:
        st.info("3D view requires >=3 nodes.")
        return
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos_3d = nx.spring_layout(nx_graph, dim=3, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x, edge_y, edge_z = [], [], []
    for u, v in nx_graph.edges():
        x0, y0, z0 = pos_3d[u]; x1, y1, z1 = pos_3d[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None]); edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                              line=dict(width=2, color=theme['edge_unknown']), hoverinfo='skip')
    node_x, node_y, node_z, node_text, node_size, node_color, node_labels = [], [], [], [], [], [], []
    for i, node in enumerate(nx_graph.nodes()):
        x, y, z = pos_3d[node]
        node_x.append(x); node_y.append(y); node_z.append(z)
        deg = nx_graph.degree(node); freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}")
        node_size.append(max(6, min(25, deg * 2 + 8)))
        node_color.append(cmap_colors[i])
        node_labels.append(node)
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text',
                                marker=dict(size=node_size, color=node_color, opacity=0.9),
                                text=node_labels, textposition="top center",
                                textfont=dict(size=8, color=theme['font']),
                                hovertext=node_text, hoverinfo='text')
    fig_data = [edge_trace, node_trace]

    if show_edge_weights:
        for u, v in nx_graph.edges():
            x0, y0, z0 = pos_3d[u]
            x1, y1, z1 = pos_3d[v]
            w = nx_graph[u][v].get('weight', 1)
            mid_x, mid_y, mid_z = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
            fig_data.append(go.Scatter3d(
                x=[mid_x], y=[mid_y], z=[mid_z],
                mode='text',
                text=[f"{w:.1f}"],
                textfont=dict(size=7, color=theme['font']),
                hoverinfo='skip',
                showlegend=False
            ))

    fig = go.Figure(data=fig_data,
                    layout=go.Layout(scene=dict(xaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color']),
                                                 yaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color']),
                                                 zaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color'])),
                                     margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
                                     paper_bgcolor=theme['plotly_paper']))
    st.plotly_chart(fig, use_container_width=True)

def render_graph_fallback(nx_graph, concept_abstract_map, theme=None, show_edge_weights=False):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    st.markdown(f"### Graph Summary (Text View)")
    st.markdown(f"- **Nodes**: {len(nx_graph.nodes())}")
    st.markdown(f"- **Edges**: {len(nx_graph.edges())}")
    if len(nx_graph.edges()) > 0:
        edge_list = [(u, v, nx_graph[u][v].get('weight', 1)) for u, v in nx_graph.edges()]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**Top 20 Strongest Connections:**")
        for i, (u, v, w) in enumerate(edge_list[:20], 1):
            edge_type = nx_graph[u][v].get('edge_type', 'unknown')
            weight_badge = f"<span style='background:{theme.get('edge_cooccurrence','#38bdf8') if edge_type=='cooccurrence' else theme.get('edge_semantic','#fb923c') if edge_type=='semantic' else theme.get('edge_unknown','#94a3b8')};color:white;padding:1px 5px;border-radius:4px;font-size:11px;'>W={w:.1f}</span>" if show_edge_weights else ""
            st.markdown(f"{i}. `{u}` + `{v}` {weight_badge} (weight: {w:.2f}, type: {edge_type})", unsafe_allow_html=True)
    if len(concept_abstract_map) > 0:
        freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in nx_graph.nodes()]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**Top Concepts by Frequency:**")
        st.dataframe(pd.DataFrame(freq_data[:15], columns=["Concept", "Abstract Count"]), use_container_width=True)

# ==========================================
# SUNBURST & RADAR CHARTS
# ==========================================
def build_category_hierarchy(valid_concepts: List[str], concept_abstract_map: Dict, top_n_per_category: int = 40):
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    category_map = abstract_concepts_to_categories(valid_concepts)
    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        freq = len(concept_abstract_map.get(concept, []))
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
        labels.append(parent); parents.append(""); values.append(data["count"])
        for child, cnt in data["children"]:
            labels.append(child); parents.append(parent); values.append(cnt)
    return labels, parents, values

def render_sunburst_chart(labels, parents, values, cmap_name="viridis", label_size=11, width=800, height=600, theme=None, branchvalues="total"):
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    n_items = len(labels)
    use_remainder = n_items > 80
    unique_ids = []; seen = {}
    for i, lab in enumerate(labels):
        base = lab[:25] + ("..." if len(lab) > 25 else "")
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base); seen[base] = 1
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
    colors = get_colormap_colors(cmap_name, len(unique_ids))
    bv = branchvalues if branchvalues in ["total", "remainder"] else ("remainder" if use_remainder else "total")
    fig = go.Figure(go.Sunburst(
        labels=unique_ids, parents=parent_ids, values=values, ids=unique_ids,
        branchvalues=bv,
        marker=dict(colors=colors, line=dict(width=0.5, color="white")),
        textinfo="label+percent entry+value",
        insidetextorientation="radial",
        textfont=dict(size=label_size),
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Parent: %{parent}<extra></extra>'
    ))
    fig.update_layout(
        title="<b>CoCrFeNi HEA & Laser AM Research Domain Hierarchy</b><br><i>Size = concept frequency</i>",
        font=dict(size=label_size, family="Arial"),
        paper_bgcolor="white", plot_bgcolor="white",
        width=width, height=height,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_radar_chart(concept_scores_df: pd.DataFrame, top_k: int = 15, cmap_name: str = "viridis", theme=None):
    if concept_scores_df.empty or len(concept_scores_df) < 2:
        st.info("Not enough concepts for radar chart.")
        return
    metrics = ['frequency', 'semantic_density', 'coherence_score', 'distillation_efficiency']
    available_metrics = [m for m in metrics if m in concept_scores_df.columns]
    if not available_metrics:
        st.warning("No metrics available for radar chart.")
        return
    top_concepts = concept_scores_df.nlargest(top_k, 'distillation_efficiency')
    normalized = top_concepts.copy()
    for m in available_metrics:
        col = normalized[m]
        if col.max() > col.min():
            normalized[m] = (col - col.min()) / (col.max() - col.min())
        else:
            normalized[m] = 0.5
    categories = available_metrics
    fig = go.Figure()
    colors = get_colormap_colors(cmap_name, len(normalized))
    for idx, (_, row) in enumerate(normalized.iterrows()):
        concept = row['concept']
        values = [row[m] for m in categories]
        values += values[:1]
        angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself', name=concept[:20],
            line=dict(width=2, color=colors[idx]), fillcolor=colors[idx], opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Top Concepts: Multi-Dimensional Comparison",
        showlegend=True, width=750, height=600,
        paper_bgcolor=theme["plotly_paper"] if theme else "#ffffff",
        font=dict(color=theme["font"] if theme else "#000000"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# EXPORT FUNCTIONS (ENHANCED)
# ==========================================
def export_graph(nx_graph, concept_abstract_map, format_type: str):
    if format_type == "GraphML":
        try:
            nx.write_graphml_lxml(nx_graph, "hea_laser_graph.graphml")
        except:
            nx.write_graphml(nx_graph, "hea_laser_graph.graphml")
        with open("hea_laser_graph.graphml", "rb") as f:
            return f.read(), "application/graphml+xml", "hea_laser_graph.graphml"
    elif format_type == "JSON":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "hea_laser_graph.json"
    elif format_type == "CSV (Edges)":
        edge_data = []
        for u, v, data in nx_graph.edges(data=True):
            row = {"source": u, "target": v}
            row.update({k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))})
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "hea_laser_edges.csv"
    elif format_type == "CSV (Nodes)":
        node_data = []
        for node in nx_graph.nodes():
            row = {"concept": node, "frequency": len(concept_abstract_map.get(node, [])),
                   "degree": nx_graph.degree(node)}
            row.update({k: v for k, v in nx_graph.nodes[node].items()})
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "hea_laser_nodes.csv"
    elif format_type == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [get_nanomaterials_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buf.seek(0); plt.close()
            return buf.read(), "image/png", "hea_laser_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif format_type == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=150)
            node_colors = [get_nanomaterials_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
            buf.seek(0); plt.close()
            return buf.read(), "image/svg+xml", "hea_laser_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    return None, None, None

# ==========================================
# GRAPH METRICS DASHBOARD
# ==========================================
def compute_graph_metrics(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0,
        "connected_components": nx.number_connected_components(G),
        "avg_clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
    }
    try:
        bc = nx.betweenness_centrality(G, normalized=True, k=min(100, G.number_of_nodes()))
        top_bridges = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_bridges"] = top_bridges
        metrics["avg_betweenness"] = np.mean(list(bc.values()))
    except Exception:
        metrics["top_bridges"] = []
    return metrics

def display_metric_dashboard(metrics: dict, theme=None):
    if not metrics:
        st.warning("No graph metrics available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"]); col2.metric("Edges", metrics["edges"])
    col3.metric("Density", f"{metrics['density']:.3f}"); col4.metric("Avg Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6, col7 = st.columns(3)
    col5.metric("Clustering", f"{metrics['clustering']:.3f}")
    col6.metric("Components", metrics["connected_components"])
    col7.metric("Avg Betweenness", f"{metrics.get('avg_betweenness', 0):.3f}")
    if metrics.get("top_bridges"):
        st.markdown("**Top Bridge Concepts (High Betweenness)**")
        bridge_df = pd.DataFrame(metrics["top_bridges"], columns=["Concept", "Bridge Score"])
        st.dataframe(bridge_df, use_container_width=True)

# ==========================================
# THEME CONFIGURATION
# ==========================================
THEME_PRESETS = {
    "Bright (Default)": {
        "bg": "#ffffff", "font": "#1e293b", "tooltip_bg": "rgba(255,255,255,0.95)",
        "tooltip_border": "#cbd5e1", "tooltip_text": "#1e293b",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.45)",
        "edge_semantic": "rgba(251, 146, 60, 0.40)",
        "edge_bridge": "rgba(250, 204, 21, 0.55)",
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
# INTERACTIVE GRAPH EDITING (WITH UNDO/REDO)
# ==========================================
def apply_graph_edits(nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map,
                       nodes_to_remove=None, nodes_to_merge=None, merge_name=None,
                       new_edge=None, new_edge_weight=1.0, min_degree=0, min_freq=0):
    edited = False
    if nodes_to_remove:
        for node in nodes_to_remove:
            if node in nx_graph:
                nx_graph.remove_node(node)
                edited = True
        valid_concepts = [c for c in valid_concepts if c not in nodes_to_remove]
        for node in nodes_to_remove:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
    if nodes_to_merge and merge_name and len(nodes_to_merge) >= 2:
        merged_edges = {}
        merged_freq = 0
        merged_abstracts = set()
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
                            merged_edges[neighbor] = {'weight': w, 'cooccurrence': cooc, 'semantic': sem, 'edge_type': etype}
                merged_freq += nx_graph.nodes[node].get('frequency', 0)
                if node in concept_abstract_map:
                    merged_abstracts.update(concept_abstract_map[node])
                nx_graph.remove_node(node)
        nx_graph.add_node(merge_name, frequency=merged_freq)
        for neighbor, edge_data in merged_edges.items():
            nx_graph.add_edge(merge_name, neighbor, **edge_data)
        concept_abstract_map[merge_name] = list(merged_abstracts)
        valid_concepts = [c for c in valid_concepts if c not in nodes_to_merge]
        if merge_name not in valid_concepts:
            valid_concepts.append(merge_name)
        for node in nodes_to_merge:
            if node in concept_abstract_map and node != merge_name:
                del concept_abstract_map[node]
        edited = True
    if new_edge and len(new_edge) == 2:
        u, v = new_edge
        if u in nx_graph and v in nx_graph and not nx_graph.has_edge(u, v):
            nx_graph.add_edge(u, v, weight=new_edge_weight, cooccurrence=0, semantic=0, edge_type='manual')
            edited = True
    if min_degree > 0:
        low_degree = [n for n in nx_graph.nodes() if nx_graph.degree(n) < min_degree]
        for node in low_degree:
            nx_graph.remove_node(node)
        valid_concepts = [c for c in valid_concepts if c not in low_degree]
        for node in low_degree:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
        edited = True
    if min_freq > 0:
        low_freq = [n for n in nx_graph.nodes() if nx_graph.nodes[n].get('frequency', 0) < min_freq]
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
    return nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, edited

# ==========================================
# EXTRA VISUALIZATIONS
# ==========================================
def render_concept_timeline(df_filtered, valid_concepts, concept_abstract_map, theme=None):
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
    top_concepts = sorted(valid_concepts, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)[:10]
    timeline_data = []
    for year in year_range:
        year_mask = df_filtered["Year"] == year
        year_df = df_filtered[year_mask]
        year_text = ""
        for idx, row in year_df.iterrows():
            for col in df_filtered.columns:
                if pd.notna(row[col]):
                    year_text += " " + str(row[col])
        for concept in top_concepts:
            count = len(re.findall(r'' + re.escape(concept) + r'', year_text, re.I))
            timeline_data.append({"Year": year, "Concept": concept, "Count": count})
    if not timeline_data:
        st.info("No timeline data to display.")
        return
    timeline_df = pd.DataFrame(timeline_data)
    fig = px.line(timeline_df, x="Year", y="Count", color="Concept",
                  title="Concept Frequency Over Time",
                  labels={"Count": "Mentions", "Year": "Publication Year"},
                  template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark")
    fig.update_layout(paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
                      plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
                      font_color=theme.get("font", "#000000"))
    st.plotly_chart(fig, use_container_width=True)

def render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map, top_n=30, theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    top_concepts = sorted(valid_concepts, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)[:top_n]
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
    fig = px.imshow(matrix, x=top_concepts, y=top_concepts,
                    labels=dict(x="Concept", y="Concept", color="Co-occurrence"),
                    title=f"Co-occurrence Heatmap (Top {n} Concepts)",
                    color_continuous_scale="Viridis")
    fig.update_layout(paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
                      font_color=theme.get("font", "#000000"))
    st.plotly_chart(fig, use_container_width=True)

def render_tsne_projection(valid_concepts, concept_abstract_map, embed_model, theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 5:
        st.info("Need at least 5 concepts for t-SNE projection.")
        return
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        perplexity = min(30, len(valid_concepts) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        coords = tsne.fit_transform(embeddings)
        category_map = abstract_concepts_to_categories(valid_concepts)
        categories = [category_map.get(c, 'general') for c in valid_concepts]
        frequencies = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
        df_tsne = pd.DataFrame({
            'x': coords[:, 0], 'y': coords[:, 1],
            'concept': valid_concepts, 'category': categories, 'frequency': frequencies
        })
        fig = px.scatter(df_tsne, x='x', y='y', color='category', size='frequency',
                         hover_data=['concept', 'frequency'],
                         title='t-SNE Projection of Concept Embeddings',
                         labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                         template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark")
        fig.update_layout(paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
                          font_color=theme.get("font", "#000000"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"t-SNE projection failed: {e}")

def render_community_detection(nx_graph, valid_concepts, concept_abstract_map, theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(nx_graph.nodes()) < 3:
        st.info("Need at least 3 nodes for community detection.")
        return
    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(nx_graph))
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        pos = nx.spring_layout(nx_graph, seed=42)
        cmap_colors = get_colormap_colors("tab20", max(len(communities), 1))
        edge_x, edge_y = [], []
        for u, v in nx_graph.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                line=dict(width=0.8, color=theme['edge_unknown']),
                                hoverinfo='none')
        node_traces = []
        for i, comm in enumerate(communities):
            comm_nodes = list(comm)
            node_x, node_y, node_text, node_size = [], [], [], []
            for node in comm_nodes:
                x, y = pos[node]
                node_x.append(x); node_y.append(y)
                deg = nx_graph.degree(node)
                freq = len(concept_abstract_map.get(node, []))
                node_text.append(f"{node}<br>Community {i}<br>Degree: {deg}<br>Freq: {freq}")
                node_size.append(max(10, min(30, deg * 2 + 8)))
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                    marker=dict(size=node_size, color=cmap_colors[i % len(cmap_colors)],
                                               line=dict(width=1.5, color='white')),
                                    text=comm_nodes, textposition="bottom center",
                                    textfont=dict(size=8, color=theme['font']),
                                    hovertext=node_text, hoverinfo='text',
                                    name=f"Community {i} ({len(comm_nodes)})")
            node_traces.append(node_trace)
        fig = go.Figure(data=[edge_trace] + node_traces,
                        layout=go.Layout(showlegend=True, hovermode='closest',
                                         title=f"Community Detection ({len(communities)} communities)",
                                         margin=dict(b=0, l=0, r=0, t=40),
                                         plot_bgcolor=theme['plotly_bg'], paper_bgcolor=theme['plotly_paper'],
                                         font=dict(color=theme['font'])))
        st.plotly_chart(fig, use_container_width=True)
        comm_data = []
        for i, comm in enumerate(communities):
            comm_data.append({
                "Community": i,
                "Size": len(comm),
                "Top Concepts": ", ".join(sorted(comm, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)[:5])
            })
        st.dataframe(pd.DataFrame(comm_data), use_container_width=True)
    except Exception as e:
        st.warning(f"Community detection failed: {e}")

def render_concept_growth(df_filtered, valid_concepts, concept_abstract_map, theme=None):
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
    top_concepts = sorted(valid_concepts, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)[:15]
    growth_data = []
    for concept in top_concepts:
        early_count = 0
        recent_count = 0
        for idx, row in early_df.iterrows():
            text = " ".join([str(row[col]) for col in df_filtered.columns if pd.notna(row[col])])
            early_count += len(re.findall(r'' + re.escape(concept) + r'', text, re.I))
        for idx, row in recent_df.iterrows():
            text = " ".join([str(row[col]) for col in df_filtered.columns if pd.notna(row[col])])
            recent_count += len(re.findall(r'' + re.escape(concept) + r'', text, re.I))
        growth_rate = ((recent_count - early_count) / max(early_count, 1)) * 100 if early_count > 0 else 0
        growth_data.append({
            "Concept": concept,
            "Early Count": early_count,
            "Recent Count": recent_count,
            "Growth Rate (%)": growth_rate
        })
    growth_df = pd.DataFrame(growth_data).sort_values("Growth Rate (%)", ascending=False)
    fig = px.bar(growth_df, x="Concept", y="Growth Rate (%)",
                 color="Growth Rate (%)", color_continuous_scale="RdYlGn",
                 title=f"Concept Growth Rate (Early <={mid_year} vs Recent >{mid_year})",
                 labels={"Growth Rate (%)": "Growth Rate (%)"},
                 template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark")
    fig.update_layout(paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
                      font_color=theme.get("font", "#000000"),
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(growth_df, use_container_width=True)

def render_bubble_chart(nx_graph, valid_concepts, concept_abstract_map, distill_df, theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 3:
        st.info("Need at least 3 concepts for bubble chart.")
        return
    category_map = abstract_concepts_to_categories(valid_concepts)
    bubble_data = []
    for concept in valid_concepts:
        degree = nx_graph.degree(concept) if concept in nx_graph else 0
        freq = len(concept_abstract_map.get(concept, []))
        efficiency = distill_df[distill_df['concept'] == concept]['distillation_efficiency'].values
        efficiency = float(efficiency[0]) if len(efficiency) > 0 else 0.0
        category = category_map.get(concept, 'general')
        bubble_data.append({
            "Concept": concept, "Degree": degree, "Frequency": freq,
            "Distillation Efficiency": efficiency, "Category": category
        })
    bubble_df = pd.DataFrame(bubble_data)
    fig = px.scatter(bubble_df, x="Degree", y="Frequency", size="Distillation Efficiency",
                     color="Category", hover_data=["Concept"],
                     title="Concept Importance Bubble Chart",
                     size_max=50,
                     template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark")
    fig.update_layout(paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
                      font_color=theme.get("font", "#000000"))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# SIDEBAR CONFIGURATION (ENHANCED)
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0
        )
        theme = THEME_PRESETS[st.session_state['theme']]

        st.subheader("HEA & Laser AM Focus Areas")
        st.markdown("- CoCrFeNi high-entropy alloys (HEA/MPEA)")
        st.markdown("- Laser Powder Bed Fusion (LPBF) & Additive Manufacturing")
        st.markdown("- Thermodynamic Data Tensors (TDT) & CPD")
        st.markdown("- Phase-Field Modeling (Allen-Cahn, KKS)")
        st.markdown("- Marangoni Fluid Dynamics & Melt Pool")
        st.markdown("- AI Surrogates & Cross-Attention Mechanisms")
        st.markdown("- Digital Twins & Physics-Informed ML")
        st.markdown("- Microstructural Evolution & Elemental Partitioning")
        st.markdown("- Computational Methods (FEA/MOOSE, ALS)")

        st.subheader("Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:", ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"], index=0
        )
        st.session_state['show_edge_weights'] = st.toggle(
            "Show edge weights", value=False,
            help="Display numerical weight labels on graph edges. Useful for quantitative analysis but may clutter dense graphs."
        )
        st.session_state['edge_label_mode'] = st.selectbox(
            "Edge label mode:", ["hover", "threshold", "all"], index=0,
            help="hover=tooltip only, threshold=top 20% edges, all=all edges"
        )
        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:", options=list(SUPPORTED_COLORMAPS.keys()), index=0
        )

        st.subheader("Physics & Layout")
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

        st.subheader("Display Limits")
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

        st.subheader("Graph Parameters")
        st.session_state['min_freq'] = st.slider("Min concept frequency", 1, 20, 1)
        st.session_state['min_words'] = st.slider("Min words per concept", 2, 5, 2)
        st.session_state['sim_threshold'] = st.slider("Semantic threshold", 0.6, 0.95, 0.85, step=0.05)
        st.session_state['cooc_weight'] = st.slider("Co-occurrence weight", 0.5, 1.0, 0.9, step=0.1)
        st.session_state['sem_weight'] = st.slider("Semantic weight", 0.0, 0.5, 0.1, step=0.1)

        st.subheader("Statistics")
        st.session_state['bootstrap_samples'] = st.slider("Bootstrap samples", 100, 2000, 500, step=100)
        st.session_state['alpha_level'] = st.selectbox("Significance alpha", [0.01, 0.05, 0.10], index=1)

        # Graph Editing Section
        st.markdown("---")
        st.subheader("Graph Editing")
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
                if st.button("Undo", key="undo_btn") and st.session_state['edit_history'].can_undo():
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
                if st.button("Redo", key="redo_btn") and st.session_state['edit_history'].can_redo():
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
        st.subheader("Sunburst Options")
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

        st.markdown("---")
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"Device: {gpu_info}")

# ==========================================
# MAIN APPLICATION (ENHANCED)
# ==========================================
def main():
    st.title("HEA-LaserConceptGraph: CoCrFeNi HEA & Laser AM Explorer")
    st.caption("Large-corpus concept graph builder for CoCrFeNi high-entropy alloys & laser additive manufacturing | Optimized for TDT/CPD thermodynamics, phase-field modeling, Marangoni fluid dynamics, and AI surrogate models")
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
    st.header("Data Loading")
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

    # --- RUN ANALYSIS ---
    if st.button("Build Concept Graph", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status = st.status("Initializing analysis...", expanded=True)
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
                config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.9)
                config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.1)
                st.write(f"Adaptive config: {config}")
                progress_bar.progress(0.15)
                st.write("Extracting concepts from abstracts...")
                all_concepts, all_metrics = extract_concepts_from_abstracts(df_filtered, selected_text_cols)
                st.write(f"Extracted concepts from {len(all_concepts)} documents")
                progress_bar.progress(0.30)
                st.write("Filtering and normalizing concepts...")
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts, config)
                st.write(f"**{len(valid_concepts)}** valid concepts retained")
                progress_bar.progress(0.45)
                if len(valid_concepts) < 5:
                    st.error("Too few concepts extracted. Try lowering frequency thresholds.")
                    return
                st.write("Building concept graph...")
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

                # NEW: Advanced analytics
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

                st.success("Analysis complete!")
                progress_bar.progress(1.00)
                status.update(label="Analysis complete!", state="complete", expanded=False)
                st.session_state.analysis_data = {
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

        viz_tab, distill_tab, scores_tab, valid_tab, export_tab, extra_viz_tab, advanced_tab = st.tabs([
            "Visualization", "Distillation", "Research Directions", "Validation", "Export", "Extra Viz", "Advanced Analytics"
        ])

        with viz_tab:
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

            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=physics,
                                   cmap_name=cmap, top_n_nodes=top_n,
                                   theme=theme, physics_preset=physics_preset,
                                   show_edge_weights=show_weights, edge_label_mode=edge_label_mode)
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

        with distill_tab:
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

        with scores_tab:
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

        with valid_tab:
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

        with export_tab:
            st.subheader("Export & Post-Processing")
            export_format = st.selectbox("Format:", ["GraphML", "JSON", "CSV (Edges)", "CSV (Nodes)", "PNG", "SVG"])
            if st.button("Generate Export"):
                result = export_graph(nx_graph, concept_abstract_map, export_format)
                if result[0]:
                    data_bytes, mime, filename = result
                    st.download_button("Save File", data=data_bytes, file_name=filename, mime=mime)

            # Publication figure export
            st.markdown("---")
            st.subheader("Publication-Ready Figure")
            pub_dpi = st.slider("DPI", 150, 600, 300, step=50)
            pub_figsize = st.selectbox("Figure size:", [(10, 8), (12, 10), (14, 12), (16, 14)], index=2)
            if st.button("Generate Publication Figure"):
                pub_bytes = export_publication_figure(nx_graph, valid_concepts, concept_abstract_map,
                                                       cmap_name=cmap, dpi=pub_dpi, figsize=pub_figsize)
                if pub_bytes:
                    st.download_button("Download Publication PNG", data=pub_bytes,
                                     file_name="hea_laser_publication.png", mime="image/png")

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
                st.download_button("Download Report (Markdown)", data=report.encode('utf-8'),
                                  file_name="hea_laser_analysis_report.md", mime="text/markdown")
                with st.expander("Preview Report"):
                    st.markdown(report)

            concept_list_df = pd.DataFrame({
                'concept': valid_concepts,
                'frequency': [len(concept_abstract_map.get(c, [])) for c in valid_concepts],
                'degree': [nx_graph.degree(c) for c in valid_concepts],
                'category': [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Concept List (CSV)", data=csv_concepts,
                              file_name="concepts.csv", mime="text/csv")

        with extra_viz_tab:
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

        # NEW: Advanced Analytics Tab
        with advanced_tab:
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

                    # Correlation heatmap of centralities
                    corr_cols = ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']
                    available = [c for c in corr_cols if c in centrality_df.columns]
                    if len(available) >= 2:
                        corr_matrix = centrality_df[available].corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                        title="Centrality Correlation Matrix",
                                        color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig, use_container_width=True)

                    # Degree distribution
                    fig = plot_degree_distribution(nx_graph, theme=theme)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No centrality data available.")

if __name__ == "__main__":
    main()
