
# FBM Pipeline Overview (Analysis + Clustering)

The FBM pipeline is organized into two main stages:

* **01_FBM_Analysis**: preprocessing, trial extraction, ERSP computation, anatomical reconstruction, and report generation
* **02_FBM_Clustering**: feature extraction from ERSPs, clustering, atlas projection, and dynamic visualization

The workflow is sequential: raw electrophysiology data are first transformed into trial-aligned ERSP representations, which are then used to derive higher-level structure through blob-based feature extraction and clustering. These clusters are subsequently mapped onto a common anatomical space to enable interpretation at the population level, both statically (figures, PDFs) and dynamically (time-resolved activity maps and videos).

While the architecture is conceptually modular, some notebooks still contain legacy or exploratory code; the structure below reflects the intended functional pipeline.

---

# 01_FBM_Analysis

## Purpose

Transforms raw electrophysiology recordings into structured ERSP outputs and anatomical visualizations, and compiles per-patient summaries.

## Structure

```text
01_FBM_Analysis/
├── 01_PD_extract_*.ipynb
│   ├── Extract photodiode-based trial onsets/offsets
│   ├── Handles different datasets (PAT, EL, MicroEPI)
│   └── Outputs trial TSV files to prep0/
│
├── 04_ERSP_LM.ipynb
│   ├── Loads raw data + trial TSVs
│   ├── Computes ERSP and high-gamma activity
│   ├── Applies masking / statistical filtering
│   └── Saves ERSP plots and condition-wise outputs
│
├── 05_Brain_Atlas.ipynb
│   ├── Loads FreeSurfer anatomy and electrode coordinates
│   ├── Renders electrode locations on cortical surfaces
│   └── Saves glass-brain and mosaic visualizations
│
├── 06_PDF_Summary_LM.ipynb
│   ├── Aggregates ERSP and atlas outputs
│   ├── Creates per-patient summary PDFs
│   └── Supports selected or full-electrode reports
│
├── 07_ClusteringOutput.ipynb
│   ├── Recomputes / formats ERSPs for clustering
│   ├── Exports clean ERSP matrices and images
│   └── Saves clustering-ready datasets
│
├── lf_io_utils.py
├── lf_trials.py
├── lf_ersp.py
├── lf_recon.py
│   └── Core helper modules for I/O, trials, ERSP, and reconstruction
│
└── outputs/
    ├── 04_ersp_LM/
    ├── Paper1_recons/
    └── 07_ClusteringOutput/
```

## Key idea

This stage standardizes heterogeneous raw data into:

* **trial-aligned signals**
* **ERSP representations (time × frequency)**
* **anatomical electrode mappings**

These outputs form the input to clustering.

---

# 02_FBM_Clustering

## Purpose

Transforms ERSP outputs into structured features, identifies clusters, and maps them onto a shared anatomical space for interpretation.

## Structure

```text
02_FBM_Clustering/
├── 230_blob_clustering.ipynb
│   ├── Loads ERSP outputs from 01_FBM_Analysis
│   ├── Extracts blob-based features
│   ├── Applies gating / QC on samples
│   ├── Performs embedding (e.g., UMAP) and clustering
│   └── Saves clustering runs, metadata, and QC outputs
│
├── 240_blob_cluster_recon.ipynb
│   ├── Loads a selected clustering run
│   ├── Builds atlas-aligned inputs (electrode → brain space)
│   ├── Renders cluster distributions on brain surfaces
│   ├── Saves cluster-level ERSP summaries
│   └── Generates static reports and figures
│
├── 241_atlas_activity_plots.ipynb
│   ├── Loads clustering + atlas outputs
│   ├── Aggregates activity over time-frequency windows
│   ├── Renders dynamic brain activity maps
│   ├── Saves frame sequences
│   └── Compiles MP4 videos
│
├── functions
│   ├── *.py to be filled after cleaning the folder of redundancy
│
└── outputs/
    ├── 230_blob_clustering_runs/
    ├── 240_blob_cluster_recon/
    └── 241_atlas_activity_plots/
```

## Key idea

This stage converts ERSP data into:

* **feature vectors (blob representations)**
* **cluster assignments**
* **atlas-level interpretations**

and enables both:

* **static interpretation** (cluster maps, PDFs)
* **dynamic interpretation** (time-resolved brain activity videos)

---

# End-to-end pipeline summary

```text
Raw data (ns6 / h5 / mat)
        ↓
01_PD_extract_*  → trial timing (TSV)
        ↓
04_ERSP_LM       → ERSP (time × frequency)
        ↓
07_ClusteringOutput → cleaned ERSP matrices
        ↓
230_blob_clustering → features + clusters
        ↓
240_blob_cluster_recon → atlas cluster maps
        ↓
241_atlas_activity_plots → dynamic brain activity videos
```
