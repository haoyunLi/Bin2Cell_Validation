# Bin2Cell Validation Method for Visium HD Data

This repository provides a validation pipeline to test the performance of different bin-level to cell-level assignment tools for Visium HD spatial transcriptomics data.

## Overview

This validation method creates ground truth data by combining whole-cell segmentation, cell type annotation, and single-cell RNA-seq data assignment. The goal is to evaluate whether bin-to-cell tools can accurately recover cell-level gene expression and cell boundaries from binned Visium HD data.

## Pipeline Overview

The validation pipeline consists of three main steps:

### Step 1: Generate Whole-Cell Segmentation (Ground Truth Masks)

**Objective:** Perform unbiased whole-cell segmentation on the full H&E image to obtain cell boundaries.

- **Tool Used:** [CSGO (Cell Segmentation with Globally Optimized boundaries)](https://github.com/QBRC/CSGO)
- **Why CSGO:** Provides whole-cell segmentation without bias or information leakage, as it's independent of the transcriptomic data
- **Input:** Full-resolution Visium HD H&E image
- **Output:** Cell segmentation masks with boundaries for nucleus, cytoplasm, and cell membrane

**Implementation:**
- Script: `csgo_segmentation_tiled.py` - Processes large images in tiles to handle memory constraints
- SBATCH: `run_csgo_segmentation.sbatch` - HPC submission script

### Step 2: Cell Type Annotation via CellTypist

**Objective:** Determine the cell type composition in the tissue to guide single-cell data selection.

- **Tool Used:** [CellTypist](https://www.celltypist.org/)
- **Input:** Visium HD 8μm binned data with Space Ranger clustering results
- **Model:** Mouse_Whole_Brain.pkl (for mouse brain tissue) or based on the tissue user use
- **Output:** Cell type annotations for each bin, showing the distribution of cell types in the tissue

**Implementation:**
- Script: `run_celltypist_annotation.py` - Annotates clusters and generates UMAP visualizations
- SBATCH: `run_cellannotation.sbatch` - HPC submission script

### Step 3: Single-Cell Data Assignment to Segmented Cells

**Objective:** Create ground truth cell-level gene expression by assigning single-cell RNA-seq profiles to segmented cells.

**Workflow:**
1. **Find matching single-cell dataset:** Identify a single-cell RNA-seq dataset from the same tissue type with matching cell type composition
2. **Gene localization analysis:** Perform GO (Gene Ontology) analysis to determine which genes are expressed in:
   - Nucleus
   - Cytoplasm
   - Cell membrane
3. **Spatial assignment:** Assign single-cell profiles to segmented cells based on:
   - Cell type matching
   - Spatial location in tissue
   - One-to-one mapping (no cell reuse)
   - Subcellular compartment-specific gene expression (genes assigned to nucleus, cytoplasm, or membrane regions of the segmented cell based on their localization)

**Output:**
- Cell-level ground truth with known gene expression profiles
- Spatial coordinates of each cell
- Subcellular gene localization information

## Ground Truth Data Structure

The final ground truth dataset consists of three components:

1. **Cell-level data (from single-cell assignment):**
   - Gene expression matrix at single-cell resolution
   - Cell type labels
   - Cell boundaries and spatial coordinates

2. **Bin-level data (from Visium HD):**
   - Gene expression from 2μm bins
   - Spatial coordinates of bins
   - Known overlap with segmented cells

3. **Segmentation masks (from CSGO):**
   - Whole-cell boundaries
   - Subcellular compartments (nucleus, cytoplasm, membrane)
   - Pixel-level annotations

## Validation Approach

With this ground truth, we can evaluate bin-to-cell assignment tools by:

1. **Input:** Provide binned Visium HD data (2μm or 8μm resolution) to the bin-to-cell tool
2. **Tool Output:** The tool assigns bin-level data to predicted cells
3. **Comparison:** Compare the tool's output against our ground truth:
   - **Gene expression recovery:** How well does the tool recover the true cell-level gene expression from single-cell data?
   - **Cell boundary accuracy:** How well do the predicted cell boundaries match the CSGO segmentation masks?
   - **Cell type assignment:** Are bins correctly assigned to the right cell types?

## Performance Metrics

Tools can be evaluated on:
- Gene expression correlation (predicted vs. ground truth)
- Cell boundary overlap (IoU with CSGO masks)
- Cell type accuracy (concordance with CellTypist annotations)
- Spatial accuracy (correct bin-to-cell assignments)

## Installation

### 1. Create Conda Environment

```bash
# Create environment in local directory (to save space in home)
conda create --prefix ./Bin2Cell_Validation python=3.12 -y
conda activate ./Bin2Cell_Validation

# Install dependencies
pip install -r requirements.txt
```

### 2. Install CSGO

```bash
git clone git@github.com:QBRC/CSGO.git
cd CSGO
conda env create -f environment.yml
conda activate cell-seg-go
```