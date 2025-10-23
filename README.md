# Bin2Cell Validation Method for Visium HD Data

This repository provides a validation pipeline to test the performance of different bin-level to cell-level assignment tools for Visium HD spatial transcriptomics data.

## Overview

This validation method creates ground truth data by combining whole-cell segmentation, cell type annotation, and single-cell RNA-seq data assignment. The goal is to evaluate whether bin-to-cell tools can accurately recover cell-level gene expression and cell boundaries from binned Visium HD data.

## Pipeline Overview

The validation pipeline consists of three main steps:

### Step 1: Generate Whole-Cell Segmentation (Ground Truth Masks)

**Objective:** Perform unbiased whole-cell segmentation on the full H&E image to obtain cell boundaries.

- **Tool Used:** [Cellpose-SAM](https://cellpose.readthedocs.io/)
- **Why Cellpose-SAM:** Provides accurate whole-cell segmentation with state-of-the-art performance using a foundation model, independent of transcriptomic data to avoid information leakage
- **Models Used:**
  - **Cellpose-SAM model**: For whole-cell segmentation
  - **Cellpose nuclei model**: For nuclear segmentation
- **Input:** Full-resolution Visium HD H&E image
- **Output:** Cell segmentation masks with subcellular compartment assignments:
  - **Boundary** (cell membrane regions)
  - **Nuclear** (nucleus regions)
  - **Cytoplasm** (interior - nuclear)
  - Background pixels

**Implementation:**

- Script: `cellpose_sam_segmentation.py` - Processes tissue images for whole-cell and nuclear segmentation
- **Output Files:**
  - `*_cell_masks.npy` - Full-resolution whole-cell segmentation masks
  - `*_nuclear_masks.npy` - Nuclear segmentation masks
  - `*_boundary_mask.npy` - Cell boundary masks
  - `*_pixel_to_cell_mapping_full.csv.gz` - Compressed pixel-level mapping with columns:
    - `x, y`: pixel coordinates
    - `cell_id`: which cell this pixel belongs to
    - `is_boundary`: membrane regions
    - `is_nuclear`: nucleus regions
    - `is_cytoplasm`: cytoplasm regions (interior - nuclear)
    - `is_interior`: all non-boundary regions
  - Visualization files (downsampled for performance)

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

**Implementation:**

- Script: `gene_localization_GO_analysis.py` - Classify genes by subcellular localization (nucleus, cytoplasm, cell membrane) using GO term analysis

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

3. **Segmentation masks (from Cellpose-SAM):**
   - Whole-cell boundaries
   - Subcellular compartments:
     - **Boundary** (cell membrane)
     - **Nuclear** (nucleus)
     - **Cytoplasm** (computed as interior - nuclear)
   - Pixel-level annotations for all compartments

## Validation Approach

With this ground truth, we can evaluate bin-to-cell assignment tools by:

1. **Input:** Provide binned Visium HD data (2μm or 8μm resolution) to the bin-to-cell tool
2. **Tool Output:** The tool assigns bin-level data to predicted cells
3. **Comparison:** Compare the tool's output against our ground truth:
   - **Gene expression recovery:** How well does the tool recover the true cell-level gene expression from single-cell data?
   - **Cell boundary accuracy:** How well do the predicted cell boundaries match the Cellpose-SAM segmentation masks?
   - **Cell type assignment:** Are bins correctly assigned to the right cell types?

## Performance Metrics

Tools can be evaluated on:

- Gene expression correlation (predicted vs. ground truth)
- Cell boundary overlap (IoU with Cellpose-SAM masks)
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

### 2. Install Cellpose-SAM

```bash
# Create Cellpose environment
conda create --prefix ./cellpose python=3.10 -y
conda activate ./cellpose

# Install Cellpose with SAM support
pip install cellpose[gui]

# Optional: Install additional dependencies for large image processing
pip install scikit-image pandas
```
