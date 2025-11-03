# Bin2Cell Validation Method for Visium HD Data

This repository provides a validation pipeline to test the performance of different bin-level to cell-level assignment tools for Visium HD spatial transcriptomics data.

## Overview

This validation method creates ground truth data by combining whole-cell segmentation, cell type annotation, and single-cell RNA-seq data assignment. The goal is to evaluate whether bin-to-cell tools can accurately recover cell-level gene expression and cell boundaries from binned Visium HD data.

## Pipeline Overview

The validation pipeline consists of the following steps:

### Step 0: Crop Visium HD Image to Tissue Region

**Objective:** Extract and crop the full-resolution microscope image to only the tissue-covered region.

- **Tool:** `crop_visium_hd_image.py` module
- **Input:**
  - Full-resolution microscope image (.tif, .tiff, .btf formats)
  - Tissue positions parquet file from Space Ranger output
- **Output:**
  - Cropped image containing only Visium HD tissue region
  - Pixel-to-spot mapping array
  - Filtered tissue positions dataframe
  - Visualization of crop boundaries

**Implementation:**

- Module: `crop_visium_hd_image.py` - Core cropping functionality
- Script: `crop_visium_hd_script.py` - Batch processing script
- Notebook: `crop_visium_hd_notebook.ipynb` - Interactive tutorial

**Configuration:** Edit paths in `crop_visium_hd_script.py` or `crop_visium_hd_notebook.ipynb`:

```python
df_path = 'your_data/binned_outputs/square_002um/spatial/tissue_positions.parquet'
img_path = 'your_data/input/tissue_image.tif'
output_path = 'cropped_image.png'
```

### Step 1: Generate Whole-Cell Segmentation (Ground Truth Masks)

**Objective:** Perform unbiased whole-cell segmentation on the full H&E image to obtain cell boundaries with subcellular compartment classification.

- **Tool Used:** [Cellpose cyto3](https://cellpose.readthedocs.io/)
- **Why Cellpose cyto3:**
  - Specifically trained for brightfield/H&E images with built-in nuclear detection
  - Provides accurate whole-cell segmentation with state-of-the-art performance
  - Independent of transcriptomic data to avoid information leakage
  - Single model for both cell and nuclear segmentation (more consistent)

**Model Configuration:**

- **Cell Segmentation:**

  - Model: `cyto3` (optimized for brightfield images)
  - `flow_threshold = 0.4` (default, maintains cell quality)
  - `cellprob_threshold = -2.0` (more permissive, includes dim pixels for bigger/complete cells)
  - `diameter = None` (auto-estimate)

- **Nuclear Segmentation:**

  - Method: Extract from cyto3's built-in nuclear probability map
  - `nuclear_threshold = 0.65` (balanced detection, aims for ~70% nuclear, ~30% cytoplasm)
  - Uses label-based component detection for individual nuclei

- **Boundary Detection:**
  - Method: 4-connectivity
  - Result: Thin, precise boundaries suitable for transcript assignment

**Input:** Full-resolution Visium HD H&E image (cropped tissue region)

**Output:** Cell segmentation masks with subcellular compartment assignments:

- **Boundary** (cell membrane regions, 4-connected)
- **Nuclear** (nucleus regions matched 1:1 with cells)
- **Cytoplasm** (interior - nuclear)
- Background pixels

**Implementation:**

- Script: `cellpose_segmentation.py` - Optimized pipeline for whole-cell and nuclear segmentation

**Output Files:**

- `*_cell_masks.tif/.npy` - Full-resolution whole-cell segmentation masks
- `*_nuclear_masks.tif/.npy` - Nuclear segmentation masks (matched to cells)
- `*_boundary_mask.npy` - Cell boundary masks (4-connected)
- `*_nuclear_binary_mask.npy` - Binary nuclear mask (matched to cells, interior only)
- `*_pixel_to_cell_mapping_full.csv.gz` - Compressed pixel-level mapping with columns:
  - `x, y`: pixel coordinates
  - `cell_id`: which cell this pixel belongs to (0 = background)
  - `is_boundary`: 1 if membrane pixel (4-connected)
  - `is_nuclear`: 1 if nuclear pixel (matched to cell, interior only)
  - `is_cytoplasm`: 1 if cytoplasm pixel (interior - nuclear)
  - `is_interior`: 1 if non-boundary pixel inside cell
- `*_cell_nuclear_overlay_downsampled_1x.png` - Visualization (cells with nuclei overlaid)

### Step 1.5 (Optional): Alternative Nuclear Segmentation with StarDist

**Objective:** Run alternative nuclear segmentation using StarDist to compare with Cellpose cyto3's built-in nuclear detection.

- **Tool Used:** [StarDist](https://github.com/stardist/stardist) - Pre-trained model for H&E images
- **Why StarDist:**
  - Independent nuclear segmentation method (similar to SMURF pipeline)
  - Useful for comparison and validation of Cellpose nuclear results
  - Pre-trained on H&E images with robust performance

**Model Configuration:**

- Model: `2D_versatile_he` (pre-trained for H&E images)
- `prob_thresh = 0.6` (nucleus detection confidence)
- `nms_thresh = 0.3` (non-maximum suppression)
- Tiled processing for large images (tile_size=4700px, overlap=80px)

**Input:**

- Full-resolution Visium HD H&E image (cropped tissue region)
- Cellpose cell masks (for nucleus-to-cell matching)

**Output:**

- `*_stardist_nuclear_masks.npy` - StarDist nuclear segmentation
- `*_stardist_nuclear_binary_matched.npy` - Binary nuclear mask matched to cells
- `*_stardist_pixel_to_cell_mapping_full.csv.gz` - Pixel-level mapping with StarDist nuclei
- Comparison visualizations and statistics

**Implementation:**

- Script: `stardist_nuclear_segmentation.py` - StarDist nuclear segmentation with cell matching

**Note:** This is optional - Cellpose cyto3's built-in nuclear detection is already high quality. Use StarDist if you want to compare different nuclear segmentation methods or validate results.

### Step 2: Cell Type Annotation via CellTypist

**Objective:** Determine the cell type composition in the tissue to guide single-cell data selection.

- **Tool Used:** [CellTypist](https://www.celltypist.org/)
- **Input:** Visium HD 8μm binned data with Space Ranger clustering results
- **Model:** Mouse_Whole_Brain.pkl (for mouse brain tissue) or based on the tissue user use
- **Output:** Cell type annotations for each bin, showing the distribution of cell types in the tissue

**Implementation:**

- Script: `run_celltypist_annotation.py` - Annotates clusters and generates UMAP visualizations
- SBATCH: `run_cellannotation.sbatch` - HPC submission script

### Step 2.5: Map Cell Type Annotations to Segmented Cells

**Objective:** Assign cell type labels from 8μm cluster annotations to pixel-level Cellpose segmentation results.

- **Process:** Maps each 2μm pixel from Cellpose segmentation to its corresponding 8μm bin, then assigns cell types via majority vote
- **Input:**
  - Cellpose pixel-to-cell mapping CSV (2μm resolution, ~600M pixels)
  - 8μm tissue positions with cluster assignments (from Space Ranger)
  - Cluster-to-cell-type mapping (from CellTypist)
  - Annotated h5ad file with cluster assignments

**Algorithm:**

1. Load 8μm bin positions and cluster-to-cell-type mapping
2. Create vectorized mapping from 2μm pixels to 8μm bins (divide coordinates by 4)
3. For each pixel, look up its corresponding 8μm bin barcode
4. Assign cluster ID and cell type to each pixel based on bin annotation
5. Use majority vote to assign final cell type to each segmented cell

**Output:**

- `*_pixel_to_cell_mapping_with_celltype.csv.gz` - Enhanced pixel mapping with columns:
  - `x, y`: Pixel coordinates (2μm resolution)
  - `cell_id`: Cell identifier from Cellpose
  - `is_boundary`, `is_interior`, `is_nuclear`, `is_cytoplasm`: Compartment flags
  - `cell_type`: Assigned cell type (from 8μm cluster annotation)
- Cell type distribution statistics

**Implementation:**

- Script: `map_cells_to_clusters.py` - Vectorized cell type assignment pipeline

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

3. **Segmentation masks (from Cellpose):**
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

### 2. Install Cellpose

```bash
# Create Cellpose environment
conda create --prefix ./cellpose python=3.10 -y
conda activate ./cellpose

# Install Cellpose
pip install cellpose[gui]

# Optional: Install additional dependencies for large image processing
pip install scikit-image pandas
```
