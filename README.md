# Bin2Cell Validation Method for Visium HD Data

This repository provides a validation pipeline to test the performance of different bin-level to cell-level assignment tools for Visium HD spatial transcriptomics data.

## Overview

This validation method creates ground truth data by combining whole-cell segmentation, cell type annotation, and single-cell RNA-seq data assignment. The goal is to evaluate whether bin-to-cell tools can accurately recover cell-level gene expression and cell boundaries from binned Visium HD data.

## Pipeline Overview

The validation pipeline consists of the following steps:

## Recommended Execution Order & Commands

Follow this order whenever you regenerate the full benchmark. Paths below use the
current kidney/colorectal examples—update them to match your dataset.

1. **Crop Visium HD image first**

   ```bash
   # Uses paths defined inside crop_visium_hd_script.py
   python crop_visium_hd_script.py
   ```

2. **Run Cellpose whole-cell segmentation**

   ```bash
   conda activate ./cellpose
   python cellpose_segmentation.py \
       --img_path cropped_visium_hd_human_kidney.png \
       --output_dir cellpose_sam_human_kidney_output
   conda deactivate
   ```

3. **(Optional) Validate nuclei with StarDist** – provides the "stardist" comparison the
   SMURF pipeline expects.

   ```bash
   # Update default paths inside the script if needed
   python stardist_nuclear_segmentation.py
   ```

4. **Annotate cells (choose one path)**

   - **STHD Annotation (recommended):**

     ```bash
     source sthd_env/bin/activate
     python build_sthd_lambda_file.py \
         --h5_file colorectal_sc_data/CRC_GSE166555_expression.h5 \
         --metadata_file colorectal_sc_data/CRC_GSE166555_CellMetainfo_table.tsv \
         --output lambda_colorectal_major_lineage.txt \
         --celltype_column "Celltype (major-lineage)"
     deactivate
     ```

     ```bash
     source sthd_env/bin/activate
     python sthd_annotation.py \
         --pixel_csv cellpose_sam_human_colorectal_output/cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_full.csv.gz \
         --visium_dir Human_Colorectal/output/binned_outputs/square_002um \
         --lambda_file lambda_colorectal_major_lineage.txt \
         --output cellpose_sam_human_colorectal_output/cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_with_celltype.csv.gz \
         --microns_per_pixel 0.2737012522439323
     deactivate
     ```

   - **CellTypist → cluster mapping (fallback):**

     ```bash
     python run_celltypist_annotation.py \
         --data_path Human_kidney/output/binned_outputs/square_008um/filtered_feature_bc_matrix.h5 \
         --clustering_path Human_kidney/output/binned_outputs/square_008um/analysis/clustering/gene_expression_graphclust/clusters.csv \
         --umap_path Human_kidney/output/binned_outputs/square_008um/analysis/umap/gene_expression_2_components/projection.csv \
         --output_dir cellannotation_results_human_kidney

     python map_cells_to_clusters.py \
         --cellpose_csv cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_full.csv.gz \
         --tissue_positions Human_kidney/output/binned_outputs/square_008um/spatial/tissue_positions.parquet \
         --h5ad_path cellannotation_results_human_kidney/visium_hd_annotated.h5ad \
         --cluster_mapping cellannotation_results_human_kidney/cluster_to_celltype_mapping.csv \
         --output_csv cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_with_celltype.csv.gz
     ```

5. **Run nuclear expansion** (required before pseudo HD creation):

   ```bash
   python nuclear_expansion.py \
       --input_csv cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_with_celltype.csv.gz \
       --output_csv cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_expanded.csv.gz \
       --output_vis cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_cell_nuclear_expanded_overlay.png \
       --nuclear_range_csv cell_based_nuclear_range.csv \
       --original_image cropped_visium_hd_human_kidney.png
   ```

6. **Generate pseudo Visium HD data** (uses `config_pseudo_hd.py` for paths):

   ```bash
   python create_pseudo_visium_hd.py
   ```

7. **Validate pseudo HD cell assignments** to confirm the ground-truth mapping:

   ```bash
   python validate_cell_assignments.py \
       --pseudo_hd_dir pseudo_visium_hd_outpu_full \
       --output_dir validation_results \
       --sample_size 100
   ```

8. **Run SMURF segmentation + deconvolution over pseudo HD data:**

   ```bash
   conda activate smurf
   python SMURF_analyze_spatial.py \
       --tissue_positions Human_kidney/output/binned_outputs/square_002um/spatial/tissue_positions.parquet \
       --tissue_image Human_kidney/input/Visium_HD_Human_Kidney_FFPE_tissue_image.tif \
       --pseudo_hd_dir pseudo_visium_hd_outpu_full \
       --output_dir smurf_result
   conda deactivate
   ```

9. **Validate SMURF output** (nuclear overlap, whole-cell IoU, gene correlation):

   ```bash
   conda activate smurf
   python validate_smurf_results.py \
       --smurf_dir smurf_result \
       --pseudo_hd_dir pseudo_visium_hd_outpu_full \
       --pixel_file cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_expanded.csv.gz \
       --sc_h5_file kidney_sc_data/KIRC_GSE159115_expression.h5 \
       --sc_meta_file kidney_sc_data/KIRC_GSE159115_CellMetainfo_table.tsv \
       --output_dir smurf_validation_output
   conda deactivate
   ```

Steps 3 and 4b are optional, but the rest should be executed sequentially to keep
metadata synchronized for downstream validation and SMURF benchmarking.

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
  - `nuclear_threshold = 0.6` 
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

### Step 1.6: Cell Expansion Based on Nuclear:Cytoplasm Ratios

**Objective:** Expand cell boundaries around nuclei to match biologically accurate nuclear:cytoplasm (N:C) ratios for each cell type.

- **Tool Used:** `nuclear_expansion.py`
- **Method:**
  - Keeps nuclear regions EXACTLY the same (no changes to nuclei)
  - Expands cytoplasm and cell boundaries outward from current edges
  - Uses cell type-specific N:C ratio ranges from literature
  - Expands each cell iteratively until it reaches the target N:C ratio for its cell type
  - Uses circular morphological dilation (disk structuring element) for natural, elliptical cell shapes
  - Handles overlapping cell boundaries during expansion (first-come-first-served for boundary pixels)
  - Processes cells in batches for memory efficiency

**Why This Matters:**
- Cellpose often under-segments cells (conservative boundaries)
- Many cell types have specific N:C ratios based on biology:
  - Epithelial cells: 20-30% nuclear
  - Immune cells: 40-60% nuclear (large nuclei, less cytoplasm)
  - Neurons: 10-20% nuclear (extensive cytoplasm/dendrites)
- Expanding cells to match biological ratios improves spatial accuracy

**Input:**
- **Pixel-to-cell mapping**: `*_pixel_to_cell_mapping_with_celltype.csv.gz` (with cell type annotations)
- **N:C ratio ranges**: `cell_based_nuclear_range.csv` (cell type-specific nuclear percentage ranges)
- **Nuclear masks**: Original nuclear segmentation (kept unchanged as anchor)

**Output:**
- **Expanded pixel mapping**: `*_pixel_to_cell_mapping_expanded.csv.gz` with updated:
  - `is_cytoplasm`: Expanded cytoplasm regions
  - `is_boundary`: Updated cell boundaries
  - `is_interior`: Expanded cell interiors
  - `is_nuclear`: Unchanged (nuclei stay the same)
- **Expanded cell masks**: `*_cell_masks_expanded.npy` (updated cell segmentation)
- **Visualization**: `*_nuclear_overlay_expanded.png` (nuclei in blue, expanded cytoplasm in red)
- **Statistics**: Per-cell-type N:C ratio before/after expansion

**Algorithm:**
1. Load cell type-specific N:C ratio ranges from CSV
2. Calculate current N:C ratio for each cell
3. Identify cells that need expansion (below target ratio)
4. For each cell:
   - Calculate target expansion to reach desired N:C ratio
   - Apply iterative morphological dilation (circular kernel)
   - Stop when target ratio reached or max iterations exceeded
5. Update pixel-to-cell mapping with expanded boundaries
6. Generate visualization showing before/after comparison


**Implementation:**
- Script: `nuclear_expansion.py` - Cell boundary expansion using morphological dilation
- Config: `cell_based_nuclear_range.csv` - Cell type-specific N:C ratio ranges

**Example N:C Ratio Ranges:**
```csv
cell_type,min_nuclear_percentage,max_nuclear_percentage
Epithelial cells,0.20,0.30
Immune cells,0.40,0.60
Fibroblasts,0.15,0.25
Endothelial cells,0.25,0.35
```

### Step 2: Cell Type Annotation via CellTypist

**Objective:** Determine the cell type composition in the tissue to guide single-cell data selection.

- **Tool Used:** [CellTypist](https://www.celltypist.org/)
- **Input:** Visium HD 8μm binned data with Space Ranger clustering results
- **Model:** Mouse_Whole_Brain.pkl (for mouse brain tissue) or based on the tissue user use
- **Output:** Cell type annotations for each bin, showing the distribution of cell types in the tissue

**Implementation:**

- Script: `run_celltypist_annotation.py` - Annotates clusters and generates UMAP visualizations

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
2. **Gene localization analysis:** Classify genes by subcellular compartment using multi-source annotation
3. **Spatial assignment:** Assign single-cell profiles to segmented cells based on:
   - Cell type matching
   - Spatial location in tissue
   - One-to-one mapping (no cell reuse)
   - Subcellular compartment-specific gene expression (genes assigned to nucleus, cytoplasm, or membrane regions of the segmented cell based on their localization)

**Implementation:**

- Script: `gene_localization_GO_analysis.py` - Classify genes by subcellular localization using GENCODE, RNALocate, and UniProt annotations

#### Gene Localization Classification Logic

The script classifies genes into three mutually exclusive subcellular compartments using multiple annotation databases:

**Data Sources:**

1. **GENCODE/Ensembl GTF** - Gene biotype annotations (lncRNA, protein_coding, snRNA, etc.)
2. **RNALocate** - Experimental RNA subcellular localization database (~1.26M entries)
3. **UniProt** - Protein subcellular location annotations via REST API

**Classification Rules:**

**Nuclear Genes:**
- **Criteria:** Non-coding genes AND RNALocate nuclear localization (BOTH required)
  - **Non-coding biotypes:** lncRNA, antisense, processed_transcript, snRNA, snoRNA, scaRNA, misc_RNA, sense_intronic, sense_overlapping, bidirectional_promoter_lncRNA
  - **AND** RNALocate location contains: "nucleus", "nucleoplasm", "nuclear speckle", "nucleolus", or "chromatin"
- **Manual additions:** MALAT1, NEAT1, XIST, SNHG*, SNORD*, SNORA*, RNU* (well-known nuclear RNAs)
- **Expected:** ~1,000-1,500 genes (~4-6% of total genes)

**Membrane Genes:**
- **Criteria:** Protein-coding genes AND UniProt membrane keywords (BOTH required)
  - **Biotype:** protein_coding
  - **AND** UniProt annotation contains: "Secreted", "Signal peptide", "Transmembrane", "GPI-anchor", "Cell membrane", "Extracellular", "Endoplasmic reticulum", or "Golgi apparatus"
- **Expected:** ~7,000-9,000 genes (~30-40% of total genes)

**Cytoplasm Genes:**
- **Criteria:** All remaining genes (Total genes - Nuclear genes - Membrane genes)
- **Expected:** ~13,000-15,000 genes (~55-65% of total genes)
- **Note:** Most protein-coding genes that are not membrane-associated fall into this category


**Output Files:**

- `gene_localization_results/genes_nucleus.txt` - Nuclear-enriched genes
- `gene_localization_results/genes_cell_membrane.txt` - Membrane/secretory genes
- `gene_localization_results/genes_cytoplasm.txt` - Cytoplasmic genes


## Pseudo Visium HD Data Generation

**Objective:** Create synthetic Visium HD data from single-cell RNA-seq data with known ground truth for validation purposes.

- **Tool Used:** `create_pseudo_visium_hd.py`
- **Configuration:** `config_pseudo_hd.py`

**Purpose:**
- Generate pseudo-spatial transcriptomics data where the ground truth is completely known
- Assign single-cell RNA-seq profiles to specific spatial locations based on cell type matching
- Distribute genes to 2μm bins using compartment-specific spatial kernels
- Create data in 10X Visium HD format matching real data structure
- Provides a controlled dataset for validating bin-to-cell assignment tools

**Key Features:**
- **Memory-efficient sparse matrix processing**: Handles 100K+ cells without OOM
- **Subcellular gene distribution**: Uses biologically-informed spatial kernels:
  - **Nuclear genes**: Gaussian distribution (center-weighted)
  - **Cytoplasm genes**: Uniform distribution (equal probability)
  - **Membrane genes**: Reverse Gaussian (higher near cell boundary)
- **Overlapping bin handling**: Bins can belong to multiple cells (realistic!)
- **Bin aggregation optimization**: Pre-aggregates pixels to 2μm bins for 50-100x speedup
- **Chunked processing**: Processes matrix in chunks to avoid memory issues
- **10X format output**: Matches original Visium HD structure (MTX + barcodes + features)

**Workflow:**
1. Load gene subcellular localization (nucleus, cytoplasm, membrane) from GO analysis
2. Load single-cell RNA-seq data (H5 format) with cell type annotations
3. Load pixel-to-cell mapping with expanded cell boundaries
4. Assign single-cell profiles to segmented cells by cell type (random sampling with replacement)
5. Distribute genes to 2μm bins using compartment-specific spatial kernels
6. Aggregate expression using sparse matrix operations (memory efficient)
7. Output in 10X Visium HD format matching original spatial structure

**Input:**
- **Pixel-to-cell mapping**: `*_pixel_to_cell_mapping_expanded.csv.gz` (with cell types and compartments)
- **Gene localization**: `gene_localization_results/` (genes_nucleus.txt, genes_cytoplasm.txt, genes_cell_membrane.txt)
- **Single-cell data**:
  - Expression matrix: `KIRC_GSE159115_expression.h5` (custom H5 format)
  - Metadata: `KIRC_GSE159115_CellMetainfo_table.tsv` (with cell type column)
- **Original Visium HD structure**: `binned_outputs/square_002um/` (for barcode/feature matching)
- **Configuration**: `config_pseudo_hd.py` (paths, parameters, optional cropping)

**Output:**
- **10X format matrix**:
  - `filtered_feature_bc_matrix/matrix.mtx.gz` - Sparse expression matrix (genes × bins)
  - `filtered_feature_bc_matrix/barcodes.tsv.gz` - Bin barcodes (format: s_002um_XXXXX_YYYYY-1)
  - `filtered_feature_bc_matrix/features.tsv.gz` - Gene information (ENSG IDs + symbols)
- **Ground truth mapping**: `ground_truth_cell_assignments.csv` (cell_id → sc_cell mapping)
- **Spatial metadata**: Copied from original Visium HD data

**Implementation:**
- Script: `create_pseudo_visium_hd.py` - Memory-optimized pseudo data generation pipeline
- Config: `config_pseudo_hd.py` - Paths and parameters (supports spatial cropping for testing)

**Example Usage:**
```bash
# Full tissue processing
python create_pseudo_visium_hd.py

# Or edit config_pseudo_hd.py for cropped region (faster testing):
CROP_REGION = {
    'quantile': 0.2,  # Process first 20% of tissue
    'min_bin_x': 0,
    'min_bin_y': 0,
    'max_bin_x': None,  # Auto-determined from quantile
    'max_bin_y': None
}
```

### Validate Cell Assignment Function

**Objective:** Verify that the pseudo Visium HD data generation is working correctly by comparing bin expression to original single-cell profiles.

- **Tool Used:** `validate_cell_assignments.py`
- **Purpose:** Quality control script to validate that gene expression was correctly distributed to bins

**Key Features:**
- **Memory-efficient streaming**: Processes 33GB matrix files without OOM errors
- **Sparse matrix operations**: Keeps data in sparse format throughout
- **Subset loading**: Only loads bins needed for sampled cells (~1% of data)
- **Vectorized validation**: Pre-computes lookups for 100x speedup

**Validation Methods:**
1. **Bin-to-cell assignment validation**:
   - Samples cells from ground truth
   - Identifies bins belonging to those cells
   - Sums expression across all bins for each cell
   - Compares to original single-cell expression
   - Calculates Pearson correlation (expected: >0.7 for good quality)

2. **Overlapping bins validation**:
   - Identifies bins assigned to multiple cells
   - Verifies these bins have higher expression than single-cell bins
   - Expected: Overlapping bins have ~2x expression (additive)

**Usage:**
```bash
# Basic validation (100 cells, ~15-20 min total)
python validate_cell_assignments.py \
    --pseudo_hd_dir pseudo_visium_hd_outpu_full \
    --output_dir validation_results \
    --sample_size 100

# Large validation (1000 cells, ~20-25 min total)
python validate_cell_assignments.py \
    --pseudo_hd_dir pseudo_visium_hd_outpu_full \
    --output_dir validation_results \
    --sample_size 1000
```

**Parameters:**
- `--pseudo_hd_dir`: Directory containing pseudo Visium HD output (with ground_truth_cell_assignments.csv)
- `--output_dir`: Output directory for validation results (default: validation_results)
- `--sample_size`: Number of cells to sample for validation (default: 100)
  - Larger sample = more robust validation but longer runtime
  - Each cell typically has 100-500 bins

**Input Requirements:**
- Pseudo HD matrix: `pseudo_hd_dir/filtered_feature_bc_matrix/` (10X format)
- Ground truth: `pseudo_hd_dir/ground_truth_cell_assignments.csv`
- Pixel mapping: `cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_expanded.csv.gz`
- Single-cell data: `kidney_sc_data/KIRC_GSE159115_expression.h5` and metadata

**Output:**
- **Validation metrics**:
  - Mean/median correlation (bin expression vs SC expression)
  - % cells with correlation > 0.5, > 0.7
  - Per-cell-type correlation breakdown
  - Overlapping bin statistics
- **CSV file**: `validation_results/cell_validation_details.csv` with per-cell results:
  - `cell_id`, `cell_type`, `n_bins`
  - `correlation_all`, `correlation_nonzero`
  - `total_counts_bins`, `total_counts_sc`
- **Visualization**: `validation_results/validation_results.png`
  - Correlation distribution histogram
  - Correlation by cell type
  - Total counts comparison (SC vs bins)
  - Overlapping vs single-cell bins expression


## Ground Truth Data Structure

The pseudo Visium HD ground truth dataset consists of:

1. **Pseudo Visium HD binned data (2μm resolution):**
   - Gene expression matrix in 10X Visium HD format (MTX + features + barcodes)
   - Spatially-aware gene distribution based on subcellular localization:
     - **Nuclear genes**: Gaussian distribution (center-weighted, more genes near nucleus center)
     - **Cytoplasm genes**: Uniform distribution (equal probability across cytoplasm)
     - **Membrane genes**: Reverse Gaussian (0-2μm from boundary, higher concentration near cell edge)
   - Handles overlapping bins (bins can belong to multiple cells)
   - Matches original Visium HD spatial structure and barcode format

2. **Ground truth cell assignments:**
   - Mapping of segmented cell IDs to assigned single-cell profiles
   - Saved as `ground_truth_cell_assignments.csv` with columns:
     - `cell_id`: Segmented cell ID from Cellpose
     - `sc_cell_index`: Which single-cell profile was assigned to this cell
     - `cell_type`: Cell type annotation (matched between segmentation and scRNA-seq)
     - `sc_cell_barcode`: Original single-cell barcode from scRNA-seq data
   - Enables validation by comparing predicted assignments to known truth

3. **Expanded cell segmentation with compartments:**
   - Pixel-to-cell mapping with subcellular compartment flags
   - Cells expanded based on cell type-specific N:C ratios
   - Compartment assignments used for spatial gene distribution:
     - `is_nuclear`: Nuclear regions (kept unchanged from original segmentation)
     - `is_cytoplasm`: Cytoplasm regions (expanded outward from nuclei)
     - `is_boundary`: Cell membrane/boundary regions
     - `is_interior`: Non-boundary pixels inside cell
   - Aggregated to 2μm bin resolution for computational efficiency

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
# Create shared environment
conda create -n bin2cell python=3.12 -y
conda activate bin2cell

# Install the project dependencies
pip install -r requirements.txt
```

### 2. Install Cellpose

```bash
# Create Cellpose environment
conda create --prefix ./cellpose python=3.10 -y
conda activate ./cellpose

# Install Cellpose
pip install cellpose[gui]

# Install required dependencies for cellpose_segmentation.py
pip install matplotlib scikit-image pandas scipy Pillow
```

**Note:** The `cellpose_segmentation.py` script requires matplotlib, scipy, and other dependencies to run. Make sure these are installed in the cellpose environment before running the script.

### 3. Install STHD (cell type deconvolution)

STHD requires Python ≥ 3.8. Create a dedicated virtual environment so it does not
interfere with the rest of the toolchain.

```bash
# Create a Python 3.12 virtual environment inside this repo
python3.12 -m venv sthd_env
source sthd_env/bin/activate
python -m pip install --upgrade pip
```

You have two installation options:

**Option A – PyPI package**

```bash
pip install STHD
```

**Option B – Install from source**

```bash
git clone git@github.com:yi-zhang/STHD.git
pip install -r STHD/requirements.txt

# Make sure the STHD repo is on your Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/STHD"
# or inside scripts: sys.path.append('./STHD')
```

After installation, you can import any module via `from STHD import …`. Build or
update the lambda reference file once per dataset:

```bash
python build_sthd_lambda_file.py \
    --h5_file colorectal_sc_data/CRC_GSE166555_expression.h5 \
    --metadata_file colorectal_sc_data/CRC_GSE166555_CellMetainfo_table.tsv \
    --output lambda_colorectal_major_lineage.txt \
    --celltype_column "Celltype (major-lineage)"
```

Keep the generated `lambda_*.txt` files under version control so STHD runs are
reproducible, and always activate `sthd_env` before invoking `sthd_annotation.py`.

### 4. Install SMURF (nuclear segmentation + benchmarking)

It is recommended to run SMURF inside its own Conda environment—this mirrors the
cluster scripts and keeps TensorFlow/StarDist isolated from the rest of the repo.

```bash
conda create -n smurf python=3.10 -y
conda activate smurf
python -m pip install --upgrade pip
```

Choose the build that matches your hardware:

- **Lite version (CPU-only):**

  ```bash
  pip install pysmurf
  ```

- **Full version (GPU, recommended for production):**

  ```bash
  pip install "pysmurf[full]"
  ```

  The full install expects CUDA-enabled drivers and will pull the optional GPU
  dependencies required by the “full” SMURF pipeline.
