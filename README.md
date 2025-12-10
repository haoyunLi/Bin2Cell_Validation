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
       --pixel_file cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_expanded.csv.gz \
       --sc_h5_file kidney_sc_data/KIRC_GSE159115_expression.h5 \
       --sc_meta_file kidney_sc_data/KIRC_GSE159115_CellMetainfo_table.tsv \
       --sample_size 100 \
       --microns_per_pixel 0.2739038899725172
   ```

8. **Run SMURF segmentation + deconvolution over pseudo HD data:**

   ```bash
   conda activate smurf
   python SMURF_analyze_spatial.py \
       --tissue_positions Human_kidney/output/binned_outputs/square_002um/spatial/tissue_positions.parquet \
       --tissue_image Human_kidney/input/Visium_HD_Human_Kidney_FFPE_tissue_image.tif \
       --pseudo_hd_dir pseudo_visium_hd_outpu_full \
       --output_dir smurf_result \
       --microns_per_pixel 0.2739038899725172
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
       --output_dir smurf_validation_output \
       --microns_per_pixel 0.2739038899725172
   conda deactivate
   ```

Steps 3 and 4b are optional, but the rest should be executed sequentially to keep
metadata synchronized for downstream validation and SMURF benchmarking.



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
