"""
Configuration file for pseudo Visium HD generation.
Edit the paths below to match your data.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path.cwd()

# Input paths
CELLPOSE_DIR = BASE_DIR / 'cellpose_sam_human_kidney_output'
PIXEL_MAPPING_CSV = CELLPOSE_DIR / 'cropped_visium_hd_human_kidney_pixel_to_cell_mapping_expanded.csv.gz'
GENE_LOCALIZATION_DIR = BASE_DIR / 'gene_localization_results_kidney'

# Single-cell data (H5 + TSV format)
SC_DATA_DIR = BASE_DIR / 'kidney_sc_data'
SC_H5_PATH = SC_DATA_DIR / 'KIRC_GSE159115_expression.h5'
SC_METADATA_PATH = SC_DATA_DIR / 'KIRC_GSE159115_CellMetainfo_table.tsv'

# Output paths
OUTPUT_DIR = BASE_DIR / 'pseudo_visium_hd_outpu_full'

# Parameters
MICRONS_PER_PIXEL = 0.2739038899725172  # Image resolution
MEMBRANE_CUTOFF = 1  # Membrane zone: 0 to this many 2µm bins FROM boundary (closer to boundary = more reads)
                     # NOTE: Was 10 pixels (2.7µm) → now 1 bin (2µm) due to bin-level optimization

# Spatial cropping (to process subset of tissue for testing)
# Set to None to process entire tissue, or specify region in 2µm bins
CROP_REGION = {
    'min_bin_x': 0,      # Start X coordinate (in 2µm bins)
    'max_bin_x': None,   # End X coordinate (None = use quantile below)
    'min_bin_y': 0,      # Start Y coordinate (in 2µm bins)
    'max_bin_y': None,   # End Y coordinate (None = use quantile below)
    'quantile': 1      # If max_bin_x/y is None, use this quantile (0.2 = first 20%)
}
# Set CROP_REGION = None to disable cropping and process entire tissue

# Cell type column name in single-cell data
SC_CELL_TYPE_COLUMN = 'cell_type'  # Change if your column has a different name

# Original Visium HD data (to match barcodes and structure)
# Set to the extracted binned_outputs/square_002um directory from the original data
# If None, will generate new synthetic data without matching original structure
ORIGINAL_HD_DIR = BASE_DIR / 'Human_kidney' / 'output' / 'binned_outputs' / 'square_002um'

# Random seed for reproducibility
RANDOM_SEED = 42
