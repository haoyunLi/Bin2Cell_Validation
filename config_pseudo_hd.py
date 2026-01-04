"""
Configuration file for pseudo Visium HD generation.
Edit the paths below to match your data.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path.cwd()

# Input paths
CELLPOSE_DIR = BASE_DIR / 'cellpose_sam_human_colorectal_output'
PIXEL_MAPPING_CSV = CELLPOSE_DIR / 'cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_expanded.csv.gz'

# Single-cell data (H5 + TSV format)
SC_DATA_DIR = BASE_DIR / 'colorectal_sc_data'
SC_H5_PATH = SC_DATA_DIR / 'CRC_GSE166555_raw_tumor.h5'
SC_METADATA_PATH = SC_DATA_DIR / 'GSE166555_meta_data_tumor.tsv'

# Cell type column name in single-cell data(EX.Celltype (major-lineage))
SC_CELL_TYPE_COLUMN = 'sct_cell_type'  

# Original Visium HD data (to match barcodes and structure)
# Set to the extracted binned_outputs/square_002um directory from the original data
# If None, will generate new synthetic data without matching original structure
ORIGINAL_HD_DIR = 'Human_Colorectal/output/binned_outputs/square_002um'

# Random seed for reproducibility
RANDOM_SEED = 42

# Output paths
OUTPUT_DIR = BASE_DIR / 'colorectal_pseudo_visium_hd_output_full'

# Parameters
MICRONS_PER_PIXEL = 0.2737012522439323  # Image resolution

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



