#!/usr/bin/env python
"""
Create Pseudo Visium HD Data by Assigning Single-Cell Gene Expression to Segmented Cells

This script:
1. Loads cell segmentation masks with subcellular compartment assignments
2. Loads gene subcellular localization (nucleus, cytoplasm, membrane)
3. Loads single-cell RNA-seq data
4. Matches single-cell profiles to segmented cells by cell type
5. Distributes genes to 2μm bins using spatial kernels:
   - Nuclear: Gaussian distribution (center-weighted, more genes near center)
   - Cytoplasm: Uniform distribution (equal probability)
   - Membrane: Reverse Gaussian (5-10 pixels inward from boundary, more genes near boundary)
6. Handles overlapping bins (bins assigned to multiple cells via append, not overwrite)
7. Generates pseudo HD data in same format as original 2μm Visium HD data

Usage:
    python create_pseudo_visium_hd.py
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import logging
import sys
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix  # For fast sparse matrix aggregation
import gzip
import gc  # For garbage collection
from collections import defaultdict  # For faster dictionary aggregation

# Load configuration globally so helper functions can access it
import config_pseudo_hd as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_pseudo_visium_hd.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Old file-based gene localization removed - now using splicing model only


def load_single_cell_data(sc_h5_path, sc_metadata_path):
    """
    Load single-cell RNA-seq data from H5 matrix + TSV metadata.

    Args:
        sc_h5_path: Path to H5 gene expression matrix
        sc_metadata_path: Path to TSV metadata file with cell type info

    Returns:
        AnnData object with expression and cell type annotations
    """
    logger.info(f"Loading single-cell data...")
    logger.info(f"  Expression matrix: {sc_h5_path}")
    logger.info(f"  Metadata: {sc_metadata_path}")

    # Load H5 expression matrix (custom format with 'matrix' group)
    logger.info("  Reading H5 expression matrix...")
    import h5py
    from scipy.sparse import csr_matrix

    with h5py.File(sc_h5_path, 'r') as f:
        # Read matrix data
        matrix_group = f['matrix']
        data = matrix_group['data'][:]
        indices = matrix_group['indices'][:]
        indptr = matrix_group['indptr'][:]
        shape = matrix_group['shape'][:]

        # Read gene names
        gene_names = matrix_group['features']['name'][:]
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in gene_names]

        # Read barcodes
        barcodes = matrix_group['barcodes'][:]
        barcodes = [b.decode() if isinstance(b, bytes) else b for b in barcodes]

        # Matrix is stored with indptr matching cells, so shape is (cells, genes)
        # Create sparse matrix as cells x genes (swap shape dimensions)
        X = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

    # Create AnnData (already cells x genes)
    adata = sc.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=gene_names))

    logger.info(f"    Cells: {adata.n_obs}")
    logger.info(f"    Genes: {adata.n_vars}")

    # Load metadata TSV
    logger.info("  Reading metadata TSV...")
    metadata = pd.read_csv(sc_metadata_path, sep='\t')
    logger.info(f"    Metadata rows: {len(metadata)}")
    logger.info(f"    Metadata columns: {list(metadata.columns)}")

    # Merge metadata with adata.obs
    # Assuming first column in metadata is cell barcode
    cell_col = metadata.columns[0]
    metadata = metadata.set_index(cell_col)

    # Match cells between expression and metadata
    common_cells = adata.obs.index.intersection(metadata.index)
    logger.info(f"  Matched {len(common_cells)} cells between expression and metadata")

    if len(common_cells) == 0:
        logger.warning("  WARNING: No matching cells! Check barcode format")
        logger.info(f"    Example expression barcode: {adata.obs.index[0]}")
        logger.info(f"    Example metadata barcode: {metadata.index[0]}")

    # Subset to common cells
    adata = adata[common_cells, :].copy()

    # Add metadata to adata.obs
    for col in metadata.columns:
        adata.obs[col] = metadata.loc[adata.obs.index, col]

    # Use cell type column from config
    from config_pseudo_hd import SC_CELL_TYPE_COLUMN

    if SC_CELL_TYPE_COLUMN in adata.obs.columns:
        adata.obs['cell_type'] = adata.obs[SC_CELL_TYPE_COLUMN]
        logger.info(f"  Using '{SC_CELL_TYPE_COLUMN}' as cell_type column")
        logger.info(f"  Cell types found:")
        for ct, count in adata.obs['cell_type'].value_counts().items():
            logger.info(f"    {ct}: {count} cells")
    else:
        logger.error(f"  ERROR: Column '{SC_CELL_TYPE_COLUMN}' not found in metadata!")
        logger.info(f"  Available columns: {list(adata.obs.columns)}")
        raise ValueError(f"Column '{SC_CELL_TYPE_COLUMN}' not found in metadata. Please check the column name.")

    return adata


def load_pixel_to_cell_mapping(csv_path, aggregate_to_bins=True, microns_per_pixel=0.2739038899725172):
    """
    Load pixel-to-cell mapping with compartment and cell type annotations.
    Optionally aggregate pixels to 2µm bins IMMEDIATELY for 100x faster processing.

    Args:
        csv_path: Path to compressed CSV file
        aggregate_to_bins: If True, aggregate to 2µm bins immediately (MUCH faster!)
        microns_per_pixel: Resolution for bin conversion

    Returns:
        DataFrame with columns: bin_x, bin_y, cell_id, is_boundary, is_interior, is_nuclear, is_cytoplasm, cell_type
        (If aggregate_to_bins=False, returns original pixel-level data with x, y)
    """
    logger.info(f"Loading pixel-to-cell mapping from {csv_path}...")

    # Read in chunks for memory efficiency
    chunk_size = 1_000_000
    chunks = []

    with gzip.open(csv_path, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=chunk_size):
            chunks.append(chunk)
            logger.info(f"  Loaded chunk with {len(chunk):,} pixels...")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Total pixels loaded: {len(df):,}")
    logger.info(f"  Unique cells: {df['cell_id'].nunique()}")

    if aggregate_to_bins:
        logger.info("  OPTIMIZATION: Aggregating pixels to 2µm bins immediately...")
        logger.info("    This reduces data size by ~50x and speeds up processing by 100x!")
        logger.info("    Handling mixed compartments: bins with both nuclear+cytoplasm will be split into separate rows")

        # Convert pixels to bin coordinates
        df['bin_x'] = (df['x'] * microns_per_pixel / 2.0).astype(int)
        df['bin_y'] = (df['y'] * microns_per_pixel / 2.0).astype(int)

        # Count pixels by compartment for each (bin_x, bin_y, cell_id) combination
        pixel_counts = df.groupby(['bin_x', 'bin_y', 'cell_id']).agg({
            'is_nuclear': 'sum',      # Count nuclear pixels
            'is_cytoplasm': 'sum',    # Count cytoplasm pixels
            'is_boundary': 'sum',     # Count boundary pixels
            'is_interior': 'sum',     # Count interior pixels
            'cell_type': 'first'
        }).reset_index()

        # Split bins with mixed compartments into separate rows (VECTORIZED - no loop!)
        logger.info("  Splitting mixed compartments (vectorized)...")

        # Calculate total pixels per bin
        pixel_counts['total_pixels'] = (
            pixel_counts['is_nuclear'] +
            pixel_counts['is_cytoplasm'] +
            pixel_counts['is_boundary']
        )

        # Filter out empty bins
        pixel_counts = pixel_counts[pixel_counts['total_pixels'] > 0].copy()

        # Create nuclear compartment rows (where nuclear_count > 0)
        nuclear_bins = pixel_counts[pixel_counts['is_nuclear'] > 0].copy()
        nuclear_bins['weight'] = nuclear_bins['is_nuclear'] / nuclear_bins['total_pixels']
        nuclear_bins['is_nuclear'] = 1
        nuclear_bins['is_cytoplasm'] = 0
        nuclear_bins['is_boundary'] = 0
        nuclear_bins['is_interior'] = 1

        # Create cytoplasm compartment rows (where cytoplasm_count > 0)
        cytoplasm_bins = pixel_counts[pixel_counts['is_cytoplasm'] > 0].copy()
        cytoplasm_bins['weight'] = cytoplasm_bins['is_cytoplasm'] / cytoplasm_bins['total_pixels']
        cytoplasm_bins['is_nuclear'] = 0
        cytoplasm_bins['is_cytoplasm'] = 1
        cytoplasm_bins['is_boundary'] = 0
        cytoplasm_bins['is_interior'] = 1

        # Create boundary compartment rows (only if no nuclear or cytoplasm)
        boundary_bins = pixel_counts[
            (pixel_counts['is_boundary'] > 0) &
            (pixel_counts['is_nuclear'] == 0) &
            (pixel_counts['is_cytoplasm'] == 0)
        ].copy()
        boundary_bins['weight'] = 1.0
        boundary_bins['is_nuclear'] = 0
        boundary_bins['is_cytoplasm'] = 0
        boundary_bins['is_boundary'] = 1
        boundary_bins['is_interior'] = 0

        # Combine all compartments
        df_bins = pd.concat([nuclear_bins, cytoplasm_bins, boundary_bins], ignore_index=True)

        # Select only needed columns
        df_bins = df_bins[['bin_x', 'bin_y', 'cell_id', 'cell_type',
                           'is_nuclear', 'is_cytoplasm', 'is_boundary', 'is_interior', 'weight']]

        # Log statistics
        mixed_bins = pixel_counts[(pixel_counts['is_nuclear'] > 0) & (pixel_counts['is_cytoplasm'] > 0)]
        logger.info(f"  Aggregated {len(df):,} pixels → {len(df_bins):,} bin-compartment pairs (~{len(df)/len(df_bins):.1f}x reduction)")
        logger.info(f"  Bins with mixed compartments (nuclear+cytoplasm): {len(mixed_bins):,}")
        logger.info(f"  Memory saved: ~{(len(df) - len(df_bins)) * 50 / 1e6:.1f} MB")

        return df_bins
    else:
        return df


def compute_nuclear_kernel(df_cell, use_bins=True):
    """
    Compute Gaussian kernel for nuclear compartment.
    More genes near center, fewer at edges.

    Args:
        df_cell: DataFrame for single cell with nuclear bins/pixels
        use_bins: If True, df_cell has bin_x/bin_y columns (OPTIMIZED!)
                  If False, df_cell has x/y pixel columns (old behavior)

    Returns:
        Dictionary mapping (bin_x, bin_y) -> probability weight
    """
    nuclear_regions = df_cell[df_cell['is_nuclear'] == 1]

    if len(nuclear_regions) == 0:
        return {}

    # Get coordinates (bins or pixels)
    if use_bins:
        x_coords = nuclear_regions['bin_x'].values
        y_coords = nuclear_regions['bin_y'].values
    else:
        x_coords = nuclear_regions['x'].values
        y_coords = nuclear_regions['y'].values

    # Get bin weights if available (for mixed compartment bins)
    bin_weights = nuclear_regions['weight'].values if 'weight' in nuclear_regions.columns else np.ones(len(nuclear_regions))

    # Find centroid of nucleus (weighted by bin_weights)
    centroid_x = np.average(x_coords, weights=bin_weights)
    centroid_y = np.average(y_coords, weights=bin_weights)

    # Calculate distance from each bin/pixel to centroid
    distances = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)

    # Use Gaussian distribution: probability ~ exp(-distance^2 / (2 * sigma^2))
    # Sigma based on nuclear size (use std of distances)
    sigma = np.std(distances) if len(distances) > 1 else 1.0
    if sigma == 0:
        sigma = 1.0

    # Gaussian weights (center = high probability, edges = low probability)
    weights = np.exp(-distances**2 / (2 * sigma**2))

    # Multiply by bin weights (to account for mixed compartment bins)
    weights = weights * bin_weights

    # Normalize to probabilities
    weights = weights / np.sum(weights)

    # Build dictionary (optimized with dict constructor)
    coords = list(zip(x_coords.astype(int), y_coords.astype(int)))
    kernel = dict(zip(coords, weights))

    return kernel


def compute_cytoplasm_kernel(df_cell, use_bins=True):
    """
    Compute uniform kernel for cytoplasm compartment.
    Equal probability for all cytoplasm bins/pixels (weighted by bin fraction if mixed compartments).

    Args:
        df_cell: DataFrame for single cell with cytoplasm bins/pixels
        use_bins: If True, df_cell has bin_x/bin_y columns (OPTIMIZED!)
                  If False, df_cell has x/y pixel columns (old behavior)

    Returns:
        Dictionary mapping (bin_x, bin_y) -> probability weight
    """
    cytoplasm_regions = df_cell[df_cell['is_cytoplasm'] == 1]

    if len(cytoplasm_regions) == 0:
        return {}

    if use_bins:
        x_vals = cytoplasm_regions['bin_x'].values.astype(int)
        y_vals = cytoplasm_regions['bin_y'].values.astype(int)
    else:
        x_vals = cytoplasm_regions['x'].values.astype(int)
        y_vals = cytoplasm_regions['y'].values.astype(int)

    # Get bin weights if available (for mixed compartment bins)
    bin_weights = cytoplasm_regions['weight'].values if 'weight' in cytoplasm_regions.columns else np.ones(len(cytoplasm_regions))

    # Uniform distribution weighted by bin fractions
    weights = bin_weights / np.sum(bin_weights)

    coords = list(zip(x_vals, y_vals))
    kernel = dict(zip(coords, weights))

    return kernel


def compute_membrane_kernel(df_cell, membrane_cutoff=1, use_bins=True):
    """
    Compute reverse Gaussian kernel for membrane compartment.
    Membrane zone = 0 to cutoff bins/pixels FROM boundary.
    Closer to boundary = MORE reads (reverse Gaussian).

    Args:
        df_cell: DataFrame for single cell
        membrane_cutoff: Maximum distance from boundary in BINS (if use_bins=True) or PIXELS (if use_bins=False).
                         Default=1 bin (2µm), which is equivalent to ~10 pixels at 0.27µm/pixel resolution.
        use_bins: If True, df_cell has bin_x/bin_y columns (OPTIMIZED!)
                  If False, df_cell has x/y pixel columns (old behavior)

    Returns:
        Dictionary mapping (bin_x, bin_y) -> probability weight
    """
    # NOTE: membrane_cutoff is now in BINS (not pixels) when use_bins=True
    # No conversion needed - caller should pass cutoff in appropriate units

    # Get cell dimensions
    if use_bins:
        max_y = df_cell['bin_y'].max()
        max_x = df_cell['bin_x'].max()
        min_y = df_cell['bin_y'].min()
        min_x = df_cell['bin_x'].min()
        y_coords = df_cell['bin_y'].values
        x_coords = df_cell['bin_x'].values
    else:
        max_y = df_cell['y'].max()
        max_x = df_cell['x'].max()
        min_y = df_cell['y'].min()
        min_x = df_cell['x'].min()
        y_coords = df_cell['y'].values
        x_coords = df_cell['x'].values

    height = max_y - min_y + 1
    width = max_x - min_x + 1

    # Create binary mask for this cell
    cell_mask = np.zeros((height, width), dtype=bool)
    y_rel = (y_coords - min_y).astype(int)
    x_rel = (x_coords - min_x).astype(int)
    cell_mask[y_rel, x_rel] = True

    # Compute distance transform (distance to nearest boundary FROM INSIDE)
    # For each interior bin/pixel, this gives distance to nearest boundary
    distance_map = distance_transform_edt(cell_mask)

    # Select bins/pixels within the membrane zone (0 to membrane_cutoff from boundary)
    membrane_mask = (distance_map > 0) & (distance_map <= membrane_cutoff)

    if not np.any(membrane_mask):
        # No bins/pixels in membrane zone, fallback to boundary bins/pixels
        boundary_regions = df_cell[df_cell['is_boundary'] == 1]
        if len(boundary_regions) == 0:
            return {}

        uniform_prob = 1.0 / len(boundary_regions)
        if use_bins:
            x_vals = boundary_regions['bin_x'].values.astype(int)
            y_vals = boundary_regions['bin_y'].values.astype(int)
        else:
            x_vals = boundary_regions['x'].values.astype(int)
            y_vals = boundary_regions['y'].values.astype(int)
        coords = list(zip(x_vals, y_vals))
        kernel = {coord: uniform_prob for coord in coords}
        return kernel

    # Get membrane bin/pixel coordinates and distances
    membrane_y, membrane_x = np.where(membrane_mask)
    membrane_distances = distance_map[membrane_mask]

    # Reverse Gaussian: closer to boundary (SMALLER distance) = HIGHER weight
    # Distance = 0 (at boundary) → highest weight
    # Distance = cutoff (far from boundary) → lowest weight
    # Weight = exp(-distance^2 / (2 * sigma^2))

    sigma = membrane_cutoff / 3.0  # Sigma based on cutoff range
    if sigma == 0:
        sigma = 1.0

    # Direct Gaussian on distance (smaller distance = higher weight)
    weights = np.exp(-membrane_distances**2 / (2 * sigma**2))
    weights = weights / np.sum(weights)

    # Build kernel (optimized)
    x_abs = (membrane_x + min_x).astype(int)
    y_abs = (membrane_y + min_y).astype(int)
    coords = list(zip(x_abs, y_abs))
    kernel = dict(zip(coords, weights))

    return kernel


def assign_sc_to_cells(df_pixels, adata_sc):
    """
    Assign single-cell profiles to segmented cells based on cell type matching.
    Random sampling with replacement (since we may not have enough cells).

    Args:
        df_pixels: Pixel-to-cell mapping with cell_type column
        adata_sc: Single-cell AnnData object with cell_type in obs

    Returns:
        Dictionary mapping cell_id -> sc_cell_index
    """
    logger.info("Assigning single-cell profiles to segmented cells...")

    # Get unique cells and their cell types
    cell_info = df_pixels[df_pixels['cell_id'] > 0].groupby('cell_id')['cell_type'].first().to_dict()
    logger.info(f"  Total segmented cells: {len(cell_info)}")

    # Group segmented cells by cell type
    cells_by_type = {}
    for cell_id, cell_type in cell_info.items():
        if pd.notna(cell_type):
            if cell_type not in cells_by_type:
                cells_by_type[cell_type] = []
            cells_by_type[cell_type].append(cell_id)

    logger.info(f"  Cell types in segmented data:")
    for ct, cells in cells_by_type.items():
        logger.info(f"    {ct}: {len(cells)} cells")

    # Create mapping
    cell_to_sc = {}

    for cell_type, cell_ids in cells_by_type.items():
        # Find matching sc cells
        sc_cells_of_type = adata_sc.obs[adata_sc.obs['cell_type'] == cell_type].index.tolist()

        if len(sc_cells_of_type) == 0:
            logger.warning(f"  No sc cells found for cell type: {cell_type}")
            continue

        logger.info(f"  Assigning {len(cell_ids)} segmented {cell_type} cells to {len(sc_cells_of_type)} sc cells")

        # Random sampling with replacement
        np.random.seed(42)
        assigned_sc_indices = np.random.choice(sc_cells_of_type, size=len(cell_ids), replace=True)

        for cell_id, sc_idx in zip(cell_ids, assigned_sc_indices):
            cell_to_sc[cell_id] = sc_idx

    logger.info(f"  Total assignments: {len(cell_to_sc)}")
    return cell_to_sc


def distribute_genes_to_bins(df_pixels, adata_sc, cell_to_sc_mapping,
                              dropout_probs, membrane_genes,
                              microns_per_pixel=0.2739038899725172):
    """
    Distribute genes from single-cell data to 2μm bins using splicing-based per-cell localization.

    Args:
        df_pixels: Pixel-to-cell mapping (or bin-to-cell mapping if aggregated)
        adata_sc: Single-cell AnnData
        cell_to_sc_mapping: Dict mapping cell_id -> sc_cell_index
        dropout_probs: Dict containing p0 and global_alpha from splicing model
        membrane_genes: Set of membrane genes from UniProt
        microns_per_pixel: Resolution

    Returns:
        DataFrame with columns: bin_x, bin_y, gene, count
    """
    logger.info("Distributing genes to 2μm bins using PER-CELL splicing model...")

    # Get list of genes in sc data
    sc_genes = np.array(adata_sc.var_names.tolist())
    logger.info(f"  Single-cell data has {len(sc_genes)} genes")

    logger.info(f"  Baseline probs (p0) available for {len(dropout_probs['p0'])} genes")
    logger.info(f"  Global alpha: {dropout_probs['global_alpha']:.6f}")
    logger.info(f"  Membrane genes: {len(membrane_genes)} genes")

    # Import the per-cell assignment function and parameters
    from gene_localization_GO_analysis import (
        assign_nuclear_genes_per_cell,
        SIGMA_CELL_FACTOR,
        C_CAPTURE_EFFICIENCY,
        CLIP_EPS
    )

    # Extract p0 and global_alpha
    p0 = dropout_probs['p0']
    global_alpha = dropout_probs['global_alpha']

    # OPTIMIZATION: Pre-compute p0 array aligned with sc_genes (10x faster than dict lookups!)
    logger.info(f"  Pre-computing p0 array for {len(sc_genes)} genes...")
    p0_array = np.array([p0.get(g, 0.0) for g in sc_genes])
    logger.info(f"    p0 array: {(p0_array > 0).sum()} genes with non-zero baseline probability")

    # Pre-compute membrane gene mask (for fast vectorized operations)
    membrane_gene_mask = np.array([g in membrane_genes for g in sc_genes])
    logger.info(f"    Membrane mask: {membrane_gene_mask.sum()} membrane genes")

    # Check if data is already in bin format (optimized path) or pixel format (old path)
    use_bins = 'bin_x' in df_pixels.columns and 'bin_y' in df_pixels.columns

    if use_bins:
        logger.info("  Data is already in BIN format (OPTIMIZED!) - no conversion needed")
    else:
        logger.info("  Data is in PIXEL format - converting to bins...")
        # Add bin coordinates to pixel dataframe
        df_pixels['bin_x'] = (df_pixels['x'] * microns_per_pixel / 2.0).astype(int)
        df_pixels['bin_y'] = (df_pixels['y'] * microns_per_pixel / 2.0).astype(int)

    # Pre-group bins/pixels by cell_id (MUCH faster than filtering 127K times!)
    logger.info(f"  Grouping {'bins' if use_bins else 'pixels'} by cell_id (this saves hours of filtering)...")
    df_cells_only = df_pixels[df_pixels['cell_id'] > 0]
    cells_grouped = {cell_id: group for cell_id, group in df_cells_only.groupby('cell_id')}

    unique_cells = list(cells_grouped.keys())
    logger.info(f"  Processing {len(unique_cells)} cells...")

    # Draw cell-specific factors b_i ~ LogNormal(0, σ²) for ALL cells at once
    # With mean=0, sigma=0.2: E[b_i] = exp(0 + σ²/2) = exp(0.02) ≈ 1.0202
    np.random.seed(config.RANDOM_SEED)
    num_cells = len(unique_cells)
    mean_log = 0.0  # Mean of underlying normal distribution
    cell_factors = np.random.lognormal(mean=mean_log, sigma=SIGMA_CELL_FACTOR, size=num_cells)
    cell_id_to_factor = {cell_id: cell_factors[i] for i, cell_id in enumerate(unique_cells)}
    logger.info(f"    Cell factors: mean={cell_factors.mean():.4f}, std={cell_factors.std():.4f}, range=[{cell_factors.min():.4f}, {cell_factors.max():.4f}]")

    # Extract expression matrix once (for sharing across processes)
    sc_gene_expr_matrix = adata_sc.X.toarray()

    # Create barcode-to-index mapping (for fast lookup)
    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(adata_sc.obs.index)}

    # OPTIMIZED APPROACH: Use SPARSE MATRIX for aggregation (10-50x faster!)
    logger.info(f"  Processing {len(unique_cells)} cells with SPARSE MATRIX aggregation...")
    logger.info(f"  This uses scipy.sparse which is optimized for exactly this use case!")
    logger.info(f"  Progress will be logged every 1000 cells")

    # Create gene name to index mapping (for fast lookup)
    logger.info(f"  Creating gene index mapping...")
    gene_to_idx = {gene: idx for idx, gene in enumerate(sc_genes)}

    # Pre-create gene index array for vectorized lookups
    # This allows numpy-based indexing instead of dict lookups in the loop
    gene_idx_array = np.arange(len(sc_genes), dtype=np.int32)

    # Calculate bin dimensions
    logger.info(f"  Calculating spatial dimensions...")
    max_bin_x = int(df_pixels['bin_x'].max()) + 1
    max_bin_y = int(df_pixels['bin_y'].max()) + 1
    num_bins = max_bin_x * max_bin_y
    logger.info(f"    Bin dimensions: {max_bin_x} x {max_bin_y} = {num_bins:,} bins")
    logger.info(f"    Genes: {len(sc_genes):,}")
    logger.info(f"    Matrix shape: {num_bins:,} bins x {len(sc_genes):,} genes")

    # Pre-allocate numpy arrays instead of Python lists (MUCH faster!)
    # Python lists have reallocation overhead - numpy arrays are pre-allocated
    # Estimate: 127K cells × 60K entries/cell = 7.62B entries (with 20% safety margin to avoid expensive reallocation)
    estimated_entries = len(unique_cells) * 60000
    logger.info(f"  Pre-allocating numpy arrays for ~{estimated_entries:,} entries (~91 GB)...")

    # Store bin_x and bin_y separately to avoid 1D conversion issues
    bin_x_indices = np.empty(estimated_entries, dtype=np.int32)
    bin_y_indices = np.empty(estimated_entries, dtype=np.int32)
    gene_indices = np.empty(estimated_entries, dtype=np.int32)
    counts = np.empty(estimated_entries, dtype=np.float32)

    current_idx = 0  # Track current position in arrays

    # Track min/max coordinates to compute proper sparse matrix dimensions
    min_bin_x = float('inf')
    max_bin_x = float('-inf')
    min_bin_y = float('inf')
    max_bin_y = float('-inf')

    for idx, cell_id in enumerate(unique_cells):
        if cell_id not in cell_to_sc_mapping:
            continue

        # Get pixels for this cell (from pre-grouped dict - instant!)
        df_cell = cells_grouped[cell_id]

        # Get SC expression for this cell
        sc_barcode = cell_to_sc_mapping[cell_id]
        sc_idx = barcode_to_idx[sc_barcode]
        sc_gene_expr = sc_gene_expr_matrix[sc_idx, :]

        # Compute spatial kernels (pass use_bins flag)
        nuclear_kernel = compute_nuclear_kernel(df_cell, use_bins=use_bins)
        cytoplasm_kernel = compute_cytoplasm_kernel(df_cell, use_bins=use_bins)
        membrane_kernel = compute_membrane_kernel(df_cell, use_bins=use_bins)

        # Determine gene compartments for THIS cell using splicing model
        # Get cell-specific factor for this cell
        cell_factor = cell_id_to_factor.get(cell_id, 1.0)

        # PER-CELL assignment using VECTORIZED splicing model (10-50x faster!)
        nuclear_genes_cell = assign_nuclear_genes_per_cell(
            sc_gene_expr,
            sc_genes,
            p0,
            cell_factor,
            global_alpha,
            C=C_CAPTURE_EFFICIENCY,
            eps=CLIP_EPS,
            p0_array=p0_array  # OPTIMIZATION: Use pre-computed array
        )

        # VECTORIZED membrane/cytosol assignment (10x faster than sets!)
        # Create boolean masks directly without set operations
        expressed_mask = sc_gene_expr > 0
        nuclear_mask_cell = np.array([g in nuclear_genes_cell for g in sc_genes])

        # Membrane: expressed AND in membrane_genes AND NOT nuclear
        membrane_mask_cell = expressed_mask & membrane_gene_mask & ~nuclear_mask_cell

        # Cytosol: expressed AND NOT nuclear AND NOT membrane
        cytoplasm_mask_cell = expressed_mask & ~nuclear_mask_cell & ~membrane_mask_cell

        # Process all genes in each compartment
        for gene_mask, kernel in [
            (nuclear_mask_cell, nuclear_kernel),
            (cytoplasm_mask_cell, cytoplasm_kernel),
            (membrane_mask_cell, membrane_kernel)
        ]:
            if len(kernel) == 0 or gene_mask.sum() == 0:
                continue

            compartment_genes = sc_genes[gene_mask]
            compartment_counts = sc_gene_expr[gene_mask]

            nonzero_mask = compartment_counts > 0
            if nonzero_mask.sum() == 0:
                continue

            compartment_genes = compartment_genes[nonzero_mask]
            compartment_counts = compartment_counts[nonzero_mask]

            # Convert kernel dict to arrays (fully vectorized)
            # Use items() to get both at once (slightly faster)
            kernel_items = list(kernel.items())
            coords = np.array([k for k, v in kernel_items], dtype=np.int32)
            probs = np.array([v for k, v in kernel_items], dtype=np.float32)

            if use_bins:
                # Kernel already outputs bin coordinates!
                bin_coords_x = coords[:, 0]
                bin_coords_y = coords[:, 1]
            else:
                # Convert pixel coordinates to bin coordinates
                bin_coords_x = (coords[:, 0] * microns_per_pixel / 2.0).astype(int)
                bin_coords_y = (coords[:, 1] * microns_per_pixel / 2.0).astype(int)

            # VECTORIZED: Compute gene-bin count matrix
            gene_counts_matrix = compartment_counts[:, np.newaxis] * probs[np.newaxis, :]
            nonzero_rows, nonzero_cols = np.where(gene_counts_matrix > 0)

            if len(nonzero_rows) == 0:
                continue

            # OPTIMIZED VECTORIZED DICTIONARY AGGREGATION
            # Extract all bin coordinates, genes, and counts at once (vectorized!)
            bin_x_vals = bin_coords_x[nonzero_cols]
            bin_y_vals = bin_coords_y[nonzero_cols]
            gene_vals = compartment_genes[nonzero_rows]
            count_vals = gene_counts_matrix[nonzero_rows, nonzero_cols]

            # FULLY VECTORIZED: Store bin_x and bin_y separately (no 1D conversion!)
            # This avoids issues with cropped data where bins don't start at 0
            gene_idx_vals = gene_idx_array[gene_mask][nonzero_mask][nonzero_rows]

            # Write directly to pre-allocated numpy arrays (no reallocation!)
            n_entries = len(bin_x_vals)

            # Check if we need to expand arrays (handles case when estimate is too low)
            if current_idx + n_entries > len(bin_x_indices):
                expansion_size = max(n_entries, len(bin_x_indices) // 10)  # Expand by at least 10% or n_entries
                logger.info(f"  Expanding arrays: current size {len(bin_x_indices):,}, need {current_idx + n_entries:,}, adding {expansion_size:,}")
                bin_x_indices = np.concatenate([bin_x_indices, np.empty(expansion_size, dtype=np.int32)])
                bin_y_indices = np.concatenate([bin_y_indices, np.empty(expansion_size, dtype=np.int32)])
                gene_indices = np.concatenate([gene_indices, np.empty(expansion_size, dtype=np.int32)])
                counts = np.concatenate([counts, np.empty(expansion_size, dtype=np.float32)])

            bin_x_indices[current_idx:current_idx+n_entries] = bin_x_vals.astype(np.int32)
            bin_y_indices[current_idx:current_idx+n_entries] = bin_y_vals.astype(np.int32)
            gene_indices[current_idx:current_idx+n_entries] = gene_idx_vals
            counts[current_idx:current_idx+n_entries] = count_vals.astype(np.float32)
            current_idx += n_entries

            # Track coordinate bounds
            min_bin_x = min(min_bin_x, bin_x_vals.min())
            max_bin_x = max(max_bin_x, bin_x_vals.max())
            min_bin_y = min(min_bin_y, bin_y_vals.min())
            max_bin_y = max(max_bin_y, bin_y_vals.max())

        # Progress logging every 1000 cells
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(unique_cells)} cells processed ({100*(idx+1)/len(unique_cells):.1f}%), "
                       f"{current_idx:,} entries accumulated")

        # Gentle garbage collection every 10000 cells
        if (idx + 1) % 10000 == 0:
            gc.collect()

    # All cells processed! Now create sparse matrix and aggregate
    logger.info(f"  Completed processing all {len(unique_cells)} cells!")
    logger.info(f"  Total entries accumulated: {current_idx:,}")
    logger.info(f"  Creating sparse matrix and aggregating duplicates...")

    # Log coordinate bounds (for debugging)
    logger.info(f"  Pseudo data coordinate bounds:")
    logger.info(f"    bin_x: [{min_bin_x}, {max_bin_x}]")
    logger.info(f"    bin_y: [{min_bin_y}, {max_bin_y}]")

    # Trim arrays to actual size (we may have over-estimated)
    logger.info(f"  Step 1/3: Trimming arrays to actual size...")
    bin_x_arr = bin_x_indices[:current_idx]
    bin_y_arr = bin_y_indices[:current_idx]
    gene_indices_arr = gene_indices[:current_idx]
    counts_arr = counts[:current_idx]

    # Clear original arrays to free memory
    del bin_x_indices, bin_y_indices, gene_indices, counts
    gc.collect()

    # SPARSE MATRIX APPROACH (FAST!)
    # Convert bin coordinates to 0-based indices by subtracting minimum
    # This handles cropped data where coordinates don't start at 0
    logger.info(f"  Step 2/3: Creating sparse matrix with offset coordinates...")
    bin_x_offset = bin_x_arr - min_bin_x  # Now starts at 0
    bin_y_offset = bin_y_arr - min_bin_y  # Now starts at 0

    # Dimensions of the sparse matrix
    n_bins_x = int(max_bin_x - min_bin_x + 1)
    n_bins_y = int(max_bin_y - min_bin_y + 1)
    n_genes = len(sc_genes)

    logger.info(f"    Spatial dimensions: {n_bins_x} × {n_bins_y} = {n_bins_x * n_bins_y:,} bins")
    logger.info(f"    Gene dimension: {n_genes:,} genes")

    # Convert 2D bin coords to 1D indices (using OFFSET coordinates!)
    bin_1d_indices = bin_x_offset.astype(np.int64) * n_bins_y + bin_y_offset.astype(np.int64)

    # Create 2D sparse matrix: rows = bins (1D indexed), cols = genes
    # This will automatically aggregate duplicates!
    logger.info(f"    Building COO sparse matrix from {len(counts_arr):,} entries...")
    num_entries_with_duplicates = len(counts_arr)

    from scipy.sparse import coo_matrix
    sparse_matrix = coo_matrix(
        (counts_arr, (bin_1d_indices, gene_indices_arr)),
        shape=(n_bins_x * n_bins_y, n_genes),
        dtype=np.float32
    )

    # Free intermediate arrays
    del bin_x_arr, bin_y_arr, bin_x_offset, bin_y_offset, bin_1d_indices, gene_indices_arr, counts_arr
    gc.collect()

    # Convert to CSR and aggregate duplicates (SUPER FAST!)
    logger.info(f"  Step 3/3: Converting to CSR format and aggregating duplicates...")
    sparse_matrix = sparse_matrix.tocsr()
    sparse_matrix.sum_duplicates()  # This aggregates all duplicate (bin, gene) pairs!

    # Convert back to COO to extract aggregated data
    coo = sparse_matrix.tocoo()
    logger.info(f"    After aggregation: {coo.nnz:,} unique bin-gene pairs")
    logger.info(f"    Reduction: {num_entries_with_duplicates/coo.nnz:.1f}x")

    # Return the sparse matrix COO and metadata (DON'T create DataFrame to save memory!)
    logger.info(f"  SUCCESS! Returning sparse representation to avoid OOM")

    return {
        'coo': coo,
        'sc_genes': sc_genes,
        'min_bin_x': min_bin_x,
        'min_bin_y': min_bin_y,
        'n_bins_y': n_bins_y
    }


def create_visium_hd_format(sparse_data, output_dir, original_hd_dir=None):
    """
    Convert bin-level gene counts to Visium HD format matching original 10X structure.

    Args:
        sparse_data: Dict with 'coo' (sparse matrix), 'sc_genes', coordinate offsets
        output_dir: Output directory for pseudo HD data
        original_hd_dir: Path to original Visium HD binned_outputs/square_002um/ directory
                        If provided, will use original barcodes and spatial structure
    """
    import scipy.io as sio
    import gzip

    logger.info("Converting to Visium HD format...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if original_hd_dir is not None:
        # Load original Visium HD structure and replace expression values
        logger.info(f"  Loading original Visium HD structure from {original_hd_dir}...")
        original_hd_dir = Path(original_hd_dir)

        # Load original barcodes
        barcodes_file = original_hd_dir / 'filtered_feature_bc_matrix' / 'barcodes.tsv.gz'
        with gzip.open(barcodes_file, 'rt') as f:
            original_barcodes = [line.strip() for line in f]
        logger.info(f"  Loaded {len(original_barcodes):,} original barcodes")

        # Load original features (genes)
        features_file = original_hd_dir / 'filtered_feature_bc_matrix' / 'features.tsv.gz'
        with gzip.open(features_file, 'rt') as f:
            original_features = [line.strip().split('\t') for line in f]
        gene_ids = [f[0] for f in original_features]  # ENSG IDs
        gene_names = [f[1] for f in original_features]  # Gene symbols
        logger.info(f"  Loaded {len(gene_names):,} original genes")

        # Create barcode to index mapping
        barcode_to_idx = {bc: i for i, bc in enumerate(original_barcodes)}

        # Create gene name to index mapping (our data uses gene symbols)
        gene_to_idx = {name: i for i, name in enumerate(gene_names)}

        # Parse barcodes to extract bin_x, bin_y (OPTIMIZED - vectorized)
        # Format: s_002um_00658_01498-1 -> bin_x=658, bin_y=1498
        logger.info("  Parsing barcode coordinates...")
        barcode_coords = [
            (int(bc.split('_')[2]), int(bc.split('_')[3].split('-')[0]))
            for bc in original_barcodes
        ]

        # Create mapping from (bin_x, bin_y) to barcode index
        coords_to_barcode_idx = {coord: i for i, coord in enumerate(barcode_coords)}

        # Log original data bounds
        all_x = [c[0] for c in barcode_coords]
        all_y = [c[1] for c in barcode_coords]
        logger.info(f"  Original Visium HD data bounds:")
        logger.info(f"    bin_x range: [{min(all_x)}, {max(all_x)}]")
        logger.info(f"    bin_y range: [{min(all_y)}, {max(all_y)}]")
        logger.info(f"    Total barcodes: {len(barcode_coords):,}")

        # Extract sparse data components
        coo = sparse_data['coo']
        sc_genes = sparse_data['sc_genes']
        min_bin_x = sparse_data['min_bin_x']
        min_bin_y = sparse_data['min_bin_y']
        n_bins_y = sparse_data['n_bins_y']

        logger.info("  Building expression matrix from pseudo data (chunk-based to avoid OOM)...")
        logger.info(f"    Pseudo data: {coo.nnz:,} bin-gene pairs")

        # Process in chunks with VECTORIZED filtering and INCREMENTAL matrix building
        chunk_size = 100_000_000
        num_chunks = (coo.nnz + chunk_size - 1) // chunk_size
        logger.info(f"    Processing {num_chunks} chunks (vectorized + incremental, no memory accumulation)...")

        # Build barcode lookup once
        logger.info("  Creating barcode lookup array...")
        max_x_orig = max(coord[0] for coord in coords_to_barcode_idx.keys()) + 1
        max_y_orig = max(coord[1] for coord in coords_to_barcode_idx.keys()) + 1
        barcode_lookup = np.full((max_x_orig, max_y_orig), -1, dtype=np.int32)
        for (x, y), idx in coords_to_barcode_idx.items():
            barcode_lookup[x, y] = idx

        # Create gene name to index lookup
        valid_genes_set = set(gene_to_idx.keys())
        gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}

        # Initialize sparse matrix (will build incrementally)
        from scipy.sparse import coo_matrix
        logger.info("  Building sparse matrix incrementally (100x faster, OOM-proof)...")
        matrix = None
        total_valid_entries = 0

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, coo.nnz)

            logger.info(f"    Chunk {chunk_idx+1}/{num_chunks}: entries {start_idx:,} to {end_idx:,}...")

            # Extract chunk from COO
            chunk_rows = coo.row[start_idx:end_idx]
            chunk_cols = coo.col[start_idx:end_idx]
            chunk_data = coo.data[start_idx:end_idx]

            # Convert to bin coordinates (add back offset)
            bin_x_chunk = (chunk_rows // n_bins_y).astype(np.int32) + int(min_bin_x)
            bin_y_chunk = (chunk_rows % n_bins_y).astype(np.int32) + int(min_bin_y)
            gene_names_chunk = sc_genes[chunk_cols]

            # VECTORIZED gene filtering (100x faster than Python loop!)
            valid_genes = np.isin(gene_names_chunk, list(valid_genes_set))

            # VECTORIZED coordinate filtering
            valid_coords = (
                (bin_x_chunk >= 0) & (bin_x_chunk < max_x_orig) &
                (bin_y_chunk >= 0) & (bin_y_chunk < max_y_orig)
            )

            # Combine filters
            potential_valid = valid_genes & valid_coords

            # Get barcode indices for potentially valid entries (only for those passing first filters)
            potential_indices = np.where(potential_valid)[0]
            if len(potential_indices) > 0:
                barcode_indices = barcode_lookup[
                    bin_x_chunk[potential_indices],
                    bin_y_chunk[potential_indices]
                ]

                # Final filter: barcode must be valid (>= 0)
                valid_barcodes = barcode_indices >= 0
                final_valid_indices = potential_indices[valid_barcodes]

                num_valid = len(final_valid_indices)
                total_valid_entries += num_valid
                logger.info(f"      Retained {num_valid:,} valid entries")

                if num_valid > 0:
                    # Get gene indices (vectorized)
                    valid_gene_names = gene_names_chunk[final_valid_indices]
                    gene_indices = np.array([gene_name_to_idx[g] for g in valid_gene_names], dtype=np.int32)

                    # Get valid barcode indices and counts
                    valid_barcode_indices = barcode_indices[valid_barcodes]
                    valid_counts = chunk_data[final_valid_indices]

                    # Create chunk sparse matrix
                    chunk_matrix = coo_matrix(
                        (valid_counts, (gene_indices, valid_barcode_indices)),
                        shape=(len(gene_names), len(original_barcodes))
                    )

                    # Add to accumulated matrix (incremental!)
                    if matrix is None:
                        matrix = chunk_matrix
                    else:
                        matrix = matrix + chunk_matrix

                    del gene_indices, valid_barcode_indices, valid_counts, chunk_matrix
            else:
                logger.info(f"      Retained 0 valid entries")

            # Clean up chunk data
            del chunk_rows, chunk_cols, chunk_data, bin_x_chunk, bin_y_chunk, gene_names_chunk
            del valid_genes, valid_coords, potential_valid
            gc.collect()

        logger.info(f"    Total valid entries: {total_valid_entries:,}")

        # Clean up lookup
        del barcode_lookup
        gc.collect()

        # If no valid entries, create empty matrix
        if matrix is None:
            logger.warning("  No valid entries found! Creating empty matrix...")
            matrix = coo_matrix(([], ([], [])), shape=(len(gene_names), len(original_barcodes)))

        logger.info(f"  Created matrix: {matrix.shape[0]} genes x {matrix.shape[1]} barcodes")
        logger.info(f"  Non-zero entries: {matrix.nnz:,}")

        # Save in 10X format
        matrix_dir = output_dir / 'filtered_feature_bc_matrix'
        matrix_dir.mkdir(exist_ok=True, parents=True)

        # Save matrix.mtx.gz
        logger.info("  Saving matrix.mtx.gz...")
        matrix_file = matrix_dir / 'matrix.mtx.gz'
        with gzip.open(matrix_file, 'wb') as f:
            sio.mmwrite(f, matrix, comment='metadata_json: {"software_version": "pseudo-hd-1.0.0", "format_version": 2}')

        # Save features.tsv.gz
        logger.info("  Saving features.tsv.gz...")
        features_out = matrix_dir / 'features.tsv.gz'
        with gzip.open(features_out, 'wt') as f:
            for gene_id, gene_name, feature_type in original_features:
                f.write(f"{gene_id}\t{gene_name}\t{feature_type}\n")

        # Save barcodes.tsv.gz
        logger.info("  Saving barcodes.tsv.gz...")
        barcodes_out = matrix_dir / 'barcodes.tsv.gz'
        with gzip.open(barcodes_out, 'wt') as f:
            for bc in original_barcodes:
                f.write(f"{bc}\n")

        # Copy spatial directory from original
        logger.info("  Copying spatial metadata from original...")
        import shutil
        spatial_orig = original_hd_dir / 'spatial'
        spatial_out = output_dir / 'spatial'
        if spatial_orig.exists():
            shutil.copytree(spatial_orig, spatial_out, dirs_exist_ok=True)

        logger.info(f"  Saved pseudo Visium HD data in 10X format to {output_dir}")

    else:
        # Original simple CSV output (fallback)
        logger.info("  Creating simple CSV format (no original structure provided)...")

        # Create bin identifiers
        df_bins['bin_id'] = df_bins['bin_x'].astype(str) + '_' + df_bins['bin_y'].astype(str)

        # Pivot to matrix format
        matrix = df_bins.pivot_table(
            index='bin_id',
            columns='gene',
            values='count',
            aggfunc='sum',
            fill_value=0
        )

        logger.info(f"  Matrix shape: {matrix.shape[0]} bins x {matrix.shape[1]} genes")

        # Save as CSV
        matrix_file = output_dir / 'pseudo_hd_gene_expression.csv.gz'
        logger.info(f"  Saving gene expression matrix to {matrix_file}...")
        matrix.to_csv(matrix_file, compression='gzip')

        # Save spatial coordinates
        spatial_file = output_dir / 'pseudo_hd_spatial_coords.csv'
        logger.info(f"  Saving spatial coordinates to {spatial_file}...")

        spatial_df = df_bins[['bin_id', 'bin_x', 'bin_y']].drop_duplicates()
        spatial_df.to_csv(spatial_file, index=False)

        logger.info(f"  Saved pseudo Visium HD data to {output_dir}")


def save_ground_truth_mapping(cell_to_sc_mapping, df_pixels, adata_sc, output_dir):
    """
    Save ground truth mapping of which sc cell was assigned to which segmented cell.
    This is the GROUND TRUTH for validation!

    Args:
        cell_to_sc_mapping: Dict mapping cell_id -> sc_cell_index
        df_pixels: Pixel-to-cell mapping (to get cell types)
        adata_sc: Single-cell AnnData (to get sc cell info)
        output_dir: Output directory

    Saves:
        ground_truth_cell_assignments.csv with columns:
            - cell_id: Segmented cell ID
            - sc_cell_index: Which sc cell was assigned to this cell
            - cell_type: Cell type
            - sc_cell_barcode: Original sc cell barcode (if available)
    """
    logger.info("Saving ground truth cell-to-sc mapping...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get cell types for segmented cells
    cell_info = df_pixels[df_pixels['cell_id'] > 0].groupby('cell_id')['cell_type'].first().to_dict()

    # Build mapping dataframe
    rows = []
    for cell_id, sc_idx in cell_to_sc_mapping.items():
        cell_type = cell_info.get(cell_id, 'Unknown')

        # Get sc cell barcode if available
        sc_barcode = adata_sc.obs.index[adata_sc.obs.index == sc_idx].tolist()
        sc_barcode = sc_barcode[0] if len(sc_barcode) > 0 else sc_idx

        rows.append({
            'cell_id': cell_id,
            'sc_cell_index': sc_idx,
            'cell_type': cell_type,
            'sc_cell_barcode': sc_barcode
        })

    df_ground_truth = pd.DataFrame(rows)

    # Check if empty
    if len(df_ground_truth) == 0:
        logger.error("  ERROR: No cells were assigned!")
        logger.error("  This means no cell types matched between segmentation and SC data.")
        logger.error("  Check the cell type names in both datasets.")
        return

    # Save
    ground_truth_file = output_dir / 'ground_truth_cell_assignments.csv'
    df_ground_truth.to_csv(ground_truth_file, index=False)

    logger.info(f"  Saved ground truth mapping to {ground_truth_file}")
    logger.info(f"  Total assignments: {len(df_ground_truth)}")

    # Summary by cell type
    if len(df_ground_truth) > 0 and 'cell_type' in df_ground_truth.columns:
        logger.info("  Ground truth summary by cell type:")
        for cell_type, count in df_ground_truth['cell_type'].value_counts().items():
            logger.info(f"    {cell_type}: {count} cells")


def main():
    """Main pipeline for creating pseudo Visium HD data."""

    logger.info("="*70)
    logger.info("Create Pseudo Visium HD Data from Single-Cell Assignment")
    logger.info("="*70)

    # Configuration is loaded globally from config_pseudo_hd.py

    logger.info(f"\nInput files:")
    logger.info(f"  Pixel mapping: {config.PIXEL_MAPPING_CSV}")
    logger.info(f"  GTF annotation: {getattr(config, 'GTF_FILE', 'gencode.v38.annotation.gtf')}")
    logger.info(f"  Single-cell H5: {config.SC_H5_PATH}")
    logger.info(f"  Single-cell metadata: {config.SC_METADATA_PATH}")
    logger.info(f"\nOutput directory: {config.OUTPUT_DIR}")
    logger.info(f"\nParameters:")
    logger.info(f"  Microns per pixel: {config.MICRONS_PER_PIXEL}")
    logger.info(f"  Membrane zone: 0-{config.MEMBRANE_CUTOFF} bins (~{config.MEMBRANE_CUTOFF * 2}µm) from boundary (closer = more reads)")
    logger.info(f"  Random seed: {config.RANDOM_SEED}")

    if hasattr(config, 'CROP_REGION') and config.CROP_REGION is not None:
        logger.info(f"  CROP MODE ENABLED:")
        crop_cfg = config.CROP_REGION
        logger.info(f"    Region: bin_x=[{crop_cfg.get('min_bin_x', 0)}, {crop_cfg.get('max_bin_x', 'auto')}], " +
                   f"bin_y=[{crop_cfg.get('min_bin_y', 0)}, {crop_cfg.get('max_bin_y', 'auto')}]")
        if crop_cfg.get('max_bin_x') is None or crop_cfg.get('max_bin_y') is None:
            logger.info(f"    Quantile: {crop_cfg.get('quantile', 0.2)} (will take first {crop_cfg.get('quantile', 0.2)*100:.0f}% of tissue)")
    else:
        logger.info(f"  Processing ENTIRE tissue (no cropping)")

    logger.info("")

    # Set random seed
    np.random.seed(config.RANDOM_SEED)

    # Step 1 & 2: Load single-cell data FIRST (needed for gene counts in splicing model)
    adata_sc = load_single_cell_data(config.SC_H5_PATH, config.SC_METADATA_PATH)

    # Compute gene counts for splicing model
    logger.info("Computing gene counts for splicing model...")
    gene_counts_array = np.array(adata_sc.X.sum(axis=0)).flatten()
    gene_counts = {gene: int(count) for gene, count in zip(adata_sc.var_names, gene_counts_array)}

    # Step 1: Compute gene localization using splicing model
    logger.info("Computing gene localization using SPLICING-BASED MODEL...")

    gtf_file = getattr(config, 'GTF_FILE', 'gencode.v38.annotation.gtf')
    organism = getattr(config, 'ORGANISM', 'human')

    # Import and call the classification function
    from gene_localization_GO_analysis import classify_genes_aggregate

    results = classify_genes_aggregate(
        genes=list(adata_sc.var_names),
        gtf_file=gtf_file,
        gene_counts=gene_counts,
        organism=organism
    )

    # Extract parameters for per-cell use
    dropout_probs = results  # Contains p0, global_alpha, etc.
    membrane_genes = results['membrane_genes_all']

    # Step 3: Load pixel-to-cell mapping (with bin aggregation optimization)
    df_pixels = load_pixel_to_cell_mapping(
        config.PIXEL_MAPPING_CSV,
        aggregate_to_bins=True,  # OPTIMIZATION: Aggregate to 2µm bins immediately for 100x speedup!
        microns_per_pixel=config.MICRONS_PER_PIXEL
    )

    # Step 3.5: Crop to spatial region if specified (TEST MODE)
    if hasattr(config, 'CROP_REGION') and config.CROP_REGION is not None:
        logger.info("CROP MODE: Subsetting to spatial region for testing...")
        crop_cfg = config.CROP_REGION

        # Determine max coordinates
        if crop_cfg.get('max_bin_x') is None:
            max_x = df_pixels['bin_x'].quantile(crop_cfg.get('quantile', 0.2))
            logger.info(f"  Using quantile {crop_cfg.get('quantile', 0.2)}: max_bin_x = {max_x}")
        else:
            max_x = crop_cfg['max_bin_x']

        if crop_cfg.get('max_bin_y') is None:
            max_y = df_pixels['bin_y'].quantile(crop_cfg.get('quantile', 0.2))
            logger.info(f"  Using quantile {crop_cfg.get('quantile', 0.2)}: max_bin_y = {max_y}")
        else:
            max_y = crop_cfg['max_bin_y']

        min_x = crop_cfg.get('min_bin_x', 0)
        min_y = crop_cfg.get('min_bin_y', 0)

        # Apply crop
        logger.info(f"  Cropping to region: bin_x=[{min_x}, {max_x}], bin_y=[{min_y}, {max_y}]")
        original_bins = len(df_pixels)
        df_pixels = df_pixels[
            (df_pixels['bin_x'] >= min_x) & (df_pixels['bin_x'] <= max_x) &
            (df_pixels['bin_y'] >= min_y) & (df_pixels['bin_y'] <= max_y)
        ].copy()

        logger.info(f"  Cropped: {original_bins:,} → {len(df_pixels):,} bin-compartment pairs ({100*len(df_pixels)/original_bins:.1f}%)")
        logger.info(f"  Unique cells in cropped region: {df_pixels['cell_id'].nunique():,}")

    # Step 4: Standardize cell type names
    logger.info("Standardizing cell type names...")

    # Get unique cell types from both datasets
    seg_types = set(df_pixels[df_pixels['cell_id'] > 0]['cell_type'].dropna().unique())
    sc_types = set(adata_sc.obs['cell_type'].dropna().unique())

    logger.info(f"  Segmentation cell types: {seg_types}")
    logger.info(f"  SC data cell types: {sc_types}")

    # Auto-generate mapping by removing suffixes
    name_mapping = {}
    for seg_type in seg_types:
        # Try exact match
        if seg_type in sc_types:
            name_mapping[seg_type] = seg_type
            continue

        # Try removing " cells" or " cell" suffix
        for suffix in [' cells', ' cell']:
            cleaned = seg_type.replace(suffix, '')
            if cleaned in sc_types:
                name_mapping[seg_type] = cleaned
                logger.info(f"  Auto-mapping: '{seg_type}' → '{cleaned}'")
                break
        else:
            # No match found
            if seg_type != 'Unknown':
                logger.warning(f"  No SC match for '{seg_type}' - these cells will be skipped")
            name_mapping[seg_type] = seg_type

    # Apply mapping
    df_pixels['cell_type'] = df_pixels['cell_type'].replace(name_mapping)

    # Step 4: Assign sc cells to segmented cells
    cell_to_sc_mapping = assign_sc_to_cells(df_pixels, adata_sc)

    # Step 5: Save ground truth mapping (cell_id -> sc_cell)
    save_ground_truth_mapping(cell_to_sc_mapping, df_pixels, adata_sc, config.OUTPUT_DIR)

    # Step 6: Distribute genes to bins using splicing model
    sparse_data = distribute_genes_to_bins(
        df_pixels,
        adata_sc,
        cell_to_sc_mapping,
        dropout_probs,
        membrane_genes,
        config.MICRONS_PER_PIXEL
    )

    # Step 7: Create Visium HD format output
    logger.info("Step 7: Converting to Visium HD format...")

    original_hd_dir = getattr(config, 'ORIGINAL_HD_DIR', None)
    create_visium_hd_format(sparse_data, config.OUTPUT_DIR, original_hd_dir=original_hd_dir)

    logger.info("\n" + "="*70)
    logger.info("Processing Complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
