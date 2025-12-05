#!/usr/bin/env python
"""
Annotate cells using STHD (Spatial Transcriptomics HD) deconvolution.

This script:
1. Loads Cellpose pixel-to-cell mapping with nuclear information
2. Runs STHD on 2μm Visium HD bins to get cell type predictions
3. Assigns each cell to the cell type that has the majority of its NUCLEAR bins
4. Saves output in the same format as map_cells_to_clusters.py for compatibility with nuclear_expansion.py

The key difference from CellTypist approach:
- STHD uses spatial deconvolution with scRNA-seq reference (lambda file)
- Focuses on NUCLEAR bins only (where RNA is localized)
- More accurate for spatial transcriptomics vs CellTypist

Usage:
    # First activate the STHD environment
    source sthd_env/bin/activate

    # Then run the script
    python run_sthd_annotation.py \
        --pixel_csv cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_full.csv.gz \
        --visium_dir Human_Colorectal/output/binned_outputs/square_002um \
        --lambda_file lambda_colorectal_major_lineage.txt \
        --output cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_with_celltype.csv.gz \
        --scale_factor 0.07973422

Requirements:
    - STHD (installed in sthd_env)
    - scanpy, anndata, pandas, numpy
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_aggregate_visium_hd_data(visium_dir):
    """
    Load Visium HD 2μm data and aggregate to 8μm bins for STHD processing.

    Args:
        visium_dir: Path to square_002um directory

    Returns:
        adata_8um: AnnData with 8μm bins (for STHD)
        mapping_df: DataFrame with columns [array_row, array_col, bin_8um_id] for mapping back
    """
    logger.info(f"Loading Visium HD 2μm data from {visium_dir}...")

    import scanpy as sc
    from scipy.sparse import csr_matrix
    import anndata

    # Load 10x format data
    adata = sc.read_10x_mtx(Path(visium_dir)/'filtered_feature_bc_matrix', var_names='gene_symbols', cache=False)

    # Load spatial coordinates
    spatial_dir = Path(visium_dir) / 'spatial'
    tissue_positions = pd.read_parquet(spatial_dir / 'tissue_positions.parquet')

    # Match barcodes
    common_barcodes = list(set(adata.obs_names) & set(tissue_positions['barcode']))
    logger.info(f"  Found {len(common_barcodes):,} common 2μm barcodes")

    # Filter and reorder
    adata = adata[common_barcodes, :].copy()
    tissue_positions = tissue_positions.set_index('barcode').loc[common_barcodes]

    # Add spatial coordinates
    adata.obs['array_row'] = tissue_positions['array_row'].values
    adata.obs['array_col'] = tissue_positions['array_col'].values
    adata.obs['pxl_row_in_fullres'] = tissue_positions['pxl_row_in_fullres'].values
    adata.obs['pxl_col_in_fullres'] = tissue_positions['pxl_col_in_fullres'].values

    logger.info(f"  Loaded {adata.n_obs:,} bins x {adata.n_vars:,} genes")

    # Aggregate to 8μm bins (4x4 aggregation)
    logger.info("  Aggregating 2μm bins → 8μm bins (4x4)...")

    adata.obs['bin_8um_row'] = (adata.obs['array_row'] // 4).astype(int)
    adata.obs['bin_8um_col'] = (adata.obs['array_col'] // 4).astype(int)
    adata.obs['bin_8um_id'] = (adata.obs['bin_8um_row'].astype(str) + '_' +
                                 adata.obs['bin_8um_col'].astype(str))

    # Create mapping DataFrame from 2μm coords to 8μm bin ID (VECTORIZED)
    logger.info("    Creating 2μm → 8μm mapping...")
    mapping_df = adata.obs[['array_row', 'array_col', 'bin_8um_id']].copy()
    mapping_df['array_row'] = mapping_df['array_row'].astype(int)
    mapping_df['array_col'] = mapping_df['array_col'].astype(int)

    # Aggregate counts by 8μm bins (VECTORIZED using sparse matrix multiplication)

    bin_ids = adata.obs['bin_8um_id'].unique()
    logger.info(f"    Aggregating {len(adata):,} 2μm bins → {len(bin_ids):,} 8μm bins")

    bin_to_idx = {bid: i for i, bid in enumerate(bin_ids)}
    bin_indices = adata.obs['bin_8um_id'].map(bin_to_idx).values.astype(np.int32)

    # Sum counts per 8μm bin using sparse matrix multiplication (FULLY VECTORIZED!)
    n_bins_8um = len(bin_ids)
    n_bins_2um = adata.n_obs

    logger.info("    Summing counts per 8μm bin (vectorized)...")
    # Create aggregation matrix: (n_8um_bins x n_2um_bins)
    # Each row sums all 2μm bins belonging to that 8μm bin
    agg_matrix = csr_matrix(
        (np.ones(n_bins_2um, dtype=np.float32), (bin_indices, np.arange(n_bins_2um))),
        shape=(n_bins_8um, n_bins_2um)
    )
    # Multiply: (n_8um x n_2um) @ (n_2um x n_genes) = (n_8um x n_genes)
    aggregated_X = agg_matrix @ adata.X

    # Create 8μm AnnData
    adata_8um = anndata.AnnData(
        X=csr_matrix(aggregated_X),
        var=adata.var.copy(),
        obs=pd.DataFrame(index=bin_ids)
    )

    # Add spatial coordinates (use mean of 2μm bins)
    spatial_info = adata.obs.groupby('bin_8um_id').agg({
        'pxl_row_in_fullres': 'mean',
        'pxl_col_in_fullres': 'mean',
        'bin_8um_row': 'first',
        'bin_8um_col': 'first'
    })
    adata_8um.obs = adata_8um.obs.join(spatial_info)
    adata_8um.obs['array_row'] = adata_8um.obs['bin_8um_row'].astype(int)
    adata_8um.obs['array_col'] = adata_8um.obs['bin_8um_col'].astype(int)

    # Add spatial coordinates to obsm (required by STHD/squidpy for spatial graph)
    adata_8um.obsm['spatial'] = adata_8um.obs[['array_col', 'array_row']].values

    logger.info(f"  Aggregated to {adata_8um.n_obs:,} 8μm bins x {adata_8um.n_vars:,} genes")

    return adata_8um, mapping_df


def run_sthd_deconvolution(adata_8um, lambda_file):
    """
    Run REAL STHD deconvolution with spatial smoothness on 8μm bins.

    Args:
        adata_8um: AnnData with 8μm aggregated Visium HD data
        lambda_file: Path to lambda reference file

    Returns:
        adata_8um: AnnData with cell type predictions added to .obs
    """
    logger.info("Running STHD deconvolution (with spatial smoothness)...")
    logger.info(f"  Lambda file: {lambda_file}")

    try:
        from STHD import sthdio, refscrna, model
    except ImportError:
        logger.error("STHD not found! Please activate the sthd_env virtual environment:")
        logger.error("  source sthd_env/bin/activate")
        sys.exit(1)

    # Load lambda reference using STHD's API
    logger.info("  Loading lambda reference...")
    lambda_df = refscrna.load_scrna_ref(lambda_file)
    logger.info(f"    {lambda_df.shape[0]:,} genes x {lambda_df.shape[1]} cell types")
    logger.info(f"    Cell types: {list(lambda_df.columns)}")

    # Create STHD data object from 8μm adata
    logger.info("  Creating STHD data object...")
    sthd_data = sthdio.STHD(
        spatial_path=adata_8um,
        counts_data=None,
        full_res_image_path=None,  # Not needed for deconvolution
        load_type='preload'
    )

    # Match reference to spatial data
    logger.info("  Matching reference to spatial data...")
    sthd_data.match_refscrna(lambda_df, cutgene=True)

    common_genes = sthd_data.adata.shape[1]
    logger.info(f"  Found {common_genes:,} common genes between data and reference")

    if common_genes < 100:
        logger.error(f"Too few common genes ({common_genes})! Check gene naming.")
        sys.exit(1)

    # Prepare constants and training weights for STHD model
    logger.info("  Preparing model constants (with spatial graph)...")
    X, Y, Z, F, Acsr_row, Acsr_col = model.prepare_constants(sthd_data)
    logger.info(f"    X={X:,} spots, Y={Y:,} genes, Z={Z} cell types")

    logger.info("  Preparing training weights...")
    W, eW, P, Phi, ll_wat, ce_wat, m, v = model.prepare_training_weights(X, Y, Z)

    # Train STHD model
    logger.info("  Training STHD model (this may take a while)...")
    logger.info("    n_iter=20, step_size=1, beta=0.1")
    metrics = model.train(
        n_iter=20,
        step_size=1,  # learning rate
        beta=0.1,  # weight of cross-entropy w.r.t log-likelihood
        constants=(X, Y, Z, F, Acsr_row, Acsr_col),
        weights=(W, eW, P, Phi, ll_wat, ce_wat, m, v),
        early_stop=True
    )
    logger.info(f"  Training completed in {len(metrics)} iterations")

    # Extract predictions from P matrix
    # P is the probability matrix: spots x cell types
    logger.info("  Extracting cell type predictions...")

    # Get cell type names from lambda reference
    cell_type_names = list(lambda_df.columns)

    # Create DataFrame with proportions
    proportions = pd.DataFrame(
        P,
        index=sthd_data.adata.obs_names,
        columns=cell_type_names
    )

    # Assign cell type with maximum proportion
    adata_8um.obs['sthd_cell_type'] = proportions.idxmax(axis=1).values
    adata_8um.obs['sthd_max_proportion'] = proportions.max(axis=1).values

    # Add all proportions to obsm
    adata_8um.obsm['sthd_proportions'] = proportions.values

    logger.info("  STHD deconvolution complete")
    logger.info(f"  Predicted cell types:")
    for ct, count in adata_8um.obs['sthd_cell_type'].value_counts().items():
        pct = 100 * count / len(adata_8um)
        logger.info(f"    {ct}: {count:,} bins ({pct:.1f}%)")

    return adata_8um


def assign_cell_types_from_nuclear_bins(cellpose_csv, adata_with_predictions, output_csv,
                                         microns_per_pixel=0.2737012522439323):
    """
    Assign cell types to cells based on majority vote of their NUCLEAR bins.

    This is similar to map_cells_to_clusters.py but uses STHD predictions instead of clusters.

    Args:
        cellpose_csv: Path to Cellpose pixel mapping CSV (with cropped coordinates starting from 0)
        adata_with_predictions: AnnData with STHD cell type predictions
        output_csv: Path to save annotated CSV
        microns_per_pixel: Pixel to micron conversion
    """
    logger.info("Assigning cell types from nuclear bins...")

    # Load Cellpose CSV
    logger.info(f"  Loading Cellpose pixel mapping from {cellpose_csv}...")
    chunks = []
    for chunk in pd.read_csv(cellpose_csv, compression='gzip', chunksize=1_000_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"    Loaded {len(df):,} pixels")

    # Filter to nuclear pixels only (this is where RNA is!)
    df_nuclear = df[df['is_nuclear'] == 1].copy()
    logger.info(f"  Filtering to {len(df_nuclear):,} nuclear pixels")

    # Convert pixels to 2μm bins
    logger.info("  Mapping nuclear pixels to 2μm bins...")
    BIN_SIZE_MICRONS = 2.0
    PIXELS_PER_BIN = BIN_SIZE_MICRONS / microns_per_pixel

    # Use pixel coordinates directly (already cropped, no offset needed)
    pixel_rows = df_nuclear['y'].values.astype(np.int32)
    pixel_cols = df_nuclear['x'].values.astype(np.int32)

    # Compute bin indices
    bin_row_idx = (pixel_rows / PIXELS_PER_BIN).astype(np.int32)
    bin_col_idx = (pixel_cols / PIXELS_PER_BIN).astype(np.int32)

    # Create barcode lookup from adata (FULLY VECTORIZED!)
    logger.info("  Creating bin-to-celltype lookup (vectorized)...")

    # Extract array coordinates (vectorized)
    if 'array_row' in adata_with_predictions.obs.columns:
        array_rows = adata_with_predictions.obs['array_row'].values.astype(np.int32)
        array_cols = adata_with_predictions.obs['array_col'].values.astype(np.int32)
    else:
        # Fall back to pixel coordinates
        pxl_rows = adata_with_predictions.obs['pxl_row_in_fullres'].values.astype(np.int32)
        pxl_cols = adata_with_predictions.obs['pxl_col_in_fullres'].values.astype(np.int32)
        array_rows = (pxl_rows / PIXELS_PER_BIN).astype(np.int32)
        array_cols = (pxl_cols / PIXELS_PER_BIN).astype(np.int32)

    cell_types = adata_with_predictions.obs['sthd_cell_type'].values

    # Create DataFrame for fast merge (much faster than dict lookup!)
    bin_lookup_df = pd.DataFrame({
        'array_row': array_rows,
        'array_col': array_cols,
        'sthd_cell_type': cell_types
    })

    logger.info(f"    Created lookup for {len(bin_lookup_df):,} bins")

    # Map nuclear pixels to cell types (VECTORIZED MERGE - 100x faster!)
    logger.info("  Mapping nuclear pixels to cell types (vectorized merge)...")
    df_nuclear['array_row'] = bin_row_idx
    df_nuclear['array_col'] = bin_col_idx

    # Merge with bin lookup (vectorized!)
    df_nuclear = df_nuclear.merge(
        bin_lookup_df,
        on=['array_row', 'array_col'],
        how='left'
    )
    df_nuclear['bin_celltype'] = df_nuclear['sthd_cell_type'].fillna('Unknown')
    df_nuclear = df_nuclear.drop(columns=['sthd_cell_type', 'array_row', 'array_col'])

    mapped_count = (df_nuclear['bin_celltype'] != 'Unknown').sum()
    logger.info(f"    Mapped {mapped_count:,} / {len(df_nuclear):,} nuclear pixels ({100*mapped_count/len(df_nuclear):.1f}%)")

    # Majority vote per cell based on nuclear pixels
    logger.info("  Computing majority vote per cell (nuclear bins only)...")

    def get_majority_celltype(cell_types):
        """Get cell type with most nuclear pixels."""
        if len(cell_types) == 0:
            return 'Unknown'

        counts = cell_types.value_counts()

        if len(counts) == 1 and counts.index[0] == 'Unknown':
            return 'Unknown'

        # Remove Unknown from counts
        counts_without_unknown = counts[counts.index != 'Unknown']

        if len(counts_without_unknown) == 0:
            return 'Unknown'

        return counts_without_unknown.index[0]

    # Group by cell_id and find majority cell type
    cell_type_votes = df_nuclear[df_nuclear['cell_id'] > 0].groupby('cell_id')['bin_celltype'].agg(get_majority_celltype)
    cell_to_celltype = cell_type_votes.to_dict()

    logger.info(f"  Assigned cell types to {len(cell_to_celltype):,} cells")

    # Apply to all pixels in original dataframe
    logger.info("  Applying cell types to all pixels...")
    df['cell_type'] = df['cell_id'].map(cell_to_celltype)
    df['cell_type'] = df['cell_type'].fillna('Unknown')

    # Save output
    logger.info(f"  Saving annotated CSV to {output_csv}...")
    df.to_csv(output_csv, index=False, compression='gzip')
    logger.info(f"    Saved {len(df):,} pixels")

    # Log cell type distribution
    logger.info("\n  Cell Type Distribution (by cells):")
    celltype_counts = df[df['cell_id'] > 0].groupby('cell_id')['cell_type'].first().value_counts()
    for celltype, count in celltype_counts.items():
        pct = 100 * count / celltype_counts.sum()
        logger.info(f"    {celltype}: {count:,} cells ({pct:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Annotate cells using STHD deconvolution',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--pixel_csv', type=str, required=True,
                       help='Path to Cellpose pixel mapping CSV')
    parser.add_argument('--visium_dir', type=str, required=True,
                       help='Path to square_002um directory')
    parser.add_argument('--lambda_file', type=str, required=True,
                       help='Path to lambda reference file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV path')
    parser.add_argument('--microns_per_pixel', type=float, default=0.2737012522439323,
                       help='Microns per pixel (default: 0.2737012522439323 for colorectal)')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("STHD-Based Cell Type Annotation (8μm strategy)")
    logger.info("="*70)

    # Step 1: Load 2μm data and aggregate to 8μm bins
    adata_8um, mapping_df = load_and_aggregate_visium_hd_data(args.visium_dir)

    # Step 2: Run STHD on 8μm bins (with spatial smoothness!)
    adata_8um = run_sthd_deconvolution(adata_8um, args.lambda_file)

    # Step 3: Map 8μm predictions back to 2μm bins (VECTORIZED!)
    logger.info("Mapping 8μm predictions back to 2μm bins...")
    import anndata

    # Create lookup DataFrame: 8μm bin_id -> cell_type
    celltype_lookup = adata_8um.obs[['sthd_cell_type']].copy()
    celltype_lookup.index.name = 'bin_8um_id'
    celltype_lookup = celltype_lookup.reset_index()

    # Merge 2μm mapping with cell types (VECTORIZED!)
    df_2um = mapping_df.merge(celltype_lookup, on='bin_8um_id', how='left')
    df_2um['sthd_cell_type'] = df_2um['sthd_cell_type'].fillna('Unknown')

    logger.info(f"  Created {len(df_2um):,} 2μm bin predictions from {len(adata_8um):,} 8μm bins")

    # Create minimal adata-like object for compatibility with assign_cell_types_from_nuclear_bins
    adata_2um = anndata.AnnData(obs=df_2um.reset_index(drop=True))
    adata_2um.obs.index = adata_2um.obs.index.astype(str)

    # Step 4: Assign cell types based on nuclear bins (uses 2μm resolution)
    df_annotated = assign_cell_types_from_nuclear_bins(
        args.pixel_csv,
        adata_2um,
        args.output,
        microns_per_pixel=args.microns_per_pixel
    )

    logger.info("\n" + "="*70)
    logger.info("STHD Annotation Complete!")
    logger.info("="*70)
    logger.info(f"\nOutput: {args.output}")
    logger.info("\nColumns in output CSV:")
    logger.info("  - x, y: Pixel coordinates")
    logger.info("  - cell_id: Cell ID from Cellpose")
    logger.info("  - is_boundary, is_interior, is_nuclear, is_cytoplasm: Segmentation flags")
    logger.info("  - cell_type: Assigned cell type from STHD (based on nuclear bins)")


if __name__ == "__main__":
    main()
