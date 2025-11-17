#!/usr/bin/env python
"""
Add cell type annotations to Cellpose pixel-level segmentation results.

This script:
1. Loads Cellpose pixel-to-cell mapping CSV
2. Loads 8μm cluster annotations from tissue_positions.parquet
3. Maps each pixel DIRECTLY to 8μm bins using scaling factor (8μm / 0.2739 μm/pixel ≈ 29.2)
4. Assigns cell type to each cell based on majority vote
5. Adds 'cell_type' column to the Cellpose CSV

Usage:
    python map_cells_to_clusters.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_8um_annotations(tissue_positions_path, cluster_mapping_path):
    """
    Load 8μm bin positions and cluster-to-celltype mappings.

    Args:
        tissue_positions_path: Path to square_008um/spatial/tissue_positions.parquet
        cluster_mapping_path: Path to cluster_to_celltype_mapping.csv

    Returns:
        tissue_df: DataFrame with 8μm bin positions and cell types
        cluster_to_celltype: Dictionary mapping cluster -> cell_type
    """
    logger.info(f"Loading 8μm tissue positions from {tissue_positions_path}...")
    tissue_df = pd.read_parquet(tissue_positions_path, engine='fastparquet')
    logger.info(f"  Loaded {len(tissue_df):,} bins (8μm)")

    # Load cluster-to-celltype mapping
    logger.info(f"Loading cluster mapping from {cluster_mapping_path}...")
    cluster_mapping = pd.read_csv(cluster_mapping_path, index_col=0)
    cluster_to_celltype = cluster_mapping['cell_type'].to_dict()
    logger.info(f"  Loaded {len(cluster_to_celltype)} cluster types")

    # Load cluster assignments (from h5ad or separate file)
    # Assuming cluster info is in the mapping file or we need to load it separately

    return tissue_df, cluster_to_celltype


def assign_cell_types(cellpose_csv_path, bins_8um_df, cluster_assignments, cluster_to_celltype, output_path):
    """
    Load Cellpose CSV, map pixels DIRECTLY to 8μm bins using scaling factor, assign cell types, and save.

    This approach uses direct pixel-to-bin scaling (8μm / 0.2739 μm/pixel) which is simple and fast.

    Args:
        cellpose_csv_path: Path to Cellpose pixel_to_cell_mapping_full.csv.gz
        bins_8um_df: DataFrame with 8μm bin positions (from square_008um/spatial/tissue_positions.parquet)
        cluster_assignments: Dictionary mapping 8μm barcode -> cluster_id
        cluster_to_celltype: Dictionary mapping cluster_id -> cell_type
        output_path: Path to save output CSV with cell_type column
    """
    logger.info(f"Loading Cellpose pixel mapping from {cellpose_csv_path}...")

    # Load in chunks to handle large file
    chunk_size = 1_000_000
    chunks = []

    for chunk in pd.read_csv(cellpose_csv_path, compression='gzip', chunksize=chunk_size):
        logger.info(f"  Processing chunk with {len(chunk):,} pixels...")
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Loaded {len(df):,} total pixels")

    # Strategy: Direct pixel → 8μm bin mapping using scaling factor
    # Scaling: 8μm / 0.2739 μm/pixel ≈ 29.2 pixels per 8μm bin

    logger.info("Mapping pixels DIRECTLY to 8μm bins using scaling factor...")

    # Scaling factor from metadata
    MICRONS_PER_PIXEL = 0.2739038899725172
    BIN_SIZE_MICRONS = 8.0
    PIXELS_PER_8UM_BIN = BIN_SIZE_MICRONS / MICRONS_PER_PIXEL  # ≈ 29.206

    logger.info(f"  Scaling: {PIXELS_PER_8UM_BIN:.3f} pixels per 8μm bin")

    # Crop offsets
    CROP_OFFSET_Y = 12519  # row_left
    CROP_OFFSET_X = 3157   # col_up

    logger.info(f"  Applying crop offset: y+{CROP_OFFSET_Y}, x+{CROP_OFFSET_X}")

    # Extract pixel coordinates and convert to full-resolution
    pixel_rows = df['y'].values.astype(np.int32) + CROP_OFFSET_Y
    pixel_cols = df['x'].values.astype(np.int32) + CROP_OFFSET_X

    # Compute 8μm bin indices: divide by scaling factor and round down with int()
    bin_8um_row_idx = (pixel_rows / PIXELS_PER_8UM_BIN).astype(np.int32)
    bin_8um_col_idx = (pixel_cols / PIXELS_PER_8UM_BIN).astype(np.int32)

    logger.info(f"  Computed 8μm bin indices for {len(df):,} pixels")

    # Vectorized mapping using pandas merge
    logger.info("  Preparing 8μm bin lookup (vectorized)...")
    bins_8um_df['row_idx'] = (bins_8um_df['pxl_row_in_fullres'] / PIXELS_PER_8UM_BIN).astype(np.int32)
    bins_8um_df['col_idx'] = (bins_8um_df['pxl_col_in_fullres'] / PIXELS_PER_8UM_BIN).astype(np.int32)

    # Create a lookup dataframe with only necessary columns
    bins_lookup = bins_8um_df[['row_idx', 'col_idx', 'barcode']].copy()
    logger.info(f"  Created lookup for {len(bins_lookup):,} 8μm bins")

    # Add bin indices to pixel dataframe
    df['bin_row_idx'] = bin_8um_row_idx
    df['bin_col_idx'] = bin_8um_col_idx

    # Vectorized merge: map pixels to 8μm barcodes
    logger.info("  Mapping pixels to 8μm barcodes (vectorized merge)...")
    df = df.merge(
        bins_lookup,
        left_on=['bin_row_idx', 'bin_col_idx'],
        right_on=['row_idx', 'col_idx'],
        how='left'
    )
    df = df.rename(columns={'barcode': 'bin_8um'})
    df = df.drop(columns=['row_idx', 'col_idx', 'bin_row_idx', 'bin_col_idx'])

    # Count mapped pixels
    mapped_8um = df['bin_8um'].notna().sum()
    logger.info(f"  Mapped {mapped_8um:,} / {len(df):,} pixels to 8μm bins ({100*mapped_8um/len(df):.1f}%)")

    # Map bins to clusters (vectorized)
    logger.info("Mapping bins to clusters (vectorized)...")
    df['cluster'] = df['bin_8um'].map(cluster_assignments)

    # Map clusters to cell types (vectorized)
    logger.info("Assigning cell types (vectorized)...")
    df['cell_type'] = df['cluster'].map(cluster_to_celltype)

    # For cells with multiple clusters, use majority vote (most pixels wins)
    logger.info("Assigning cell types by majority vote for each cell...")

    def get_majority_celltype(cell_types):
        """Get cell type with most pixels. If tie or all Unknown, return 'Unknown'."""
        if len(cell_types) == 0:
            return 'Unknown'

        # Count occurrences
        counts = cell_types.value_counts()

        # If only Unknown values, return Unknown
        if len(counts) == 1 and counts.index[0] == 'Unknown':
            return 'Unknown'

        # Remove Unknown from counts for majority vote
        counts_without_unknown = counts[counts.index != 'Unknown']

        if len(counts_without_unknown) == 0:
            return 'Unknown'

        # Return cell type with most pixels
        return counts_without_unknown.index[0]

    # Group by cell_id and find majority cell_type (only for cells with pixels)
    logger.info("  Counting pixel votes per cell...")
    cell_type_votes = df[df['cell_id'] > 0].groupby('cell_id')['cell_type'].agg(get_majority_celltype)

    # Create cell_id -> cell_type mapping
    cell_to_celltype = cell_type_votes.to_dict()

    logger.info(f"  Assigned cell types to {len(cell_to_celltype):,} cells")

    # Fill in cell_type for all pixels based on their cell_id (vectorized)
    df['cell_type_assigned'] = df['cell_id'].map(cell_to_celltype)
    df['cell_type_assigned'] = df['cell_type_assigned'].fillna('Unknown')

    # Drop intermediate columns
    df_final = df[['x', 'y', 'cell_id', 'is_boundary', 'is_interior', 'is_nuclear', 'is_cytoplasm', 'cell_type_assigned']]
    df_final = df_final.rename(columns={'cell_type_assigned': 'cell_type'})

    # Save output
    logger.info(f"Saving annotated CSV to {output_path}...")
    df_final.to_csv(output_path, index=False, compression='gzip')
    logger.info(f"  Saved {len(df_final):,} pixels")

    # Log cell type distribution
    logger.info("\nCell Type Distribution:")
    celltype_counts = df_final[df_final['cell_id'] > 0].groupby('cell_id')['cell_type'].first().value_counts()
    for celltype, count in celltype_counts.items():
        pct = 100 * count / celltype_counts.sum()
        logger.info(f"  {celltype}: {count:,} cells ({pct:.1f}%)")

    return df_final


def load_cluster_assignments_from_h5ad(h5ad_path):
    """
    Load cluster assignments from annotated h5ad file.

    Args:
        h5ad_path: Path to visium_hd_annotated.h5ad

    Returns:
        cluster_assignments: Dictionary mapping barcode -> cluster_id
    """
    logger.info(f"Loading cluster assignments from {h5ad_path}...")

    import scanpy as sc
    adata = sc.read_h5ad(h5ad_path)

    # Extract barcode -> cluster mapping
    cluster_assignments = adata.obs['cluster'].to_dict()

    logger.info(f"  Loaded cluster assignments for {len(cluster_assignments):,} bins")
    logger.info(f"  Number of unique clusters: {adata.obs['cluster'].nunique()}")

    return cluster_assignments


def main():
    """Main pipeline to add cell types to Cellpose pixel-level CSV."""

    logger.info("="*70)
    logger.info("Add Cell Type Annotations to Cellpose Pixel-Level Results")
    logger.info("="*70)

    # Define paths
    current_dir = Path.cwd()

    # Input files
    cellpose_dir = current_dir / 'cellpose_sam_human_kidney_output'
    annotation_dir = current_dir / 'cellannotation_results_human_kidney'
    visium_8um_dir = current_dir / 'Human_kidney' / 'output' / 'binned_outputs' / 'square_008um'

    cellpose_csv = cellpose_dir / 'cropped_visium_hd_human_kidney_pixel_to_cell_mapping_full.csv.gz'
    tissue_positions_8um = visium_8um_dir / 'spatial' / 'tissue_positions.parquet'
    h5ad_path = annotation_dir / 'visium_hd_annotated.h5ad'
    cluster_mapping_csv = annotation_dir / 'cluster_to_celltype_mapping.csv'

    # Output
    output_dir = current_dir / 'cellpose_sam_human_kidney_output'
    output_csv = output_dir / 'cropped_visium_hd_human_kidney_pixel_to_cell_mapping_with_celltype.csv.gz'

    logger.info(f"\nInput files:")
    logger.info(f"  Cellpose CSV: {cellpose_csv}")
    logger.info(f"  8μm positions: {tissue_positions_8um}")
    logger.info(f"  Annotations: {h5ad_path}")
    logger.info(f"  Cluster mapping: {cluster_mapping_csv}")
    logger.info(f"\nOutput file:")
    logger.info(f"  {output_csv}")
    logger.info("")

    # Step 1: Load 8μm bin positions and cluster mappings
    tissue_df_8um, cluster_to_celltype = load_8um_annotations(tissue_positions_8um, cluster_mapping_csv)

    # Step 2: Load cluster assignments from h5ad
    cluster_assignments = load_cluster_assignments_from_h5ad(h5ad_path)

    # Step 3: Load Cellpose CSV, map directly to 8μm bins using scaling factor, assign cell types, and save
    df_annotated = assign_cell_types(cellpose_csv, tissue_df_8um, cluster_assignments, cluster_to_celltype, output_csv)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("Processing Complete!")
    logger.info("="*70)
    logger.info(f"\nOutput file: {output_csv}")
    logger.info(f"\nColumns in output CSV:")
    logger.info(f"  - x, y: Pixel coordinates in cropped image")
    logger.info(f"  - cell_id: Cell identifier from Cellpose")
    logger.info(f"  - is_boundary: 1 if pixel is cell boundary")
    logger.info(f"  - is_interior: 1 if pixel is cell interior")
    logger.info(f"  - is_nuclear: 1 if pixel is nucleus")
    logger.info(f"  - is_cytoplasm: 1 if pixel is cytoplasm")
    logger.info(f"  - cell_type: Assigned cell type (mapped via scaling: pixel → 8μm bin → cluster → cell type)")


if __name__ == "__main__":
    main()
