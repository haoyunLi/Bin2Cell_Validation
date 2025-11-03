#!/usr/bin/env python
"""
Add cell type annotations to Cellpose pixel-level segmentation results.

This script:
1. Loads Cellpose pixel-to-cell mapping CSV (2μm resolution pixels)
2. Loads 8μm cluster annotations from tissue_positions.parquet
3. Maps each 2μm pixel to its corresponding 8μm bin
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


def create_pixel_to_bin_mapping(tissue_df):
    """
    Create vectorized arrays for fast pixel-to-bin mapping.

    For Visium HD:
    - 2μm bins: each bin is 1 pixel at full resolution
    - 8μm bins: each bin is 4x4 pixels at 2μm resolution

    Args:
        tissue_df: DataFrame with columns [barcode, pxl_col_in_fullres, pxl_row_in_fullres, array_row, array_col]

    Returns:
        Tuple of (bin_centers_array, barcodes_array) for vectorized lookup
    """
    logger.info("Creating vectorized pixel-to-bin mapping (2μm -> 8μm)...")

    # Extract arrays for vectorized operations
    barcodes = tissue_df['barcode'].values
    center_cols = tissue_df['pxl_col_in_fullres'].values.astype(np.int32)
    center_rows = tissue_df['pxl_row_in_fullres'].values.astype(np.int32)

    # Stack into (N, 2) array for vectorized distance computation
    bin_centers = np.column_stack([center_rows, center_cols])

    logger.info(f"  Created vectorized mapping for {len(barcodes):,} bins")

    return bin_centers, barcodes


def assign_cell_types(cellpose_csv_path, bin_centers, barcodes, cluster_assignments, cluster_to_celltype, output_path):
    """
    Load Cellpose CSV, map pixels to 8μm bins using vectorized operations, assign cell types, and save.

    Args:
        cellpose_csv_path: Path to Cellpose pixel_to_cell_mapping_full.csv.gz
        bin_centers: numpy array (N, 2) of [row, col] for 8μm bin centers
        barcodes: numpy array of barcodes corresponding to bin_centers
        cluster_assignments: Dictionary mapping barcode -> cluster_id
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

    # Map each pixel to its 8μm bin using VECTORIZED operations
    logger.info("Mapping pixels to 8μm bins (fully vectorized)...")

    # Extract pixel coordinates as numpy arrays
    pixel_rows = df['y'].values.astype(np.int32)
    pixel_cols = df['x'].values.astype(np.int32)

    # For each pixel, find nearest 8μm bin center
    # Each 8μm bin covers a 4x4 region, so we divide by 4 and round to find the bin
    # Then find which bin center is closest

    # Convert pixel coordinates to bin indices (divide by 4 since 8μm = 4 * 2μm)
    pixel_bin_rows = (pixel_rows / 4).astype(np.int32)
    pixel_bin_cols = (pixel_cols / 4).astype(np.int32)

    # Also compute bin centers for comparison
    bin_center_rows = (bin_centers[:, 0] / 4).astype(np.int32)
    bin_center_cols = (bin_centers[:, 1] / 4).astype(np.int32)

    # Create a dictionary for fast lookup: (bin_row, bin_col) -> barcode
    logger.info("  Building bin coordinate lookup table...")
    bin_coord_to_barcode = {}
    for i in range(len(barcodes)):
        key = (bin_center_rows[i], bin_center_cols[i])
        bin_coord_to_barcode[key] = barcodes[i]

    logger.info(f"  Lookup table size: {len(bin_coord_to_barcode):,} unique bin coordinates")

    # Vectorized lookup using numpy
    logger.info("  Performing vectorized bin assignment...")
    bin_8um_list = []
    batch_size = 10_000_000  # Process 10M pixels at a time

    for i in range(0, len(pixel_bin_rows), batch_size):
        end_idx = min(i + batch_size, len(pixel_bin_rows))
        batch_rows = pixel_bin_rows[i:end_idx]
        batch_cols = pixel_bin_cols[i:end_idx]

        # Look up barcodes for this batch
        batch_barcodes = [
            bin_coord_to_barcode.get((r, c), None)
            for r, c in zip(batch_rows, batch_cols)
        ]
        bin_8um_list.extend(batch_barcodes)

        if (i // batch_size) % 10 == 0:
            logger.info(f"    Processed {end_idx:,} / {len(pixel_bin_rows):,} pixels...")

    df['bin_8um'] = bin_8um_list

    # Count how many pixels were mapped
    mapped_pixels = df['bin_8um'].notna().sum()
    logger.info(f"  Mapped {mapped_pixels:,} / {len(df):,} pixels to 8μm bins ({100*mapped_pixels/len(df):.1f}%)")

    # Map bins to clusters (vectorized)
    logger.info("Mapping bins to clusters (vectorized)...")
    df['cluster'] = df['bin_8um'].map(cluster_assignments)

    # Map clusters to cell types (vectorized)
    logger.info("Assigning cell types (vectorized)...")
    df['cell_type'] = df['cluster'].map(cluster_to_celltype)

    # For pixels without cluster assignment, use majority vote within each cell
    logger.info("Assigning cell types by majority vote for each cell...")

    # Group by cell_id and find majority cell_type
    cell_type_votes = df[df['cell_type'].notna()].groupby('cell_id')['cell_type'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
    )

    # Create cell_id -> cell_type mapping
    cell_to_celltype = cell_type_votes.to_dict()

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
    tissue_df, cluster_to_celltype = load_8um_annotations(tissue_positions_8um, cluster_mapping_csv)

    # Step 2: Load cluster assignments from h5ad
    cluster_assignments = load_cluster_assignments_from_h5ad(h5ad_path)

    # Step 3: Create pixel-to-bin mapping (2μm -> 8μm) - vectorized
    bin_centers, barcodes = create_pixel_to_bin_mapping(tissue_df)

    # Step 4: Load Cellpose CSV, assign cell types, and save - fully vectorized
    df_annotated = assign_cell_types(cellpose_csv, bin_centers, barcodes, cluster_assignments, cluster_to_celltype, output_csv)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("Processing Complete!")
    logger.info("="*70)
    logger.info(f"\nOutput file: {output_csv}")
    logger.info(f"\nColumns in output CSV:")
    logger.info(f"  - x, y: Pixel coordinates (2μm resolution)")
    logger.info(f"  - cell_id: Cell identifier from Cellpose")
    logger.info(f"  - is_boundary: 1 if pixel is cell boundary")
    logger.info(f"  - is_interior: 1 if pixel is cell interior")
    logger.info(f"  - is_nuclear: 1 if pixel is nucleus")
    logger.info(f"  - is_cytoplasm: 1 if pixel is cytoplasm")
    logger.info(f"  - cell_type: Assigned cell type from 8μm cluster annotation")


if __name__ == "__main__":
    main()
