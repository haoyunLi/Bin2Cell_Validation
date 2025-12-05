#!/usr/bin/env python
"""
Build STHD λ (lambda) File from scRNA-seq Data

This script creates a lambda reference for STHD (Spatial Transcriptomics HD)
from single-cell RNA-seq data using per-cell gene normalization.

The normalization follows STHD's approach (from STHD/refscrna.py):
1. For each cell, normalize gene counts so they sum to 1 (fraction of transcripts)
2. Average across all cells of the same type

Output format:
- Tab-delimited text file
- First column = gene IDs
- First row = cell-type names
- Values = average fraction of transcripts per gene per cell type
- Each column (cell type) sums to 1.0

Usage:
    python build_sthd_lambda_file.py \
        --h5_file colorectal_sc_data/CRC_GSE166555_expression.h5 \
        --metadata_file colorectal_sc_data/CRC_GSE166555_CellMetainfo_table.tsv \
        --output lambda_colorectal.txt \
        --celltype_column "Celltype (major-lineage)"
"""

import pandas as pd
import numpy as np
import h5py
from scipy.sparse import csr_matrix
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sc_data(h5_file, metadata_file, celltype_column):
    """
    Load single-cell data from H5 file and metadata.

    Returns:
        genes: list of gene names
        barcodes: list of cell barcodes
        matrix: sparse CSR matrix (cells x genes)
        cell_types: pandas Series of cell type labels indexed by barcode
    """
    logger.info(f"Loading single-cell data from {h5_file}...")

    with h5py.File(h5_file, 'r') as f:
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

        # Create sparse matrix (cells x genes)
        # Note: stored as (genes x cells), so we need to transpose
        matrix = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

    logger.info(f"  Loaded {len(barcodes):,} cells x {len(gene_names):,} genes")
    logger.info(f"  Matrix shape: {matrix.shape}")

    # Load metadata
    logger.info(f"Loading metadata from {metadata_file}...")
    df_meta = pd.read_csv(metadata_file, sep='\t')

    # Check available columns
    logger.info(f"  Available columns: {list(df_meta.columns)}")

    if celltype_column not in df_meta.columns:
        raise ValueError(f"Column '{celltype_column}' not found in metadata. Available: {list(df_meta.columns)}")

    # Set index to Cell barcode
    if 'Cell' not in df_meta.columns:
        raise ValueError("Metadata must have a 'Cell' column with cell barcodes")

    df_meta.index = df_meta['Cell']

    # Get cell type labels
    cell_types = df_meta.loc[:, celltype_column]

    # Filter to cells that are in both matrix and metadata
    common_barcodes = list(set(barcodes) & set(cell_types.index))
    logger.info(f"  Found {len(common_barcodes):,} cells in both matrix and metadata")

    # Reorder to match
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    indices = [barcode_to_idx[bc] for bc in common_barcodes]

    matrix = matrix[indices, :]
    cell_types = cell_types.loc[common_barcodes]

    logger.info(f"  Cell types found:")
    for ct, count in cell_types.value_counts().items():
        logger.info(f"    {ct}: {count:,} cells")

    return gene_names, common_barcodes, matrix, cell_types


def compute_lambda(matrix, gene_names, cell_types):
    """
    Compute STHD lambda using per-cell gene normalization.

    For each cell type:
    1. Normalize each cell's gene counts to sum to 1 (fraction of transcripts)
    2. Average across all cells of that type

    This gives lambda[gene, celltype] = average fraction of transcripts
    that gene makes up in cells of that type.

    Reference: STHD/refscrna.py lines 24-26:
        genenorm = (genebycell/(genebycell.sum(axis=0)))  # per-cell normalization
        genenorm = genenorm.mean(axis=1)  # average across cells
    """
    logger.info("Computing STHD lambda (per-cell gene normalization, then average)...")

    unique_celltypes = list(cell_types.unique())
    n_genes = len(gene_names)
    n_celltypes = len(unique_celltypes)

    # Initialize result matrix
    lambda_matrix = np.zeros((n_genes, n_celltypes))

    for i, ct in enumerate(unique_celltypes):
        # Get cells of this type
        cell_mask = (cell_types == ct).values
        ct_matrix = matrix[cell_mask, :]  # cells x genes

        # Per-cell normalization: each cell's genes sum to 1
        cell_totals = np.asarray(ct_matrix.sum(axis=1)).flatten()
        cell_totals[cell_totals == 0] = 1  # avoid division by zero

        # Normalize each cell (row) by its total
        ct_array = ct_matrix.toarray()  # cells x genes
        cell_normalized = ct_array / cell_totals[:, np.newaxis]

        # Average across cells to get lambda for this cell type
        lambda_matrix[:, i] = cell_normalized.mean(axis=0)

        logger.info(f"    {ct}: {cell_mask.sum():,} cells, lambda range [{lambda_matrix[:, i].min():.6f}, {lambda_matrix[:, i].max():.6f}]")

    # Create DataFrame
    df_lambda = pd.DataFrame(
        lambda_matrix,
        index=gene_names,
        columns=unique_celltypes
    )

    # Verify: column sums should all be ~1.0
    col_sums = df_lambda.sum(axis=0)
    logger.info(f"  Column sums (should all be ~1.0):")
    for ct in unique_celltypes[:5]:
        logger.info(f"    {ct}: {col_sums[ct]:.6f}")

    return df_lambda


def save_lambda_file(df_lambda, output_file):
    """
    Save lambda file in STHD format.

    Format:
    - Tab-delimited
    - First row: empty cell, then cell type names
    - Subsequent rows: gene name, then expression values
    """
    logger.info(f"Saving lambda file to {output_file}...")

    df_lambda.to_csv(output_file, sep='\t', index=True, header=True)

    logger.info(f"  Saved {df_lambda.shape[0]:,} genes x {df_lambda.shape[1]} cell types")
    logger.info(f"  File size: {Path(output_file).stat().st_size / 1e6:.2f} MB")

    # Show preview
    logger.info("\n  Preview of output file (first 5 genes, first 3 cell types):")
    preview = df_lambda.iloc[:5, :3]
    logger.info("\n" + preview.to_string())


def main():
    parser = argparse.ArgumentParser(
        description='Build STHD lambda file from scRNA-seq data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--h5_file', type=str, required=True,
                       help='Path to H5 expression file')
    parser.add_argument('--metadata_file', type=str, required=True,
                       help='Path to metadata TSV file')
    parser.add_argument('--output', type=str, default='lambda_reference.txt',
                       help='Output lambda file (default: lambda_reference.txt)')
    parser.add_argument('--celltype_column', type=str, default='Celltype (major-lineage)',
                       help='Column name for cell type labels (default: "Celltype (major-lineage)")')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("Building STHD Lambda Reference File")
    logger.info("="*70)

    # Load data
    gene_names, barcodes, matrix, cell_types = load_sc_data(
        args.h5_file,
        args.metadata_file,
        args.celltype_column
    )

    # Compute lambda with per-cell normalization
    df_lambda = compute_lambda(matrix, gene_names, cell_types)

    # Save
    save_lambda_file(df_lambda, args.output)

    logger.info("\n=== Complete ===")
    logger.info(f"Lambda file saved to: {args.output}")


if __name__ == '__main__':
    main()
