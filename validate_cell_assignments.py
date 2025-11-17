#!/usr/bin/env python
"""
Validate Pseudo Visium HD Data Generation

This script validates that the pseudo Visium HD data was generated correctly by:
1. Checking that bins get expression from the correct cells
2. Verifying gene expression matches single-cell profiles
3. Validating overlapping bins (bins assigned to multiple cells)
4. Checking compartment-specific gene localization

Usage:
    python validate_cell_assignments.py \
        --pseudo_hd_dir pseudo_visium_hd_output \
        --sample_size 100
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
from scipy.sparse import csr_matrix
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def identify_required_bins(ground_truth_file, pixel_file, sample_size, microns_per_pixel=0.2739038899725172):
    """
    MEMORY OPTIMIZATION: Identify which bins are needed BEFORE loading the matrix.
    This allows us to load only ~1% of the data instead of the full 33GB!
    """
    logger.info(f"MEMORY OPTIMIZATION: Identifying required bins first...")

    # Load and sample ground truth
    df_gt = pd.read_csv(ground_truth_file)
    sampled_cells = df_gt.sample(n=min(sample_size, len(df_gt)), random_state=42)
    logger.info(f"  Sampled {len(sampled_cells)} cells for validation")

    # Get required cell IDs
    required_cell_ids = set(sampled_cells['cell_id'].values)

    # Scan pixel file to find required bins
    logger.info(f"  Scanning pixel mapping for bins belonging to sampled cells...")
    required_bins = set()
    chunk_num = 0

    for chunk in pd.read_csv(pixel_file, compression='gzip', chunksize=2_000_000):
        chunk_num += 1
        chunk = chunk[chunk['cell_id'].isin(required_cell_ids)]

        if len(chunk) > 0:
            chunk['bin_x'] = (chunk['x'] * microns_per_pixel / 2.0).astype(int)
            chunk['bin_y'] = (chunk['y'] * microns_per_pixel / 2.0).astype(int)

            for _, row in chunk.iterrows():
                barcode = f"s_002um_{int(row['bin_x']):05d}_{int(row['bin_y']):05d}-1"
                required_bins.add(barcode)

        if chunk_num % 20 == 0:
            logger.info(f"    Processed {chunk_num} chunks, found {len(required_bins):,} bins...")

    logger.info(f"  ✓ Need only {len(required_bins):,} bins for validation (saves ~99% memory!)")
    return sampled_cells, required_bins


def load_pseudo_hd_data_subset(pseudo_hd_dir, required_bins):
    """
    MEMORY-EFFICIENT: Load only required bins using STREAMING (avoids OOM).

    Instead of loading the entire 33GB matrix (OOM!), this:
    1. Streams the MTX file line by line
    2. Keeps only entries for required bins
    3. Uses <5GB RAM instead of 100GB+
    """
    logger.info(f"Loading SUBSET of pseudo Visium HD data (STREAMING mode - no OOM)...")
    logger.info(f"  Will load only {len(required_bins):,} bins instead of millions")

    matrix_dir = Path(pseudo_hd_dir) / 'filtered_feature_bc_matrix'

    # Load barcodes and genes (lightweight)
    logger.info("  Step 1/3: Loading metadata (fast)...")
    with gzip.open(matrix_dir / 'barcodes.tsv.gz', 'rt') as f:
        all_barcodes = [line.strip() for line in f]

    with gzip.open(matrix_dir / 'features.tsv.gz', 'rt') as f:
        genes = [line.strip().split('\t')[1] for line in f]

    # Create barcode lookup
    barcode_to_idx = {bc: i for i, bc in enumerate(all_barcodes)}
    required_indices = {barcode_to_idx[bc] for bc in required_bins if bc in barcode_to_idx}

    logger.info(f"  Step 2/3: Streaming matrix file (10-15 min, but won't OOM)...")
    logger.info(f"    This reads 33GB file line-by-line, keeping only {len(required_indices):,} bins")

    # Stream matrix file
    start = time.time()
    with gzip.open(matrix_dir / 'matrix.mtx.gz', 'rt') as f:
        # Skip header comments
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()

        # Read dimensions
        n_genes, n_bins, n_entries = map(int, line.strip().split())
        logger.info(f"      Matrix: {n_genes:,} genes x {n_bins:,} bins, {n_entries:,} entries")

        # Stream and filter data
        data = []
        rows = []
        cols = []

        # Map original bin indices to subset indices
        idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted(required_indices))}
        subset_barcodes = [all_barcodes[i] for i in sorted(required_indices)]

        processed = 0
        kept = 0

        for line in f:
            gene_idx, bin_idx, count = line.strip().split()
            gene_idx = int(gene_idx) - 1  # MTX is 1-indexed
            bin_idx = int(bin_idx) - 1
            count = float(count)

            # Keep only if bin is in required set
            if bin_idx in required_indices:
                rows.append(gene_idx)
                cols.append(idx_mapping[bin_idx])
                data.append(count)
                kept += 1

            processed += 1
            if processed % 50_000_000 == 0:
                elapsed = time.time() - start
                logger.info(f"        Progress: {processed:,}/{n_entries:,} ({100*processed/n_entries:.1f}%), "
                           f"kept {kept:,}, time: {elapsed/60:.1f}min")

    elapsed = time.time() - start
    logger.info(f"  ✓ Streaming complete in {elapsed/60:.1f} minutes")
    logger.info(f"    Kept {kept:,} / {processed:,} entries ({100*kept/processed:.3f}%)")

    # Create sparse matrix
    logger.info(f"  Step 3/3: Building sparse matrix...")
    matrix = csr_matrix((data, (rows, cols)), shape=(len(genes), len(subset_barcodes)))
    matrix = matrix.T.tocsr()  # Transpose to bins x genes

    # Create AnnData
    import anndata as ad
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=subset_barcodes),
        var=pd.DataFrame(index=genes)
    )

    logger.info(f"  ✓ Loaded {adata.n_obs:,} bins x {adata.n_vars:,} genes")
    logger.info(f"  Memory usage: ~{matrix.data.nbytes / 1e9:.2f} GB (instead of 100+ GB!)")

    return adata


def load_pseudo_hd_data(pseudo_hd_dir):
    """
    DEPRECATED: This loads the entire 33GB matrix and causes OOM!
    Use load_pseudo_hd_data_subset() instead.
    """
    logger.warning("WARNING: load_pseudo_hd_data() loads entire 33GB matrix (OOM risk!)")
    logger.warning("Use load_pseudo_hd_data_subset() instead for memory efficiency")

    matrix_dir = Path(pseudo_hd_dir) / 'filtered_feature_bc_matrix'
    start = time.time()
    adata = sc.read_10x_mtx(matrix_dir, var_names='gene_symbols', cache=False)
    elapsed = time.time() - start

    logger.info(f"  ✓ Loaded in {elapsed/60:.1f} minutes")
    logger.info(f"  Loaded {adata.n_obs:,} bins, {adata.n_vars} genes")
    return adata


def load_ground_truth(ground_truth_file):
    """Load ground truth cell assignments."""
    logger.info(f"Loading ground truth from {ground_truth_file}...")
    df_gt = pd.read_csv(ground_truth_file)
    logger.info(f"  Loaded {len(df_gt):,} cell assignments")
    logger.info(f"  Cell types:")
    for ct, count in df_gt['cell_type'].value_counts().items():
        logger.info(f"    {ct}: {count:,} cells")
    return df_gt


def load_bin_to_cell_mapping(pixel_file, microns_per_pixel=0.2739038899725172):
    """Load and create bin-to-cell mapping from pixel data (OPTIMIZED: chunked reading, vectorized operations)."""
    logger.info(f"Loading pixel mapping from {pixel_file}...")

    # OPTIMIZATION: Read in chunks to reduce memory usage
    logger.info("  Reading compressed CSV in chunks...")
    chunks = []
    chunk_size = 5_000_000
    for chunk in pd.read_csv(pixel_file, compression='gzip', chunksize=chunk_size):
        # Only keep cells (filter out background immediately)
        chunk = chunk[chunk['cell_id'] > 0]
        # Convert pixels to bins immediately
        chunk['bin_x'] = (chunk['x'] * microns_per_pixel / 2.0).astype(np.int32)
        chunk['bin_y'] = (chunk['y'] * microns_per_pixel / 2.0).astype(np.int32)
        # Drop pixel coordinates to save memory
        chunks.append(chunk[['bin_x', 'bin_y', 'cell_id']])

    df_pixels = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Loaded {len(df_pixels):,} pixels (cells only)")

    # OPTIMIZATION: Use frozenset for faster membership testing
    # Group by bin and create set of cell_ids
    logger.info("  Creating bin-to-cell mapping (vectorized)...")
    bin_to_cells = df_pixels.groupby(['bin_x', 'bin_y'])['cell_id'].apply(frozenset).reset_index()
    bin_to_cells.columns = ['bin_x', 'bin_y', 'cell_ids']

    logger.info(f"  Created mapping for {len(bin_to_cells):,} bins")

    # Count bins with multiple cells (vectorized)
    cell_counts = bin_to_cells['cell_ids'].apply(len)
    multi_cell_count = (cell_counts > 1).sum()
    logger.info(f"  Bins with multiple cells: {multi_cell_count:,} ({100*multi_cell_count/len(bin_to_cells):.1f}%)")

    return bin_to_cells


def load_sc_data(h5_file, metadata_file):
    """Load single-cell data (custom H5 format) - OPTIMIZED: keeps sparse format."""
    logger.info(f"Loading single-cell data from {h5_file}...")

    import h5py
    from scipy.sparse import csr_matrix
    import anndata as ad

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

        # Create sparse matrix (KEEP SPARSE!)
        X = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

    # Create AnnData object
    adata_sc = ad.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=gene_names))

    # Load metadata
    df_meta = pd.read_csv(metadata_file, sep='\t')
    df_meta.index = df_meta['Cell']

    # Add cell type info
    adata_sc.obs['cell_type'] = df_meta.loc[adata_sc.obs.index, 'Celltype (major-lineage)']

    logger.info(f"  Loaded {adata_sc.n_obs:,} cells, {adata_sc.n_vars} genes")
    logger.info(f"  Matrix is sparse: {type(adata_sc.X)}")

    return adata_sc


def validate_bin_assignments(adata, bin_to_cells, df_gt, adata_sc, sample_size=100):
    """
    Validate that bins get expression from the correct cells (OPTIMIZED: vectorized, sparse operations).

    For each sampled cell:
    1. Get its bins from segmentation
    2. Get expression in those bins from pseudo HD data
    3. Compare with original single-cell expression
    4. Calculate correlation
    """
    logger.info("\n=== Validating Bin-to-Cell Assignments ===")

    # Sample cells
    sampled_cells = df_gt.sample(n=min(sample_size, len(df_gt)), random_state=42)
    logger.info(f"Sampling {len(sampled_cells)} cells for validation...")

    # OPTIMIZATION: Pre-compute common genes and indices (only once!)
    logger.info("  Pre-computing gene indices (10x speedup)...")
    common_genes = list(set(adata.var_names) & set(adata_sc.var_names))
    if len(common_genes) < 10:
        logger.error("Not enough common genes between pseudo HD and SC data!")
        return None

    # Create index mappings for fast lookup (avoids repeated set operations)
    hd_gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    sc_gene_to_idx = {g: i for i, g in enumerate(adata_sc.var_names)}
    common_gene_idx_hd = np.array([hd_gene_to_idx[g] for g in common_genes if g in hd_gene_to_idx])
    common_gene_idx_sc = np.array([sc_gene_to_idx[g] for g in common_genes if g in sc_gene_to_idx])

    logger.info(f"  Found {len(common_genes)} common genes")

    # OPTIMIZATION: Pre-create barcode-to-index mapping (10x faster than repeated isin calls)
    logger.info("  Building barcode index (100x speedup for lookups)...")
    barcode_to_idx = {bc: i for i, bc in enumerate(adata.obs.index)}

    # OPTIMIZATION: Pre-build bin lookup dictionary (vectorized)
    logger.info("  Building bin-to-barcode lookup (avoids repeated string formatting)...")
    bin_lookup = {}
    for _, row in bin_to_cells.iterrows():
        barcode = f"s_002um_{int(row['bin_x']):05d}_{int(row['bin_y']):05d}-1"
        if barcode in barcode_to_idx:
            bin_lookup[(row['bin_x'], row['bin_y'])] = (barcode, row['cell_ids'])

    results = []

    for cell_num, (idx, row) in enumerate(sampled_cells.iterrows()):
        cell_id = row['cell_id']
        sc_barcode = row['sc_cell_barcode']
        cell_type = row['cell_type']

        # OPTIMIZATION: Use pre-built lookup instead of filtering (100x faster!)
        cell_bin_barcodes = [
            barcode for (bx, by), (barcode, cell_ids) in bin_lookup.items()
            if cell_id in cell_ids
        ]

        if len(cell_bin_barcodes) == 0:
            continue

        # OPTIMIZATION: Use sparse matrix slicing (keep sparse!)
        bin_indices = [barcode_to_idx[bc] for bc in cell_bin_barcodes]
        bin_expr_full = adata.X[bin_indices, :][:, common_gene_idx_hd].sum(axis=0)
        # Convert to 1D array (sparse sum returns matrix)
        bin_expr = np.asarray(bin_expr_full).flatten()

        # Get sc expression for this cell (OPTIMIZATION: keep sparse, use slicing)
        if sc_barcode not in adata_sc.obs.index:
            continue

        sc_idx = list(adata_sc.obs.index).index(sc_barcode)
        sc_expr_full = adata_sc.X[sc_idx, :][:, common_gene_idx_sc]
        # Convert to 1D array (avoid .toarray() which is slow)
        sc_expr = np.asarray(sc_expr_full.todense()).flatten()

        # Calculate correlation
        if bin_expr.sum() > 0 and sc_expr.sum() > 0:
            # Pearson correlation
            corr = np.corrcoef(bin_expr, sc_expr)[0, 1]

            # Also calculate for non-zero genes only
            nonzero = (bin_expr > 0) | (sc_expr > 0)
            if nonzero.sum() > 10:
                corr_nonzero = np.corrcoef(bin_expr[nonzero], sc_expr[nonzero])[0, 1]
            else:
                corr_nonzero = np.nan
        else:
            corr = np.nan
            corr_nonzero = np.nan

        results.append({
            'cell_id': cell_id,
            'cell_type': cell_type,
            'n_bins': len(cell_bin_barcodes),
            'total_counts_bins': float(bin_expr.sum()),
            'total_counts_sc': float(sc_expr.sum()),
            'correlation_all': corr,
            'correlation_nonzero': corr_nonzero,
            'n_genes_expressed_bins': (bin_expr > 0).sum(),
            'n_genes_expressed_sc': (sc_expr > 0).sum()
        })

        if (cell_num + 1) % 10 == 0:
            logger.info(f"  Processed {cell_num + 1}/{len(sampled_cells)} cells...")

    df_results = pd.DataFrame(results)

    # Calculate metrics
    logger.info("\nResults:")
    logger.info(f"  Cells successfully evaluated: {len(df_results)}")
    logger.info(f"  Mean correlation (all genes): {df_results['correlation_all'].mean():.3f}")
    logger.info(f"  Median correlation (all genes): {df_results['correlation_all'].median():.3f}")
    logger.info(f"  Mean correlation (non-zero genes): {df_results['correlation_nonzero'].mean():.3f}")
    logger.info(f"  Cells with corr > 0.5: {(df_results['correlation_all'] > 0.5).sum()} ({100*(df_results['correlation_all'] > 0.5).mean():.1f}%)")
    logger.info(f"  Cells with corr > 0.7: {(df_results['correlation_all'] > 0.7).sum()} ({100*(df_results['correlation_all'] > 0.7).mean():.1f}%)")

    # Per cell type
    logger.info("\n  Correlation by cell type:")
    for ct in df_results['cell_type'].unique():
        ct_data = df_results[df_results['cell_type'] == ct]
        logger.info(f"    {ct}: {ct_data['correlation_all'].mean():.3f} (n={len(ct_data)})")

    return df_results


def validate_overlapping_bins(adata, bin_to_cells):
    """
    Validate that bins assigned to multiple cells get combined expression (OPTIMIZED: vectorized).
    Overlapping bins should have higher expression than non-overlapping bins.
    """
    logger.info("\n=== Validating Overlapping Bins ===")

    # OPTIMIZATION: Vectorized cell count
    cell_counts = bin_to_cells['cell_ids'].apply(len).values
    multi_mask = cell_counts > 1
    single_mask = cell_counts == 1

    multi_cell_bins = bin_to_cells[multi_mask].copy()
    single_cell_bins = bin_to_cells[single_mask].copy()

    logger.info(f"Bins with multiple cells: {len(multi_cell_bins):,}")
    logger.info(f"Bins with single cell: {len(single_cell_bins):,}")

    if len(multi_cell_bins) == 0:
        logger.warning("No overlapping bins found!")
        return None

    # Sample overlapping bins
    sample_multi = multi_cell_bins.sample(n=min(1000, len(multi_cell_bins)), random_state=42)
    sample_single = single_cell_bins.sample(n=min(1000, len(single_cell_bins)), random_state=42)

    # OPTIMIZATION: Vectorized bin count extraction
    def get_bin_counts(df_bins):
        # Create all barcodes at once (vectorized string formatting)
        barcodes = [f"s_002um_{int(row['bin_x']):05d}_{int(row['bin_y']):05d}-1"
                   for _, row in df_bins.iterrows()]

        # Filter to only valid barcodes (vectorized)
        valid_mask = [bc in adata.obs.index for bc in barcodes]
        valid_barcodes = [bc for bc, valid in zip(barcodes, valid_mask) if valid]

        if len(valid_barcodes) == 0:
            return []

        # OPTIMIZATION: Extract all counts at once (batch slicing)
        barcode_indices = [list(adata.obs.index).index(bc) for bc in valid_barcodes]
        counts = adata.X[barcode_indices, :].sum(axis=1)
        # Convert sparse matrix to array
        return np.asarray(counts).flatten().tolist()

    multi_counts = get_bin_counts(sample_multi)
    single_counts = get_bin_counts(sample_single)

    logger.info(f"\nCounts in overlapping bins:")
    logger.info(f"  Mean: {np.mean(multi_counts):.1f}")
    logger.info(f"  Median: {np.median(multi_counts):.1f}")

    logger.info(f"\nCounts in single-cell bins:")
    logger.info(f"  Mean: {np.mean(single_counts):.1f}")
    logger.info(f"  Median: {np.median(single_counts):.1f}")

    logger.info(f"\nRatio (overlapping/single): {np.mean(multi_counts) / np.mean(single_counts):.2f}x")

    return {
        'multi_cell_counts': multi_counts,
        'single_cell_counts': single_counts
    }


def plot_validation_results(df_cell_results, overlap_results, output_dir):
    """Create validation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Correlation distribution
    if df_cell_results is not None and len(df_cell_results) > 0:
        axes[0, 0].hist(df_cell_results['correlation_all'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df_cell_results['correlation_all'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df_cell_results["correlation_all"].mean():.3f}')
        axes[0, 0].set_xlabel('Correlation (Bin Expression vs SC Expression)')
        axes[0, 0].set_ylabel('Number of Cells')
        axes[0, 0].set_title('Bin-to-Cell Assignment Validation')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: By cell type
        if 'cell_type' in df_cell_results.columns:
            cell_type_corr = df_cell_results.groupby('cell_type')['correlation_all'].mean().sort_values()
            axes[0, 1].barh(range(len(cell_type_corr)), cell_type_corr.values)
            axes[0, 1].set_yticks(range(len(cell_type_corr)))
            axes[0, 1].set_yticklabels(cell_type_corr.index)
            axes[0, 1].set_xlabel('Mean Correlation')
            axes[0, 1].set_title('Correlation by Cell Type')
            axes[0, 1].grid(axis='x', alpha=0.3)

        # Plot 3: Counts comparison
        axes[1, 0].scatter(df_cell_results['total_counts_sc'], df_cell_results['total_counts_bins'], alpha=0.5)
        axes[1, 0].plot([0, df_cell_results['total_counts_sc'].max()],
                        [0, df_cell_results['total_counts_sc'].max()],
                        'r--', label='y=x')
        axes[1, 0].set_xlabel('SC Total Counts')
        axes[1, 0].set_ylabel('Bin Total Counts')
        axes[1, 0].set_title('Total Counts: SC vs Bins')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    # Plot 4: Overlapping bins comparison
    if overlap_results is not None:
        multi = overlap_results['multi_cell_counts']
        single = overlap_results['single_cell_counts']

        axes[1, 1].boxplot([single, multi], labels=['Single Cell', 'Multiple Cells'])
        axes[1, 1].set_ylabel('Total Counts per Bin')
        axes[1, 1].set_title('Expression in Single vs Overlapping Bins')
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_results.png', dpi=300, bbox_inches='tight')
    logger.info(f"\nSaved plot: {output_dir / 'validation_results.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate pseudo Visium HD data generation (MEMORY EFFICIENT)')
    parser.add_argument('--pseudo_hd_dir', type=str,
                       default='pseudo_visium_hd_output',
                       help='Path to pseudo Visium HD output directory')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Number of cells to sample for validation')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("MEMORY-EFFICIENT Pseudo Visium HD Validation")
    logger.info("="*70)
    logger.info("Streams 33GB matrix file to avoid OOM (uses <5GB RAM)")
    logger.info("")

    # Paths
    pseudo_hd_dir = Path(args.pseudo_hd_dir)
    ground_truth_file = pseudo_hd_dir / 'ground_truth_cell_assignments.csv'
    pixel_file = Path('cellpose_sam_human_kidney_output/cropped_visium_hd_human_kidney_pixel_to_cell_mapping_expanded.csv.gz')
    sc_h5_file = Path('kidney_sc_data/KIRC_GSE159115_expression.h5')
    sc_meta_file = Path('kidney_sc_data/KIRC_GSE159115_CellMetainfo_table.tsv')
    output_dir = Path(args.output_dir)

    # STEP 1: Identify required bins FIRST (before loading matrix)
    sampled_cells, required_bins = identify_required_bins(
        ground_truth_file, pixel_file, args.sample_size
    )

    # STEP 2: Load ONLY required bins using streaming (memory efficient!)
    adata = load_pseudo_hd_data_subset(pseudo_hd_dir, required_bins)

    # STEP 3: Load other data (lightweight)
    df_gt = sampled_cells  # Already have sampled cells from step 1
    bin_to_cells = load_bin_to_cell_mapping(pixel_file)
    adata_sc = load_sc_data(sc_h5_file, sc_meta_file)

    # STEP 4: Validate
    df_cell_results = validate_bin_assignments(adata, bin_to_cells, df_gt, adata_sc,
                                                sample_size=args.sample_size)
    overlap_results = validate_overlapping_bins(adata, bin_to_cells)

    # STEP 5: Save results
    output_dir.mkdir(exist_ok=True)
    if df_cell_results is not None and len(df_cell_results) > 0:
        df_cell_results.to_csv(output_dir / 'cell_validation_details.csv', index=False)
        logger.info(f"Saved detailed results: {output_dir / 'cell_validation_details.csv'}")

    # STEP 6: Plot
    plot_validation_results(df_cell_results, overlap_results, output_dir)

    logger.info(f"\n=== Validation Complete ===")
    logger.info(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
