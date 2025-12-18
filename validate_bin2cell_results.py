#!/usr/bin/env python
"""
Validate Bin2Cell Results Against Ground Truth

This script validates Bin2Cell cell segmentation and gene assignment results by comparing:

VALIDATION STRATEGY:
1. Cell Matching: Uses NUCLEAR overlap to match ground truth cells to Bin2Cell cells
   - More accurate because nuclei are distinct and don't overlap
   - Finds best Bin2Cell nucleus for each ground truth nucleus
   - ANY nuclear bin overlap counts as a correct match (ground truth)

2. Spatial Validation: Calculates WHOLE CELL overlap for matched cell pairs
   - Once cells are matched by nuclei, validates the whole cell boundary detection
   - Measures IoU (Intersection over Union) for complete cell regions

3. Gene Correlation: Compares gene expression profiles between matched cells
   - For each matched cell pair, calculates Pearson and Spearman correlation
   - Uses ground truth single-cell expression vs Bin2Cell reconstructed expression

Bin2Cell Outputs Used:
- bin2cell_results/human_colorectal_bin2cell_assignments.csv: Bin-to-cell assignments
- bin2cell_results/human_colorectal_bin2cell_cells.h5ad: Cell-level gene expression data
- bin2cell_results/human_colorectal_bin2cell_cell_summary.csv: Cell summary statistics

Ground Truth Data:
- pseudo_visium_hd_outpu_full/ground_truth_cell_assignments.csv: Maps cell_id -> sc_cell_barcode
- cellpose_sam_human_kidney_output/...pixel_to_cell_mapping_expanded.csv.gz: Maps pixels -> cell_id (with compartment info)
- kidney_sc_data/KIRC_GSE159115_expression.h5: Single-cell ground truth gene expression

Usage:
    python validate_bin2cell_results.py \
        --bin2cell_dir bin2cell_results_colorectal \
        --pseudo_hd_dir pseudo_visium_hd_outpu_full \
        --pixel_file cellpose_sam_human_colorectal_output/cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_expanded.csv.gz \
        --sc_h5_file colorectal_sc_data/CRC_GSE166555_expression.h5 \
        --sc_meta_file colorectal_sc_data/CRC_GSE166555_CellMetainfo_table.tsv \
        --output_dir bin2cell_validation_output
"""

import numpy as np
import pandas as pd
import scanpy as sc
import logging
import sys
import argparse
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gzip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validate_bin2cell_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_bin2cell_results(bin2cell_dir, microns_per_pixel=0.2739038899725172):
    """
    Load Bin2Cell output files.

    Returns:
        bin2cell_nuclear_bins: Dict mapping cell_id -> set of nuclear (bin_x, bin_y)
        bin2cell_all_bins: Dict mapping cell_id -> set of all (bin_x, bin_y)
        adata_bin2cell: AnnData with final cell-level gene expression
    """
    logger.info("Loading Bin2Cell results...")
    bin2cell_path = Path(bin2cell_dir)

    # Load bin-to-cell assignments
    logger.info("  Loading bin2cell_assignments.csv...")
    assignments_file = list(bin2cell_path.glob('*_bin2cell_assignments.csv'))[0]
    df_assignments = pd.read_csv(assignments_file)
    logger.info(f"    Loaded {len(df_assignments):,} bin assignments")

    # Load cell-level gene expression
    logger.info("  Loading bin2cell_cells.h5ad (cell gene expression)...")
    cells_file = list(bin2cell_path.glob('*_bin2cell_cells.h5ad'))[0]
    adata_bin2cell = sc.read_h5ad(cells_file)
    logger.info(f"    Loaded {adata_bin2cell.n_obs} cells, {adata_bin2cell.n_vars} genes")

    # Load cell summary
    logger.info("  Loading bin2cell_cell_summary.csv...")
    summary_file = list(bin2cell_path.glob('*_bin2cell_cell_summary.csv'))[0]
    df_summary = pd.read_csv(summary_file)
    logger.info(f"    Loaded summary for {len(df_summary)} cells")

    # Build bin sets from assignments
    logger.info("  Building nuclear and whole cell bin sets...")
    bin_size = 2.0  # 2 micron bins

    bin2cell_nuclear_bins = defaultdict(set)
    bin2cell_all_bins = defaultdict(set)

    for _, row in df_assignments.iterrows():
        cell_id = row['cell_id']
        # FIX: Swap coordinates to match ground truth coordinate system
        # Ground truth uses (x, y) where x is the wider dimension
        # Bin2Cell uses (array_row, array_col) where array_col is the wider dimension
        # Therefore: x = array_row, y = array_col (SWAPPED from original)
        bin_x = row['array_row']  # FIXED: was array_col
        bin_y = row['array_col']  # FIXED: was array_row
        is_nuclear = row['is_nuclear']

        bin2cell_all_bins[cell_id].add((bin_x, bin_y))
        if is_nuclear:
            bin2cell_nuclear_bins[cell_id].add((bin_x, bin_y))

    logger.info(f"  ✓ Built bin sets for {len(bin2cell_all_bins)} cells")
    logger.info(f"    Cells with nuclear bins: {len(bin2cell_nuclear_bins)}")
    logger.info(f"    Average bins per cell: {np.mean([len(v) for v in bin2cell_all_bins.values()]):.1f}")
    logger.info(f"    Average nuclear bins per cell: {np.mean([len(v) for v in bin2cell_nuclear_bins.values() if len(v) > 0]):.1f}")

    return bin2cell_nuclear_bins, bin2cell_all_bins, adata_bin2cell


def load_ground_truth(gt_file, pixel_file, sc_h5_path, sc_metadata_path):
    """
    Load ground truth data.

    Returns:
        df_gt: Ground truth cell assignments (cell_id -> sc_cell_barcode)
        df_pixels: Pixel-to-cell mapping (x, y -> cell_id)
        adata_sc_gt: Single-cell ground truth gene expression
    """
    logger.info("Loading ground truth data...")

    # Load ground truth cell assignments
    logger.info("  Loading ground_truth_cell_assignments.csv...")
    df_gt = pd.read_csv(gt_file)
    logger.info(f"    Ground truth cells: {len(df_gt)}")
    logger.info(f"    Cell types: {df_gt['cell_type'].value_counts().to_dict()}")

    # Load pixel-to-cell mapping (this is large, may need chunking)
    logger.info("  Loading pixel-to-cell mapping (this may take a while)...")
    df_pixels = pd.read_csv(pixel_file, compression='gzip')
    logger.info(f"    Total pixels: {len(df_pixels)}")

    # Load single-cell ground truth gene expression
    logger.info("  Loading single-cell ground truth gene expression...")
    import h5py

    with h5py.File(sc_h5_path, 'r') as f:
        matrix_group = f['matrix']
        data = matrix_group['data'][:]
        indices = matrix_group['indices'][:]
        indptr = matrix_group['indptr'][:]
        shape = matrix_group['shape'][:]

        gene_names = matrix_group['features']['name'][:]
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in gene_names]

        barcodes = matrix_group['barcodes'][:]
        barcodes = [b.decode() if isinstance(b, bytes) else b for b in barcodes]

        X = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

    adata_sc_gt = sc.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=gene_names))

    # Add metadata
    metadata = pd.read_csv(sc_metadata_path, sep='\t')
    cell_col = metadata.columns[0]
    metadata = metadata.set_index(cell_col)
    adata_sc_gt.obs = adata_sc_gt.obs.join(metadata, how='left')

    logger.info(f"    Single cells: {adata_sc_gt.n_obs}")
    logger.info(f"    Genes: {adata_sc_gt.n_vars}")

    return df_gt, df_pixels, adata_sc_gt


def build_ground_truth_cell_regions(df_pixels, microns_per_pixel=0.2739038899725172):
    """
    Build bin sets for each ground truth cell.
    Returns both nuclear regions (for matching) and whole cell regions (for validation).

    Returns:
        cell_nuclear_dict: Dictionary mapping cell_id -> set of nuclear (bin_x, bin_y) tuples
        cell_whole_dict: Dictionary mapping cell_id -> set of whole cell (bin_x, bin_y) tuples
    """
    logger.info("Building ground truth cell regions...")

    cell_nuclear_dict = defaultdict(set)
    cell_whole_dict = defaultdict(set)

    # Convert pixels to bins (2μm) - vectorized
    df_pixels['bin_x'] = (df_pixels['x'] * microns_per_pixel / 2.0).astype(int)
    df_pixels['bin_y'] = (df_pixels['y'] * microns_per_pixel / 2.0).astype(int)

    # Filter to only cells (vectorized)
    df_cells = df_pixels[df_pixels['cell_id'] > 0].copy()
    logger.info(f"  Processing {len(df_cells):,} pixels from {df_cells['cell_id'].nunique()} cells...")

    # Build whole cell regions
    logger.info("  Building whole cell regions...")
    for cell_id, group in df_cells.groupby('cell_id'):
        coords = set(zip(group['bin_x'].values, group['bin_y'].values))
        cell_whole_dict[cell_id] = coords

    # Build nuclear regions
    logger.info("  Building nuclear regions...")
    if 'is_nuclear' in df_cells.columns:
        df_nuclear = df_cells[df_cells['is_nuclear'] == 1]
        for cell_id, group in df_nuclear.groupby('cell_id'):
            coords = set(zip(group['bin_x'].values, group['bin_y'].values))
            cell_nuclear_dict[cell_id] = coords
    else:
        logger.warning("  No 'is_nuclear' column found in pixel mapping!")

    logger.info(f"  ✓ Built regions for {len(cell_whole_dict)} cells")
    logger.info(f"    Cells with nuclear regions: {len(cell_nuclear_dict)}")

    return cell_nuclear_dict, cell_whole_dict


def match_cells_by_nuclear_overlap(gt_nuclear_regions, bin2cell_nuclear_regions, overlap_threshold=0.0):
    """
    Match ground truth cells to Bin2Cell cells based on nuclear overlap.

    Strategy: For each GT cell nucleus, find the Bin2Cell nucleus with maximum overlap.
    Any overlap counts as a match (threshold=0.0 for ground truth validation).

    Returns:
        cell_matches: Dict mapping gt_cell_id -> bin2cell_cell_id
        match_df: DataFrame with detailed matching statistics
    """
    logger.info(f"\nMatching cells by NUCLEAR overlap (threshold={overlap_threshold})...")
    logger.info("  Strategy: ANY nuclear bin overlap = correct match (ground truth validation)")

    cell_matches = {}
    match_details = []

    for gt_cell_id, gt_nuclear_bins in gt_nuclear_regions.items():
        if len(gt_nuclear_bins) == 0:
            continue

        best_match = None
        best_overlap = 0
        best_iou = 0

        # Find Bin2Cell cell with maximum nuclear overlap
        for bin2cell_id, bin2cell_nuclear_bins in bin2cell_nuclear_regions.items():
            if len(bin2cell_nuclear_bins) == 0:
                continue

            overlap_bins = gt_nuclear_bins & bin2cell_nuclear_bins
            overlap_count = len(overlap_bins)

            if overlap_count > best_overlap:
                best_overlap = overlap_count
                union_count = len(gt_nuclear_bins | bin2cell_nuclear_bins)
                best_iou = overlap_count / union_count if union_count > 0 else 0
                best_match = bin2cell_id

        # Any overlap counts as a match for ground truth validation
        if best_overlap > 0:
            cell_matches[gt_cell_id] = best_match
            match_details.append({
                'gt_cell_id': gt_cell_id,
                'bin2cell_cell_id': best_match,
                'nuclear_overlap_bins': best_overlap,
                'nuclear_iou': best_iou,
                'gt_nuclear_bins': len(gt_nuclear_bins),
                'bin2cell_nuclear_bins': len(bin2cell_nuclear_regions[best_match])
            })

    match_df = pd.DataFrame(match_details)

    logger.info(f"\n  NUCLEAR MATCHING RESULTS:")
    logger.info(f"    Ground truth cells with nuclei: {len(gt_nuclear_regions)}")
    logger.info(f"    Bin2Cell cells with nuclei: {len(bin2cell_nuclear_regions)}")
    logger.info(f"    Successfully matched: {len(cell_matches)} ({100*len(cell_matches)/len(gt_nuclear_regions):.1f}%)")

    if len(match_df) > 0:
        logger.info(f"    Mean nuclear IoU: {match_df['nuclear_iou'].mean():.3f}")
        logger.info(f"    Median nuclear IoU: {match_df['nuclear_iou'].median():.3f}")
        logger.info(f"    Nuclear IoU > 0.5: {(match_df['nuclear_iou'] > 0.5).sum()} ({100*(match_df['nuclear_iou'] > 0.5).mean():.1f}%)")

    return cell_matches, match_df


def calculate_spatial_overlap(gt_whole_regions, bin2cell_whole_regions, cell_matches):
    """
    Calculate WHOLE CELL spatial overlap (IoU) for matched cell pairs.

    This validates how well Bin2Cell reconstructed the complete cell boundaries,
    AFTER matching cells by their nuclei.
    """
    logger.info("\nCalculating WHOLE CELL spatial overlap for matched cells...")

    overlap_results = []

    for gt_cell_id, bin2cell_cell_id in cell_matches.items():
        gt_bins = gt_whole_regions.get(gt_cell_id, set())
        bin2cell_bins = bin2cell_whole_regions.get(bin2cell_cell_id, set())

        if len(gt_bins) == 0 or len(bin2cell_bins) == 0:
            continue

        # Calculate IoU for whole cell
        intersection = gt_bins & bin2cell_bins
        union = gt_bins | bin2cell_bins

        iou = len(intersection) / len(union) if len(union) > 0 else 0
        precision = len(intersection) / len(bin2cell_bins) if len(bin2cell_bins) > 0 else 0
        recall = len(intersection) / len(gt_bins) if len(gt_bins) > 0 else 0

        overlap_results.append({
            'gt_cell_id': gt_cell_id,
            'bin2cell_cell_id': bin2cell_cell_id,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'gt_bins': len(gt_bins),
            'bin2cell_bins': len(bin2cell_bins),
            'intersection_bins': len(intersection)
        })

    df_overlap = pd.DataFrame(overlap_results)

    if len(df_overlap) > 0:
        logger.info(f"\n  WHOLE CELL OVERLAP RESULTS:")
        logger.info(f"    Matched cells evaluated: {len(df_overlap)}")
        logger.info(f"    Mean IoU: {df_overlap['iou'].mean():.3f}")
        logger.info(f"    Median IoU: {df_overlap['iou'].median():.3f}")
        logger.info(f"    Mean Precision: {df_overlap['precision'].mean():.3f}")
        logger.info(f"    Mean Recall: {df_overlap['recall'].mean():.3f}")
        logger.info(f"    IoU > 0.5: {(df_overlap['iou'] > 0.5).sum()} ({100*(df_overlap['iou'] > 0.5).mean():.1f}%)")
        logger.info(f"    IoU > 0.7: {(df_overlap['iou'] > 0.7).sum()} ({100*(df_overlap['iou'] > 0.7).mean():.1f}%)")

    return df_overlap


def calculate_gene_correlation(df_gt, adata_sc_gt, adata_bin2cell, df_overlap):
    """
    Calculate gene expression correlation between ground truth and Bin2Cell for matched cells.
    """
    logger.info("\nCalculating gene expression correlation...")

    # Find common genes
    common_genes = list(set(adata_sc_gt.var_names) & set(adata_bin2cell.var_names))
    logger.info(f"  Common genes: {len(common_genes)}")

    if len(common_genes) < 10:
        logger.warning("  Not enough common genes for correlation!")
        return pd.DataFrame()

    # Create gene index mappings
    gt_gene_idx = {g: i for i, g in enumerate(adata_sc_gt.var_names)}
    bin2cell_gene_idx = {g: i for i, g in enumerate(adata_bin2cell.var_names)}

    common_gt_idx = [gt_gene_idx[g] for g in common_genes]
    common_bin2cell_idx = [bin2cell_gene_idx[g] for g in common_genes]

    correlation_results = []

    for _, row in df_overlap.iterrows():
        gt_cell_id = row['gt_cell_id']
        bin2cell_cell_id = row['bin2cell_cell_id']

        # Get ground truth sc_cell_barcode
        gt_row = df_gt[df_gt['cell_id'] == gt_cell_id]
        if len(gt_row) == 0:
            continue

        sc_barcode = gt_row.iloc[0]['sc_cell_barcode']
        cell_type = gt_row.iloc[0]['cell_type']

        # Get ground truth expression
        if sc_barcode not in adata_sc_gt.obs.index:
            continue

        gt_expr = adata_sc_gt[sc_barcode, :].X[:, common_gt_idx].toarray().flatten()

        # Get Bin2Cell expression
        bin2cell_cell_str = str(int(bin2cell_cell_id))  # Convert to int first to remove .0
        if bin2cell_cell_str not in adata_bin2cell.obs.index:
            continue

        bin2cell_expr = adata_bin2cell[bin2cell_cell_str, :].X[:, common_bin2cell_idx].toarray().flatten()

        # Calculate correlations
        if gt_expr.sum() > 0 and bin2cell_expr.sum() > 0:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(gt_expr, bin2cell_expr)

            # Spearman correlation
            spearman_r, spearman_p = spearmanr(gt_expr, bin2cell_expr)

            # Non-zero gene correlation
            nonzero_mask = (gt_expr > 0) | (bin2cell_expr > 0)
            if nonzero_mask.sum() > 10:
                pearson_nonzero, _ = pearsonr(gt_expr[nonzero_mask], bin2cell_expr[nonzero_mask])
            else:
                pearson_nonzero = np.nan

            correlation_results.append({
                'gt_cell_id': gt_cell_id,
                'bin2cell_cell_id': bin2cell_cell_id,
                'cell_type': cell_type,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pearson_nonzero': pearson_nonzero,
                'gt_total_counts': gt_expr.sum(),
                'bin2cell_total_counts': bin2cell_expr.sum(),
                'gt_genes_detected': (gt_expr > 0).sum(),
                'bin2cell_genes_detected': (bin2cell_expr > 0).sum()
            })

    df_corr = pd.DataFrame(correlation_results)

    if len(df_corr) > 0:
        logger.info(f"\n  GENE CORRELATION RESULTS:")
        logger.info(f"    Cells with valid correlations: {len(df_corr)}")
        logger.info(f"    Mean Pearson r: {df_corr['pearson_r'].mean():.3f}")
        logger.info(f"    Median Pearson r: {df_corr['pearson_r'].median():.3f}")
        logger.info(f"    Mean Spearman r: {df_corr['spearman_r'].mean():.3f}")
        logger.info(f"    Pearson r > 0.5: {(df_corr['pearson_r'] > 0.5).sum()} ({100*(df_corr['pearson_r'] > 0.5).mean():.1f}%)")
        logger.info(f"    Pearson r > 0.7: {(df_corr['pearson_r'] > 0.7).sum()} ({100*(df_corr['pearson_r'] > 0.7).mean():.1f}%)")

        # Per cell type
        logger.info(f"\n  Correlation by cell type:")
        for ct in df_corr['cell_type'].unique():
            ct_data = df_corr[df_corr['cell_type'] == ct]
            logger.info(f"    {ct}: Pearson={ct_data['pearson_r'].mean():.3f}, n={len(ct_data)}")

    return df_corr


def assign_ground_truth_annotations(cell_matches, df_gt, adata_bin2cell):
    """
    Assign ground truth cell type and metadata to matched Bin2Cell cells.
    """
    logger.info("\nAssigning ground truth annotations to Bin2Cell cells...")

    # Create reverse mapping: bin2cell_cell_id -> gt_cell_id
    reverse_matches = {v: k for k, v in cell_matches.items()}

    # Add columns to adata_bin2cell.obs
    adata_bin2cell.obs['gt_cell_id'] = -1
    adata_bin2cell.obs['gt_cell_type'] = 'Unmatched'
    adata_bin2cell.obs['gt_sc_barcode'] = 'None'

    annotation_records = []

    for bin2cell_id_str in adata_bin2cell.obs.index:
        bin2cell_id = int(bin2cell_id_str)

        if bin2cell_id in reverse_matches:
            gt_cell_id = reverse_matches[bin2cell_id]

            # Get ground truth info
            gt_row = df_gt[df_gt['cell_id'] == gt_cell_id]
            if len(gt_row) > 0:
                cell_type = gt_row.iloc[0]['cell_type']
                sc_barcode = gt_row.iloc[0]['sc_cell_barcode']

                adata_bin2cell.obs.loc[bin2cell_id_str, 'gt_cell_id'] = gt_cell_id
                adata_bin2cell.obs.loc[bin2cell_id_str, 'gt_cell_type'] = cell_type
                adata_bin2cell.obs.loc[bin2cell_id_str, 'gt_sc_barcode'] = sc_barcode

                annotation_records.append({
                    'bin2cell_cell_id': bin2cell_id,
                    'gt_cell_id': gt_cell_id,
                    'cell_type': cell_type,
                    'sc_barcode': sc_barcode
                })

    annotation_df = pd.DataFrame(annotation_records)

    logger.info(f"  Annotated {len(annotation_records)} Bin2Cell cells with ground truth")
    logger.info(f"  Cell types: {annotation_df['cell_type'].value_counts().to_dict()}")

    return adata_bin2cell, annotation_df


def generate_validation_report(df_overlap, df_corr, output_dir):
    """
    Generate validation summary plots and report.
    """
    logger.info("\nGenerating validation report...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: IoU distribution
    if len(df_overlap) > 0:
        axes[0, 0].hist(df_overlap['iou'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df_overlap['iou'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df_overlap["iou"].mean():.3f}')
        axes[0, 0].set_xlabel('IoU (Whole Cell)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Spatial Overlap (IoU)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    # Plot 2: Precision vs Recall
    if len(df_overlap) > 0:
        axes[0, 1].scatter(df_overlap['recall'], df_overlap['precision'], alpha=0.5)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    # Plot 3: Gene correlation distribution
    if len(df_corr) > 0:
        axes[0, 2].hist(df_corr['pearson_r'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 2].axvline(df_corr['pearson_r'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df_corr["pearson_r"].mean():.3f}')
        axes[0, 2].set_xlabel('Pearson Correlation')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Gene Expression Correlation')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

    # Plot 4: Correlation by cell type
    if len(df_corr) > 0 and 'cell_type' in df_corr.columns:
        cell_type_corr = df_corr.groupby('cell_type')['pearson_r'].mean().sort_values()
        axes[1, 0].barh(range(len(cell_type_corr)), cell_type_corr.values)
        axes[1, 0].set_yticks(range(len(cell_type_corr)))
        axes[1, 0].set_yticklabels(cell_type_corr.index)
        axes[1, 0].set_xlabel('Mean Pearson r')
        axes[1, 0].set_title('Correlation by Cell Type')
        axes[1, 0].grid(axis='x', alpha=0.3)

    # Plot 5: Counts comparison
    if len(df_corr) > 0:
        axes[1, 1].scatter(df_corr['gt_total_counts'], df_corr['bin2cell_total_counts'], alpha=0.5)
        max_count = max(df_corr['gt_total_counts'].max(), df_corr['bin2cell_total_counts'].max())
        axes[1, 1].plot([0, max_count], [0, max_count], 'r--', label='y=x')
        axes[1, 1].set_xlabel('Ground Truth Total Counts')
        axes[1, 1].set_ylabel('Bin2Cell Total Counts')
        axes[1, 1].set_title('Total Counts: GT vs Bin2Cell')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    # Plot 6: IoU vs Correlation
    if len(df_overlap) > 0 and len(df_corr) > 0:
        # Merge dataframes on cell IDs
        merged = df_overlap.merge(df_corr[['gt_cell_id', 'pearson_r']], on='gt_cell_id', how='inner')
        if len(merged) > 0:
            axes[1, 2].scatter(merged['iou'], merged['pearson_r'], alpha=0.5)
            axes[1, 2].set_xlabel('Spatial IoU')
            axes[1, 2].set_ylabel('Gene Correlation (Pearson r)')
            axes[1, 2].set_title('Spatial Accuracy vs Gene Correlation')
            axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'bin2cell_validation_summary.png', dpi=300, bbox_inches='tight')
    logger.info(f"  Saved validation plot: {output_path / 'bin2cell_validation_summary.png'}")
    plt.close()

    # Save text summary
    with open(output_path / 'validation_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("Bin2Cell Validation Summary\n")
        f.write("="*80 + "\n\n")

        f.write("SPATIAL OVERLAP METRICS:\n")
        if len(df_overlap) > 0:
            f.write(f"  Matched cells: {len(df_overlap)}\n")
            f.write(f"  Mean IoU: {df_overlap['iou'].mean():.3f}\n")
            f.write(f"  Median IoU: {df_overlap['iou'].median():.3f}\n")
            f.write(f"  Mean Precision: {df_overlap['precision'].mean():.3f}\n")
            f.write(f"  Mean Recall: {df_overlap['recall'].mean():.3f}\n")
            f.write(f"  IoU > 0.5: {(df_overlap['iou'] > 0.5).sum()} ({100*(df_overlap['iou'] > 0.5).mean():.1f}%)\n")
            f.write(f"  IoU > 0.7: {(df_overlap['iou'] > 0.7).sum()} ({100*(df_overlap['iou'] > 0.7).mean():.1f}%)\n")
        f.write("\n")

        f.write("GENE CORRELATION METRICS:\n")
        if len(df_corr) > 0:
            f.write(f"  Cells evaluated: {len(df_corr)}\n")
            f.write(f"  Mean Pearson r: {df_corr['pearson_r'].mean():.3f}\n")
            f.write(f"  Median Pearson r: {df_corr['pearson_r'].median():.3f}\n")
            f.write(f"  Mean Spearman r: {df_corr['spearman_r'].mean():.3f}\n")
            f.write(f"  Pearson r > 0.5: {(df_corr['pearson_r'] > 0.5).sum()} ({100*(df_corr['pearson_r'] > 0.5).mean():.1f}%)\n")
            f.write(f"  Pearson r > 0.7: {(df_corr['pearson_r'] > 0.7).sum()} ({100*(df_corr['pearson_r'] > 0.7).mean():.1f}%)\n")
        f.write("\n")

    logger.info(f"  Saved text summary: {output_path / 'validation_summary.txt'}")

    # Save detailed results
    if len(df_overlap) > 0:
        df_overlap.to_csv(output_path / 'spatial_overlap_results.csv', index=False)
        logger.info(f"  Saved spatial overlap: {output_path / 'spatial_overlap_results.csv'}")

    if len(df_corr) > 0:
        df_corr.to_csv(output_path / 'gene_correlation_results.csv', index=False)
        logger.info(f"  Saved gene correlation: {output_path / 'gene_correlation_results.csv'}")


def create_overlay_visualization(df_pixels, bin2cell_all_bins, cell_matches, df_overlap, output_dir, microns_per_pixel=0.2739038899725172):
    """
    Create overlay visualization of ground truth and Bin2Cell segmentations.
    Shows matched cells and overlap quality.

    Args:
        df_pixels: Ground truth pixel-to-cell mapping
        bin2cell_all_bins: Dict mapping bin2cell_cell_id -> set of (bin_x, bin_y)
        cell_matches: Dictionary mapping gt_cell_id -> bin2cell_cell_id
        df_overlap: DataFrame with overlap metrics
        output_dir: Output directory
        microns_per_pixel: Conversion factor
    """
    from PIL import Image

    logger.info("\nCreating overlay visualizations...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get image dimensions from df_pixels
    logger.info("  Determining image dimensions...")
    max_x = df_pixels['x'].max() + 1
    max_y = df_pixels['y'].max() + 1
    logger.info(f"  Image dimensions: {max_x} x {max_y} pixels")

    # Build ground truth segmentation mask at pixel level
    logger.info("  Building ground truth segmentation mask...")
    gt_mask = np.zeros((max_y, max_x), dtype=np.int32)

    # Use chunked processing for large dataframe
    chunk_size = 1000000
    for start_idx in range(0, len(df_pixels), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df_pixels))
        chunk = df_pixels.iloc[start_idx:end_idx]

        # Filter to cells only
        chunk_cells = chunk[chunk['cell_id'] > 0]
        if len(chunk_cells) > 0:
            x_coords = chunk_cells['x'].values
            y_coords = chunk_cells['y'].values
            cell_ids = chunk_cells['cell_id'].values

            # Bounds check
            valid = (y_coords >= 0) & (y_coords < gt_mask.shape[0]) & (x_coords >= 0) & (x_coords < gt_mask.shape[1])
            gt_mask[y_coords[valid], x_coords[valid]] = cell_ids[valid]

        if start_idx > 0 and start_idx % 10000000 == 0:
            logger.info(f"    Processed {start_idx:,} pixels...")

    logger.info(f"  Ground truth mask built: {gt_mask.shape}")

    # Build Bin2Cell segmentation mask from bins
    logger.info("  Building Bin2Cell segmentation mask from bins...")
    bin2cell_mask = np.zeros((max_y, max_x), dtype=np.int32)

    bin_size_microns = 2.0
    for cell_id, bins in bin2cell_all_bins.items():
        for bin_x, bin_y in bins:
            # Convert bin coordinates to pixel coordinates
            x_start = int(bin_x * bin_size_microns / microns_per_pixel)
            x_end = int((bin_x + 1) * bin_size_microns / microns_per_pixel)
            y_start = int(bin_y * bin_size_microns / microns_per_pixel)
            y_end = int((bin_y + 1) * bin_size_microns / microns_per_pixel)

            # Bounds check
            x_start = max(0, min(x_start, max_x))
            x_end = max(0, min(x_end, max_x))
            y_start = max(0, min(y_start, max_y))
            y_end = max(0, min(y_end, max_y))

            # Fill the bin region
            bin2cell_mask[y_start:y_end, x_start:x_end] = cell_id

    logger.info(f"  Bin2Cell mask built: {bin2cell_mask.shape}")

    # Create color-coded overlay - FULLY VECTORIZED
    logger.info("  Creating fully vectorized overlay...")

    # Create RGB image
    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

    # Simple 2-color overlay: GT in green, Bin2Cell in red, overlap in yellow
    gt_cell_mask = (gt_mask > 0)
    bin2cell_cell_mask = (bin2cell_mask > 0)

    # Green channel: Ground truth
    overlay[gt_cell_mask, 1] = 255

    # Red channel: Bin2Cell
    overlay[bin2cell_cell_mask, 0] = 255

    # Where both overlap, you get yellow (red + green = yellow)
    # Where only GT: green
    # Where only Bin2Cell: red

    logger.info("  Overlay created (Green=GT, Red=Bin2Cell, Yellow=Overlap)")

    # Save as a single full-resolution image
    logger.info("  Saving full resolution overlay image...")
    overlay_img = Image.fromarray(overlay)
    overlay_path = output_path / 'segmentation_overlay_full_resolution.png'
    overlay_img.save(overlay_path, compress_level=6)  # PNG compression

    logger.info(f"  Saved full resolution overlay: {overlay_path}")
    logger.info(f"  Image size: {overlay.shape[1]} x {overlay.shape[0]} pixels")

    # Also save a legend as a separate small image
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', label='Overlap (GT + Bin2Cell agree)'),
        Patch(facecolor='green', label='Ground Truth only'),
        Patch(facecolor='red', label='Bin2Cell only'),
        Patch(facecolor='black', label='Background (no cells)')
    ]
    legend = ax.legend(handles=legend_elements, loc='center', fontsize=14, frameon=True)
    ax.set_title('Overlay Color Legend', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path / 'overlay_legend.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved legend to {output_path / 'overlay_legend.png'}")


def main():
    """Main validation workflow"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate Bin2Cell results against ground truth')
    parser.add_argument('--bin2cell_dir', type=str,
                       default='bin2cell_results_colorectal',
                       help='Path to Bin2Cell results directory')
    parser.add_argument('--pseudo_hd_dir', type=str,
                       default='pseudo_visium_hd_outpu_full',
                       help='Path to pseudo Visium HD output directory')
    parser.add_argument('--pixel_file', type=str,
                       default='cellpose_sam_human_colorectal_output/cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_expanded.csv.gz',
                       help='Path to pixel-to-cell mapping CSV (gzipped)')
    parser.add_argument('--sc_h5_file', type=str,
                       default='colorectal_sc_data/CRC_GSE166555_expression.h5',
                       help='Path to single-cell H5 expression file')
    parser.add_argument('--sc_meta_file', type=str,
                       default='colorectal_sc_data/CRC_GSE166555_CellMetainfo_table.tsv',
                       help='Path to single-cell metadata TSV file')
    parser.add_argument('--output_dir', type=str,
                       default='bin2cell_validation_output',
                       help='Output directory for validation results')
    parser.add_argument('--microns_per_pixel', type=float,
                       default=0.2739038899725172,
                       help='Microns per pixel conversion factor')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Bin2Cell Results Validation")
    logger.info("="*80)

    # Paths from arguments
    bin2cell_dir = Path(args.bin2cell_dir)
    pseudo_hd_dir = Path(args.pseudo_hd_dir)
    output_dir = Path(args.output_dir)

    # File paths
    gt_file = pseudo_hd_dir / 'ground_truth_cell_assignments.csv'
    pixel_file = Path(args.pixel_file)
    sc_h5_path = Path(args.sc_h5_file)
    sc_metadata_path = Path(args.sc_meta_file)

    # Step 1: Load Bin2Cell results
    bin2cell_nuclear_bins, bin2cell_all_bins, adata_bin2cell = load_bin2cell_results(
        bin2cell_dir, microns_per_pixel=args.microns_per_pixel
    )

    # Step 2: Load ground truth
    df_gt, df_pixels, adata_sc_gt = load_ground_truth(gt_file, pixel_file, sc_h5_path, sc_metadata_path)

    # Step 3: Build spatial regions (both nuclear and whole cell)
    gt_nuclear_regions, gt_whole_regions = build_ground_truth_cell_regions(
        df_pixels, microns_per_pixel=args.microns_per_pixel
    )

    # Step 4: Match cells based on NUCLEAR overlap
    cell_matches, match_df = match_cells_by_nuclear_overlap(
        gt_nuclear_regions, bin2cell_nuclear_bins,
        overlap_threshold=0.0
    )

    # Step 5: Assign ground truth annotations to matched Bin2Cell cells
    adata_bin2cell_annotated, annotation_df = assign_ground_truth_annotations(
        cell_matches, df_gt, adata_bin2cell
    )

    # Step 6: Calculate WHOLE CELL spatial overlap
    df_overlap = calculate_spatial_overlap(gt_whole_regions, bin2cell_all_bins, cell_matches)

    # Step 7: Calculate gene correlation
    df_corr = calculate_gene_correlation(df_gt, adata_sc_gt, adata_bin2cell, df_overlap)

    # Step 8: Generate validation report
    generate_validation_report(df_overlap, df_corr, output_dir)

    # Step 9: Create overlay visualization
    if len(cell_matches) > 0:
        create_overlay_visualization(df_pixels, bin2cell_all_bins, cell_matches, df_overlap, output_dir,
                                    microns_per_pixel=args.microns_per_pixel)
    else:
        logger.warning("WARNING: No cells matched! Skipping overlay visualization.")

    # Step 10: Save annotated Bin2Cell data
    logger.info("Saving annotated Bin2Cell data...")
    adata_bin2cell_annotated.obs['gt_cell_id'] = adata_bin2cell_annotated.obs['gt_cell_id'].astype(str)
    adata_bin2cell_annotated.write(output_dir / 'adata_bin2cell_annotated.h5ad')
    annotation_df.to_csv(output_dir / 'bin2cell_cell_annotations.csv', index=False)
    match_df.to_csv(output_dir / 'nuclear_bin_matches.csv', index=False)
    logger.info(f"  Annotated data saved to {output_dir / 'adata_bin2cell_annotated.h5ad'}")
    logger.info(f"  Annotation table saved to {output_dir / 'bin2cell_cell_annotations.csv'}")
    logger.info(f"  Match details saved to {output_dir / 'nuclear_bin_matches.csv'}")

    logger.info("="*80)
    logger.info("Validation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
