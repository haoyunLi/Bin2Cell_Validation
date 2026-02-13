#!/usr/bin/env python
"""
Compare Bin2Cell and SMURF Results - Plotting Only

This script creates comparison plots from pre-computed validation results.

Input CSV files (from each validation output directory):
- spatial_overlap_results.csv
- gene_correlation_results.csv
- genewise_correlation_by_celltype.csv
- cell_count_error_by_celltype.csv

Output:
- 4 segmentation comparison boxplots (IoU, F1, Precision, Recall)
- 1 per-cell gene correlation comparison boxplot (Spearman)
- 1 gene-wise correlation comparison boxplot (Spearman)
- 1 cell count fraction comparison barplot

Usage:
    python validate_comparison.py \
        --bin2cell_dir bin2cell_validation_output_0.25 \
        --smurf_dir smurf_validation_output_0.25 \
        --output_dir comparison_validation_output
"""

import numpy as np
import pandas as pd
import logging
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validate_comparison.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_ground_truth(gt_file):
    """Load ground truth cell assignments to get cell_type mapping."""
    logger.info(f"Loading ground truth from {gt_file}...")
    df_gt = pd.read_csv(gt_file)
    logger.info(f"  Loaded {len(df_gt)} cells with {df_gt['cell_type'].nunique()} cell types")
    return df_gt


def load_validation_results(validation_dir, df_gt):
    """Load all CSV files from a validation output directory and merge with ground truth."""
    logger.info(f"Loading validation results from {validation_dir}...")
    val_path = Path(validation_dir)

    results = {}

    # Load spatial overlap results
    spatial_file = val_path / 'spatial_overlap_results.csv'
    if spatial_file.exists():
        results['spatial'] = pd.read_csv(spatial_file)
        # Merge with ground truth to get cell_type
        if 'cell_type' not in results['spatial'].columns:
            results['spatial'] = results['spatial'].merge(
                df_gt[['cell_id', 'cell_type']],
                left_on='gt_cell_id', right_on='cell_id', how='left'
            )
        logger.info(f"  Loaded spatial_overlap_results.csv: {len(results['spatial'])} rows")
    else:
        logger.warning(f"  spatial_overlap_results.csv not found!")
        results['spatial'] = pd.DataFrame()

    # Load gene correlation results
    corr_file = val_path / 'gene_correlation_results.csv'
    if corr_file.exists():
        results['correlation'] = pd.read_csv(corr_file)
        # Merge with ground truth to get cell_type
        if 'cell_type' not in results['correlation'].columns:
            results['correlation'] = results['correlation'].merge(
                df_gt[['cell_id', 'cell_type']],
                left_on='gt_cell_id', right_on='cell_id', how='left'
            )
        logger.info(f"  Loaded gene_correlation_results.csv: {len(results['correlation'])} rows")
    else:
        logger.warning(f"  gene_correlation_results.csv not found!")
        results['correlation'] = pd.DataFrame()

    # Load gene-wise correlation results
    genewise_file = val_path / 'genewise_correlation_by_celltype.csv'
    if genewise_file.exists():
        results['genewise'] = pd.read_csv(genewise_file)
        logger.info(f"  Loaded genewise_correlation_by_celltype.csv: {len(results['genewise'])} rows")
    else:
        logger.warning(f"  genewise_correlation_by_celltype.csv not found!")
        results['genewise'] = pd.DataFrame()

    # Load cell count error results
    count_file = val_path / 'cell_count_error_by_celltype.csv'
    if count_file.exists():
        results['cell_count'] = pd.read_csv(count_file)
        logger.info(f"  Loaded cell_count_error_by_celltype.csv: {len(results['cell_count'])} rows")
    else:
        logger.warning(f"  cell_count_error_by_celltype.csv not found!")
        results['cell_count'] = pd.DataFrame()

    return results


# =============================================================================
# Comparison Plotting Functions
# =============================================================================

def create_comparison_segmentation_boxplots(df_bin2cell, df_smurf, output_dir):
    """Create comparison box plots for segmentation metrics."""
    logger.info("Creating comparison segmentation box plots...")
    output_path = Path(output_dir)

    # Check for cell_type column
    if 'cell_type' not in df_bin2cell.columns or 'cell_type' not in df_smurf.columns:
        logger.warning("  cell_type column not found in spatial overlap data!")
        return

    # Match validation filtering: keep only matched cells when available
    df_b2c = df_bin2cell.copy()
    df_smf = df_smurf.copy()
    if 'nuclear_match' in df_b2c.columns:
        df_b2c = df_b2c[df_b2c['nuclear_match']].copy()
    if 'nuclear_match' in df_smf.columns:
        df_smf = df_smf[df_smf['nuclear_match']].copy()

    # Filter valid cell types
    df_b2c = df_b2c[df_b2c['cell_type'].notna()].copy()
    df_smf = df_smf[df_smf['cell_type'].notna()].copy()

    # Get common cell types
    common_types = sorted(set(df_b2c['cell_type'].unique()) & set(df_smf['cell_type'].unique()))

    if len(common_types) == 0:
        logger.warning("No common cell types found!")
        return

    logger.info(f"  Found {len(common_types)} common cell types")

    metrics = ['iou', 'f1_score', 'precision', 'recall']
    metric_names = ['IoU', 'F1 Score', 'Precision', 'Recall']

    for metric, metric_name in zip(metrics, metric_names):
        if metric not in df_b2c.columns or metric not in df_smf.columns:
            logger.warning(f"  {metric} column not found, skipping...")
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        # Sort cell types by Bin2Cell median
        medians = {ct: np.median(df_b2c[df_b2c['cell_type'] == ct][metric].values) for ct in common_types}
        cell_types_sorted = sorted(common_types, key=lambda x: medians[x], reverse=True)

        # Prepare data
        data_bin2cell = [df_b2c[df_b2c['cell_type'] == ct][metric].values for ct in cell_types_sorted]
        data_smurf = [df_smf[df_smf['cell_type'] == ct][metric].values for ct in cell_types_sorted]

        # Positions for grouped boxes
        n_types = len(cell_types_sorted)
        group_width = 0.6
        box_width = 0.25
        positions_b2c = np.arange(n_types) * group_width
        positions_smf = positions_b2c + box_width + 0.05

        # Create box plots
        bp1 = ax.boxplot(data_bin2cell, positions=positions_b2c, widths=box_width,
                         patch_artist=True, medianprops=dict(color='black'))
        bp2 = ax.boxplot(data_smurf, positions=positions_smf, widths=box_width,
                         patch_artist=True, medianprops=dict(color='black'))

        # Color boxes
        for patch in bp1['boxes']:
            patch.set_facecolor('#1f77b4')  # Blue for Bin2Cell
        for patch in bp2['boxes']:
            patch.set_facecolor('#ff7f0e')  # Orange for SMURF

        # Set labels
        ax.set_xticks((positions_b2c + positions_smf) / 2)
        ax.set_xticklabels(cell_types_sorted, rotation=45, ha='right', fontsize=16, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=18, fontweight='bold')
        ax.set_xlim(-0.3, (n_types - 1) * group_width + group_width)
        plt.yticks(fontsize=16, fontweight='bold')

        # Legend
        ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Bin2Cell', 'SMURF'],
                  loc='upper right', fontsize=12)

        plt.tight_layout()
        output_file = output_path / f'comparison_segmentation_{metric}.tiff'
        plt.savefig(output_file, format='tiff', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {output_file}")


def create_comparison_gene_correlation_boxplot(df_corr_b2c, df_corr_smf, output_dir):
    """Create comparison box plot for per-cell gene correlation."""
    logger.info("Creating comparison gene correlation box plot...")
    output_path = Path(output_dir)

    # Determine the spearman column name (different between bin2cell and smurf)
    b2c_spearman_col = 'spearman_r' if 'spearman_r' in df_corr_b2c.columns else 'spearman_corr'
    smf_spearman_col = 'spearman_r' if 'spearman_r' in df_corr_smf.columns else 'spearman_corr'

    if b2c_spearman_col not in df_corr_b2c.columns:
        logger.warning(f"  Spearman column not found in Bin2Cell data!")
        return
    if smf_spearman_col not in df_corr_smf.columns:
        logger.warning(f"  Spearman column not found in SMURF data!")
        return

    df_b2c = df_corr_b2c[df_corr_b2c['cell_type'].notna()].copy()
    df_smf = df_corr_smf[df_corr_smf['cell_type'].notna()].copy()

    common_types = sorted(set(df_b2c['cell_type'].unique()) & set(df_smf['cell_type'].unique()))

    if len(common_types) == 0:
        logger.warning("No common cell types found for correlation!")
        return

    logger.info(f"  Found {len(common_types)} common cell types")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Sort by Bin2Cell median
    medians = {ct: np.median(df_b2c[df_b2c['cell_type'] == ct][b2c_spearman_col].values) for ct in common_types}
    cell_types_sorted = sorted(common_types, key=lambda x: medians[x], reverse=True)

    data_bin2cell = [df_b2c[df_b2c['cell_type'] == ct][b2c_spearman_col].values for ct in cell_types_sorted]
    data_smurf = [df_smf[df_smf['cell_type'] == ct][smf_spearman_col].values for ct in cell_types_sorted]

    n_types = len(cell_types_sorted)
    group_width = 0.6
    box_width = 0.25
    positions_b2c = np.arange(n_types) * group_width
    positions_smf = positions_b2c + box_width + 0.05

    bp1 = ax.boxplot(data_bin2cell, positions=positions_b2c, widths=box_width,
                     patch_artist=True, medianprops=dict(color='black'))
    bp2 = ax.boxplot(data_smurf, positions=positions_smf, widths=box_width,
                     patch_artist=True, medianprops=dict(color='black'))

    for patch in bp1['boxes']:
        patch.set_facecolor('#1f77b4')
    for patch in bp2['boxes']:
        patch.set_facecolor('#ff7f0e')

    ax.set_xticks((positions_b2c + positions_smf) / 2)
    ax.set_xticklabels(cell_types_sorted, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax.set_ylabel('Spearman Correlation', fontsize=18, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlim(-0.3, (n_types - 1) * group_width + group_width)
    plt.yticks(fontsize=16, fontweight='bold')

    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Bin2Cell', 'SMURF'],
              loc='upper right', fontsize=12)

    plt.tight_layout()
    output_file = output_path / 'comparison_gene_correlation_spearman.tiff'
    plt.savefig(output_file, format='tiff', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {output_file}")


def create_comparison_genewise_correlation_boxplot(df_gw_b2c, df_gw_smf, output_dir):
    """Create comparison box plot for gene-wise correlation."""
    logger.info("Creating comparison gene-wise correlation box plot...")
    output_path = Path(output_dir)

    if len(df_gw_b2c) == 0 or len(df_gw_smf) == 0:
        logger.warning("No gene-wise correlation data available!")
        return

    # Check column name
    spearman_col = 'spearman_corr'
    if spearman_col not in df_gw_b2c.columns or spearman_col not in df_gw_smf.columns:
        logger.warning(f"  {spearman_col} column not found!")
        return

    common_types = sorted(set(df_gw_b2c['cell_type'].unique()) & set(df_gw_smf['cell_type'].unique()))

    if len(common_types) == 0:
        logger.warning("No common cell types found for gene-wise correlation!")
        return

    logger.info(f"  Found {len(common_types)} common cell types")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Sort by Bin2Cell median
    medians = {ct: np.median(df_gw_b2c[df_gw_b2c['cell_type'] == ct][spearman_col].values) for ct in common_types}
    cell_types_sorted = sorted(common_types, key=lambda x: medians[x], reverse=True)

    data_bin2cell = [df_gw_b2c[df_gw_b2c['cell_type'] == ct][spearman_col].values for ct in cell_types_sorted]
    data_smurf = [df_gw_smf[df_gw_smf['cell_type'] == ct][spearman_col].values for ct in cell_types_sorted]

    n_types = len(cell_types_sorted)
    group_width = 0.6
    box_width = 0.25
    positions_b2c = np.arange(n_types) * group_width
    positions_smf = positions_b2c + box_width + 0.05

    bp1 = ax.boxplot(data_bin2cell, positions=positions_b2c, widths=box_width,
                     patch_artist=True, medianprops=dict(color='black'))
    bp2 = ax.boxplot(data_smurf, positions=positions_smf, widths=box_width,
                     patch_artist=True, medianprops=dict(color='black'))

    for patch in bp1['boxes']:
        patch.set_facecolor('#1f77b4')
    for patch in bp2['boxes']:
        patch.set_facecolor('#ff7f0e')

    ax.set_xticks((positions_b2c + positions_smf) / 2)
    ax.set_xticklabels(cell_types_sorted, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax.set_ylabel('Gene-wise Spearman Correlation', fontsize=18, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlim(-0.3, (n_types - 1) * group_width + group_width)
    plt.yticks(fontsize=16, fontweight='bold')

    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Bin2Cell', 'SMURF'],
              loc='upper right', fontsize=12)

    plt.tight_layout()
    output_file = output_path / 'comparison_genewise_correlation_spearman.tiff'
    plt.savefig(output_file, format='tiff', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {output_file}")


def create_comparison_cell_count_barplot(df_count_b2c, df_count_smf, output_dir):
    """Create comparison bar plot for cell count fraction."""
    logger.info("Creating comparison cell count fraction bar plot...")
    output_path = Path(output_dir)

    if len(df_count_b2c) == 0 or len(df_count_smf) == 0:
        logger.warning("No cell count data available!")
        return

    common_types = sorted(set(df_count_b2c['cell_type']) & set(df_count_smf['cell_type']))

    if len(common_types) == 0:
        logger.warning("No common cell types found for cell count!")
        return

    logger.info(f"  Found {len(common_types)} common cell types")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Compute count_ratio = predicted/true = 1 + count_error
    df_count_b2c = df_count_b2c.copy()
    df_count_smf = df_count_smf.copy()
    df_count_b2c['count_ratio'] = 1 + df_count_b2c['count_error']
    df_count_smf['count_ratio'] = 1 + df_count_smf['count_error']

    # Sort by Bin2Cell ratio
    df_b2c_sorted = df_count_b2c[df_count_b2c['cell_type'].isin(common_types)].copy()
    df_b2c_sorted = df_b2c_sorted.sort_values('count_ratio', ascending=False)
    cell_types_sorted = df_b2c_sorted['cell_type'].tolist()

    # Get values in sorted order
    b2c_values = [df_count_b2c[df_count_b2c['cell_type'] == ct]['count_ratio'].values[0] for ct in cell_types_sorted]
    smf_values = [df_count_smf[df_count_smf['cell_type'] == ct]['count_ratio'].values[0] for ct in cell_types_sorted]

    n_types = len(cell_types_sorted)
    group_width = 0.6
    bar_width = 0.25
    positions_b2c = np.arange(n_types) * group_width
    positions_smf = positions_b2c + bar_width + 0.05

    bars1 = ax.bar(positions_b2c, b2c_values, width=bar_width, color='#1f77b4',
                   edgecolor='black', linewidth=1.5, label='Bin2Cell')
    bars2 = ax.bar(positions_smf, smf_values, width=bar_width, color='#ff7f0e',
                   edgecolor='black', linewidth=1.5, label='SMURF')

    ax.set_xticks((positions_b2c + positions_smf) / 2)
    ax.set_xticklabels(cell_types_sorted, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax.set_ylabel('Predicted Cell Count Ratio\n(Predicted/True)', fontsize=18, fontweight='bold')
    ax.set_xlim(-0.3, (n_types - 1) * group_width + group_width)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_ylim(0, 0.4)
    plt.yticks(fontsize=16, fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars1, b2c_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=12)
    for bar, val in zip(bars2, smf_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=12)

    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    output_file = output_path / 'comparison_cell_count_fraction.tiff'
    plt.savefig(output_file, format='tiff', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {output_file}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare Bin2Cell and SMURF validation results')
    parser.add_argument('--bin2cell_dir', type=str, default='bin2cell_validation_output_0.25',
                        help='Path to Bin2Cell validation output directory')
    parser.add_argument('--smurf_dir', type=str, default='smurf_validation_output_0.25',
                        help='Path to SMURF validation output directory')
    parser.add_argument('--gt_file', type=str,
                        default='lung_pseudo_visium_hd_output_full_0.25/ground_truth_cell_assignments.csv',
                        help='Path to ground truth cell assignments CSV')
    parser.add_argument('--output_dir', type=str, default='comparison_validation_output',
                        help='Output directory for comparison plots')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Bin2Cell vs SMURF Comparison Plotting")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load ground truth for cell type mapping
    df_gt = load_ground_truth(args.gt_file)

    # Load pre-computed validation results
    bin2cell_results = load_validation_results(args.bin2cell_dir, df_gt)
    smurf_results = load_validation_results(args.smurf_dir, df_gt)

    # Create comparison plots
    logger.info("\n" + "=" * 40)
    logger.info("Creating comparison plots...")
    logger.info("=" * 40)

    # 1-4. Segmentation boxplots (IoU, F1, Precision, Recall)
    if len(bin2cell_results['spatial']) > 0 and len(smurf_results['spatial']) > 0:
        create_comparison_segmentation_boxplots(
            bin2cell_results['spatial'],
            smurf_results['spatial'],
            output_dir
        )

    # 5. Per-cell gene correlation boxplot
    if len(bin2cell_results['correlation']) > 0 and len(smurf_results['correlation']) > 0:
        create_comparison_gene_correlation_boxplot(
            bin2cell_results['correlation'],
            smurf_results['correlation'],
            output_dir
        )

    # 6. Gene-wise correlation boxplot
    if len(bin2cell_results['genewise']) > 0 and len(smurf_results['genewise']) > 0:
        create_comparison_genewise_correlation_boxplot(
            bin2cell_results['genewise'],
            smurf_results['genewise'],
            output_dir
        )

    # 7. Cell count fraction barplot
    if len(bin2cell_results['cell_count']) > 0 and len(smurf_results['cell_count']) > 0:
        create_comparison_cell_count_barplot(
            bin2cell_results['cell_count'],
            smurf_results['cell_count'],
            output_dir
        )

    logger.info("\n" + "=" * 80)
    logger.info("Comparison plotting completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
