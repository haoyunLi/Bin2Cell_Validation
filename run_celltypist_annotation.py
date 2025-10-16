#!/usr/bin/env python
"""
Run CellTypist for cell type annotation on Visium HD spatial transcriptomics data.

CellTypist is designed for single-cell data, but can be adapted for spatial transcriptomics
by treating each spatial spot as a "pseudo-cell".

Usage:
    python run_celltypist_annotation.py

Requirements:
    - celltypist
    - scanpy
    - anndata
"""

import scanpy as sc
import celltypist
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('celltypist_annotation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_visium_hd_data(data_path, clustering_path=None):
    """
    Load Visium HD data from 10x Space Ranger output.

    Parameters
    ----------
    data_path : str or Path
        Path to the filtered_feature_bc_matrix.h5 file
    clustering_path : str or Path, optional
        Path to clusters.csv file with pre-computed clustering

    Returns
    -------
    adata : AnnData
        Annotated data object with expression and metadata
    """
    logger.info(f"Loading Visium HD data from {data_path}")

    # Load the data
    adata = sc.read_10x_h5(data_path)

    # Make variable names unique
    adata.var_names_make_unique()

    logger.info(f"Loaded data: {adata.n_obs} spots, {adata.n_vars} genes")

    # Load clustering if provided
    if clustering_path is not None:
        logger.info(f"Loading clustering from {clustering_path}")
        clusters = pd.read_csv(clustering_path, index_col=0)

        # Match barcodes
        common_barcodes = adata.obs_names.intersection(clusters.index)
        logger.info(f"Found {len(common_barcodes)} common barcodes")

        adata = adata[common_barcodes, :].copy()
        adata.obs['cluster'] = clusters.loc[common_barcodes, 'Cluster']

    return adata


def preprocess_for_celltypist(adata, n_top_genes=3000):
    """
    Preprocess spatial data for CellTypist annotation.

    Parameters
    ----------
    adata : AnnData
        Raw count data
    n_top_genes : int
        Number of highly variable genes to use

    Returns
    -------
    adata : AnnData
        Preprocessed data with normalized counts
    """
    logger.info("Preprocessing data for CellTypist...")

    # Basic QC
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Normalization (CellTypist expects log-normalized data)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized data
    adata.raw = adata.copy()

    # Feature selection
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')

    logger.info(f"Selected {adata.var['highly_variable'].sum()} highly variable genes")

    return adata


def run_celltypist(adata, model='Mouse_Whole_Brain.pkl', majority_voting=True):
    """
    Run CellTypist cell type prediction.

    Parameters
    ----------
    adata : AnnData
        Preprocessed expression data
    model : str
        CellTypist model to use. Options:
        - 'Mouse_Whole_Brain.pkl' (default for mouse brain - includes all brain regions)
        - 'Developing_Mouse_Brain.pkl' (developing mouse brain)
        - 'Mouse_Isocortex_Hippocampus.pkl' (mouse cortex and hippocampus)
        - 'Mouse_Dentate_Gyrus.pkl' (mouse dentate gyrus)
        - 'Immune_All_Low.pkl' (immune cells)
        See celltypist.models.models_description() for full list
    majority_voting : bool
        Whether to use majority voting for refinement

    Returns
    -------
    adata : AnnData
        Data with cell type predictions added to .obs
    predictions : celltypist prediction object
    """
    logger.info(f"Running CellTypist with model: {model}")

    # Download model if needed
    try:
        celltypist.models.download_models(model=model, force_update=False)
    except Exception as e:
        logger.warning(f"Could not download model {model}: {e}")
        logger.info("Using default model instead")
        model = None

    # Run prediction
    logger.info("Running cell type prediction...")
    predictions = celltypist.annotate(
        adata,
        model=model,
        majority_voting=majority_voting
    )

    # Add predictions to adata
    adata.obs['celltypist_cell_type'] = predictions.predicted_labels['predicted_labels']

    if majority_voting:
        adata.obs['celltypist_majority_voting'] = predictions.predicted_labels['majority_voting']

    logger.info("Cell type prediction complete")
    logger.info(f"Identified {adata.obs['celltypist_cell_type'].nunique()} cell types")

    return adata, predictions


def load_existing_umap(adata, umap_projection_path):
    """
    Load existing UMAP coordinates from Space Ranger analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data
    umap_projection_path : Path
        Path to Space Ranger UMAP projection.csv file

    Returns
    -------
    adata : AnnData
        Data with UMAP coordinates loaded
    """
    logger.info(f"Loading existing UMAP coordinates from {umap_projection_path}")

    # Read UMAP projection
    umap_df = pd.read_csv(umap_projection_path, index_col=0)

    # Match barcodes between adata and UMAP coordinates
    common_barcodes = adata.obs_names.intersection(umap_df.index)
    logger.info(f"Found {len(common_barcodes)} barcodes with UMAP coordinates")

    # Reorder UMAP coordinates to match adata
    umap_coords = umap_df.loc[adata.obs_names]

    # Add UMAP coordinates to adata
    adata.obsm['X_umap'] = umap_coords.values

    logger.info("UMAP coordinates loaded successfully")

    return adata


def compute_umap(adata, n_neighbors=15, n_pcs=30):
    """
    Compute UMAP embedding (fallback if pre-computed UMAP not available).

    Parameters
    ----------
    adata : AnnData
        Preprocessed data
    n_neighbors : int
        Number of neighbors for UMAP
    n_pcs : int
        Number of principal components to use

    Returns
    -------
    adata : AnnData
        Data with UMAP coordinates
    """
    logger.info("Computing PCA...")
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')

    logger.info("Computing neighborhood graph...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    logger.info("Computing UMAP embedding...")
    sc.tl.umap(adata, min_dist=0.3, spread=1.0)

    logger.info("UMAP computation complete")

    return adata


def create_visualizations(adata, output_dir):
    """
    Create UMAP and other visualizations.

    Parameters
    ----------
    adata : AnnData
        Annotated data with UMAP
    output_dir : Path
        Directory to save figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("Creating visualizations...")

    # Set plotting parameters
    sc.settings.set_figure_params(dpi=150, facecolor='white', figsize=(8, 8))

    # 1. UMAP colored by cell type
    logger.info("Creating UMAP by cell type...")
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(
        adata,
        color='celltypist_cell_type',
        ax=ax,
        show=False,
        legend_loc='right margin',
        title='UMAP colored by CellTypist cell type',
        frameon=False,
        palette='tab20'
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_celltypes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. UMAP colored by cluster (if available)
    if 'cluster' in adata.obs.columns:
        logger.info("Creating UMAP by cluster...")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color='cluster',
            ax=ax,
            show=False,
            title='UMAP colored by original clustering',
            frameon=False,
            palette='tab20'
        )
        plt.tight_layout()
        plt.savefig(output_dir / 'umap_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Side-by-side comparison
        logger.info("Creating comparison plot...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sc.pl.umap(
            adata,
            color='cluster',
            ax=axes[0],
            show=False,
            title='Original Clustering',
            frameon=False,
            palette='tab20',
            legend_loc='on data',
            legend_fontsize=8
        )
        sc.pl.umap(
            adata,
            color='celltypist_cell_type',
            ax=axes[1],
            show=False,
            title='CellTypist Cell Types',
            frameon=False,
            palette='tab20',
            legend_loc='right margin',
            legend_fontsize=8
        )
        plt.tight_layout()
        plt.savefig(output_dir / 'umap_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Cluster-cell type heatmap
        logger.info("Creating cluster-cell type heatmap...")
        cluster_celltype = pd.crosstab(
            adata.obs['cluster'],
            adata.obs['celltypist_cell_type'],
            normalize='index'
        ) * 100

        fig, ax = plt.subplots(figsize=(max(10, len(cluster_celltype.columns) * 0.5),
                                         max(8, len(cluster_celltype) * 0.5)))
        sns.heatmap(
            cluster_celltype,
            cmap='YlOrRd',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Percentage (%)'},
            linewidths=0.5,
            ax=ax
        )
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_title('Cell Type Composition per Cluster', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_celltype_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Cell type proportions bar plot
    logger.info("Creating cell type proportions plot...")
    celltype_counts = adata.obs['celltypist_cell_type'].value_counts()
    celltype_props = (celltype_counts / len(adata) * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(celltype_props) * 0.4)))
    colors = sns.color_palette('tab20', n_colors=len(celltype_props))
    celltype_props.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Percentage of spots (%)', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title('Cell Type Composition', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (celltype, prop) in enumerate(celltype_props.items()):
        ax.text(prop + 0.5, i, f'{prop:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'celltype_proportions.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("All visualizations created successfully")


def summarize_results(adata, output_dir):
    """
    Create summary statistics and visualizations.

    Parameters
    ----------
    adata : AnnData
        Annotated data with predictions
    output_dir : Path
        Directory to save outputs
    """
    logger.info("Creating summary statistics...")

    # Cell type composition
    cell_type_counts = adata.obs['celltypist_cell_type'].value_counts()
    cell_type_pct = (cell_type_counts / len(adata) * 100).round(2)

    summary_df = pd.DataFrame({
        'Cell_Type': cell_type_counts.index,
        'Count': cell_type_counts.values,
        'Percentage': cell_type_pct.values
    })

    summary_path = output_dir / 'cell_type_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    # Print summary
    logger.info("\nCell Type Composition:")
    for idx, row in summary_df.iterrows():
        logger.info(f"  {row['Cell_Type']}: {row['Count']} spots ({row['Percentage']}%)")

    # If clustering was provided, create cluster-cell type mapping
    if 'cluster' in adata.obs.columns:
        cluster_celltype = pd.crosstab(
            adata.obs['cluster'],
            adata.obs['celltypist_cell_type'],
            normalize='index'
        ) * 100

        cluster_path = output_dir / 'cluster_celltype_composition.csv'
        cluster_celltype.to_csv(cluster_path)
        logger.info(f"Cluster-cell type mapping saved to {cluster_path}")

    return summary_df


def main():
    """Main workflow for CellTypist annotation of Visium HD data."""

    # Paths
    base_dir = Path.cwd()
    visium_dir = base_dir / 'brain' / 'output' / 'binned_outputs' / 'square_008um'
    data_path = visium_dir / 'filtered_feature_bc_matrix.h5'
    clustering_path = visium_dir / 'analysis' / 'clustering' / 'gene_expression_graphclust' / 'clusters.csv'
    umap_path = visium_dir / 'analysis' / 'umap' / 'gene_expression_2_components' / 'projection.csv'
    output_dir = base_dir / 'cellannotation_results'
    output_dir.mkdir(exist_ok=True)

    logger.info("="*70)
    logger.info("CellTypist Annotation for Visium HD Mouse Brain")
    logger.info("="*70)

    # Load data
    adata = load_visium_hd_data(data_path, clustering_path)

    # Preprocess
    adata = preprocess_for_celltypist(adata)

    # Run CellTypist
    # Available mouse brain models:
    # - 'Mouse_Whole_Brain.pkl': Very detailed (231 cell types) - High resolution
    # - 'Developing_Mouse_Brain.pkl': Developmental stages - May have fewer types
    # - 'Mouse_Isocortex_Hippocampus.pkl': Focused on cortex/hippocampus only
    adata, predictions = run_celltypist(
        adata,
        model='Developing_Mouse_Brain.pkl',
        majority_voting=True
    )

    # Load existing UMAP or compute new one
    logger.info("\nLoading UMAP coordinates for visualization...")
    if umap_path.exists():
        logger.info("Found existing UMAP coordinates from Space Ranger analysis")
        adata = load_existing_umap(adata, umap_path)
    else:
        logger.info("UMAP coordinates not found, computing new UMAP...")
        adata = compute_umap(adata, n_neighbors=15, n_pcs=30)

    # Save annotated data
    logger.info("Saving annotated data...")
    adata_path = output_dir / 'visium_hd_annotated.h5ad'
    adata.write_h5ad(adata_path)
    logger.info(f"Annotated data saved to {adata_path}")

    # Save predictions
    pred_path = output_dir / 'celltypist_predictions.csv'
    predictions.predicted_labels.to_csv(pred_path)
    logger.info(f"Predictions saved to {pred_path}")

    # Create summary
    summary_df = summarize_results(adata, output_dir)

    # Create visualizations
    logger.info("\n" + "="*70)
    logger.info("Creating UMAP visualizations...")
    logger.info("="*70)
    create_visualizations(adata, output_dir)

    logger.info("\n" + "="*70)
    logger.info("CellTypist annotation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  - visium_hd_annotated.h5ad")
    logger.info("  - celltypist_predictions.csv")
    logger.info("  - cell_type_summary.csv")
    logger.info("  - umap_celltypes.png")
    logger.info("  - umap_confidence.png")
    logger.info("  - umap_clusters.png")
    logger.info("  - umap_comparison.png")
    logger.info("  - cluster_celltype_heatmap.png")
    logger.info("  - celltype_proportions.png")
    logger.info("="*70)

    return adata, predictions


if __name__ == "__main__":
    adata, predictions = main()
