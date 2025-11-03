#!/usr/bin/env python
"""
Run StarDist nuclear segmentation on Visium HD images and compare with Cellpose results.

This script uses StarDist (similar to SMURF pipeline) for nuclear segmentation,
then matches nuclei to cells from Cellpose cell segmentation.

Usage:
    conda activate <stardist_env>
    python stardist_nuclear_segmentation.py
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import sys
from PIL import Image

# Disable PIL decompression bomb protection
Image.MAX_IMAGE_PIXELS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stardist_segmentation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def match_nuclei_to_cells(cell_masks, nuclear_masks):
    """
    Match nuclear masks to cell masks ensuring 1 cell = 1 nucleus.

    Args:
        cell_masks: Cell segmentation masks from Cellpose (2D array with cell IDs)
        nuclear_masks: Nuclear segmentation masks from StarDist (2D array with nucleus IDs)

    Returns:
        nuclear_mask_matched: Binary mask of nuclear pixels matched to cells
        stats: Dictionary with matching statistics
    """
    logger.info("Matching StarDist nuclei to Cellpose cells (1 cell = 1 nucleus)...")

    from scipy.sparse import coo_matrix

    # Flatten masks
    cell_flat = cell_masks.flatten()
    nuclear_flat = nuclear_masks.flatten()

    # Only consider pixels with both cell and nucleus
    mask = (cell_flat > 0) & (nuclear_flat > 0)
    cell_ids = cell_flat[mask]
    nuclear_ids = nuclear_flat[mask]

    logger.info(f"  Processing {len(cell_ids):,} pixels with both cell and nucleus labels...")

    # Create sparse overlap matrix
    num_cells = np.max(cell_masks)
    num_nuclei = np.max(nuclear_masks)

    overlap_matrix = coo_matrix(
        (np.ones(len(cell_ids), dtype=np.int32), (cell_ids, nuclear_ids)),
        shape=(num_cells + 1, num_nuclei + 1)
    ).tocsr()

    logger.info(f"  Built overlap matrix: {num_cells:,} cells × {num_nuclei:,} nuclei")

    # Find best nucleus for each cell
    best_nuclei_indices = np.zeros(num_cells + 1, dtype=np.int32)

    for cell_id in range(1, num_cells + 1):
        row = overlap_matrix[cell_id]
        if row.nnz > 0:
            best_nuclei_indices[cell_id] = row.indices[np.argmax(row.data)]

    cells_with_nucleus = np.sum(best_nuclei_indices > 0)
    cells_without_nucleus = num_cells - cells_with_nucleus

    logger.info(f"  Cells with nucleus: {cells_with_nucleus:,}")
    logger.info(f"  Cells without nucleus: {cells_without_nucleus:,}")

    # Create matched nuclear mask
    best_nucleus_per_pixel = best_nuclei_indices[cell_masks]
    nuclear_mask_matched = ((nuclear_masks > 0) & (nuclear_masks == best_nucleus_per_pixel)).astype(np.uint8)

    logger.info(f"  Total nuclear pixels (matched): {nuclear_mask_matched.sum():,}")

    stats = {
        'num_cells': num_cells,
        'num_nuclei': num_nuclei,
        'cells_with_nucleus': cells_with_nucleus,
        'cells_without_nucleus': cells_without_nucleus,
        'nuclear_pixels': nuclear_mask_matched.sum(),
    }

    return nuclear_mask_matched, stats


def run_stardist_segmentation_tiled(img, prob_thresh=0.6, nms_thresh=0.3, tile_size=4700, overlap=80):
    """
    Run StarDist 2D nuclear segmentation on large images using tiling.

    Args:
        img: Input image (H&E stained)
        prob_thresh: Probability threshold (default 0.6, higher = more selective)
        nms_thresh: Non-maximum suppression threshold
        tile_size: Size of each tile for processing
        overlap: Overlap between tiles for merging

    Returns:
        nuclear_masks: 2D array with nucleus IDs
    """
    logger.info(f"Running StarDist segmentation (tiled) with prob_thresh={prob_thresh}, nms_thresh={nms_thresh}...")
    logger.info(f"  Tile size: {tile_size}, overlap: {overlap}")

    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    # Load pre-trained model
    model = StarDist2D.from_pretrained("2D_versatile_he")

    h, w = img.shape[:2]
    logger.info(f"  Image size: {w} x {h}")

    # Initialize final segmentation array
    segmentation_final = np.zeros((h, w), dtype=np.uint32)
    current_max_id = 0

    # Process tiles
    num_tiles_y = int(np.ceil(h / tile_size))
    num_tiles_x = int(np.ceil(w / tile_size))
    logger.info(f"  Processing {num_tiles_y} x {num_tiles_x} = {num_tiles_y * num_tiles_x} tiles...")

    tile_results = {}

    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            # Calculate tile boundaries with overlap
            i_start = max(0, i - overlap)
            i_end = min(h, i + tile_size + overlap)
            j_start = max(0, j - overlap)
            j_end = min(w, j + tile_size + overlap)

            # Extract tile
            tile = img[i_start:i_end, j_start:j_end]

            # Normalize tile
            axis_norm = (0, 1, 2)
            tile_norm = normalize(tile, 1, 99.8, axis=axis_norm)

            # Segment tile
            tile_masks, _ = model.predict_instances(
                tile_norm,
                nms_thresh=nms_thresh,
                prob_thresh=prob_thresh
            )

            # Store result
            tile_results[(i, j)] = {
                'masks': tile_masks,
                'i_start': i_start,
                'i_end': i_end,
                'j_start': j_start,
                'j_end': j_end,
            }

            logger.info(f"    Tile ({i}, {j}): {len(np.unique(tile_masks)) - 1} nuclei")

    # Merge tiles with unique IDs
    logger.info("  Merging tiles...")
    for idx, ((i, j), tile_data) in enumerate(tile_results.items()):
        tile_masks = tile_data['masks']
        i_start = tile_data['i_start']
        i_end = tile_data['i_end']
        j_start = tile_data['j_start']
        j_end = tile_data['j_end']

        # Adjust nucleus IDs to be unique across all tiles
        tile_masks_adjusted = tile_masks.copy()
        tile_masks_adjusted[tile_masks > 0] = tile_masks[tile_masks > 0] + current_max_id
        current_max_id = tile_masks_adjusted.max()

        # Determine the region to copy (without overlap for non-edge tiles)
        if i == 0:
            copy_i_start = 0
        else:
            copy_i_start = overlap

        if j == 0:
            copy_j_start = 0
        else:
            copy_j_start = overlap

        copy_i_end = tile_masks_adjusted.shape[0]
        copy_j_end = tile_masks_adjusted.shape[1]

        # Calculate destination indices
        dest_i_start = i_start + copy_i_start
        dest_i_end = min(h, i_start + copy_i_end)
        dest_j_start = j_start + copy_j_start
        dest_j_end = min(w, j_start + copy_j_end)

        # Copy non-overlapping region
        tile_region = tile_masks_adjusted[copy_i_start:copy_i_end, copy_j_start:copy_j_end]
        segmentation_final[dest_i_start:dest_i_end, dest_j_start:dest_j_end] = tile_region[:dest_i_end-dest_i_start, :dest_j_end-dest_j_start]

    num_nuclei = len(np.unique(segmentation_final)) - 1
    nuclear_pixels = (segmentation_final > 0).sum()

    logger.info(f"  StarDist segmentation complete!")
    logger.info(f"  Number of nuclei detected: {num_nuclei:,}")
    logger.info(f"  Nuclear pixels: {nuclear_pixels:,}")

    return segmentation_final


def compare_nuclear_segmentations(cellpose_nuclear, stardist_nuclear, cell_masks):
    """
    Compare Cellpose and StarDist nuclear segmentations.

    Args:
        cellpose_nuclear: Binary nuclear mask from Cellpose (matched to cells)
        stardist_nuclear: Binary nuclear mask from StarDist (matched to cells)
        cell_masks: Cell segmentation from Cellpose

    Returns:
        comparison_stats: Dictionary with comparison statistics
    """
    logger.info("Comparing Cellpose vs StarDist nuclear segmentation...")

    # Calculate interior pixels (non-boundary)
    from scipy import ndimage
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    neighbor_max = ndimage.maximum_filter(cell_masks, footprint=structure)
    neighbor_min = ndimage.minimum_filter(cell_masks, footprint=structure)
    cell_boundaries = ((cell_masks > 0) & ((neighbor_max != cell_masks) | (neighbor_min != cell_masks))).astype(np.uint8)
    is_interior = ((cell_masks > 0) & (cell_boundaries == 0)).astype(np.uint8)

    interior_pixels = is_interior.sum()

    # Cellpose nuclear stats
    cellpose_nuclear_pixels = cellpose_nuclear.sum()
    cellpose_nuclear_pct = 100.0 * cellpose_nuclear_pixels / interior_pixels if interior_pixels > 0 else 0

    # StarDist nuclear stats
    stardist_nuclear_pixels = stardist_nuclear.sum()
    stardist_nuclear_pct = 100.0 * stardist_nuclear_pixels / interior_pixels if interior_pixels > 0 else 0

    # Overlap
    overlap_pixels = (cellpose_nuclear & stardist_nuclear).sum()
    overlap_pct_cellpose = 100.0 * overlap_pixels / cellpose_nuclear_pixels if cellpose_nuclear_pixels > 0 else 0
    overlap_pct_stardist = 100.0 * overlap_pixels / stardist_nuclear_pixels if stardist_nuclear_pixels > 0 else 0

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPARISON: Cellpose vs StarDist Nuclear Segmentation")
    logger.info(f"{'='*70}")
    logger.info(f"Interior pixels (total):           {interior_pixels:,}")
    logger.info(f"")
    logger.info(f"Cellpose Nuclear:")
    logger.info(f"  Nuclear pixels:                  {cellpose_nuclear_pixels:,}")
    logger.info(f"  % of cell interior:              {cellpose_nuclear_pct:.1f}%")
    logger.info(f"")
    logger.info(f"StarDist Nuclear:")
    logger.info(f"  Nuclear pixels:                  {stardist_nuclear_pixels:,}")
    logger.info(f"  % of cell interior:              {stardist_nuclear_pct:.1f}%")
    logger.info(f"")
    logger.info(f"Overlap:")
    logger.info(f"  Overlapping pixels:              {overlap_pixels:,}")
    logger.info(f"  % of Cellpose nuclear:           {overlap_pct_cellpose:.1f}%")
    logger.info(f"  % of StarDist nuclear:           {overlap_pct_stardist:.1f}%")
    logger.info(f"{'='*70}\n")

    comparison_stats = {
        'interior_pixels': interior_pixels,
        'cellpose_nuclear_pixels': cellpose_nuclear_pixels,
        'cellpose_nuclear_pct': cellpose_nuclear_pct,
        'stardist_nuclear_pixels': stardist_nuclear_pixels,
        'stardist_nuclear_pct': stardist_nuclear_pct,
        'overlap_pixels': overlap_pixels,
        'overlap_pct_cellpose': overlap_pct_cellpose,
        'overlap_pct_stardist': overlap_pct_stardist,
    }

    return comparison_stats


def create_comparison_visualization(img, cell_masks, cellpose_nuclear, stardist_nuclear, output_dir, image_name):
    """
    Create visualization comparing Cellpose and StarDist nuclear segmentations.
    """
    logger.info("Creating comparison visualization...")

    from skimage.color import label2rgb

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original H&E Image", fontsize=14)
    axes[0, 0].axis('off')

    # Cell segmentation
    cell_colored = label2rgb(cell_masks, bg_label=0)
    axes[0, 1].imshow(cell_colored)
    axes[0, 1].set_title(f"Cellpose Cells ({len(np.unique(cell_masks)) - 1:,} cells)", fontsize=14)
    axes[0, 1].axis('off')

    # Cellpose nuclei
    axes[0, 2].imshow(img)
    axes[0, 2].imshow(cellpose_nuclear, cmap='Reds', alpha=0.5)
    axes[0, 2].set_title(f"Cellpose Nuclei ({cellpose_nuclear.sum():,} pixels)", fontsize=14)
    axes[0, 2].axis('off')

    # StarDist nuclei
    axes[1, 0].imshow(img)
    axes[1, 0].imshow(stardist_nuclear, cmap='Blues', alpha=0.5)
    axes[1, 0].set_title(f"StarDist Nuclei ({stardist_nuclear.sum():,} pixels)", fontsize=14)
    axes[1, 0].axis('off')

    # Overlay both
    overlay = np.zeros((*img.shape[:2], 3))
    overlay[:, :, 0] = cellpose_nuclear  # Red channel
    overlay[:, :, 2] = stardist_nuclear   # Blue channel
    axes[1, 1].imshow(img)
    axes[1, 1].imshow(overlay, alpha=0.5)
    axes[1, 1].set_title("Overlay: Cellpose (Red) vs StarDist (Blue)", fontsize=14)
    axes[1, 1].axis('off')

    # Difference map
    overlap = (cellpose_nuclear & stardist_nuclear).astype(float)
    cellpose_only = (cellpose_nuclear & ~stardist_nuclear.astype(bool)).astype(float)
    stardist_only = (stardist_nuclear & ~cellpose_nuclear.astype(bool)).astype(float)

    diff_map = np.zeros((*img.shape[:2], 3))
    diff_map[:, :, 1] = overlap  # Green = overlap
    diff_map[:, :, 0] = cellpose_only  # Red = Cellpose only
    diff_map[:, :, 2] = stardist_only  # Blue = StarDist only

    axes[1, 2].imshow(img)
    axes[1, 2].imshow(diff_map, alpha=0.6)
    axes[1, 2].set_title("Difference: Green=Overlap, Red=Cellpose only, Blue=StarDist only", fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    comparison_path = output_dir / f'{image_name}_nuclear_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparison visualization saved: {comparison_path}")


def main():
    """Run StarDist nuclear segmentation and compare with Cellpose results."""

    # Define paths
    current_dir = Path.cwd()
    img_path = current_dir / 'cropped_visium_hd_human_kidney.png'
    cellpose_output_dir = current_dir / 'cellpose_sam_human_kidney_output'
    output_dir = current_dir / 'stardist_nuclear_output'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    logger.info("="*70)
    logger.info("StarDist Nuclear Segmentation & Comparison")
    logger.info("="*70)
    logger.info(f"Input image: {img_path}")
    logger.info(f"Cellpose output: {cellpose_output_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load image
    logger.info("\nLoading image...")
    img = np.array(Image.open(img_path))
    logger.info(f"Image loaded: shape={img.shape}, dtype={img.dtype}")

    # Load Cellpose cell masks
    logger.info("\nLoading Cellpose cell masks...")
    cellpose_cell_masks_path = cellpose_output_dir / 'cropped_visium_hd_human_kidney_cell_masks.npy'
    cell_masks = np.load(cellpose_cell_masks_path)
    logger.info(f"Cell masks loaded: {len(np.unique(cell_masks)) - 1:,} cells")

    # Load Cellpose nuclear masks
    logger.info("\nLoading Cellpose nuclear masks...")
    cellpose_nuclear_path = cellpose_output_dir / 'cropped_visium_hd_human_kidney_nuclear_binary_mask.npy'
    cellpose_nuclear_binary = np.load(cellpose_nuclear_path)
    logger.info(f"Cellpose nuclear pixels: {cellpose_nuclear_binary.sum():,}")

    # Run StarDist segmentation with prob_thresh = 0.6 (tiled to avoid OOM)
    logger.info("\n" + "="*70)
    stardist_nuclear_masks = run_stardist_segmentation_tiled(
        img,
        prob_thresh=0.6,  # More selective than default 0.01
        nms_thresh=0.3,
        tile_size=4700,  # Same as SMURF tutorial
        overlap=80  # Overlap for merging
    )

    # Match StarDist nuclei to cells
    logger.info("\n" + "="*70)
    stardist_nuclear_binary, stardist_stats = match_nuclei_to_cells(cell_masks, stardist_nuclear_masks)

    # Compare the two methods
    logger.info("\n" + "="*70)
    comparison_stats = compare_nuclear_segmentations(
        cellpose_nuclear_binary,
        stardist_nuclear_binary,
        cell_masks
    )

    # Save StarDist results
    logger.info("\nSaving StarDist results...")
    stardist_masks_path = output_dir / 'stardist_nuclear_masks.npy'
    np.save(stardist_masks_path, stardist_nuclear_masks)
    logger.info(f"StarDist nuclear masks saved: {stardist_masks_path}")

    stardist_binary_path = output_dir / 'stardist_nuclear_binary_mask.npy'
    np.save(stardist_binary_path, stardist_nuclear_binary)
    logger.info(f"StarDist nuclear binary mask saved: {stardist_binary_path}")

    # Create comparison visualization
    logger.info("\n" + "="*70)
    create_comparison_visualization(
        img,
        cell_masks,
        cellpose_nuclear_binary,
        stardist_nuclear_binary,
        output_dir,
        'human_kidney'
    )

    logger.info("\n" + "="*70)
    logger.info("Processing complete!")
    logger.info(f"\nOutput files:")
    logger.info(f"  - StarDist nuclear masks: {stardist_masks_path}")
    logger.info(f"  - StarDist binary mask: {stardist_binary_path}")
    logger.info(f"  - Comparison visualization: {output_dir / 'human_kidney_nuclear_comparison.png'}")
    logger.info(f"  - Log file: stardist_segmentation.log")

    return stardist_nuclear_masks, comparison_stats


if __name__ == "__main__":
    stardist_masks, stats = main()
