#!/usr/bin/env python
"""
Run Cellpose-SAM cell segmentation on Visium HD images.

This script uses Cellpose-SAM to segment cells in large tissue images and
saves the results including pixel-level cell assignments to CSV files.

Usage:
    conda activate ./cellpose
    python cellpose_sam_segmentation.py
"""

import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import logging
import sys

# Disable PIL decompression bomb protection for large images
Image.MAX_IMAGE_PIXELS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cellpose_sam_segmentation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

io.logger_setup()  # Cellpose logging


def save_pixel_to_cell_csv(masks, nuclear_masks, output_dir, image_name):
    """
    Save pixel-level cell assignments to CSV files.

    Args:
        masks: Cell segmentation masks from Cellpose (2D array with cell IDs)
        nuclear_masks: Nuclear segmentation masks (2D array with nucleus IDs)
        output_dir: Directory to save CSV files
        image_name: Name of the image (for naming output files)

    Returns:
        Paths to the saved CSV files
    """
    logger.info("Creating pixel-to-cell mapping CSV files...")

    height, width = masks.shape
    logger.info(f"  Image dimensions: {width} x {height}")
    logger.info(f"  Total pixels: {width * height:,}")

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    cell_id_flat = masks.flatten()

    # Detect cell boundaries for membrane identification
    logger.info("Detecting cell boundaries...")
    from scipy import ndimage
    sx = ndimage.sobel(masks.astype(float), axis=0)
    sy = ndimage.sobel(masks.astype(float), axis=1)
    sobel = np.hypot(sx, sy)
    cell_boundaries = (sobel > 0).astype(np.uint8)

    # Process nuclear masks
    logger.info("Processing nuclear segmentation...")
    nuclear_mask_binary = (nuclear_masks > 0).astype(np.uint8)

    # Create full pixel mapping DataFrame
    df_full = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'cell_id': cell_id_flat,
        'is_boundary': cell_boundaries.flatten().astype(np.uint8),
        'is_nuclear': nuclear_mask_binary.flatten().astype(np.uint8),
    })

    # Mark interior (non-boundary) pixels
    df_full['is_interior'] = ((df_full['cell_id'] > 0) & (df_full['is_boundary'] == 0)).astype(np.uint8)

    # Mark cytoplasm as interior pixels that are NOT nuclear
    df_full['is_cytoplasm'] = ((df_full['is_interior'] == 1) & (df_full['is_nuclear'] == 0)).astype(np.uint8)

    # Save full resolution (compressed)
    full_csv_path = output_dir / f'{image_name}_pixel_to_cell_mapping_full.csv.gz'
    df_full.to_csv(full_csv_path, index=False, compression='gzip')
    logger.info(f"Full-resolution mapping saved (compressed): {full_csv_path}")
    logger.info(f"  Total pixels: {len(df_full):,}")
    logger.info(f"  Boundary (membrane) pixels: {df_full['is_boundary'].sum():,}")
    logger.info(f"  Nuclear pixels: {df_full['is_nuclear'].sum():,}")
    logger.info(f"  Cytoplasm pixels: {df_full['is_cytoplasm'].sum():,}")
    logger.info(f"  Interior (total) pixels: {df_full['is_interior'].sum():,}")
    logger.info(f"  Background pixels: {(df_full['cell_id'] == 0).sum():,}")

    # Cell-level summary is optional 
    logger.info("Skipping cell-level summary (can be computed from full CSV if needed later)")
    logger.info(f"  Total cells detected: {len(np.unique(masks)) - 1:,}")
    summary_csv_path = None

    # Save masks as numpy array for easy loading
    numpy_path = output_dir / f'{image_name}_cell_masks.npy'
    np.save(numpy_path, masks)
    logger.info(f"Cell masks array saved: {numpy_path}")

    # Save boundary mask
    boundary_path = output_dir / f'{image_name}_boundary_mask.npy'
    np.save(boundary_path, cell_boundaries)
    logger.info(f"Boundary mask array saved: {boundary_path}")

    # Save nuclear masks
    nuclear_path = output_dir / f'{image_name}_nuclear_masks.npy'
    np.save(nuclear_path, nuclear_masks)
    logger.info(f"Nuclear masks array saved: {nuclear_path}")

    return full_csv_path, summary_csv_path


def create_visualization(img, masks, flows, output_dir, image_name, downsample_factor=1):
    """
    Create and save visualization of segmentation results.

    For very large images, this is downsampled to speed up visualization.

    Args:
        img: Original image
        masks: Cell masks from Cellpose
        flows: Flow results from Cellpose
        output_dir: Directory to save outputs
        image_name: Name of the image
        downsample_factor: Factor to downsample by (default 8 for speed)
    """
    logger.info(f"Creating visualization (downsampled by {downsample_factor}x for speed)...")

    # Get image dimensions
    h, w = masks.shape
    logger.info(f"Original image size: {w}x{h}")

    # Downsample for visualization only
    new_h = h // downsample_factor
    new_w = w // downsample_factor
    logger.info(f"Downsampling to {new_w}x{new_h} for faster visualization...")

    from skimage.transform import resize

    # Downsample image
    if len(img.shape) == 3:
        img_small = resize(img, (new_h, new_w, img.shape[2]), order=1, preserve_range=True, anti_aliasing=True).astype(img.dtype)
    else:
        img_small = resize(img, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True).astype(img.dtype)

    # Downsample masks (use nearest neighbor to preserve labels)
    masks_small = resize(masks, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False).astype(masks.dtype)

    # Downsample flows
    flows_small = resize(flows[0], (new_h, new_w, flows[0].shape[2]), order=1, preserve_range=True, anti_aliasing=True).astype(flows[0].dtype)

    logger.info("Creating downsampled overlay plot...")
    # Create figure with segmentation overlay
    fig = plt.figure(figsize=(20, 10))
    plot.show_segmentation(fig, img_small, masks_small, flows_small)
    plt.tight_layout()

    # Save visualization
    viz_path = output_dir / f'{image_name}_segmentation_overlay_downsampled_{downsample_factor}x.png'
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualization saved (downsampled {downsample_factor}x): {viz_path}")

    logger.info("Creating colored masks (downsampled)...")
    # Save masks as image (downsampled)
    from skimage.color import label2rgb
    mask_colored = label2rgb(masks_small, bg_label=0)
    mask_path = output_dir / f'{image_name}_masks_colored_downsampled_{downsample_factor}x.png'
    plt.imsave(mask_path, mask_colored)
    logger.info(f"Colored masks saved (downsampled {downsample_factor}x): {mask_path}")

    logger.info("Note: Full resolution masks are saved in the TIFF and numpy files")


def main():
    """Run Cellpose-SAM segmentation on the cropped Visium HD brain image."""

    # Define paths
    current_dir = Path.cwd()
    img_path = current_dir / 'cropped_visium_hd_small_intenstine.png'
    output_dir = current_dir / 'cellpose_sam_small_intenstine_output'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Check if input image exists
    if not img_path.exists():
        logger.error(f"Input image not found: {img_path}")
        raise FileNotFoundError(f"Input image not found: {img_path}")

    logger.info("="*70)
    logger.info("Cellpose-SAM Cell Segmentation")
    logger.info("="*70)
    logger.info(f"Input image: {img_path}")
    logger.info(f"Output directory: {output_dir}")

    # Check GPU availability
    use_gpu = core.use_gpu()
    if use_gpu:
        logger.info("GPU available and will be used for acceleration")
    else:
        logger.warning("GPU not available, using CPU (will be slower)")

    # Load image
    logger.info("Loading image...")
    img = io.imread(img_path)
    logger.info(f"Image loaded: shape={img.shape}, dtype={img.dtype}")

    # Determine channels
    if len(img.shape) == 3:
        logger.info(f"Multi-channel image detected with {img.shape[2]} channels")
        # For RGB/multi-channel images, Cellpose can use:
        # - channels=[0,0] for grayscale (single channel)
        # - channels=[2,3] for cytoplasm in channel 2, nucleus in channel 3
        # - channels=[0,0] with RGB will use all channels
        channels = [0, 0]  # Use all channels (grayscale mode)
    else:
        logger.info("Grayscale image detected")
        channels = [0, 0]

    # Initialize Cellpose-SAM model (newest, best generalization)
    logger.info("Initializing Cellpose-SAM model...")
    model = models.CellposeModel(gpu=use_gpu)
    logger.info("Model initialized successfully (Cellpose-SAM)")

    # Set segmentation parameters
    flow_threshold = 0.4  # Default: 0.4. Increase if missing cells, decrease if too many bad masks
    cellprob_threshold = 0.0  # Default: 0.0. Decrease if missing cells, increase if too many false detections
    diameter = None  # Let Cellpose estimate cell diameter automatically
    tile_norm_blocksize = 0  # Set to 100-200 for inhomogeneous brightness

    logger.info("Running segmentation...")
    logger.info(f"Parameters:")
    logger.info(f"  - flow_threshold: {flow_threshold}")
    logger.info(f"  - cellprob_threshold: {cellprob_threshold}")
    logger.info(f"  - diameter: {diameter} (auto-estimate)")
    logger.info(f"  - tile_norm_blocksize: {tile_norm_blocksize}")
    logger.info(f"  - channels: {channels}")

    # Run Cellpose-SAM segmentation
    masks, flows, styles = model.eval(
        img,
        channels=channels,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={'tile_norm_blocksize': tile_norm_blocksize}
    )

    logger.info(f"Cell segmentation complete!")
    logger.info(f"  - Number of cells detected: {len(np.unique(masks)) - 1}")  # -1 to exclude background

    # Run nuclear segmentation
    logger.info("Running nuclear segmentation...")
    logger.info("Initializing Cellpose nuclei model...")
    nuclei_model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')
    logger.info("Nuclei model initialized successfully")

    logger.info("Running nuclear segmentation...")
    nuclear_masks, nuclear_flows, nuclear_styles = nuclei_model.eval(
        img,
        channels=channels,
        diameter=None,  # Auto-estimate nucleus diameter
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={'tile_norm_blocksize': tile_norm_blocksize}
    )

    logger.info(f"Nuclear segmentation complete!")
    logger.info(f"  - Number of nuclei detected: {len(np.unique(nuclear_masks)) - 1}")

    # Save results
    logger.info("Saving results...")

    image_name = img_path.stem

    # Save masks in multiple formats
    masks_tif_path = output_dir / f'{image_name}_masks.tif'
    io.imsave(masks_tif_path, masks)
    logger.info(f"Masks saved (TIFF): {masks_tif_path}")

    # Save flows (optional, for debugging)
    flows_path = output_dir / f'{image_name}_flows.tif'
    io.imsave(flows_path, flows[0])
    logger.info(f"Flows saved (TIFF): {flows_path}")

    # Create visualization
    create_visualization(img, masks, flows, output_dir, image_name)

    # Save nuclear masks in TIFF format
    nuclear_masks_tif_path = output_dir / f'{image_name}_nuclear_masks.tif'
    io.imsave(nuclear_masks_tif_path, nuclear_masks)
    logger.info(f"Nuclear masks saved (TIFF): {nuclear_masks_tif_path}")

    # Save pixel-to-cell mapping CSV
    logger.info("Generating pixel-to-cell mapping CSV files...")
    full_csv_path, summary_csv_path = save_pixel_to_cell_csv(masks, nuclear_masks, output_dir, image_name)

    # Summary
    logger.info("Processing complete!")
    logger.info("Output files:")
    logger.info(f"\nSegmentation results:")
    logger.info(f"  - Cell masks (TIFF): {masks_tif_path}")
    logger.info(f"  - Nuclear masks (TIFF): {nuclear_masks_tif_path}")
    logger.info(f"  - Flows (TIFF): {flows_path}")
    logger.info(f"  - Visualization: {output_dir / f'{image_name}_segmentation_overlay.png'}")
    logger.info(f"  - Colored masks: {output_dir / f'{image_name}_masks_colored.png'}")
    logger.info(f"\nPixel-level data:")
    logger.info(f"  - Full pixel mapping (compressed): {full_csv_path}")
    logger.info(f"  - Cell summary: {summary_csv_path}")
    logger.info(f"  - Cell masks array: {output_dir / f'{image_name}_cell_masks.npy'}")
    logger.info(f"  - Nuclear masks array: {output_dir / f'{image_name}_nuclear_masks.npy'}")
    logger.info(f"  - Boundary mask array: {output_dir / f'{image_name}_boundary_mask.npy'}")
    logger.info(f"\nLog file: cellpose_sam_segmentation.log")
    logger.info("\nCSV file formats:")
    logger.info("  {image_name}_pixel_to_cell_mapping_full.csv.gz columns:")
    logger.info("    - x, y: pixel coordinates")
    logger.info("    - cell_id: which cell this pixel belongs to (0 = background)")
    logger.info("    - is_boundary: 1 if pixel is at cell boundary (membrane), 0 otherwise")
    logger.info("    - is_nuclear: 1 if pixel is in nucleus, 0 otherwise")
    logger.info("    - is_interior: 1 if pixel is inside cell (non-boundary), 0 otherwise")
    logger.info("    - is_cytoplasm: 1 if pixel is in cytoplasm (interior - nuclear), 0 otherwise")
    logger.info("\n  {image_name}_cell_summary.csv columns:")
    logger.info("    - cell_id, num_pixels, num_interior_pixels, num_boundary_pixels")
    logger.info("    - center_x, center_y, min_x, max_x, min_y, max_y")

    return masks


if __name__ == "__main__":
    masks = main()
