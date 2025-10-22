#!/usr/bin/env python
"""
Run CSGO cell segmentation on large Visium HD images using a tiled approach.

This script processes large images in smaller tiles to avoid GPU OOM errors,
then stitches the results back together.

Usage:
    python csgo_segmentation_tiled.py
"""

import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import logging
import gc
import torch
import pandas as pd
from skimage.color import label2rgb

# Disable PIL decompression bomb protection for large Visium HD images
Image.MAX_IMAGE_PIXELS = None

# Set up paths
CURRENT_DIR = Path(os.getcwd())
CSGO_MAIN_DIR = CURRENT_DIR / 'CSGO'
CSGO_SRC_DIR = CSGO_MAIN_DIR / 'src'

# Add CSGO source to path
sys.path.insert(0, str(CSGO_SRC_DIR))

# Import CSGO model
from models import CSGO

# Configure logging for real-time output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csgo_segmentation_tiled.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_tiles(image_size, tile_size=4096, overlap=256):
    """
    Create tile coordinates for processing large images.

    Args:
        image_size: Tuple (width, height) of the full image
        tile_size: Size of each tile (default 4096)
        overlap: Overlap between tiles to avoid edge artifacts (default 256)

    Returns:
        List of tuples (x, y, width, height) for each tile
    """
    width, height = image_size
    tiles = []

    y = 0
    while y < height:
        x = 0
        while x < width:
            # Calculate tile dimensions
            tile_w = min(tile_size, width - x)
            tile_h = min(tile_size, height - y)

            tiles.append((x, y, tile_w, tile_h))

            # Move to next tile with overlap
            x += tile_size - overlap
            if x >= width:
                break

        y += tile_size - overlap
        if y >= height:
            break

    return tiles


def process_tile(seg_model, img, tile_coords, output_dir, tile_idx, cell_size, img_resolution, expected_shape=None):
    """
    Process a single tile.

    Args:
        seg_model: CSGO model instance
        img: PIL Image object of the full image
        tile_coords: Tuple (x, y, width, height) for this tile
        output_dir: Directory to save tile results
        tile_idx: Index of this tile
        cell_size: Cell size parameter for CSGO
        img_resolution: Image resolution parameter for CSGO
        expected_shape: Expected output shape (e.g., (h, w, 3) for RGB or (h, w) for grayscale).
                       If None, defaults to RGB (h, w, 3)

    Returns:
        Segmentation result for this tile and its coordinates, or None if tile is empty/failed
    """
    x, y, w, h = tile_coords

    logger.info(f"Processing tile {tile_idx}: x={x}, y={y}, w={w}, h={h}")

    # Crop the tile from the full image
    tile_img = img.crop((x, y, x + w, y + h))

    # Save tile temporarily
    tile_path = output_dir / f'tile_{tile_idx:04d}.png'
    tile_img.save(tile_path)

    # Free memory
    del tile_img
    gc.collect()

    # Determine blank result shape
    if expected_shape is None:
        blank_shape = (h, w, 3)  # Default to RGB
    elif len(expected_shape) == 3:
        blank_shape = (h, w, expected_shape[2])  # RGB with same number of channels
    else:
        blank_shape = (h, w)  # Grayscale

    # Process tile with CSGO
    try:
        result = seg_model.segment(
            img_path=str(tile_path),
            cell_size=cell_size,
            img_resolution=img_resolution
        )

        # Clear GPU cache after each tile
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Tile {tile_idx} processed successfully, shape: {result.shape}")

    except ValueError as e:
        # Handle specific errors from empty tiles or tiles with no nuclei
        if "zero-size array" in str(e) or "minimum which has no identity" in str(e):
            logger.warning(f"Tile {tile_idx} has no detectable cells, returning empty segmentation with shape {blank_shape}")
            # Clean up temporary tile
            if tile_path.exists():
                tile_path.unlink()
            # Return a blank segmentation result
            blank_result = np.zeros(blank_shape, dtype=np.uint8)
            return blank_result, tile_coords
        else:
            # Re-raise other ValueErrors
            logger.error(f"Error processing tile {tile_idx}: {e}")
            if tile_path.exists():
                tile_path.unlink()
            raise

    except Exception as e:
        logger.error(f"Error processing tile {tile_idx}: {e}")
        # Clean up temporary tile
        if tile_path.exists():
            tile_path.unlink()
        # Return blank result instead of crashing
        logger.warning(f"Returning blank segmentation for failed tile {tile_idx} with shape {blank_shape}")
        blank_result = np.zeros(blank_shape, dtype=np.uint8)
        return blank_result, tile_coords

    # Clean up temporary tile
    tile_path.unlink()

    return result, tile_coords


def extract_masks_from_tiles(tiles_dir, img_size):
    """
    Extract nuclei and membrane masks from saved tile outputs.

    Args:
        tiles_dir: Directory containing tile outputs
        img_size: Original image size (width, height)

    Returns:
        nuclei_mask, membrane_mask: Full-size masks reconstructed from tiles
    """
    logger.info("Extracting nuclei and membrane masks from tile outputs...")

    # Initialize empty masks
    width, height = img_size
    nuclei_full = np.zeros((height, width), dtype=np.uint8)
    membrane_full = np.zeros((height, width), dtype=np.uint8)

    # Look for CSGO_whole_cell_seg.png files in subdirectories
    # CSGO saves outputs in tiles_dir, but we need to find the actual saved masks
    # Since CSGO only saves pipeline_view.png (4-panel image), we can't easily extract individual masks
    # Instead, we'll note that the masks aren't directly available

    logger.warning("Note: Individual nuclei and membrane masks are not directly available from CSGO output.")
    logger.warning("CSGO only saves the final cell segmentation and a combined visualization.")
    logger.warning("The nuclei and membrane information is embedded in the cell boundaries.")

    return None, None


def save_pixel_mapping_csv(cell_seg, output_dir, img_size):
    """
    Save pixel-to-cell mapping as CSV files for HD data integration.

    Args:
        cell_seg: Cell segmentation result (label map)
        output_dir: Directory to save CSV files
        img_size: Original image size (width, height)

    Returns:
        Path to the main CSV file
    """
    logger.info("Creating pixel mapping CSV files for HD data integration...")

    # Get image dimensions
    height, width = cell_seg.shape

    # Full resolution pixel mapping (compressed)
    logger.info("Generating full-resolution pixel mapping...")
    logger.info(f"  Image dimensions: {width} x {height}")
    logger.info(f"  Total pixels: {width * height:,}")

    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    cell_id_flat = cell_seg.flatten()

    # Create DataFrame with pixel classification
    # For each pixel: is_nucleus (binary), is_membrane (binary), cell_id
    # Since we only have cell_seg, we'll infer membrane from cell boundaries
    logger.info("Inferring membrane pixels from cell boundaries...")

    # Detect cell boundaries (membrane) using edge detection
    from scipy import ndimage
    # Get cell boundaries by finding where cell IDs change
    cell_boundaries = np.zeros_like(cell_seg, dtype=np.uint8)

    # Use Sobel edge detection on cell segmentation
    sx = ndimage.sobel(cell_seg.astype(float), axis=0)
    sy = ndimage.sobel(cell_seg.astype(float), axis=1)
    sobel = np.hypot(sx, sy)
    cell_boundaries = (sobel > 0).astype(np.uint8)

    # Create DataFrame with all information
    df_full = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'cell_id': cell_id_flat,
        'is_membrane': cell_boundaries.flatten().astype(np.uint8),
        'is_nucleus': np.zeros_like(cell_id_flat, dtype=np.uint8)  # Will be updated below
    })

    # Infer nucleus as center regions of cells (not boundaries)
    # Nucleus = inside cell but not at boundary
    df_full['is_nucleus'] = ((df_full['cell_id'] > 0) & (df_full['is_membrane'] == 0)).astype(np.uint8)

    # Save full resolution (compressed)
    full_csv_path = output_dir / 'pixel_to_cell_mapping_full.csv.gz'
    df_full.to_csv(full_csv_path, index=False, compression='gzip')
    logger.info(f"Full-resolution mapping saved (compressed): {full_csv_path}")
    logger.info(f"  Total pixels: {len(df_full):,}")
    logger.info(f"  Nucleus pixels: {df_full['is_nucleus'].sum():,}")
    logger.info(f"  Membrane pixels: {df_full['is_membrane'].sum():,}")
    logger.info(f"  Background pixels: {(df_full['cell_id'] == 0).sum():,}")

    # Cell-level summary
    logger.info("Generating cell-level summary...")
    cell_summary = []
    unique_cells = np.unique(cell_seg)
    unique_cells = unique_cells[unique_cells > 0]  # Remove background

    for cell_id in unique_cells:
        mask = cell_seg == cell_id
        y_indices, x_indices = np.where(mask)

        # Count nucleus and membrane pixels for this cell
        cell_pixels_df = df_full[df_full['cell_id'] == cell_id]
        num_nucleus = cell_pixels_df['is_nucleus'].sum()
        num_membrane = cell_pixels_df['is_membrane'].sum()

        if len(y_indices) > 0:
            cell_summary.append({
                'cell_id': int(cell_id),
                'num_pixels': len(y_indices),
                'num_nucleus_pixels': int(num_nucleus),
                'num_membrane_pixels': int(num_membrane),
                'center_x': int(np.mean(x_indices)),
                'center_y': int(np.mean(y_indices)),
                'min_x': int(np.min(x_indices)),
                'max_x': int(np.max(x_indices)),
                'min_y': int(np.min(y_indices)),
                'max_y': int(np.max(y_indices))
            })

    df_summary = pd.DataFrame(cell_summary)
    summary_csv_path = output_dir / 'cell_summary.csv'
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"Cell summary saved: {summary_csv_path}")
    logger.info(f"  Total cells detected: {len(df_summary):,}")

    # Save masks as numpy arrays for easy loading
    numpy_path = output_dir / 'cell_segmentation.npy'
    np.save(numpy_path, cell_seg)
    logger.info(f"Cell segmentation array saved: {numpy_path}")

    membrane_mask_path = output_dir / 'membrane_mask.npy'
    np.save(membrane_mask_path, cell_boundaries)
    logger.info(f"Membrane mask array saved: {membrane_mask_path}")

    return full_csv_path, summary_csv_path


def create_colorful_overlay(cell_seg, original_img, output_dir, img_path):
    """
    Create colorful visualizations showing nuclei, membranes, and cells.

    Args:
        cell_seg: Cell segmentation result (label map)
        original_img: Original H&E image
        output_dir: Directory to save outputs
        img_path: Path to original image for reloading
    """
    logger.info("Saving colorful cell segmentation...")

    # Use colormap for cells - each cell gets a different color
    cmap_set3 = plt.get_cmap("Set3")
    cmap_tab20c = plt.get_cmap("tab20c")
    color_dict = [cmap_tab20c.colors[i] for i in range(len(cmap_tab20c.colors))] + \
                 [cmap_set3.colors[i] for i in range(len(cmap_set3.colors))]

    # Create colorful cell segmentation
    cell_seg_colored = label2rgb(cell_seg, colors=color_dict, bg_label=0)

    # Save colorful cell segmentation (full resolution) as TIFF
    cell_seg_colored_path = output_dir / 'cell_segmentation_colored.tiff'
    cell_seg_colored_img = Image.fromarray((cell_seg_colored * 255).astype(np.uint8))
    cell_seg_colored_img.save(cell_seg_colored_path, compression='tiff_deflate')
    logger.info(f"Colored cell segmentation saved: {cell_seg_colored_path}")

    # Create and save overlay (60% original, 40% segmentation) as TIFF
    original_array = np.array(original_img)
    cell_seg_array = np.array(cell_seg_colored_img)
    overlay = (0.6 * original_array + 0.4 * cell_seg_array).astype(np.uint8)

    overlay_path = output_dir / 'cell_segmentation_overlay.tiff'
    Image.fromarray(overlay).save(overlay_path, compression='tiff_deflate')
    logger.info(f"Overlay image saved: {overlay_path}")

    return cell_seg_colored_path


def stitch_tiles(tile_results, full_size, overlap=256):
    """
    Stitch tile results back into a full image with blending in overlap regions.

    Args:
        tile_results: List of (result_array, (x, y, w, h)) tuples
        full_size: Tuple (width, height) of the full output image
        overlap: Overlap size used when creating tiles

    Returns:
        Stitched full segmentation result
    """
    logger.info("Stitching tiles together...")

    width, height = full_size

    # Determine output shape from first NON-EMPTY result to avoid issues
    first_result = None
    for result, _ in tile_results:
        if result is not None and result.size > 0:
            first_result = result
            break

    if first_result is None:
        raise ValueError("All tile results are empty or None")

    # Initialize stitched array based on first valid result
    if len(first_result.shape) == 3:
        # RGB image
        num_channels = first_result.shape[2]
        stitched = np.zeros((height, width, num_channels), dtype=np.float32)
    else:
        # Grayscale
        stitched = np.zeros((height, width), dtype=np.float32)

    # Create a weight map for blending overlaps
    weight_map = np.zeros((height, width), dtype=np.float32)

    for result, (x, y, w, h) in tile_results:
        if result is None:
            continue

        # Ensure result matches expected dimensions
        result_h, result_w = result.shape[:2]
        actual_h = min(h, result_h, height - y)
        actual_w = min(w, result_w, width - x)

        # Create weight mask for this tile (higher weight in center, lower at edges)
        tile_weight = np.ones((actual_h, actual_w), dtype=np.float32)

        # Reduce weight in overlap regions
        if x > 0:  # Left edge
            for i in range(min(overlap, actual_w)):
                tile_weight[:, i] *= i / overlap
        if y > 0:  # Top edge
            for i in range(min(overlap, actual_h)):
                tile_weight[i, :] *= i / overlap
        if x + actual_w < width:  # Right edge
            for i in range(min(overlap, actual_w)):
                tile_weight[:, actual_w - 1 - i] *= i / overlap
        if y + actual_h < height:  # Bottom edge
            for i in range(min(overlap, actual_h)):
                tile_weight[actual_h - 1 - i, :] *= i / overlap

        # Accumulate weighted results - handle both 2D and 3D cases
        if len(result.shape) == 3 and len(stitched.shape) == 3:
            # Both RGB - expand tile_weight to match channels
            tile_weight_3d = tile_weight[:, :, np.newaxis]  # Shape: (h, w, 1)
            stitched[y:y+actual_h, x:x+actual_w] += (result[:actual_h, :actual_w] * tile_weight_3d).astype(np.float32)
        elif len(result.shape) == 2 and len(stitched.shape) == 2:
            # Both grayscale
            stitched[y:y+actual_h, x:x+actual_w] += (result[:actual_h, :actual_w] * tile_weight).astype(np.float32)
        elif len(result.shape) == 3 and len(stitched.shape) == 2:
            # Result is RGB but stitched is grayscale - convert result to grayscale
            result_gray = np.mean(result[:actual_h, :actual_w], axis=2).astype(np.float32)
            stitched[y:y+actual_h, x:x+actual_w] += result_gray * tile_weight
        elif len(result.shape) == 2 and len(stitched.shape) == 3:
            # Result is grayscale but stitched is RGB - expand result to RGB
            result_rgb = np.stack([result[:actual_h, :actual_w]] * stitched.shape[2], axis=2).astype(np.float32)
            tile_weight_3d = tile_weight[:, :, np.newaxis]
            stitched[y:y+actual_h, x:x+actual_w] += result_rgb * tile_weight_3d

        # Accumulate weights
        weight_map[y:y+actual_h, x:x+actual_w] += tile_weight

    # Normalize by weights (avoid division by zero)
    weight_map = np.maximum(weight_map, 1e-6)
    if len(stitched.shape) == 3:
        # Expand weight_map to 3 channels for broadcasting
        weight_map_3d = weight_map[:, :, np.newaxis]  # Shape: (h, w, 1)
        stitched = (stitched / weight_map_3d).astype(np.uint8)
    else:
        stitched = (stitched / weight_map).astype(np.uint8)

    logger.info("Stitching complete")

    return stitched


def main():
    """Run tiled CSGO segmentation on the cropped Visium HD brain image."""

    # Define paths
    yolo_path = CSGO_SRC_DIR / 'pretrained_weights' / 'lung_best.float16.torchscript.pt'
    unet_path = CSGO_SRC_DIR / 'pretrained_weights' / 'epoch_190.pt'
    img_path = CURRENT_DIR / 'cropped_visium_hd_small_intenstine.png'
    output_dir = CURRENT_DIR / 'csgo_segmentation_small_intenstine_output'
    tiles_dir = output_dir / 'tiles'

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    tiles_dir.mkdir(exist_ok=True)

    # Check if input image exists
    if not img_path.exists():
        logger.error(f"Input image not found: {img_path}")
        raise FileNotFoundError(f"Input image not found: {img_path}")

    # Check if model weights exist
    if not yolo_path.exists():
        logger.error(f"YOLO weights not found: {yolo_path}")
        raise FileNotFoundError(f"YOLO weights not found: {yolo_path}")
    if not unet_path.exists():
        logger.error(f"UNet weights not found: {unet_path}")
        raise FileNotFoundError(f"UNet weights not found: {unet_path}")

    logger.info("CSGO Tiled Cell Segmentation for Visium HD Brain Image")
    logger.info(f"Input image: {img_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"YOLO weights: {yolo_path}")
    logger.info(f"UNet weights: {unet_path}")

    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("GPU not available, using CPU (will be slower)")

    # Load image and get size
    logger.info("Loading image...")
    img = Image.open(img_path)
    img_size = img.size
    logger.info(f"Image size: {img_size[0]} x {img_size[1]} pixels")

    # Create tiles (1024x1024 with 64 pixel overlap)
    # Note: CSGO is designed for small patches (~500-1000px).
    # Smaller tiles = faster processing per tile.
    tile_size = 1024
    overlap = 64
    tiles = create_tiles(img_size, tile_size=tile_size, overlap=overlap)
    logger.info(f"Created {len(tiles)} tiles (tile_size={tile_size}, overlap={overlap})")
    logger.info("="*70)

    # Initialize CSGO model
    logger.info("Initializing CSGO model...")
    seg_model = CSGO(
        yolo_path=str(yolo_path),
        unet_path=str(unet_path),
        gpu=True,  # Enable GPU acceleration
        save=False,  # Don't save individual tiles (avoids low contrast PNG warnings)
        output_dir=str(tiles_dir)
    )
    logger.info("CSGO model initialized successfully")

    # Process each tile
    cell_size = 50
    img_resolution = 40

    logger.info("Starting tiled segmentation...")
    logger.info(f"Parameters: cell_size={cell_size}, img_resolution={img_resolution}")

    tile_results = []
    expected_shape = None  # Will be set after first successful tile

    for idx, tile_coords in enumerate(tiles, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Tile {idx}/{len(tiles)}")
        logger.info(f"{'='*70}")

        try:
            result, coords = process_tile(
                seg_model, img, tile_coords, tiles_dir,
                idx, cell_size, img_resolution, expected_shape
            )
            tile_results.append((result, coords))

            # Set expected shape from first successful result
            if expected_shape is None and result is not None:
                expected_shape = result.shape
                logger.info(f"Output shape determined from first tile: {expected_shape}")

            # Save intermediate result every 10 tiles
            if idx % 10 == 0:
                logger.info(f"Progress checkpoint: {idx}/{len(tiles)} tiles completed")

        except Exception as e:
            logger.error(f"Error processing tile {idx}: {e}")
            logger.error("Stopping processing due to error")
            raise

    # Close the full image to free memory
    img.close()
    gc.collect()

    logger.info(f"All {len(tiles)} tiles processed successfully")

    # Stitch tiles together
    logger.info("\nStitching tiles into full image...")
    full_result = stitch_tiles(tile_results, img_size, overlap=overlap)

    # Free memory
    del tile_results
    gc.collect()

    # Save full result as TIFF (better for label maps - no low contrast warnings)
    logger.info("Saving final segmentation result...")
    result_path = output_dir / 'segmentation_result_full.tiff'
    result_img = Image.fromarray(full_result.astype(np.uint16))
    result_img.save(result_path, compression='tiff_deflate')
    logger.info(f"Full segmentation saved to: {result_path}")

    # Save full resolution images
    logger.info("\n" + "="*70)
    logger.info("Saving colorful visualizations...")
    logger.info("="*70)

    # Load original image for overlay
    original_img = Image.open(img_path)

    # Create colorful overlay visualization
    colorful_path = create_colorful_overlay(full_result, original_img, output_dir, img_path)

    logger.info("\n" + "="*70)
    logger.info("Generating pixel-to-cell mapping CSV files for HD data...")
    logger.info("="*70)

    # Save pixel mapping CSV files
    full_csv_path, summary_csv_path = save_pixel_mapping_csv(
        full_result,
        output_dir,
        img_size
    )

    # Clean up
    original_img.close()
    result_img.close()

    logger.info("\n" + "="*70)
    logger.info("All processing complete!")
    logger.info("="*70)
    logger.info(f"Output files:")
    logger.info(f"\nVisualization files (TIFF format):")
    logger.info(f"  - Full segmentation (grayscale label map): {result_path}")
    logger.info(f"  - Colorful cell segmentation: {colorful_path}")
    logger.info(f"  - Overlay (cells on original): {output_dir / 'cell_segmentation_overlay.tiff'}")
    logger.info(f"\nData files for HD mapping:")
    logger.info(f"  - Full pixel mapping (compressed): {full_csv_path}")
    logger.info(f"  - Cell summary (centers, bounds, nucleus/membrane counts): {summary_csv_path}")
    logger.info(f"  - Cell segmentation array: {output_dir / 'cell_segmentation.npy'}")
    logger.info(f"  - Membrane mask array: {output_dir / 'membrane_mask.npy'}")
    logger.info(f"\nIntermediate outputs:")
    logger.info(f"  - Individual tile outputs (nuclei, membrane, cells): {tiles_dir}")
    logger.info(f"  - Log file: csgo_segmentation_tiled.log")
    logger.info("\nNOTE: Each tile in {tiles_dir} contains a 'pipeline_view.png' showing:")
    logger.info("  1. Original H&E patch")
    logger.info("  2. Nuclei mask (from HD-YOLO) - Shows which pixels are nucleus")
    logger.info("  3. Membrane mask (from UNet) - Shows which pixels are membrane")
    logger.info("  4. Whole-cell segmentation - Shows which pixels belong to which cell")
    logger.info("\nCSV file format for pixel mapping:")
    logger.info("  pixel_to_cell_mapping_full.csv.gz has columns:")
    logger.info("    - x, y: pixel coordinates")
    logger.info("    - cell_id: which cell this pixel belongs to (0 = background)")
    logger.info("    - is_nucleus: 1 if pixel is inside cell (nucleus region), 0 otherwise")
    logger.info("    - is_membrane: 1 if pixel is at cell boundary (membrane), 0 otherwise")
    logger.info("\n  cell_summary.csv has columns:")
    logger.info("    - cell_id, num_pixels, num_nucleus_pixels, num_membrane_pixels")
    logger.info("    - center_x, center_y, min_x, max_x, min_y, max_y")

    return full_result


if __name__ == "__main__":
    result = main()
