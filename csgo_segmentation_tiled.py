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
import numpy as np
from PIL import Image
import logging
import gc
import torch

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


def process_tile(seg_model, img, tile_coords, output_dir, tile_idx, cell_size, img_resolution):
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

        logger.info(f"Tile {tile_idx} processed successfully")

    except ValueError as e:
        # Handle specific errors from empty tiles or tiles with no nuclei
        if "zero-size array" in str(e) or "minimum which has no identity" in str(e):
            logger.warning(f"Tile {tile_idx} has no detectable cells, returning empty segmentation")
            # Clean up temporary tile
            if tile_path.exists():
                tile_path.unlink()
            # Return a blank segmentation result
            blank_result = np.zeros((h, w, 3), dtype=np.uint8)
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
        logger.warning(f"Returning blank segmentation for failed tile {tile_idx}")
        blank_result = np.zeros((h, w, 3), dtype=np.uint8)
        return blank_result, tile_coords

    # Clean up temporary tile
    tile_path.unlink()

    return result, tile_coords


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

    # Determine output shape from first result
    first_result = tile_results[0][0]
    if len(first_result.shape) == 3:
        # RGB image
        stitched = np.zeros((height, width, first_result.shape[2]), dtype=np.uint8)
    else:
        # Grayscale
        stitched = np.zeros((height, width), dtype=np.uint8)

    # Create a weight map for blending overlaps
    weight_map = np.zeros((height, width), dtype=np.float32)

    for result, (x, y, w, h) in tile_results:
        # Create weight mask for this tile (higher weight in center, lower at edges)
        tile_weight = np.ones((h, w), dtype=np.float32)

        # Reduce weight in overlap regions
        if x > 0:  # Left edge
            for i in range(min(overlap, w)):
                tile_weight[:, i] *= i / overlap
        if y > 0:  # Top edge
            for i in range(min(overlap, h)):
                tile_weight[i, :] *= i / overlap
        if x + w < width:  # Right edge
            for i in range(min(overlap, w)):
                tile_weight[:, w - 1 - i] *= i / overlap
        if y + h < height:  # Bottom edge
            for i in range(min(overlap, h)):
                tile_weight[h - 1 - i, :] *= i / overlap

        # Accumulate weighted results
        if len(stitched.shape) == 3:
            # For RGB images, expand tile_weight to 3 channels
            tile_weight_3d = tile_weight[:, :, np.newaxis]  # Shape: (h, w, 1)
            for c in range(stitched.shape[2]):
                stitched[y:y+h, x:x+w, c] += (result[:h, :w, c] * tile_weight).astype(np.uint8)
        else:
            stitched[y:y+h, x:x+w] += (result[:h, :w] * tile_weight).astype(np.uint8)

        # Accumulate weights
        weight_map[y:y+h, x:x+w] += tile_weight

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
    img_path = CURRENT_DIR / 'cropped_visium_hd_brain.png'
    output_dir = CURRENT_DIR / 'csgo_segmentation_output'
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
        save=False,  # Don't save individual tiles
        output_dir=str(tiles_dir)
    )
    logger.info("CSGO model initialized successfully")

    # Process each tile
    cell_size = 50
    img_resolution = 40

    logger.info("Starting tiled segmentation...")
    logger.info(f"Parameters: cell_size={cell_size}, img_resolution={img_resolution}")

    tile_results = []
    for idx, tile_coords in enumerate(tiles, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Tile {idx}/{len(tiles)}")
        logger.info(f"{'='*70}")

        try:
            result, coords = process_tile(
                seg_model, img, tile_coords, tiles_dir,
                idx, cell_size, img_resolution
            )
            tile_results.append((result, coords))

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

    # Save full result
    logger.info("Saving final segmentation result...")
    result_path = output_dir / 'segmentation_result_full.png'
    result_img = Image.fromarray(full_result)
    result_img.save(result_path, optimize=True)
    logger.info(f"Full segmentation saved to: {result_path}")

    # Create comparison visualization (downsampled for memory efficiency)
    logger.info("Creating comparison visualization...")

    # Downsample for visualization
    max_display_size = 8000
    if max(img_size) > max_display_size:
        scale = max_display_size / max(img_size)
        new_size = (int(img_size[0] * scale), int(img_size[1] * scale))
        logger.info(f"Downsampling for visualization: {new_size[0]} x {new_size[1]}")

        original_img = Image.open(img_path).resize(new_size, Image.Resampling.LANCZOS)
        result_img_display = result_img.resize(new_size, Image.Resampling.LANCZOS)
    else:
        original_img = Image.open(img_path)
        result_img_display = result_img

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(original_img)
    axes[0].set_title('Original Cropped Image', fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(result_img_display)
    axes[1].set_title(f'CSGO Segmentation (Tiled: {len(tiles)} tiles)', fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    comparison_path = output_dir / 'segmentation_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Comparison saved to: {comparison_path}")

    # Clean up
    original_img.close()
    result_img.close()
    if 'result_img_display' in locals():
        result_img_display.close()

    logger.info("All processing complete!")
    logger.info(f"Output files:")
    logger.info(f"  - Full segmentation: {result_path}")
    logger.info(f"  - Comparison: {comparison_path}")
    logger.info(f"  - Log file: csgo_segmentation_tiled.log")

    return full_result


if __name__ == "__main__":
    result = main()
