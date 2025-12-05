#!/usr/bin/env python
"""
Expand cell boundaries and cytoplasm around nuclei based on cell type-specific N:C ratios.

This script:
1. Loads pixel-to-cell mapping with cell types and current segmentation
2. Keeps nuclear regions EXACTLY the same (no changes to nucleus)
3. Expands cytoplasm and boundary outward from current cell edge
4. Expands each cell until reaching target nuclear:interior ratio for its cell type
5. Uses cell type-specific nuclear percentage ranges (e.g., B cells 75-80%, Epithelial 25-33%)
6. Generates updated cell boundaries and cytoplasm regions
7. Creates visualization comparing before/after

The expansion is fully vectorized for performance on 600M+ pixels.

Usage:
    python nuclear_expansion.py --input_csv <input> --output_csv <output> --nuclear_range_csv <ranges>
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
from scipy.sparse import coo_matrix
from skimage.morphology import disk, ellipse
import gzip
import argparse

# Disable PIL decompression bomb protection
Image.MAX_IMAGE_PIXELS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nuclear_expansion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_nuclear_ratio_ranges(csv_path):
    """
    Load cell type-specific nuclear percentage ranges.

    Args:
        csv_path: Path to cell_based_nuclear_range.csv

    Returns:
        Dictionary mapping cell_type -> (min_percent, max_percent)
    """
    logger.info(f"Loading nuclear ratio ranges from {csv_path}...")
    df = pd.read_csv(csv_path)

    ratio_ranges = {}
    for _, row in df.iterrows():
        cell_type = row['cell_type']
        min_pct = float(row['min_nuclear_percent'])
        max_pct = float(row['max_nuclear_percent'])
        ratio_ranges[cell_type] = (min_pct, max_pct)

    logger.info(f"  Loaded ranges for {len(ratio_ranges)} cell types:")
    for cell_type, (min_pct, max_pct) in ratio_ranges.items():
        logger.info(f"    {cell_type}: {min_pct}% - {max_pct}%")

    return ratio_ranges


def process_cell_expansion_morphological(df, ratio_ranges, image_shape):
    """
    Expand all cells based on their cell type-specific N:C ratios using morphological dilation.
    Keeps nucleus EXACTLY the same, only expands cytoplasm and boundary outward.
    ALL cells are dilated simultaneously in each iteration (100-300x faster than per-cell approach).
    Uses circular kernel for natural, elliptical cell shapes.

    Args:
        df: DataFrame with pixel-level data including cell_type
        ratio_ranges: Dict mapping cell_type -> (min_pct, max_pct)
        image_shape: Tuple of (height, width) for the tissue image

    Returns:
        df_expanded: DataFrame with updated is_nuclear, is_cytoplasm, is_interior, is_boundary
    """
    logger.info("Expanding cells around nuclei using morphological dilation...")
    logger.info(f"  Nuclear regions will remain UNCHANGED")
    logger.info(f"  Image shape: {image_shape}")

    # Create full-resolution masks
    height, width = image_shape
    cell_mask = np.zeros((height, width), dtype=np.uint32)  # For initial single-cell assignment
    nuclear_mask = np.zeros((height, width), dtype=np.uint8)

    # Fill masks from dataframe (vectorized)
    logger.info("  Building initial masks from pixel data...")
    pixels_with_cells = df[df['cell_id'] > 0]
    y_coords = pixels_with_cells['y'].values.astype(np.int32)
    x_coords = pixels_with_cells['x'].values.astype(np.int32)
    cell_ids = pixels_with_cells['cell_id'].values.astype(np.uint32)
    is_nuclear = pixels_with_cells['is_nuclear'].values.astype(np.uint8)

    cell_mask[y_coords, x_coords] = cell_ids
    nuclear_mask[y_coords, x_coords] = is_nuclear

    logger.info(f"  Initial nuclear pixels: {np.sum(nuclear_mask):,}")
    logger.info(f"  Initial total cell pixels: {np.sum(cell_mask > 0):,}")

    # Calculate current nuclear:interior ratios
    total_nuclear = np.sum(nuclear_mask)
    total_interior = np.sum(cell_mask > 0)
    if total_interior > 0:
        current_ratio = 100.0 * total_nuclear / total_interior
        logger.info(f"  Current overall nuclear:interior ratio: {current_ratio:.1f}%")

    # Get unique cells (exclude background)
    unique_cells = df[df['cell_id'] > 0]['cell_id'].unique()
    num_cells = len(unique_cells)
    logger.info(f"  Processing {num_cells:,} cells...")

    # Pre-calculate target sizes for each cell (VECTORIZED)
    logger.info("  Calculating target sizes for each cell (vectorized)...")

    # Get cell type map
    cell_type_map = df[df['cell_id'] > 0].groupby('cell_id')['cell_type'].first().to_dict()

    # Count nuclear and total pixels per cell (vectorized using bincount)
    logger.info("    Counting pixels per cell (vectorized)...")
    max_cell_id = int(cell_mask.max())

    # Count total pixels per cell
    cell_total_counts = np.bincount(cell_mask.ravel(), minlength=max_cell_id+1)

    # Count nuclear pixels per cell
    nuclear_per_cell = np.zeros(max_cell_id+1, dtype=np.int64)
    nuclear_coords_y, nuclear_coords_x = np.where(nuclear_mask > 0)
    nuclear_cell_ids = cell_mask[nuclear_coords_y, nuclear_coords_x]
    nuclear_per_cell = np.bincount(nuclear_cell_ids, minlength=max_cell_id+1)

    logger.info("    Computing target sizes...")
    cell_targets = {}  # cell_id -> (current_nuclear_pixels, target_total_pixels)
    locked_cells = set()  # Initialize locked_cells early (for Unknown cells)

    for cell_id in unique_cells:
        cell_nuclear_pixels = int(nuclear_per_cell[cell_id])
        current_size = int(cell_total_counts[cell_id])

        if cell_nuclear_pixels == 0:
            # No nucleus - keep current size
            cell_targets[cell_id] = (0, current_size)
            continue

        # Get cell type
        cell_type = cell_type_map.get(cell_id, None)

        # Get target nuclear percentage for this cell type
        if cell_type not in ratio_ranges or pd.isna(cell_type) or cell_type == 'Unknown':
            # Unknown cell type - lock immediately, don't expand
            cell_targets[cell_id] = (cell_nuclear_pixels, current_size)
            locked_cells.add(cell_id)  # Lock Unknown cells immediately
            continue

        min_pct, max_pct = ratio_ranges[cell_type]
        # Randomly choose percentage within range for this specific cell
        target_pct = np.random.uniform(min_pct, max_pct)

        # Calculate target total cell size
        # target_pct = (nuclear_pixels / total_pixels) * 100
        # total_pixels = nuclear_pixels / (target_pct / 100)
        target_total_pixels = int(cell_nuclear_pixels / (target_pct / 100.0))

        cell_targets[cell_id] = (cell_nuclear_pixels, target_total_pixels)

    logger.info(f"  Set individual targets for {len(cell_targets):,} cells")
    logger.info(f"  Pre-locked {len(locked_cells):,} Unknown cells")

    # Iterative expansion using CIRCULAR DISK pattern (more natural cell shape!)
    logger.info("  Starting iterative expansion with circular disk pattern...")
    expanded_cell_mask = cell_mask.copy()
    iteration = 0
    max_iterations = 200
    min_cells_locked_per_iteration = 5
    early_stop_check_interval = 20

    # Pre-compute disk(3) offsets for circular expansion pattern
    # This creates a circular neighborhood without expensive morphological dilation
    disk_offsets = []
    radius = 3  # disk(3) - same as before, but without slow morphological dilation
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            # Check if within circular radius (Euclidean distance)
            if dy*dy + dx*dx <= radius*radius:
                disk_offsets.append((dy, dx))

    logger.info(f"  Using circular expansion with radius {radius} (disk({radius}) = {len(disk_offsets)} neighbor positions)")
    logger.info(f"  Max iterations: {max_iterations}, early stop if <{min_cells_locked_per_iteration} cells lock per {early_stop_check_interval} iterations")

    # Track progress for early stopping
    last_check_iteration = 0
    last_check_locked_count = len(locked_cells)

    # Track current sizes incrementally to avoid expensive bincount
    current_sizes = cell_total_counts.copy()  # Start with initial sizes

    # Create locked mask more efficiently
    locked_cells_array = np.zeros(max_cell_id+1, dtype=bool)
    for cell_id in locked_cells:
        locked_cells_array[cell_id] = True

    while len(locked_cells) < num_cells and iteration < max_iterations:
        iteration += 1

        # Create locked mask from cell IDs (MUCH faster than scanning full mask!)
        locked_mask = np.isin(expanded_cell_mask, np.where(locked_cells_array)[0])

        # Find empty pixels within circular distance of unlocked cells
        unlocked_cells = (expanded_cell_mask > 0) & (~locked_mask)

        # Find candidate pixels using circular neighborhood (VECTORIZED!)
        new_pixels = np.zeros((height, width), dtype=bool)
        neighbor_cells = []  # Store which cell each direction belongs to

        for dy, dx in disk_offsets:
            if dy == 0 and dx == 0:
                continue  # Skip center pixel

            # Use numpy roll to shift the masks (much simpler!)
            # Roll shifts array, wrapping around (we'll ignore wrapped edges)
            shifted_cells = np.roll(np.roll(expanded_cell_mask, -dy, axis=0), -dx, axis=1)
            shifted_unlocked = np.roll(np.roll(unlocked_cells, -dy, axis=0), -dx, axis=1)

            # Ignore wrapped edges by zeroing them out
            if dy != 0:
                if dy > 0:
                    shifted_cells[:dy, :] = 0
                    shifted_unlocked[:dy, :] = False
                else:
                    shifted_cells[dy:, :] = 0
                    shifted_unlocked[dy:, :] = False

            if dx != 0:
                if dx > 0:
                    shifted_cells[:, :dx] = 0
                    shifted_unlocked[:, :dx] = False
                else:
                    shifted_cells[:, dx:] = 0
                    shifted_unlocked[:, dx:] = False

            # Mark empty pixels that have unlocked cell neighbors at this offset
            new_pixels |= shifted_unlocked & (expanded_cell_mask == 0)
            neighbor_cells.append(shifted_cells)

        if not np.any(new_pixels):
            logger.info(f"  Iteration {iteration}: No new pixels to add, stopping")
            break

        # Assign new pixels from circular neighborhood
        # Try each direction in the circular disk pattern
        for cells_neighbor in neighbor_cells:
            # Valid: neighbor is non-zero, not locked, and pixel still needs assignment
            valid = (cells_neighbor > 0) & new_pixels & (~locked_cells_array[cells_neighbor])

            if np.any(valid):
                # Assign pixels
                expanded_cell_mask[valid] = cells_neighbor[valid]

                # Increment sizes for cells that grew (INCREMENTAL UPDATE!)
                growing_cells, pixel_counts = np.unique(cells_neighbor[valid], return_counts=True)
                for cell_id, count in zip(growing_cells, pixel_counts):
                    current_sizes[cell_id] += count

                # Mark pixels as assigned
                new_pixels[valid] = False

                if not np.any(new_pixels):
                    break

        # Check which cells reached their targets
        newly_locked = []
        for cell_id in unique_cells:
            if locked_cells_array[cell_id]:  # Already locked
                continue

            current_size = int(current_sizes[cell_id])
            _, target_size = cell_targets[cell_id]

            if current_size >= target_size:
                locked_cells.add(cell_id)
                locked_cells_array[cell_id] = True
                newly_locked.append(cell_id)

        if iteration % 10 == 0 or len(newly_locked) > 0:
            logger.info(f"  Iteration {iteration}: {len(locked_cells):,} / {num_cells:,} cells locked ({100.0*len(locked_cells)/num_cells:.1f}%)")

        # If all cells locked, stop
        if len(locked_cells) >= num_cells:
            logger.info(f"  All cells reached their targets after {iteration} iterations")
            break

        # Early stopping: Check if progress is too slow
        if iteration - last_check_iteration >= early_stop_check_interval:
            cells_locked_in_interval = len(locked_cells) - last_check_locked_count
            if cells_locked_in_interval < min_cells_locked_per_iteration:
                logger.info(f"  Early stopping at iteration {iteration}: Only {cells_locked_in_interval} cells locked in last {early_stop_check_interval} iterations")
                logger.info(f"  {num_cells - len(locked_cells):,} cells did not reach their targets (likely blocked by neighbors)")
                break
            last_check_iteration = iteration
            last_check_locked_count = len(locked_cells)

    logger.info(f"  Completed morphological dilation after {iteration} iterations")
    logger.info(f"  Locked {len(locked_cells):,} / {num_cells:,} cells")

    # Generate new pixel-level data from expanded mask
    logger.info("  Generating updated pixel-level data...")

    # Extract all pixels from expanded_cell_mask
    y_indices, x_indices = np.where(expanded_cell_mask > 0)
    cell_id_values = expanded_cell_mask[y_indices, x_indices]

    df_new = pd.DataFrame({
        'y': y_indices,
        'x': x_indices,
        'cell_id': cell_id_values
    })

    logger.info(f"  Created {len(df_new):,} pixel-cell assignments")

    # Add nuclear flag (vectorized lookup!)
    logger.info("  Adding nuclear flags (vectorized)...")
    df_new['is_nuclear'] = nuclear_mask[df_new['y'].values, df_new['x'].values]

    # Compute boundaries using FULLY VECTORIZED mask operations (NO LOOPS!)
    logger.info("  Computing boundaries (fully vectorized)...")

    # Build boundary mask from expanded_cell_mask directly
    # A pixel is a boundary if any of its 4 neighbors has a different cell_id or is empty
    boundary_mask = np.zeros((height, width), dtype=bool)

    # Check UP neighbor
    boundary_mask[1:, :] |= (expanded_cell_mask[1:, :] != expanded_cell_mask[:-1, :])

    # Check DOWN neighbor
    boundary_mask[:-1, :] |= (expanded_cell_mask[:-1, :] != expanded_cell_mask[1:, :])

    # Check LEFT neighbor
    boundary_mask[:, 1:] |= (expanded_cell_mask[:, 1:] != expanded_cell_mask[:, :-1])

    # Check RIGHT neighbor
    boundary_mask[:, :-1] |= (expanded_cell_mask[:, :-1] != expanded_cell_mask[:, 1:])

    # Lookup boundary flag for each pixel in df_new
    df_new['is_boundary'] = boundary_mask[df_new['y'].values, df_new['x'].values].astype(np.uint8)

    # Compute derived columns (vectorized)
    df_new['is_interior'] = 1 - df_new['is_boundary']
    df_new['is_cytoplasm'] = ((df_new['is_interior'] == 1) & (df_new['is_nuclear'] == 0)).astype(np.uint8)
    logger.info(f"  Computed all pixel properties")

    # Merge with cell type from original dataframe
    logger.info("  Merging cell type annotations...")
    cell_type_map = df[df['cell_id'] > 0].groupby('cell_id')['cell_type'].first().to_dict()
    df_new['cell_type'] = df_new['cell_id'].map(cell_type_map)

    logger.info(f"  Output: {len(df_new):,} pixels")
    logger.info(f"  Nuclear pixels: {df_new['is_nuclear'].sum():,}")
    logger.info(f"  Cytoplasm pixels: {df_new['is_cytoplasm'].sum():,}")
    logger.info(f"  Boundary pixels: {df_new['is_boundary'].sum():,}")

    return df_new, expanded_cell_mask, nuclear_mask


def visualize_expansion(original_image_path, expanded_cell_mask, nuclear_mask, output_path, downsample=4):
    """
    Create visualization comparing original and expanded segmentation.

    Args:
        original_image_path: Path to original H&E image
        expanded_cell_mask: Expanded cell segmentation mask
        nuclear_mask: Nuclear segmentation mask
        output_path: Path to save visualization
        downsample: Downsampling factor for visualization
    """
    logger.info(f"Creating visualization (downsampled {downsample}x)...")

    # Load original image
    if Path(original_image_path).exists():
        logger.info(f"  Loading original image from {original_image_path}...")
        img = np.array(Image.open(original_image_path))
    else:
        logger.info("  Original image not found, creating blank background...")
        img = np.ones((*expanded_cell_mask.shape, 3), dtype=np.uint8) * 255

    # Downsample
    if downsample > 1:
        img_ds = img[::downsample, ::downsample]
        cell_ds = expanded_cell_mask[::downsample, ::downsample]
        nuclear_ds = nuclear_mask[::downsample, ::downsample]
    else:
        img_ds = img
        cell_ds = expanded_cell_mask
        nuclear_ds = nuclear_mask

    # Create overlay
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Original image
    axes[0].imshow(img_ds)
    axes[0].set_title('Original H&E Image', fontsize=14)
    axes[0].axis('off')

    # Cell boundaries with nuclei
    overlay = img_ds.copy()

    # Find cell boundaries
    cell_boundary = np.zeros_like(cell_ds, dtype=bool)
    shifted_up = np.roll(cell_ds, -1, axis=0)
    shifted_down = np.roll(cell_ds, 1, axis=0)
    shifted_left = np.roll(cell_ds, -1, axis=1)
    shifted_right = np.roll(cell_ds, 1, axis=1)

    cell_boundary = (
        (cell_ds > 0) & (
            (cell_ds != shifted_up) |
            (cell_ds != shifted_down) |
            (cell_ds != shifted_left) |
            (cell_ds != shifted_right)
        )
    )

    # Overlay boundaries (green) and nuclei (red)
    overlay[cell_boundary] = [0, 255, 0]  # Green boundaries
    overlay[nuclear_ds > 0] = [255, 0, 0]  # Red nuclei

    axes[1].imshow(overlay)
    axes[1].set_title('Expanded Cells (Green) + Nuclei (Red)', fontsize=14)
    axes[1].axis('off')

    # Nuclei only
    nuclear_overlay = img_ds.copy()
    nuclear_overlay[nuclear_ds > 0] = [255, 0, 0]

    axes[2].imshow(nuclear_overlay)
    axes[2].set_title('Nuclear Segmentation Only', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved visualization to {output_path}")


def create_full_resolution_overlay(expanded_cell_mask, nuclear_mask, output_path):
    """
    Create full-resolution overlay showing nuclear (blue) vs cytoplasm (red) regions.

    Args:
        expanded_cell_mask: Full-resolution cell mask
        nuclear_mask: Full-resolution nuclear mask
        output_path: Path to save overlay image
    """
    logger.info("Creating full-resolution nuclear vs cytoplasm overlay...")

    height, width = expanded_cell_mask.shape

    # Create RGB image (white background)
    overlay = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Mark cytoplasm regions (cells - nuclei) in RED
    cytoplasm_mask = (expanded_cell_mask > 0) & (nuclear_mask == 0)
    overlay[cytoplasm_mask] = [255, 0, 0]  # Red

    # Mark nuclear regions in BLUE (on top)
    nuclear_regions = nuclear_mask > 0
    overlay[nuclear_regions] = [0, 0, 255]  # Blue

    # Save full-resolution image
    from PIL import Image
    logger.info(f"  Saving full-resolution overlay: {width} x {height} pixels...")
    img = Image.fromarray(overlay)
    img.save(output_path)

    logger.info(f"  Saved full-resolution overlay to {output_path}")
    logger.info(f"  Legend: Blue = Nuclear, Red = Cytoplasm, White = Background")


def main():
    """Main pipeline for nuclear expansion."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Expand cell boundaries based on cell type-specific N:C ratios')
    parser.add_argument('--input_csv', type=str,
                       default='cellpose_sam_human_colorectal_output/cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_with_celltype.csv.gz',
                       help='Path to input pixel-to-cell mapping CSV (gzipped)')
    parser.add_argument('--output_csv', type=str,
                       default='cellpose_sam_human_colorectal_output/cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_expanded.csv.gz',
                       help='Path to output expanded pixel mapping CSV (gzipped)')
    parser.add_argument('--output_vis', type=str,
                       default='cellpose_sam_human_colorectal_output/cropped_visium_hd_human_colorectal_cell_nuclear_expanded_overlay.png',
                       help='Path to output visualization image')
    parser.add_argument('--nuclear_range_csv', type=str,
                       default='cell_based_nuclear_range.csv',
                       help='Path to cell type nuclear range CSV')
    parser.add_argument('--original_image', type=str,
                       default='cropped_visium_hd_human_colorectal.png',
                       help='Path to original H&E image (optional, for visualization)')
    args = parser.parse_args()

    # Set paths from arguments
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_vis = Path(args.output_vis)
    nuclear_range_csv = Path(args.nuclear_range_csv)
    original_image = Path(args.original_image)

    logger.info("="*70)
    logger.info("Nuclear Expansion to Full Cell Segmentation")
    logger.info("="*70)

    logger.info(f"\nInput files:")
    logger.info(f"  Pixel mapping: {input_csv}")
    logger.info(f"  Nuclear ranges: {nuclear_range_csv}")
    logger.info(f"\nOutput files:")
    logger.info(f"  Expanded CSV: {output_csv}")
    logger.info(f"  Visualization (full-res): {output_vis}")
    logger.info("")

    # Step 1: Load nuclear ratio ranges
    ratio_ranges = load_nuclear_ratio_ranges(nuclear_range_csv)

    # Step 2: Load pixel-to-cell mapping
    logger.info(f"Loading pixel-to-cell mapping from {input_csv}...")
    logger.info("  Reading compressed CSV in chunks...")

    chunk_size = 1_000_000
    chunks = []

    for chunk in pd.read_csv(input_csv, compression='gzip', chunksize=chunk_size):
        logger.info(f"    Loaded chunk with {len(chunk):,} pixels...")
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Total pixels loaded: {len(df):,}")

    # Determine image dimensions
    max_y = df['y'].max()
    max_x = df['x'].max()
    image_shape = (max_y + 1, max_x + 1)
    logger.info(f"  Image dimensions: {image_shape[1]} x {image_shape[0]}")

    # Step 3: Expand nuclei to full cells using morphological dilation
    df_expanded, expanded_cell_mask, nuclear_mask = process_cell_expansion_morphological(
        df, ratio_ranges, image_shape
    )

    # Step 4: Save expanded CSV
    logger.info(f"Saving expanded pixel mapping to {output_csv}...")
    df_expanded.to_csv(output_csv, index=False, compression='gzip')
    logger.info(f"  Saved {len(df_expanded):,} pixels")

    # Step 5: Create full-resolution overlay (nuclear vs cytoplasm)
    create_full_resolution_overlay(
        expanded_cell_mask,
        nuclear_mask,
        output_vis
    )

    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("Expansion Summary")
    logger.info("="*70)

    # Calculate N:C ratios per cell type (VECTORIZED - much faster!)
    logger.info("\nNuclear:Cytoplasm ratios by cell type:")

    # Group by cell_id to get nuclear and total counts (VECTORIZED!)
    cell_stats = df_expanded.groupby('cell_id').agg({
        'is_nuclear': 'sum',  # Nuclear pixel count
        'cell_type': 'first'   # Cell type (all pixels in a cell have same type)
    }).rename(columns={'is_nuclear': 'nuclear_count'})

    # Count total pixels per cell
    cell_stats['total_count'] = df_expanded.groupby('cell_id').size()

    # Calculate ratio for each cell
    cell_stats['nuclear_ratio_pct'] = 100 * cell_stats['nuclear_count'] / cell_stats['total_count']

    # Group by cell type and calculate mean/std
    for cell_type in cell_stats['cell_type'].unique():
        if pd.isna(cell_type):
            continue

        type_ratios = cell_stats[cell_stats['cell_type'] == cell_type]['nuclear_ratio_pct']
        if len(type_ratios) > 0:
            mean_ratio = type_ratios.mean()
            std_ratio = type_ratios.std()
            logger.info(f"  {cell_type}: {mean_ratio:.1f}% ± {std_ratio:.1f}%")

    logger.info("\n" + "="*70)
    logger.info("Processing Complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
