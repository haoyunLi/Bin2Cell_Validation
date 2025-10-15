#!/usr/bin/env python
"""
Run CSGO cell segmentation on cropped Visium HD brain image.

This script runs the CSGO (Cell Segmentation for Generalized Organs) model
on the cropped Visium HD brain image to perform whole-cell segmentation.

Usage:
    python run_csgo_segmentation.py

The script will:
1. Load the CSGO model with pretrained weights
2. Segment the cropped_visium_hd_brain.png image
3. Save the segmentation results to output directory
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
        logging.FileHandler('csgo_segmentation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run CSGO segmentation on the cropped Visium HD brain image."""

    # Define paths
    yolo_path = CSGO_SRC_DIR / 'pretrained_weights' / 'lung_best.float16.torchscript.pt'
    unet_path = CSGO_SRC_DIR / 'pretrained_weights' / 'epoch_190.pt'
    img_path = CURRENT_DIR / 'cropped_visium_hd_brain.png'
    output_dir = CURRENT_DIR / 'csgo_segmentation_output'

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Check if input image exists
    if not img_path.exists():
        logger.error(f"Input image not found: {img_path}")
        raise FileNotFoundError(
            f"Input image not found: {img_path}\n"
            "Please run the crop_visium_hd_notebook.ipynb first to generate the cropped image."
        )

    # Check if model weights exist
    if not yolo_path.exists():
        logger.error(f"YOLO weights not found: {yolo_path}")
        raise FileNotFoundError(f"YOLO weights not found: {yolo_path}")
    if not unet_path.exists():
        logger.error(f"UNet weights not found: {unet_path}")
        raise FileNotFoundError(f"UNet weights not found: {unet_path}")

    logger.info("CSGO Cell Segmentation for Visium HD Brain Image")
    logger.info(f"Input image: {img_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"YOLO weights: {yolo_path}")
    logger.info(f"UNet weights: {unet_path}")

    # Initialize CSGO model with save option enabled
    logger.info("Initializing CSGO model...")
    seg_model = CSGO(
        yolo_path=str(yolo_path),
        unet_path=str(unet_path),
        gpu=True,  # Enable GPU acceleration
        save=True,
        output_dir=str(output_dir)
    )
    logger.info("CSGO model initialized successfully")

    # Run segmentation
    # Parameters:
    # - cell_size: Default cell diameter in pixels (adjust based on your data)
    # - img_resolution: Resolution of the input image (40x is standard)
    # For Visium HD at 2um resolution with typical H&E staining:
    # - cell_size around 20 - 60 pixels is typical
    # - img_resolution of 40x is standard

    logger.info("Starting cell segmentation...")
    logger.info("Parameters:")
    logger.info("  - Cell size (diameter): 50 pixels")
    logger.info("  - Image resolution: 40x")
    logger.info("This may take several minutes depending on image size...")

    result = seg_model.segment(
        img_path=str(img_path),
        cell_size=50,
        img_resolution=40
    )

    logger.info("Segmentation complete!")
    logger.info(f"Results saved to: {output_dir}")

    # Visualize and save the result
    logger.info("Generating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Load and display original image
    original_img = Image.open(img_path)
    axes[0].imshow(original_img)
    axes[0].set_title('Original Cropped Image', fontsize=16)
    axes[0].axis('off')

    # Display segmentation result
    axes[1].imshow(result)
    axes[1].set_title('CSGO Segmentation Result', fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()

    # Save the comparison figure
    comparison_path = output_dir / 'segmentation_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    logger.info(f"Comparison figure saved to: {comparison_path}")

    # Also save just the segmentation result
    result_path = output_dir / 'segmentation_result.png'
    fig2 = plt.figure(figsize=(15, 15))
    plt.imshow(result)
    plt.title('CSGO Segmentation Result', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)  # Close figure to free memory
    logger.info(f"Segmentation result saved to: {result_path}")

    logger.info("="*70)
    logger.info("All processing complete!")
    logger.info("="*70)

    return result


if __name__ == "__main__":
    result = main()
