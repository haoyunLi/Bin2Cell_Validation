import os
# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter all
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops messages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import smurf as su
import pickle
import copy
import gzip
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import squidpy as sq
import argparse


def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None, return_details=False):
    """StarDist segmentation function for HE images"""
    axis_norm = (0, 1, 2)  # normalize channels jointly
    img = normalize(img, 1, 99.8, axis=axis_norm)
    model = StarDist2D.from_pretrained("2D_versatile_he")

    labels, details = model.predict_instances(
        img, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )
    if return_details:
        return labels, details
    return labels


def segment_image_tiles(image, loop=4700, gap=80, prob_thresh_default=0.01, nms_thresh_default=0.3):
    """Segment large image by processing it in tiles"""
    i_max = image.shape[0]
    j_max = image.shape[1]
    segmentation_results = {}
    confidence_scores = {}

    print(f"Processing image of size {i_max} x {j_max} with tiles of size {loop} x {loop}")
    print(f"Using prob_thresh={prob_thresh_default}, nms_thresh={nms_thresh_default}")

    tiles_i = (i_max + loop - 1) // loop
    tiles_j = (j_max + loop - 1) // loop
    total_tiles = tiles_i * tiles_j
    tile_count = 0

    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            tile_count += 1
            print(f"Processing tile {tile_count}/{total_tiles} at position ({i}, {j})")

            a_t = image[max(0, i-gap):min(i+loop, i_max), max(0, j-gap):min(j+loop, j_max)]

            # Get segmentation with confidence scores
            labels, details = stardist_2D_versatile_he(a_t, nms_thresh=nms_thresh_default,
                                                      prob_thresh=prob_thresh_default,
                                                      return_details=True)

            segmentation_results[(i, j)] = labels
            # Store confidence scores for this tile
            if 'prob' in details:
                confidence_scores[(i, j)] = details['prob']
            else:
                confidence_scores[(i, j)] = np.array([])

    return segmentation_results, confidence_scores


def combine_segmentation_tiles(segmentation_results, confidence_scores, i_max, j_max, loop, gap):
    """Combine segmentation results from different tiles and track confidence scores"""
    print("Combining segmentation tiles...")

    segmentation_results1 = copy.deepcopy(segmentation_results)
    confidence_mapping = {}  # Maps final cell ID to confidence score
    num = 0

    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            mask = segmentation_results1[(i, j)] != 0
            old_labels = np.unique(segmentation_results1[(i, j)][mask])
            old_labels = old_labels[old_labels > 0]

            # Update labels and map confidence scores
            segmentation_results1[(i, j)][mask] = segmentation_results1[(i, j)][mask] + num

            # Map confidence scores to new labels
            if (i, j) in confidence_scores and len(confidence_scores[(i, j)]) > 0:
                for idx, old_label in enumerate(old_labels):
                    new_label = old_label + num
                    if idx < len(confidence_scores[(i, j)]):
                        confidence_mapping[new_label] = confidence_scores[(i, j)][idx]

            num = max(num, segmentation_results1[(i, j)].max())

    segmentation_final = np.zeros((i_max, j_max))

    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            if i == 0 and j == 0:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)]
            elif i == 0 and j != 0:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)][:, gap:]
            elif i != 0 and j == 0:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)][gap:, :]
            else:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)][gap:, gap:]

    print("Resolving tile overlaps...")
    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            if j != 0:
                if i == 0:
                    a = copy.deepcopy(segmentation_results1[(i, (j-loop))][:, -gap:])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:, :gap])
                else:
                    a = copy.deepcopy(segmentation_results1[(i, (j-loop))][gap:, -gap:])
                    b = copy.deepcopy(segmentation_results1[(i, j)][gap:, :gap])

                uni_a = np.unique(a[:, -1:])
                uni_a = uni_a[uni_a > 0]
                for ele in uni_a:
                    uni_b = np.unique(b[a == ele])
                    uni_b = uni_b[uni_b > 0]
                    if len(uni_b) > 0:
                        a[a == ele] = 0
                        for eleb in uni_b:
                            a[b == eleb] = eleb

                if i == 0:
                    segmentation_final[i:min(i+loop, i_max), min(j, j_max)-gap:min(j, j_max)] = a
                else:
                    segmentation_final[i:min(i+loop, i_max), min(j, j_max)-gap:min(j, j_max)] = a

            if i != 0:
                if j == 0:
                    a = copy.deepcopy(segmentation_results1[((i-loop), j)][-gap:, :])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:gap, :])
                else:
                    a = copy.deepcopy(segmentation_results1[((i-loop), j)][-gap:, gap:])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:gap, gap:])

                uni_a = np.unique(a[-1:, :])
                uni_a = uni_a[uni_a > 0]
                for ele in uni_a:
                    uni_b = np.unique(b[a == ele])
                    uni_b = uni_b[uni_b > 0]
                    if len(uni_b) > 0:
                        a[a == ele] = 0
                        for eleb in uni_b:
                            a[b == eleb] = eleb

                if j == 0:
                    segmentation_final[(min(i, i_max)-gap):min(i, i_max), j:min(j+loop, j_max)] = a
                else:
                    segmentation_final[(min(i, i_max)-gap):min(i, i_max), j:min(j+loop, j_max)] = a

    return segmentation_final, confidence_mapping


def filter_cells_by_confidence(segmentation_final, confidence_mapping, confidence_threshold=0.6):
    """Filter out cells with confidence scores below threshold"""
    print(f"Filtering cells with confidence < {confidence_threshold}")

    # Create filtered segmentation
    filtered_segmentation = segmentation_final.copy()
    removed_cells = []

    for cell_id, confidence in confidence_mapping.items():
        if confidence < confidence_threshold:
            filtered_segmentation[segmentation_final == cell_id] = 0
            removed_cells.append((cell_id, confidence))

    # Relabel remaining cells to be consecutive
    unique_labels = np.unique(filtered_segmentation)
    unique_labels = unique_labels[unique_labels > 0]

    final_segmentation = np.zeros_like(filtered_segmentation)
    new_confidence_mapping = {}

    for new_label, old_label in enumerate(unique_labels, 1):
        final_segmentation[filtered_segmentation == old_label] = new_label
        if old_label in confidence_mapping:
            new_confidence_mapping[new_label] = confidence_mapping[old_label]

    print(f"Removed {len(removed_cells)} cells with low confidence")
    print(f"Remaining cells: {int(final_segmentation.max())}")

    return final_segmentation, new_confidence_mapping, removed_cells


def save_confidence_scores(confidence_mapping, save_path):
    """Save confidence scores to a readable file"""
    confidence_df = pd.DataFrame([
        {'cell_id': cell_id, 'confidence_score': score}
        for cell_id, score in confidence_mapping.items()
    ])
    confidence_df = confidence_df.sort_values('confidence_score', ascending=False)

    csv_path = os.path.join(save_path, 'cell_confidence_scores.csv')
    confidence_df.to_csv(csv_path, index=False)
    print(f"Confidence scores saved to: {csv_path}")

    # Print summary statistics
    print(f"Confidence score statistics:")
    print(f"  Mean: {confidence_df['confidence_score'].mean():.3f}")
    print(f"  Median: {confidence_df['confidence_score'].median():.3f}")
    print(f"  Min: {confidence_df['confidence_score'].min():.3f}")
    print(f"  Max: {confidence_df['confidence_score'].max():.3f}")
    print(f"  Cells with confidence >= 0.8: {(confidence_df['confidence_score'] >= 0.8).sum()}")
    print(f"  Cells with confidence >= 0.9: {(confidence_df['confidence_score'] >= 0.9).sum()}")

    return confidence_df


def main():
    """Main SMURF analysis workflow."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SMURF spatial transcriptomics analysis')
    parser.add_argument('--tissue_positions', type=str,
                       default='Human_kidney/output/binned_outputs/square_002um/spatial/tissue_positions.parquet',
                       help='Path to tissue positions parquet file')
    parser.add_argument('--tissue_image', type=str,
                       default='Human_kidney/input/Visium_HD_Human_Kidney_FFPE_tissue_image.tif',
                       help='Path to tissue H&E image')
    parser.add_argument('--pseudo_hd_dir', type=str,
                       default='pseudo_visium_hd_outpu_full',
                       help='Path to pseudo Visium HD output directory')
    parser.add_argument('--output_dir', type=str,
                       default='smurf_result',
                       help='Output directory for SMURF results')
    parser.add_argument('--microns_per_pixel', type=float,
                       default=0.2739038899725172,
                       help='Microns per pixel conversion factor')
    args = parser.parse_args()

    save_path = args.output_dir
    pseudo_hd_dir = args.pseudo_hd_dir

    # Check if the directory exists
    if not os.path.exists(save_path):
        # If it doesn't exist, create it
        os.makedirs(save_path)

    so = su.prepare_dataframe_image(args.tissue_positions,
                                    args.tissue_image,
                                    'HE')
    plt.imshow(so.image_temp())
    plt.savefig(os.path.join(save_path, 'tissue_image.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save image for segmentation
    with gzip.GzipFile(os.path.join(save_path, 'image_to_segmentation.npy.gz'), 'w') as f:
        np.save(f, so.image_temp())

    print("Image saved for segmentation.")

    # Check if segmentation already exists, if not perform it
    segmentation_path = os.path.join(save_path, 'segmentation_final.npy')
    if os.path.exists(segmentation_path):
        print("Loading existing segmentation results...")
        so.segmentation_final = np.load(segmentation_path)
        print("Segmentation results loaded successfully.")
    else:
        print("Segmentation file not found. Performing nuclei segmentation...")

        # Load the saved image for segmentation
        with gzip.GzipFile(os.path.join(save_path, 'image_to_segmentation.npy.gz'), 'r') as f:
            image = np.load(f)

        print(f"Performing segmentation on image of size: {image.shape}")

        # Perform tile-based segmentation with confidence tracking
        segmentation_results, confidence_scores = segment_image_tiles(image, loop=4700, gap=80,
                                                                     prob_thresh_default=0.01,
                                                                     nms_thresh_default=0.3)

        # Combine tiles into final segmentation
        i_max, j_max = image.shape[:2]
        segmentation_final_array, confidence_mapping = combine_segmentation_tiles(
            segmentation_results, confidence_scores, i_max, j_max, loop=4700, gap=80)

        print(f"Initial segmentation completed! Found {int(segmentation_final_array.max())} nuclei.")

        # Save confidence scores
        confidence_df = save_confidence_scores(confidence_mapping, save_path)

        # Save confidence mapping
        with open(os.path.join(save_path, 'confidence_mapping.pkl'), 'wb') as f:
            pickle.dump(confidence_mapping, f)

        # Optionally filter by confidence
        # Uncomment the next lines to apply confidence filtering
        # filtered_segmentation, filtered_confidence, removed_cells = filter_cells_by_confidence(
        #     segmentation_final_array, confidence_mapping, confidence_threshold=0.85)
        # segmentation_final_array = filtered_segmentation
        # confidence_mapping = filtered_confidence

        # Save segmentation results
        np.save(segmentation_path, segmentation_final_array)
        so.segmentation_final = segmentation_final_array

        print(f"Segmentation saved to: {segmentation_path}")
        print(f"Confidence scores saved to: {os.path.join(save_path, 'cell_confidence_scores.csv')}")

    # Generate cell and spot information
    print("Generating cell and spot information...")
    so.generate_cell_spots_information()

    # Visualize segmentation results
    su.plot_results(so.image_temp(), so.segmentation_final, dpi=1500,
                    save=os.path.join(save_path, 'segmentation_overlay.png'))

    # Load gene expression data
    print("Loading gene expression data...")
    adata = sc.read_10x_mtx(os.path.join(pseudo_hd_dir, 'filtered_feature_bc_matrix'))
    adata = copy.deepcopy(adata[so.df[so.df.in_tissue == 1]['barcode']])

    # Filter genes
    sc.pp.filter_genes(adata, min_counts=100)

    # Generate nuclei*genes matrix
    print("Generating nuclei-genes matrix...")
    su.nuclei_rna(adata, so)
    adata_sc = copy.deepcopy(so.final_nuclei)

    # Filter cells
    sc.pp.filter_cells(adata_sc, min_counts=500)
    adata_raw = copy.deepcopy(adata_sc)

    # Initial single cell analysis
    print("Performing initial single cell analysis...")
    adata_sc = su.singlecellanalysis(adata_sc, resolution=0.5)

    # Iterative arrangement (this is time-consuming)
    print("Starting iterative arrangement (this may take a while)...")
    su.itering_arragement(adata_sc, adata_raw, adata, so, resolution=0.5,
                         save_folder=os.path.abspath(save_path) + '/', show=True, keep_previous=False)

    # Load iteration results
    adatas_path = os.path.join(save_path, 'adatas.h5ad')
    if not os.path.exists(adatas_path):
        # Check if files were saved with incorrect path concatenation
        adatas_path = save_path + 'adatas.h5ad'
        if not os.path.exists(adatas_path):
            raise FileNotFoundError(f"Could not find adatas.h5ad in either {os.path.join(save_path, 'adatas.h5ad')} or {save_path}adatas.h5ad")
    adatas_final = sc.read_h5ad(adatas_path)

    # Load cells_final.pkl
    cells_path = os.path.join(save_path, 'cells_final.pkl')
    if not os.path.exists(cells_path):
        cells_path = save_path + 'cells_final.pkl'
    with open(cells_path, 'rb') as file:
        cells_final = pickle.load(file)

    # Load weights_record.pkl
    weights_path = os.path.join(save_path, 'weights_record.pkl')
    if not os.path.exists(weights_path):
        weights_path = save_path + 'weights_record.pkl'
    with open(weights_path, 'rb') as file:
        weights_record = pickle.load(file)

    # Prepare data for deep learning
    print("Preparing data for deep learning optimization...")
    pct_toml_dic, spots_X_dic, celltypes_dic, cells_X_plus_dic, nonzero_indices_dic, nonzero_indices_toml, \
    cells_before_ml, cells_before_ml_x, groups_combined, spots_id_dic, spots_id_dic_prop = \
        su.make_preparation(cells_final, so, adatas_final, adata, weights_record, maximum_cells=500)

    # Setup CUDA for optimization
    import torch
    import gc

    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Enable memory fragmentation fix
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start optimization
    print("Starting deep learning optimization...")
    spot_cell_dic = su.start_optimization(spots_X_dic, celltypes_dic, cells_X_plus_dic,
                                         nonzero_indices_toml, device,
                                         num_epochs=1000, learning_rate=0.1,
                                         print_each=100, epsilon=0.00001)

    # Save optimization results
    with open(os.path.join(save_path, 'spot_cell_dic.pkl'), 'wb') as f:
        pickle.dump(spot_cell_dic, f)

    # Generate final pixel-to-cell mapping
    print("Generating final pixel-to-cell mapping...")
    su.make_pixels_cells(so, adata, cells_before_ml, spot_cell_dic, spots_id_dic,
                        spots_id_dic_prop, nonzero_indices_dic)

    # Plot final results
    su.plot_results(so.image_temp(), so.pixels_cells, dpi=1500,
                   save=os.path.join(save_path, 'final_results.pdf'))

    # Generate final cell-level data
    print("Generating final cell-level data...")
    adata_sc_final = su.get_finaldata(adata, adatas_final, spot_cell_dic, weights_record,
                                    cells_before_ml, groups_combined, pct_toml_dic,
                                    nonzero_indices_dic, spots_X_dic,
                                    cells_before_ml_x=cells_before_ml_x, so=so)

    # Save nuclear bin information for validation
    print("\nSaving nuclear bin information for each final cell...")

    # Create nuclear bin mapping from segmentation_final and pixels_cells
    # so.segmentation_final = nuclear segmentation mask (which pixels are nuclear)
    # so.pixels_cells = final pixel-to-cell mapping (which pixels belong to which cell)

    print("  Building nuclear bin information from segmentation and cell assignments...")

    # Get unique cell IDs from pixels_cells
    unique_cells = np.unique(so.pixels_cells)
    unique_cells = unique_cells[unique_cells > 0]  # Exclude background (0)

    # Bin size parameters (matching validation scripts)
    microns_per_pixel = args.microns_per_pixel
    bin_size = 2.0  # 2 micron bins

    # Dictionaries to store bin information per cell
    cell_nuclear_bins = {}  # cell_id -> set of (bin_x, bin_y) that are nuclear
    cell_all_bins = {}      # cell_id -> set of all (bin_x, bin_y)

    print(f"  Processing {len(unique_cells)} cells...")

    for cell_id in unique_cells:
        # Get pixel coordinates for this cell from pixels_cells
        y_coords, x_coords = np.where(so.pixels_cells == cell_id)

        # Convert pixels to bins and track nuclear status
        all_bins_set = set()
        nuclear_bins_set = set()

        for y, x in zip(y_coords, x_coords):
            # Convert pixel to bin coordinates
            bin_x = int(x * microns_per_pixel / bin_size)
            bin_y = int(y * microns_per_pixel / bin_size)
            all_bins_set.add((bin_x, bin_y))

            # Check if this pixel is nuclear (from initial segmentation)
            if so.segmentation_final[y, x] > 0:
                nuclear_bins_set.add((bin_x, bin_y))

        cell_all_bins[int(cell_id)] = all_bins_set
        cell_nuclear_bins[int(cell_id)] = nuclear_bins_set

    # Save nuclear bin information
    print(f"  Saving nuclear bin information for {len(cell_nuclear_bins)} cells...")
    with open(os.path.join(save_path, 'cell_nuclear_bins.pkl'), 'wb') as f:
        pickle.dump({
            'nuclear_bins': cell_nuclear_bins,
            'all_bins': cell_all_bins
        }, f)

    # Calculate and save summary statistics
    cell_stats = []
    for cell_id in unique_cells:
        n_total = len(cell_all_bins[int(cell_id)])
        n_nuclear = len(cell_nuclear_bins[int(cell_id)])
        nuclear_pct = (n_nuclear / n_total * 100) if n_total > 0 else 0.0

        cell_stats.append({
            'cell_id': int(cell_id),
            'n_total_bins': n_total,
            'n_nuclear_bins': n_nuclear,
            'nuclear_percentage': nuclear_pct
        })

    df_nuclear_stats = pd.DataFrame(cell_stats)
    df_nuclear_stats.to_csv(os.path.join(save_path, 'cell_nuclear_bins_summary.csv'), index=False)

    print(f"  Mean nuclear bins per cell: {df_nuclear_stats['n_nuclear_bins'].mean():.1f}")
    print(f"  Mean nuclear percentage: {df_nuclear_stats['nuclear_percentage'].mean():.2f}%")

    print("\nSaving final results...")
    with open(os.path.join(save_path, "so.pkl"), 'wb') as f:
        pickle.dump(so, f)

    adata_sc_final.write(os.path.join(save_path, "adata_sc_final.h5ad"))

    # Print what nuclear information is available in 'so' object
    print("\nNuclear information available in 'so' object:")
    print(f"  so.segmentation_final: Nuclear segmentation mask (shape: {so.segmentation_final.shape})")
    if hasattr(so, 'cell_spot_dic'):
        print(f"  so.cell_spot_dic: Cells -> nuclear bins mapping ({len(so.cell_spot_dic)} cells)")
    if hasattr(so, 'final_nuclei'):
        print(f"  so.final_nuclei: Nuclear gene expression matrix ({so.final_nuclei.shape})")
    if hasattr(so, 'df'):
        print(f"  so.df: Bin information ({len(so.df)} bins)")

    print("\nSMURF analysis completed successfully!")
    print(f"Final cell-level data saved as: {os.path.join(save_path, 'adata_sc_final.h5ad')}")
    print(f"Final results visualization saved as: {os.path.join(save_path, 'final_results.pdf')}")
    print(f"Nuclear bin information saved in: {os.path.join(save_path, 'so.pkl')}")


if __name__ == "__main__":
    main()
