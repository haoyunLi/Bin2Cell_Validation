#!/usr/bin/env python3
"""
Gene Localization Analysis - Correct Splicing-Based Model

Based on the paper's logic:
1. Parse GTF to get intron/exon ratio for each gene → baseline probability p_g^0
2. Draw cell-specific factors b_i ~ LogNormal(0, σ²)
3. Calculate preliminary probabilities: p_ig = p_g^0 * b_i * C
4. Compute global scaling α = T / u (where u is weighted average)
5. Scale and clip: p_hat_ig = clip(α * p_ig, ε, 1-ε)
6. Binomial split: V_ig ~ Binomial(N_ig, p_hat_ig) → nuclear counts
7. Membrane genes: Use UniProt keywords on non-nuclear genes
8. Cytosol: Everything else
"""

import os
import requests
import gzip
import re
import numpy as np
from collections import defaultdict
from typing import Dict, Set


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Model parameters
T_NUCLEAR_FRACTION = 0.15    # Target 15% unspliced transcripts globally
P_MIN = 0.02                 # Baseline probability for low-intron genes
P_MAX = 0.4                  # Baseline probability for high-intron genes
C_CAPTURE_EFFICIENCY = 1.0   # Global nuclear capture efficiency (0-1)
SIGMA_CELL_FACTOR = 0.2      # Sigma for cell-specific lognormal factors
CLIP_EPS = 1e-8              # Clipping epsilon [ε, 1-ε]

# Membrane keywords (UniProt)
MEMBRANE_KEYWORDS = {
    "Secreted", "Signal peptide", "Transmembrane", "GPI-anchor",
    "Cell membrane", "Endoplasmic reticulum", "Extracellular", "Golgi apparatus"
}


# ============================================================================
# GTF PARSING
# ============================================================================

def parse_gtf_gene_structure(gtf_file: str, genes_of_interest: Set[str]) -> Dict[str, Dict]:
    """
    Parse GTF to get intron/exon lengths.
    Returns: {gene_name: {'intron_length': int, 'gene_length': int}}
    """
    print(f"\nParsing GTF file: {gtf_file}")
    print(f"Looking for {len(genes_of_interest)} genes...")

    gene_coords = defaultdict(lambda: {'exons': [], 'start': float('inf'), 'end': 0})

    open_func = gzip.open if gtf_file.endswith('.gz') else open

    with open_func(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            attributes = parts[8]

            # Parse attributes
            attrs = {}
            for attr in attributes.split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                match = re.match(r'(\S+)\s+"([^"]+)"', attr)
                if match:
                    attrs[match.group(1)] = match.group(2)

            gene_name = attrs.get('gene_name')
            if not gene_name or gene_name not in genes_of_interest:
                continue

            if feature_type == 'gene':
                gene_coords[gene_name]['start'] = min(gene_coords[gene_name]['start'], start)
                gene_coords[gene_name]['end'] = max(gene_coords[gene_name]['end'], end)
            elif feature_type == 'exon':
                gene_coords[gene_name]['exons'].append((start, end))
                gene_coords[gene_name]['start'] = min(gene_coords[gene_name]['start'], start)
                gene_coords[gene_name]['end'] = max(gene_coords[gene_name]['end'], end)

    # Calculate lengths
    gene_structures = {}
    for gene_name, coords in gene_coords.items():
        if not coords['exons']:
            continue

        # Merge overlapping exons
        exons = sorted(coords['exons'])
        merged_exons = []
        current_start, current_end = exons[0]

        for start, end in exons[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_exons.append((current_start, current_end))
                current_start, current_end = start, end
        merged_exons.append((current_start, current_end))

        exon_length = sum(end - start + 1 for start, end in merged_exons)
        gene_length = coords['end'] - coords['start'] + 1
        intron_length = max(0, gene_length - exon_length)

        gene_structures[gene_name] = {
            'intron_length': intron_length,
            'gene_length': gene_length
        }

    print(f"Parsed {len(gene_structures)} genes")
    return gene_structures


def compute_baseline_probabilities(gene_structures: Dict[str, Dict],
                                   p_min: float = P_MIN,
                                   p_max: float = P_MAX) -> Dict[str, float]:
    """
    Compute baseline nuclear probability p_g^0 for each gene.

    Steps:
    1. r_g = intron_length / gene_length (intron ratio)
    2. q_g = (r_g - min_r) / (max_r - min_r) (normalized intron score)
    3. p_g^0 = p_min + (p_max - p_min) * q_g (baseline probability)

    Returns: {gene_name: p_g^0}
    """
    print(f"\nComputing baseline probabilities (p_min={p_min}, p_max={p_max})...")

    # Step 1: Intron ratios
    intron_ratios = {}
    for gene_name, structure in gene_structures.items():
        gene_length = structure['gene_length']
        intron_length = structure['intron_length']
        r_g = intron_length / gene_length if gene_length > 0 else 0.0
        intron_ratios[gene_name] = r_g

    # Step 2: Normalize to intron scores
    ratios = list(intron_ratios.values())
    min_r = min(ratios) if ratios else 0
    max_r = max(ratios) if ratios else 1

    print(f"  Intron ratio range: [{min_r:.4f}, {max_r:.4f}]")

    # Step 3: Baseline probabilities
    p0 = {}
    for gene_name, r_g in intron_ratios.items():
        q_g = (r_g - min_r) / (max_r - min_r) if max_r > min_r else 0.0
        p0[gene_name] = p_min + (p_max - p_min) * q_g

    return p0


# ============================================================================
# MEMBRANE GENES (UniProt)
# ============================================================================

def get_membrane_genes_uniprot(genes: list, organism: str = "human") -> Set[str]:
    """Query UniProt for membrane genes."""
    print(f"\nQuerying UniProt for membrane genes ({len(genes)} genes)...")

    membrane_genes = set()
    organism_code = "9606" if organism.lower() == "human" else "10090"

    batch_size = 100
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i+batch_size]

        gene_query = " OR ".join([f"gene:{g}" for g in batch])
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"({gene_query}) AND organism_id:{organism_code}",
            "fields": "gene_names,cc_subcellular_location,ft_transmem,ft_signal,ft_topo_dom",
            "format": "tsv",
            "size": 500
        }

        try:
            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 200:
                lines = response.text.strip().split('\n')

                for line in lines[1:]:
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    gene_names_field = parts[0]
                    if not gene_names_field:
                        continue

                    primary_gene = gene_names_field.split()[0]
                    annotation_text = ' '.join(parts[1:]).lower()

                    if any(keyword.lower() in annotation_text for keyword in MEMBRANE_KEYWORDS):
                        membrane_genes.add(primary_gene)

        except Exception as e:
            print(f"  Error in batch {i//batch_size + 1}: {e}")

        if (i + batch_size) % 500 == 0:
            print(f"  Processed {min(i+batch_size, len(genes))}/{len(genes)}...")

    print(f"Found {len(membrane_genes)} membrane genes")
    return membrane_genes


# ============================================================================
# PER-CELL ASSIGNMENT (CORRECT LOGIC)
# ============================================================================

def assign_nuclear_genes_per_cell(gene_expression: np.ndarray,
                                  gene_names: np.ndarray,
                                  p0: Dict[str, float],
                                  cell_factor: float,
                                  global_alpha: float,
                                  C: float = C_CAPTURE_EFFICIENCY,
                                  eps: float = CLIP_EPS,
                                  p0_array: np.ndarray = None) -> Set[str]:
    """
    Assign genes to nucleus for ONE cell using CORRECT paper logic.
    OPTIMIZED VERSION with vectorized operations.

    Args:
        gene_expression: N_ig counts for this cell (array, length = num_genes)
        gene_names: Gene names (array, length = num_genes)
        p0: Baseline probabilities p_g^0 (dict) - only used if p0_array is None
        cell_factor: b_i for this cell (scalar from lognormal)
        global_alpha: α (scalar, computed once globally)
        C: Global capture efficiency
        eps: Clipping epsilon
        p0_array: OPTIONAL - Pre-computed array of p0 values aligned with gene_names (FASTER!)

    Returns:
        Set of genes assigned to nucleus in this cell
    """
    # Only expressed genes
    expressed_mask = gene_expression > 0

    if not np.any(expressed_mask):
        return set()

    expressed_genes = gene_names[expressed_mask]
    expressed_counts = gene_expression[expressed_mask]

    # OPTIMIZATION: Use pre-computed p0_array if provided
    if p0_array is not None:
        # FAST PATH: Direct array indexing
        p_g0 = p0_array[expressed_mask]
        valid_mask = p_g0 > 0  # Genes with non-zero baseline probability

        if not np.any(valid_mask):
            return set()

        valid_genes = expressed_genes[valid_mask]
        valid_counts = expressed_counts[valid_mask]
        p_g0 = p_g0[valid_mask]
    else:
        # SLOW PATH: Dict lookups (backwards compatibility)
        has_prob = np.array([g in p0 for g in expressed_genes])
        valid_genes = expressed_genes[has_prob]
        valid_counts = expressed_counts[has_prob]

        if len(valid_genes) == 0:
            return set()

        p_g0 = np.array([p0[g] for g in valid_genes])

    # Step 1: Preliminary probabilities p_ig = p_g^0 * b_i * C
    p_prelim = p_g0 * cell_factor * C

    # Step 2: Scale by global alpha
    p_scaled = global_alpha * p_prelim

    # Step 3: Clip to [ε, 1-ε]
    p_hat = np.clip(p_scaled, eps, 1.0 - eps)

    # Step 4: VECTORIZED Binomial split (10-50x faster than loop!)
    # Convert counts to integers for binomial
    valid_counts_int = valid_counts.astype(int)

    # Vectorized binomial sampling
    V_all = np.random.binomial(valid_counts_int, p_hat)

    # Boolean mask for genes with nuclear counts > 0
    nuclear_mask = V_all > 0

    # Return set of nuclear genes
    nuclear_genes = set(valid_genes[nuclear_mask])

    return nuclear_genes


# ============================================================================
# COMPUTE PARAMETERS FOR PER-CELL ASSIGNMENT (for pseudo HD pipeline)
# ============================================================================

def classify_genes_aggregate(genes: list,
                            gtf_file: str,
                            gene_counts: Dict[str, int],
                            organism: str = "human") -> Dict:
    """
    Compute parameters needed for per-cell gene localization in pseudo HD pipeline.

    This function does NOT classify genes at aggregate level.
    Instead, it computes:
    - p0: Baseline probabilities for each gene (from GTF intron/exon ratios)
    - global_alpha: Global scaling factor to achieve target nuclear fraction
    - membrane_genes_all: Set of membrane genes from UniProt

    These parameters are then used by assign_nuclear_genes_per_cell() to perform
    PER-CELL gene localization with cell-specific factors.

    Args:
        genes: List of gene names
        gtf_file: Path to GTF annotation file
        gene_counts: Dict of gene -> total count (used for computing global_alpha)
        organism: "human" or "mouse"

    Returns:
        Dict with keys:
        - 'p0': Dict of gene -> baseline nuclear probability
        - 'global_alpha': Global scaling factor
        - 'membrane_genes_all': Set of membrane gene names
    """
    print("\n" + "="*70)
    print("Computing parameters for per-cell gene localization")
    print("="*70)

    genes_set = set(genes)

    # Step 1: Parse GTF to get gene structures
    gene_structures = parse_gtf_gene_structure(gtf_file, genes_set)

    # Step 2: Compute baseline probabilities p_g^0
    p0 = compute_baseline_probabilities(gene_structures)

    # Step 3: Compute global scaling factor α
    print(f"\nComputing global scaling factor α (target: {T_NUCLEAR_FRACTION*100}% of reads nuclear)...")

    # u = Σ_g (N_g * p_g^0 * C) / Σ_g N_g
    total_weighted = sum(gene_counts.get(g, 0) * p0.get(g, 0) * C_CAPTURE_EFFICIENCY for g in genes)
    total_counts = sum(gene_counts.values())
    u = total_weighted / total_counts if total_counts > 0 else 1.0

    global_alpha = T_NUCLEAR_FRACTION / u

    print(f"  u (weighted average) = {u:.6f}")
    print(f"  α (global scaling) = {global_alpha:.6f}")

    # Step 4: Get membrane genes from UniProt
    print(f"\nQuerying UniProt for membrane genes...")
    membrane_genes = get_membrane_genes_uniprot(genes, organism)

    # Return parameters for per-cell use
    results = {
        'p0': p0,
        'global_alpha': global_alpha,
        'membrane_genes_all': membrane_genes
    }

    # Summary
    print("\n" + "="*70)
    print("Parameters computed for per-cell assignment:")
    print("="*70)
    print(f"  Genes with baseline prob (p0):  {len(p0):6d}")
    print(f"  UniProt membrane genes:         {len(membrane_genes):6d}")
    print(f"  Global alpha (α):               {global_alpha:.6f}")
    print("="*70)

    return results


# Note: This module provides gene localization functions for the pseudo Visium HD pipeline.
# The classify_genes_aggregate function is called by create_pseudo_visium_hd.py
# No standalone execution needed - this is a library module.
