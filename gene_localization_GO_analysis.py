#!/usr/bin/env python3
"""Gene Localization Analysis - Classify genes using GENCODE, RNALocate, and UniProt."""

import os
import requests
import gzip
from collections import defaultdict
import re


INPUT_H5 = "kidney_sc_data/KIRC_GSE159115_expression.h5"
ORGANISM = "human"  # "mouse" or "human"
OUTPUT_DIR = "gene_localization_results_kidney"

# URLs for annotation files
GENCODE_URLS = {
    "human": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz",
    "mouse": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M35/gencode.vM35.annotation.gtf.gz"
}

# RNALocate download URLs (experimental data files)
RNALOCATE_URLS = {
    "lncRNA": "http://www.rnalocate.org/static/download/RNALocate_lncRNA_experiment.txt",
    "all": "http://www.rnalocate.org/static/download/RNALocate_all_experiment.txt"
}

# Nuclear non-coding RNA biotypes
NONCODING_BIOTYPES = {
    "lncRNA", "antisense", "processed_transcript", "snRNA", "snoRNA",
    "scaRNA", "misc_RNA", "sense_intronic", "sense_overlapping",
    "bidirectional_promoter_lncRNA", "lincRNA", "3prime_overlapping_ncRNA",
    "antisense_RNA", "macro_lncRNA", "non_coding", "retained_intron"
}

# Nuclear RNA localization keywords for RNALocate
NUCLEAR_LOCATIONS = {
    "nucleus", "nucleoplasm", "nuclear speckle", "nucleolus", "chromatin",
    "nuclear_speckle", "nuclear_body", "nuclear_membrane", "nuclear pore"
}

# Manually curated nuclear RNAs
MANUAL_NUCLEAR_GENES = {
    "MALAT1", "NEAT1", "XIST"
}

# Membrane-related UniProt keywords
MEMBRANE_KEYWORDS = {
    "Secreted", "Signal peptide", "Transmembrane", "GPI-anchor",
    "Cell membrane", "Endoplasmic reticulum",
    "Membrane"
}


def download_file(url, cache_file):
    """Download file with caching."""
    if os.path.exists(cache_file):
        print(f"Using cached {cache_file}")
        return cache_file

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(cache_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {cache_file}")
        return cache_file
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def parse_gencode_gtf(gtf_file, genes_of_interest):
    """Parse GENCODE GTF to get gene biotypes."""
    print(f"\nParsing GENCODE GTF for {len(genes_of_interest)} genes...")
    gene_biotypes = {}
    genes_set = set(genes_of_interest)

    open_func = gzip.open if gtf_file.endswith('.gz') else open

    with open_func(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue

            # Parse attributes
            attrs = {}
            for attr in parts[8].split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                match = re.match(r'(\S+)\s+"([^"]+)"', attr)
                if match:
                    attrs[match.group(1)] = match.group(2)

            gene_name = attrs.get('gene_name')
            gene_type = attrs.get('gene_type') or attrs.get('gene_biotype')

            if gene_name in genes_set and gene_type:
                gene_biotypes[gene_name] = gene_type

    print(f"Found biotypes for {len(gene_biotypes)} genes")
    return gene_biotypes


def parse_rnalocate_file(rnalocate_file, genes, organism):
    """
    Parse RNALocate experimental data file for nuclear localization.

    File format: RNALocate ID, Species, RNA Symbol, RNA Type, Subcellular Localization, GO Accession, PubMed ID, Score

    Args:
        rnalocate_file: Path to downloaded RNALocate experimental data file
        genes: List of gene symbols to check
        organism: "human" or "mouse"

    Returns:
        Set of genes with nuclear localization
    """
    print(f"\nParsing RNALocate file for nuclear genes...")

    nuclear_genes = set()
    genes_set = set(genes)

    # Species names in RNALocate
    species_name = "Homo sapiens" if organism.lower() == "human" else "Mus musculus"

    line_count = 0
    matched_count = 0

    with open(rnalocate_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1

            # Skip header if present
            if line.startswith("RNALocate ID") or line.startswith("#"):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            # Extract fields
            # Format: RNALocate ID, Species, RNA Symbol, RNA Type, Subcellular Localization, ...
            rna_species = parts[1].strip()
            rna_symbol = parts[2].strip()
            subcell_loc = parts[4].strip().lower()

            # Check if correct species
            if rna_species != species_name:
                continue

            # Check if gene is in our dataset
            if rna_symbol not in genes_set:
                continue

            # Check if subcellular location contains nuclear keywords
            if any(keyword in subcell_loc for keyword in NUCLEAR_LOCATIONS):
                nuclear_genes.add(rna_symbol)
                matched_count += 1

    print(f"  Processed {line_count} lines from RNALocate")
    print(f"  Found {len(nuclear_genes)} unique genes with nuclear localization")
    print(f"  Total entries matched: {matched_count}")

    return nuclear_genes


def get_uniprot_annotations(genes, organism):
    """Get UniProt subcellular location annotations via direct UniProt REST API."""
    print(f"\nQuerying UniProt REST API for {len(genes)} genes...")
    gene_membrane = set()

    organism_code = "9606" if organism.lower() == "human" else "10090"

    # Query UniProt directly - batch by batch
    batch_size = 100  # Smaller batches for UniProt API
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i+batch_size]

        # Build query: gene1 OR gene2 OR gene3...
        gene_query = " OR ".join([f"gene:{g}" for g in batch])

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"({gene_query}) AND organism_id:{organism_code}",
            "fields": "gene_names,cc_subcellular_location,ft_transmem,ft_signal,ft_topo_dom",
            "format": "tsv",
            "size": 500  # Max results per batch
        }

        try:
            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 200:
                lines = response.text.strip().split('\n')

                # Skip header
                for line in lines[1:]:
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    # Extract primary gene name from "Gene Names" field
                    gene_names_field = parts[0]
                    if not gene_names_field:
                        continue

                    # Primary gene name is usually the first one
                    primary_gene = gene_names_field.split()[0]

                    # Combine all annotation fields
                    annotation_text = ' '.join(parts[1:]).lower()

                    # Check for membrane/secretory keywords
                    if any(keyword.lower() in annotation_text for keyword in MEMBRANE_KEYWORDS):
                        gene_membrane.add(primary_gene)

            elif response.status_code != 200:
                print(f"  Warning: batch {i//batch_size + 1} returned status {response.status_code}")

        except Exception as e:
            print(f"  Error in batch {i//batch_size + 1}: {e}")

        print(f"  Processed {min(i+batch_size, len(genes))}/{len(genes)}")

    print(f"Found {len(gene_membrane)} membrane-associated genes from UniProt")
    return gene_membrane


def classify_genes_new_logic(genes, gene_biotypes, rnalocate_nuclear_genes, membrane_genes):
    """
    Classify genes using the new logic:
    1. Nuclear: non-coding RNAs (lncRNA, snRNA, snoRNA) + manual curation + RNALocate validated
    2. Membrane: protein-coding genes with membrane/secretory UniProt annotations (BOTH required!)
    3. Cytoplasm: everything else
    """
    print("\nClassifying genes with new logic...")

    nuclear_genes = set()
    final_membrane_genes = set()

    # Debug counters
    manual_count = 0
    pattern_count = 0
    biotype_rnalocate_count = 0
    membrane_coding_count = 0
    membrane_filtered_count = 0

    # 1. Identify nuclear genes
    for gene in genes:
        # Manual curation
        if gene in MANUAL_NUCLEAR_GENES:
            nuclear_genes.add(gene)
            manual_count += 1
            continue

        # Pattern matching for manually curated families
        if (gene.startswith("SNHG") or gene.startswith("SNORD") or
            gene.startswith("SNORA") or gene.startswith("RNU")):
            nuclear_genes.add(gene)
            pattern_count += 1
            continue

        # Check if non-coding RNA with nuclear localization (BOTH required!)
        biotype = gene_biotypes.get(gene, "")
        if biotype in NONCODING_BIOTYPES:
            # Must ALSO be in RNALocate nuclear list
            if gene in rnalocate_nuclear_genes:
                nuclear_genes.add(gene)
                biotype_rnalocate_count += 1

    # 2. Filter membrane genes to only protein-coding genes (BOTH required!)
    for gene in membrane_genes:
        if gene in genes:  # Gene is in our dataset
            biotype = gene_biotypes.get(gene, "")
            if biotype == "protein_coding":
                final_membrane_genes.add(gene)
                membrane_coding_count += 1
            else:
                membrane_filtered_count += 1

    # 3. Cytoplasmic genes = all - nuclear - membrane
    cyto_genes = set(genes) - nuclear_genes - final_membrane_genes

    results = {
        "nucleus": sorted(nuclear_genes),
        "cell_membrane": sorted(final_membrane_genes),
        "cytoplasm": sorted(cyto_genes)
    }

    print(f"\nNuclear gene breakdown:")
    print(f"  Manual curation: {manual_count}")
    print(f"  Pattern matching (SNHG*, SNORD*, etc.): {pattern_count}")
    print(f"  Non-coding + RNALocate validated: {biotype_rnalocate_count}")
    print(f"  Total nucleus: {len(results['nucleus'])} genes")

    print(f"\nMembrane gene breakdown:")
    print(f"  UniProt membrane keywords + protein-coding: {membrane_coding_count}")
    print(f"  Filtered out (non-coding): {membrane_filtered_count}")
    print(f"  Total cell_membrane: {len(results['cell_membrane'])} genes")

    print(f"\n  cytoplasm: {len(results['cytoplasm'])} genes")

    # Show some examples
    if len(results['nucleus']) > 0:
        print(f"\nExample nuclear genes: {', '.join(list(results['nucleus'])[:10])}")
    if len(results['cell_membrane']) > 0:
        print(f"Example membrane genes: {', '.join(list(results['cell_membrane'])[:10])}")

    # Verify no overlaps
    assert len(nuclear_genes & final_membrane_genes & cyto_genes) == 0, "Overlapping classifications!"

    return results


def save_results(results, output_dir):
    """Save gene lists to files."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving to {output_dir}/")

    for compartment, genes in results.items():
        filename = f"{output_dir}/genes_{compartment}.txt"
        with open(filename, 'w') as f:
            f.write("\n".join(genes))
        print(f"  {compartment}: {len(genes)} genes")


def main():
    """Run analysis pipeline."""
    print("=" * 60)
    print("Gene Localization Analysis - New Logic")
    print("Using GENCODE, RNALocate, and UniProt")
    print("=" * 60)

    # Load genes from h5 file
    import h5py
    print(f"\nLoading {INPUT_H5}...")
    with h5py.File(INPUT_H5, 'r') as f:
        gene_names = f['matrix']['features']['name'][:]
        genes = [g.decode('utf-8') if isinstance(g, bytes) else g for g in gene_names]
    print(f"Loaded {len(genes)} genes")

    # Download and cache annotation files
    cache_dir = "annotation_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 1. GENCODE GTF
    gencode_file = f"{cache_dir}/gencode_{ORGANISM}.gtf.gz"
    if not os.path.exists(gencode_file):
        download_file(GENCODE_URLS[ORGANISM], gencode_file)
    gene_biotypes = parse_gencode_gtf(gencode_file, genes)

    # Show biotype statistics
    print(f"\nGENCODE biotype statistics:")
    from collections import Counter
    biotype_counts = Counter(gene_biotypes.values())
    for biotype, count in biotype_counts.most_common(10):
        marker = " (NUCLEAR)" if biotype in NONCODING_BIOTYPES else ""
        print(f"  {biotype}: {count}{marker}")

    # 2. RNALocate (download and parse experimental data for nuclear genes from non-coding RNAs)
    # Only check genes that are non-coding RNAs
    ncrna_genes = [g for g in genes if gene_biotypes.get(g, "") in NONCODING_BIOTYPES]
    print(f"\nFound {len(ncrna_genes)} non-coding RNA genes to check against RNALocate")

    if len(ncrna_genes) > 0:
        # Download RNALocate experimental data file
        rnalocate_file = f"{cache_dir}/rnalocate_all_experiment.txt"
        if not os.path.exists(rnalocate_file):
            download_file(RNALOCATE_URLS["all"], rnalocate_file)

        # Parse for nuclear genes
        rnalocate_nuclear_genes = parse_rnalocate_file(rnalocate_file, ncrna_genes, ORGANISM)
    else:
        print("  Skipping RNALocate (no non-coding RNAs found)")
        rnalocate_nuclear_genes = set()

    # 3. UniProt
    membrane_genes = get_uniprot_annotations(genes, ORGANISM)

    # Classify genes
    results = classify_genes_new_logic(genes, gene_biotypes, rnalocate_nuclear_genes, membrane_genes)

    # Save results
    save_results(results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
