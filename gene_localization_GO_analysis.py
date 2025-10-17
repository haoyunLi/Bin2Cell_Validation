#!/usr/bin/env python3
"""Gene Localization GO Analysis - Classify genes into nucleus, cytoplasm, or cell membrane."""

import os
import requests
from collections import defaultdict


INPUT_H5AD = "cellannotation_results_small_intestine/visium_hd_annotated.h5ad"
ORGANISM = "mouse"  # "mouse" or "human"
OUTPUT_DIR = "gene_localization_results"

# Comprehensive GO terms for each compartment
GO_TERMS = {
    "nucleus": [
        "GO:0005634", "GO:0005654", "GO:0005694", "GO:0031981", "GO:0005730",
        "GO:0005635", "GO:0005643", "GO:0000785", "GO:0005657", "GO:0016607",
        "GO:0031965", "GO:0005719", "GO:0005721", "GO:0000790", "GO:0000228",
        "GO:0005739", "GO:0044613", "GO:0031981", "GO:0005654", "GO:0016604"
    ],
    "cytoplasm": [
        "GO:0005737", "GO:0005829", "GO:0043232", "GO:0005856", "GO:0005739",
        "GO:0005783", "GO:0005794", "GO:0005764", "GO:0005777", "GO:0005773",
        "GO:0005768", "GO:0005769", "GO:0031410", "GO:0005793", "GO:0005789",
        "GO:0005743", "GO:0005759", "GO:0005758", "GO:0030529", "GO:0022626",
        "GO:0005874", "GO:0005875", "GO:0005813", "GO:0005815", "GO:0031901"
    ],
    "cell_membrane": [
        "GO:0005886", "GO:0005887", "GO:0009986", "GO:0031226", "GO:0045121",
        "GO:0016020", "GO:0031224", "GO:0098590", "GO:0098589", "GO:0030054",
        "GO:0042995", "GO:0043197", "GO:0044853", "GO:0098552", "GO:0030659",
        "GO:0016021", "GO:0031225", "GO:0005902", "GO:0070161", "GO:0031982"
    ]
}

def api_request(url, data, timeout=60):
    """Make API request with error handling."""
    try:
        response = requests.post(url, data=data, timeout=timeout)
        return response if response.status_code == 200 else None
    except:
        return None


def get_ensembl_ids(genes, organism):
    """Convert gene symbols to Ensembl IDs."""
    print(f"Converting {len(genes)} genes to Ensembl IDs...")
    gene_map = {}
    species = "mouse" if organism.lower() == "mouse" else "human"

    for i in range(0, len(genes), 1000):
        batch = genes[i:i+1000]
        response = api_request(
            "http://mygene.info/v3/query",
            {"q": ",".join(batch), "scopes": "symbol,alias",
             "fields": "ensembl.gene", "species": species, "size": 1}
        )
        if response:
            for result in response.json():
                if "ensembl" in result and not result.get("notfound"):
                    symbol = result.get("query", "")
                    ensembl = result["ensembl"]
                    gene_id = ensembl[0]["gene"] if isinstance(ensembl, list) else ensembl.get("gene")
                    if gene_id:
                        gene_map[symbol] = gene_id
        print(f"  {min(i+1000, len(genes))}/{len(genes)}")

    print(f"Mapped {len(gene_map)}/{len(genes)} genes")
    return gene_map


def get_go_annotations(ensembl_ids, organism):
    """Get GO annotations for Ensembl IDs."""
    print(f"Querying GO annotations for {len(ensembl_ids)} genes...")
    dataset = "mmusculus_gene_ensembl" if organism.lower() == "mouse" else "hsapiens_gene_ensembl"
    gene_go = defaultdict(list)

    for i in range(0, len(ensembl_ids), 500):
        batch = ensembl_ids[i:i+500]
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="0">
    <Dataset name="{dataset}" interface="default">
        <Filter name="ensembl_gene_id" value="{','.join(batch)}"/>
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="go_id"/>
    </Dataset>
</Query>'''

        response = api_request("http://www.ensembl.org/biomart/martservice", {"query": xml})
        if response:
            for line in response.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1].startswith("GO:"):
                    gene_go[parts[0]].append(parts[1])
        print(f"  {min(i+500, len(ensembl_ids))}/{len(ensembl_ids)}")

    print(f"Retrieved GO for {len(gene_go)}/{len(ensembl_ids)} genes")
    return dict(gene_go)


def classify_genes(genes, gene_to_ensembl, gene_to_go, go_terms):
    """Classify genes by cellular location."""
    print("\nClassifying genes...")
    results = {k: [] for k in go_terms.keys()}
    classified = set()

    # First pass: classify genes with GO annotations
    for gene in genes:
        ensembl = gene_to_ensembl.get(gene)
        if not ensembl or ensembl not in gene_to_go:
            continue

        go_set = set(gene_to_go[ensembl])
        for compartment, go_list in go_terms.items():
            if any(go in go_set for go in go_list):
                results[compartment].append(gene)
                classified.add(gene)

    # Second pass: assign all unclassified genes to cytoplasm (default)
    unclassified = [g for g in genes if g not in classified]
    results["cytoplasm"].extend(unclassified)

    print(f"  nucleus: {len(results['nucleus'])} genes")
    print(f"  cytoplasm: {len(results['cytoplasm'])} genes ({len(unclassified)} assigned by default)")
    print(f"  cell_membrane: {len(results['cell_membrane'])} genes")

    return results


def save_results(results, output_dir):
    """Save gene lists to files."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving to {output_dir}/")

    for compartment, genes in results.items():
        filename = f"{output_dir}/genes_{compartment}.txt"
        with open(filename, 'w') as f:
            f.write("\n".join(sorted(genes)))
        print(f"  {compartment}: {len(genes)} genes")


def main():
    """Run analysis pipeline."""
    print("=" * 60)
    print("Gene Localization GO Analysis")
    print("=" * 60)

    # Load genes from h5ad
    import scanpy as sc
    print(f"\nLoading {INPUT_H5AD}...")
    adata = sc.read_h5ad(INPUT_H5AD)
    genes = adata.var_names.tolist()
    print(f"Loaded {len(genes)} genes\n")

    # Pipeline
    gene_to_ensembl = get_ensembl_ids(genes, ORGANISM)
    ensembl_ids = list(set(gene_to_ensembl.values()))
    gene_to_go = get_go_annotations(ensembl_ids, ORGANISM)
    results = classify_genes(genes, gene_to_ensembl, gene_to_go, GO_TERMS)
    save_results(results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
