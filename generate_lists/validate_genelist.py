import os
import json
import pandas as pd
import time
import csv
import typer
from pathlib import Path
import openpyxl

app = typer.Typer()

@app.command()
def validate_genes(input_oncotree_gene_file: Path = typer.Option(..., "--input_oncotree_gene_filepath", "-i", help="Path to the LLM output JSON file with OncoTree gene associations"), input_reference: Path = typer.Option(..., "--input_reference_filepath", "-ref", help="Path to the supplementary table file with reference OncoTree gene associations")):
    typer.echo(f"Input file path: {input_oncotree_gene_file}")
    typer.echo(f"Input reference file path: {input_reference}")

    if not input_oncotree_gene_file.exists():
        typer.echo(f"File not found: {input_oncotree_gene_file}")
        raise typer.Exit(code=1)

    with input_oncotree_gene_file.open("r") as f:
        oncotree_gene = json.load(f)

    if not input_reference.exists():
        typer.echo(f"File not found: {input_reference}")
        raise typer.Exit(code=1)

    tcgaset=pd.read_excel(input_reference, sheet_name='Table S1')
    tcgaset.columns=tcgaset.iloc[2].tolist()
    tcgaset=tcgaset[3:]
    tcgaset_pancan=tcgaset[tcgaset["Cancer"].str.contains("PANCAN")==True]
    pancan_gene_list=tcgaset_pancan['Gene'].tolist()
    pancan_set=set(pancan_gene_list)
    reference_gene_list=[]
    validation_results = {}
    for i, item in enumerate(oncotree_gene):
        reference_gene_list=pancan_gene_list
        if i<1000:
            if item in tcgaset["Cancer"].tolist():
                tcgaset_item=tcgaset[tcgaset["Cancer"].str.contains(item)==True]
                item_set=set(tcgaset_item['Gene'].tolist())
                item_genes_to_add=item_set-pancan_set
                reference_gene_list.extend(list(item_genes_to_add))     
            valid_genes={}
            for gene in oncotree_gene[item]['associated_genes']:
                if gene in reference_gene_list:
                    valid_genes[gene]="valid"
                else:
                    valid_genes[gene]="not in reference"
            validation_results[item] = valid_genes
    
    with open("gene_pathway_lists/validate_genes_in_reference.json", "w") as f:
        json.dump(validation_results, f, indent=2)



if __name__ == "__main__":
    app()