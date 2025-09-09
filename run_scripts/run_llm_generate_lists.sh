#!/bin/bash

# Example shell script to run llm_mine_gene_pathway_assoc_oncotree.py with configurable arguments

# Set variables
MODEL_NAME="gemini/gemini-2.0-flash"
TEMPERATURE=0.25
INPUT_FILE="assets/oncotree_latest_stable_June2025.json"
OUTPUT_FILE="gene_pathway_lists/llm_export_lists.json"
ONCOTREE_TRY1="PAAD"
ONCOTREE_TRY2="BRCA"

# Run the Python script with the specified arguments
python generate_lists/llm_mine_gene_pathway_assoc_oncotree.py \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE"\
    -model "$MODEL_NAME" \
    -temp "$TEMPERATURE" \
    -c "$ONCOTREE_TRY1" \
    -c "$ONCOTREE_TRY2"
