#!/bin/bash

# Example shell script to run validate_genelist.py with configurable arguments

# Set variables
MODEL_NAME="gemini-2.0-flash"
VALIDATION_TEMPERATURE=0.0
VALIDATION_INPUT_FILE="gene_pathway_lists/export_lists_info_6codes.json"
REFERENCE_FILE="assets/mmc1.xlsx"

# Run the Python script with the specified arguments
uv run --active python generate_lists/validate_genelist.py \
    -i "$VALIDATION_INPUT_FILE" \
    -ref "$REFERENCE_FILE" \
    -model "$MODEL_NAME" \
    -temp "$VALIDATION_TEMPERATURE"
