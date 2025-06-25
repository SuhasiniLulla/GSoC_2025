
# GSoC 2025: Generate Gene and Pathway Lists for OncoTree Codes using LLM Prompting

## Description

This project, developed as part of Google Summer of Code 2025, focuses on generating gene and pathway lists for cancer types defined by OncoTree codes. For more details, see the project description from the participating organization, cBioPortal for Cancer Genomics, here: https://github.com/cBioPortal/GSoC/issues/114. We are leveraging Large Language Models (LLMs) to generate lists of genes and pathways associated with each cancer subtype defined in the OncoTree ontology.

## Installation

### Using `uv`

This project supports `uv` for fast and reproducible Python environments.

**Setup virtual environment**:

```python3 -m venv```

**Activate virtual environment**:

```source venv/bin/activate```  # macOS/Linux

```venv\\Scripts\\activate```   # Windows

**Install `uv`**:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

**Install dependencies**:

```uv pip install -r requirements.txt```

**Run script to generate lists, adding the name of the OncoTree input JSON file**:

```uv run --no-project gsoc2025_llm_prompt_trial_6-2-25.py ONCOTREE_FILE_NAME.json```
