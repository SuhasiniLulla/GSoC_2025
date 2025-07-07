
# GSoC 2025: Generate Gene and Pathway Lists for OncoTree Codes using LLM Prompting

## Description

This project, developed as part of Google Summer of Code 2025, focuses on generating gene and pathway lists for cancer types defined by OncoTree codes. For more details, see the project description from the participating organization, cBioPortal for Cancer Genomics, here: https://github.com/cBioPortal/GSoC/issues/114. We are leveraging Large Language Models (LLMs) to generate lists of genes and pathways associated with each cancer subtype defined in the OncoTree ontology.

## Installation

### Using `uv`

This project supports `uv` for fast and reproducible Python environments.


**Install `uv`**:

   ```pip install uv```

**Install dependencies**:

To install minimal dependencies in the pyproject.toml 

```uv sync```


## 🔐 Environment Variables

This project uses environment variables to manage sensitive information like API keys.

Create a `.env` file in the root directory of the project with the following format:

```touch .env```

```LLM_API_KEY= YOUR_API_KEY_HERE```

## Run Script

**Run script to generate lists, adding the name of the OncoTree input JSON file**:

```uv run gsoc2025_llm_prompt_trial_6-2-25.py -i ONCOTREE_FILE_NAME.json```
