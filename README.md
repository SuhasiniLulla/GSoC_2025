
# GSoC 2025: Generate Gene and Pathway Lists for OncoTree Codes using LLM Prompting

## Description
Background:
cBioPortal for Cancer Genomics: An open-source platform that enables interactive exploration and visualization of large-scale cancer genomics datasets.[1,2]

Cancer Classification: Accurate classification of cancer subtypes is essential for diagnosis, prognosis, and treatment. OncoTree provides a standardized, community-driven ontology for cancer subtypes.[3]

OncoTree Integration in cBioPortal: cBioPortal uses OncoTree codes to categorize cancer samples. However, it lacks default gene/pathway recommendations per cancer type, which could enhance data exploration.


Project overview:
This project, developed as part of Google Summer of Code 2025, aims to enhance the cBioPortal platform by generating a list of recommended default genes and pathways for each OncoTree code. To achieve this goal, we use prompt engineering to query a Large Language Model (LLM) for structured lists of genes, pathways, and molecular subtypes associated with each OncoTree code, as shown here for the OncoTree code 'PAAD' (Pancreatic Adenocarcinoma):

```
{
  "PAAD": {
    "cancer_name": "Pancreatic Adenocarcinoma",
    "other_codes_used_for_data_gathering": {
      "NCIt": "C8294",
      "UMLS": "C0281361"
    },
    "associated_genes": [
      {
        "gene_symbol": "KRAS",
        "gene_info": {
          "association_strength": "very strong",
          "reference": "PMID:31582729|PMID:29625052|cBioPortal",
          "mutations": [
            "missense",
            "deletion",
            "insertion"
          ],
          "mutation_origin": "somatic",
          "diagnostic_implication": "diagnostic: Missense mutations in KRAS are associated with PAAD and used for diagnosis.",
          "therapeutic_relevance": "clinical trials such as NCT07020221 are actively testing inhibitors of the actionable missense mutation KRAS G12D which is frequent in PAAD. KRAS mutations are generally considered a negative predictive marker for EGFR inhibitors."
        }
      },
      {
        "gene_symbol": "TP53",
        "gene_info": {
          "association_strength": "very strong",
          "reference": "PMID:31582729|PMID:28726843|cBioPortal",
          "mutations": [
            "truncating",
            "missense",
            "deletion"
          ],
          "mutation_origin": "somatic",
          "diagnostic_implication": "diagnostic: Inactivation mutations in TP53 are associated with PAAD and can be used for diagnosis.",
          "therapeutic_relevance": "TP53 mutations can affect response to chemotherapy and radiation therapy."
        }
      },
      {
        "gene_symbol": "CDKN2A",
        "gene_info": {
          "association_strength": "very strong",
          "reference": "PMID:31582729|PMID:22522928|cBioPortal",
          "mutations": [
            "truncating",
            "deletion",
            "methylation"
          ],
          "mutation_origin": "germline/somatic",
          "diagnostic_implication": "diagnostic: Inactivation mutations in CDKN2A are associated with PAAD and can be used for diagnosis.",
          "therapeutic_relevance": "CDKN2A loss can lead to cell cycle dysregulation and may influence response to CDK4/6 inhibitors."
        }
      },..............
...........],
    "molecular_subtypes": [
      "Squamous",
      "Pancreatic Progenitor",
      "Immunogenic",
      "Aberrantly Differentiated Endocrine Exocrine (ADEX)"
    ],
    "associated_pathways": {
      "ar_signaling": "no",
      "ar_and_steroid_synthesis_enzymes": "no",
      "steroid_inactivating_genes": "no",
      "down_regulated_by_androgen": "no",
      "rtk_ras_pi3k_akt_signaling": "yes",
      "rb_pathway": "yes",
      "cell_cycle_pathway": "yes",
      "hippo_pathway": "yes",
      "myc_pathway": "yes",
      "notch_pathway": "yes",
      "nrf2_pathway": "yes",
      "pi3k_pathway": "yes",
      "rtk_ras_pathway": "yes",
      "tp53_pathway": "yes",
      "wnt_pathway": "yes",
      "cell_cycle_control": "yes",
      "p53_signaling": "yes",
      "notch_signaling": "yes",
      "dna_damage_response": "yes",
      "other_growth_proliferation_signaling": "yes",
      "survival_cell_death_regulation_signaling": "yes",
      "telomere_maintenance": "yes",
      "rtk_signaling_family": "yes",
      "pi3k_akt_mtor_signaling": "yes",
      "ras_raf_mek_erk_jnk_signaling": "yes",
      "angiogenesis": "yes",
      "folate_transport": "yes",
      "invasion_and_metastasis": "yes",
      "tgf_\u03b2_pathway": "yes",
      "oncogenes_associated_with_epithelial_ovarian_cancer": "no",
      "regulation_of_ribosomal_protein_synthesis_and_cell_growth": "yes"
    }
  }
```

Output valid gene, pathway, and molecular subtype sets will be used by cBioPortal to improve visualzation of datasets on the web tool. For example, valid genes will be displayed before other mutated genes in patient and study summary view tabs as illustrated below:

<img width="662" height="431" alt="Screenshot 2025-08-21 at 7 27 11 PM" src="https://github.com/user-attachments/assets/ea39cdee-1d60-4510-8e67-86093ed8bd35" />

This will aid in the identification of mutations relevant to the specific disease being studied.
For more details, see the project description from the participating organization, cBioPortal for Cancer Genomics, here: https://github.com/cBioPortal/GSoC/issues/114. 

Link to GSoC project page: https://summerofcode.withgoogle.com/myprojects/details/2AJ2V3qf

Contributor: Suhasini Lulla

GSoC project mentors: Ino de Bruijn, Dr. Karl Pichotta, Dr. Chris Fong, Dr. Augustin Luna


## Installation

### Create new virtual environment

```python -m venv llm_lists```

```source llm_lists/bin/activate```

### Clone repository

```git clone https://github.com/SuhasiniLulla/GSoC_2025```
```cd GSoC_2025```

### Using `uv`

This project supports `uv` for fast and reproducible Python environments.


**Install `uv`**:

   ```pip install uv```

**Install dependencies**:

To install minimal dependencies in the pyproject.toml 

```uv sync```


## Environment Variables

This project uses environment variables to manage sensitive information like API keys.

Create a `.env` file in the root directory of the project with the following format:

```touch .env```

```nano .env```

```LLM_API_KEY= YOUR_LLM_API_KEY_HERE```

```NCBI_API_KEY= YOUR_NCBI_API_KEY_HERE```

Ctrl+X --> Enter

## Run Script

**Run script to generate lists, adding the name of the OncoTree input JSON file**:
Query an LLM for genes, pathways, and molecular subtypes associated with each OncoTree code. (Future updates: plan to include selecting the OncoTree code(s) of choice).
For each gene association, the LLM will also output information on the strength of association, mutations, diagnostic potential, therapeutic implications, and whether mutations in this gene are observed in somatic contexts only or can be either germline or somatic.
The bash script provided has default input parameter variables set and can be run with command below:

``` bash run_scripts/run_validate_gene_pathway_subtype_lists.sh```

Parameters: 

'-i': Takes file of OncoTree codes in json format (downloadable from https://oncotree.mskcc.org/swagger-ui.html#!/tumor-types-api/tumorTypesGetTreeUsingGET)

'-o': Path to where you want to store the LLM output and filename of your choice.

'-model': Google Gemini model name of your choice (Future updates: plan to make this open to other LLM models such as those from OpenAI, etc using LiteLLM)

'-temp': Temperature setting for LLM, default set at 0.25


**Run script to validate gene, pathway, and molecular subtype lists**:

Using e-utilities to query PubMed for each gene:cancer-type, pathway:cancer-type, and molecular subtype:cancer-type association made by the LLM above. Extracting abstract text for up to 5 PMIDs and querying an LLM to validate the association using this text.

Example use case with the LLM output file and reference gene set file included in this Repo:

```bash run_scripts/run_validate_gene_pathway_subtype_lists.sh```

Parameters: 

'-i': Takes file of generated gene, pathway, molecular subtype lists as produced bythe LLM generate lists script

'ref': Expert established set of gene:cancer-type associations to validate against before using yet another LLM to validate. Here mmc1.xlsx comes from the published gene set for cancer types included in The Cancer Genome Atlas (TCGA) study (PMID:29625053)[4].

'-model': Google Gemini model name of your choice (Future updates: plan to make this open to other LLM models such as those from OpenAI, etc using LiteLLM)

'-temp': Temperature setting for LLM, default set at 0.0 for validation.


**Example structured outputs**:

LLM gene, pathway, molecular subtypes list generated for OncoTree codes PAAD, COAD, DSRCT, MNM, BRCA, NSCLC:
https://github.com/SuhasiniLulla/GSoC_2025/blob/main/gene_pathway_lists/export_lists_info_6codes.json

Validation for gene, pathway, molecular subtype associations made with OncoTree code DSRCT:
https://github.com/SuhasiniLulla/GSoC_2025/blob/main/gene_pathway_lists/validate_genes_pathways_in_references.json

--includes information on whether the association is valid or not, the PMIDs of abstracts input to a second LLM to validate the association, and LLM output with a 1-line explanation of why the association was valid or not.

Validation for gene, pathway, molecular subtype associations made with OncoTree code NSCLC_BRCA_MNM:
https://github.com/SuhasiniLulla/GSoC_2025/blob/main/gene_pathway_lists/validate_genes_pathways_in_references_NSCLC_BRCA_MNM.json


References:
1.	Cerami, E., et al., The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. Cancer Discov, 2012. 2(5): p. 401-4.
2.	Gao, J., et al., Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal. Sci Signal, 2013. 6(269): p. pl1.
3.	Kundra, R., et al., OncoTree: A Cancer Classification System for Precision Oncology. JCO Clinical Cancer Informatics, 2021(5): p. 221-230.
4.	Bailey, M.H., et al., Comprehensive Characterization of Cancer Driver Genes and Mutations. Cell, 2018. 173(2): p. 371-385.e18.


