import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Literal, get_args

import google.generativeai as genai
import pandas as pd
import requests
import typer
from dotenv import load_dotenv
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel

load_dotenv()
YOUR_API_KEY = os.getenv("LLM_API_KEY")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
genai.configure(api_key=YOUR_API_KEY)


class Answer(BaseModel):
    is_valid: Literal["yes", "no", "unknown"]
    explanation: str


def generate_gemini_compatible_schema(model: BaseModel) -> Dict[str, Any]:
    """Generates a Gemini-compatible schema from a Pydantic model."""
    schema = {"type": "object", "properties": {}, "required": []}

    for field_name, field in model.__fields__.items():
        field_type = field.annotation

        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            # Handle List[str] and List[BaseModel]
            list_arg = get_args(field_type)[0]  # Get the type of the list elements

            if isinstance(list_arg, type) and issubclass(list_arg, BaseModel):
                # Recursive call for nested Pydantic models in a list
                nested_schema = generate_gemini_compatible_schema(list_arg)
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": nested_schema,
                }
            else:
                # Handle simple lists (e.g., List[str])
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": {"type": "string"},
                }

        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Recursive call for nested Pydantic models
            schema["properties"][field_name] = generate_gemini_compatible_schema(
                field_type
            )

        else:
            # Handle primitive types (string, integer, etc.)
            schema["properties"][field_name] = {"type": "string"}  # Default to string

    schema["required"] = list(model.__fields__.keys())  # All fields are required
    return schema


# Auto-generate JSON schema from the Pydantic model
schema_json = generate_gemini_compatible_schema(Answer)

PROMPT_TEMPLATE_GENE = """Given these abstracts from PubMed below, Answer 'Yes' or 'No': Is the gene {variable} associated with the cancer type {cancer_type}. Make sure the given text mentions the exact gene and cancer types given and no other abbreviations that could resemble them. Summarize the association made in the given text in 1 line. Output your response in json format with top level keys being 'is_valid' with a literal value of 'yes' or 'no' and 'explanation' with value of not more than 1 sentence explaining association or no association based on the given text. Here is the given text = {efetch_output}.
"""

PROMPT_TEMPLATE_PATHWAY = """Given these abstracts from PubMed below, Answer 'Yes' or 'No': Is the {variable} associated with the cancer type {cancer_type}. 
Make sure the given text mentions the exact pathway and cancer types given and no other abbreviations that could resemble them. Summarize the association made 
in the given text in 1 line. Output your response in json format with top level keys being 'is_valid' with a literal value of 'yes' or 'no' and 'explanation' 
with value of not more than 1 sentence explaining association or no association based on the given text. If the association is unclear, answer 'no' and explain why under 
'explanation'. Here is the given text = {efetch_output}.
"""

PROMPT_TEMPLATE_MOLECULAR_SUBTYPE = """Given these abstracts from PubMed below, Answer 'Yes' or 'No': Is the {variable} molecular subtype associated with the cancer type {cancer_type}. 
Make sure the given text mentions the exact molecular subtype and cancer types given and no other abbreviations that could resemble them. Summarize the association made 
in the given text in 1 line. Output your response in json format with top level keys being 'is_valid' with a literal value of 'yes' or 'no' and 'explanation' 
with value of not more than 1 sentence explaining association or no association based on the given text. If the association is unclear, answer 'no' and explain why under 
'explanation'. Here is the given text = {efetch_output}.
"""


def esearch_efetch(query):
    # assemble the esearch URL
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    url = f"{base}esearch.fcgi"
    # post the esearch URL
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 5,
        "usehistory": "y",
        "api_key": NCBI_API_KEY,
    }
    response_esearch = requests.get(url, params=params, timeout=(3, 30))
    id_pattern = r"<Id>(\d+)<\/Id>"
    if id_pattern:
        all_ids = re.findall(id_pattern, response_esearch.text)
        # all_ids will now be a list of strings like: ['40823818', '40581509', ...]
        id_string = ",".join(all_ids)
        print(id_string)
        ### include this code for ESearch-EFetch
        # assemble the efetch URL
        efetch_url = f"{base}efetch.fcgi"
        # post the efetch URL
        params = {
            "db": "pubmed",
            "id": id_string,
            "rettype": "abstract",
            "usehistory": "y",
            "api_key": NCBI_API_KEY,
        }
        response_efetch = requests.get(efetch_url, params=params, timeout=(3, 30))
        time.sleep(0.3)
        output = response_efetch.text
    else:
        output = "no PMIDs found"
        id_string = "None"
    return (output, id_string)


def llm_to_validate_association(
    prompt_template, variable, cancer_type, efetch_output, llm_model, temperature
):

    generation_config = GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json",  # Ask Gemini to output JSON directly
        response_schema=schema_json,
    )

    model = genai.GenerativeModel(
        model_name=llm_model,
        generation_config=generation_config,
    )

    PROMPT_TEMPLATE = prompt_template
    current_prompt = PROMPT_TEMPLATE.format(
        variable=variable, cancer_type=cancer_type, efetch_output=efetch_output
    )

    try:
        response = model.generate_content(current_prompt)
        parsed_json_data_dict = json.loads(response.text)
        parsed_model = Answer(**parsed_json_data_dict)

    except Exception as e:
        print(f"  Error processing : {e}")
        parsed_model = Answer(is_valid="unknown", explanation="LLM parsing failed")

        # Log errors, and check response for more details if available
        if "response" in locals() and hasattr(response, "prompt_feedback"):
            print(f"    Prompt Feedback: {response.prompt_feedback}")
        if (
            "response" in locals()
            and hasattr(response, "candidates")
            and response.candidates
        ):
            print(
                f"    Candidate Finish Reason: {response.candidates[0].finish_reason}"
            )
            if response.candidates[0].finish_reason.name == "SAFETY":
                print(f"    Safety Ratings: {response.candidates[0].safety_ratings}")

    # Pause to respect the API rate limit
    time.sleep(5)
    return parsed_model.model_dump()


app = typer.Typer()


@app.command()
def validate(
    input_oncotree_gene_file: Path = typer.Option(
        ...,
        "--input_oncotree_gene_filepath",
        "-i",
        help="Path to the LLM output JSON file with OncoTree gene associations",
    ),
    input_reference: Path = typer.Option(
        ...,
        "--input_reference_filepath",
        "-ref",
        help="Path to the supplementary table file with reference OncoTree gene associations",
    ),
    llm_model: str = typer.Option(
        "--model_name",
        "-model",
        help="enter the string name of the LLM model to be used",
    ),
    temperature: float = typer.Option(
        ...,
        "--input_LLM_temperature",
        "-temp",
        help="Temperature setting for LLM: value between 0 to 1",
    ),
):
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

    tcgaset = pd.read_excel(input_reference, sheet_name="Table S1")
    tcgaset.columns = tcgaset.iloc[2].tolist()
    tcgaset = tcgaset[3:]
    tcgaset_pancan = tcgaset[tcgaset["Cancer"].str.contains("PANCAN")]
    pancan_gene_list = tcgaset_pancan["Gene"].tolist()
    pancan_set = set(pancan_gene_list)
    reference_gene_list = []
    validation_results = {}
    #llm_validation = {}
    for item in oncotree_gene:
        if item not in {"DSRCT"}:
            continue
        validation_all = {}
        reference_gene_list = pancan_gene_list
        if item in tcgaset["Cancer"].tolist():
            tcgaset_item = tcgaset[tcgaset["Cancer"].str.contains(item)]
            item_set = set(tcgaset_item["Gene"].tolist())
            item_genes_to_add = item_set - pancan_set
            reference_gene_list.extend(list(item_genes_to_add))
        valid_genes = {}
        llm_response_combined = {}
        for gene in oncotree_gene[item]["associated_genes"]:
            if gene["gene_symbol"] in reference_gene_list:
                valid_genes[gene["gene_symbol"]] = "valid"
            else:
                query = f"gene AND {gene['gene_symbol']} AND {oncotree_gene[item]['cancer_name']}"
                print(query)

                esearch_efetch_output, esearch_ids = esearch_efetch(query)
                if esearch_efetch_output == "no PMIDs found":
                    valid_genes[gene["gene_symbol"]] = "not in references queried"

                else:
                    llm_response = llm_to_validate_association(
                        PROMPT_TEMPLATE_GENE,
                        gene["gene_symbol"],
                        oncotree_gene[item]["cancer_name"],
                        esearch_efetch_output,
                        llm_model,
                        temperature,
                    )
                    if llm_response["is_valid"] == "yes":
                        valid_genes[gene["gene_symbol"]] = f"valid(PMIDs:{esearch_ids})"
                    else:
                        valid_genes[gene["gene_symbol"]] = (
                            f"not valid based on abstracts in PubMed, IDs:{esearch_ids}"
                        )

                llm_response_combined[gene["gene_symbol"]] = llm_response

        valid_pathways = {}
        for pathway, value in oncotree_gene[item]["associated_pathways"].items():
            if value == "yes":
                pathway_string = " ".join(pathway.split("_"))
                query = f"{pathway_string} AND {oncotree_gene[item]['cancer_name']}"
                print(query)
                esearch_efetch_output, esearch_ids = esearch_efetch(query)
                if esearch_efetch_output == "no PMIDs found":
                    valid_pathways[pathway] = "not in references queried"
                else:
                    llm_response = llm_to_validate_association(
                        PROMPT_TEMPLATE_PATHWAY,
                        pathway_string,
                        oncotree_gene[item]["cancer_name"],
                        esearch_efetch_output,
                        llm_model,
                        temperature,
                    )
                    if llm_response["is_valid"] == "yes":
                        valid_pathways[pathway] = f"valid(PMIDs:{esearch_ids})"
                    else:
                        valid_pathways[pathway] = (
                            f"not valid based on abstracts in PubMed, IDs:{esearch_ids}"
                        )
                llm_response_combined[pathway] = llm_response

        valid_molecular_subtypes = {}
        for molecular_subtype in oncotree_gene[item]["molecular_subtypes"]:
            query = f"{molecular_subtype} AND {oncotree_gene[item]['cancer_name']}"
            print(query)
            esearch_efetch_output, esearch_ids = esearch_efetch(query)
            if esearch_efetch_output == "no PMIDs found":
                valid_molecular_subtypes[molecular_subtype] = (
                    "not in references queried"
                )
            else:
                llm_response = llm_to_validate_association(
                    PROMPT_TEMPLATE_MOLECULAR_SUBTYPE,
                    molecular_subtype,
                    oncotree_gene[item]["cancer_name"],
                    esearch_efetch_output,
                    llm_model,
                    temperature,
                )
                if llm_response["is_valid"] == "yes":
                    valid_molecular_subtypes[molecular_subtype] = f"valid(PMIDs:{esearch_ids})"
                else:
                    valid_molecular_subtypes[molecular_subtype] = (
                        f"not valid based on abstracts in PubMed, IDs:{esearch_ids}"
                    )
            llm_response_combined[molecular_subtype] = llm_response
        # llm_validation[item] = llm_response_combined
        validation_all["valid_genes"] = valid_genes
        validation_all["valid_pathways"] = valid_pathways
        validation_all["valid_molecular_subtypes"] = valid_molecular_subtypes
        validation_all["llm_responses"] = llm_response_combined
        validation_results[item] = validation_all

    with open(
        "gene_pathway_lists/validate_genes_pathways_in_references.json", "w"
    ) as f:
        json.dump(validation_results, f, indent=2)
    # with open("gene_pathway_lists/llm_validation_responses.json", "w") as file:
    # json.dump(llm_validation, file, indent=2)


if __name__ == "__main__":
    app()
