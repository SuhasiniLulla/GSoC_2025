import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Literal, get_args

import pandas as pd
import requests
import typer
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel

load_dotenv()
YOUR_API_KEY = os.getenv("LLM_API_KEY")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# Set API keys for providers
#os.environ["OPENAI_API_KEY"] = YOUR_API_KEY
#os.environ["ANTHROPIC_API_KEY"] = YOUR_API_KEY
os.environ["GOOGLE_API_KEY"] = YOUR_API_KEY
#os.environ["MISTRAL_API_KEY"] = YOUR_API_KEY

class Answer(BaseModel):
    is_valid: Literal["yes", "no", "unknown"]
    explanation: str


def generate_json_schema(model: BaseModel) -> Dict[str, Any]:
    schema = {"type": "object", "properties": {}, "required": []}
    for field_name, field in model.__fields__.items():
        field_type = field.annotation
        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            list_arg = get_args(field_type)[0]
            if isinstance(list_arg, type) and issubclass(list_arg, BaseModel):
                nested_schema = generate_json_schema(list_arg)
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": nested_schema,
                }
            else:
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": {"type": "string"},
                }
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            schema["properties"][field_name] = generate_json_schema(field_type)
        else:
            schema["properties"][field_name] = {"type": "string"}
    schema["required"] = list(model.__fields__.keys())
    return schema


# Auto-generate JSON schema from the Pydantic model
schema_json = generate_json_schema(Answer)

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

def retry_with_backoff(func, max_retries=5, base_delay=1, jitter=True):
    """Retries a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            wait_time = base_delay * (2**attempt)
            if jitter:
                wait_time += random.uniform(0, 1)
            print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)
    raise Exception(f"Failed after {max_retries} retries")


def call_llm_with_retry(model, messages, temperature):
    """Wrapper for LiteLLM completion with retry logic."""

    def api_call():
        return completion(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    return retry_with_backoff(api_call, max_retries=5, base_delay=1)

def try_parse_json(output: str) -> dict:
    """Attempts to parse JSON with regex extraction fallback."""
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def repair_with_llm(broken_output: str, llm_model: str) -> dict:
    """Ask the LLM to fix malformed JSON."""
    repair_prompt = f"""
    The following JSON is invalid or malformed. Please fix it and return only valid JSON:

    {broken_output}
    """
    response = call_llm_with_retry(
        model=llm_model,
        messages=[{"role": "user", "content": repair_prompt}],
        temperature=0,
    )
    fixed = response.choices[0].message.content
    return try_parse_json(fixed)


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
    all_ids = re.findall(id_pattern, response_esearch.text)
    if all_ids:
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

    current_prompt = prompt_template.format(
        variable=variable, cancer_type=cancer_type, efetch_output=efetch_output
    )

    try:
        response = call_llm_with_retry(
            model=llm_model,
            messages=[{"role": "user", "content": current_prompt}],
            temperature=temperature,
        )

        response_text = (
            response.choices[0].message.content
            if hasattr(response, "choices")
            else str(response)
        )

        try:
            parsed_json_data_dict = try_parse_json(response_text)
        except json.JSONDecodeError:
            print("JSON malformed — attempting LLM repair...")
            parsed_json_data_dict = repair_with_llm(response_text, llm_model)

        parsed_model = Answer(**parsed_json_data_dict)

    except Exception as e:
        print(f"  Error processing {cancer_type}: {e}")
        parsed_model = Answer(is_valid="unknown", explanation="LLM parsing failed")
    
    # Pause to respect the API rate limit
    time.sleep(5)
    return parsed_model.model_dump()


app = typer.Typer()


@app.command()
def validate(
    input_oncotree_llmoutput_file: Path = typer.Option(
        ...,
        "--input_oncotree_llmoutput_filepath",
        "-i",
        help="Path to the LLM output JSON file with OncoTree gene, pathway, and molecular subtype associations",
    ),
    input_reference_genelist: Path = typer.Option(
        ...,
        "--input_reference_genelist_filepath",
        "-ref",
        help="Path to the supplementary table file with reference OncoTree gene associations",
    ),
    llm_model: str = typer.Option(
        "gpt-4o-mini",
        "--model_name",
        "-model",
        help="LLM model name supported by LiteLLM",
    ),
    temperature: float = typer.Option(
        0.0,
        "--input_LLM_temperature",
        "-temp",
        help="Temperature setting for LLM: 0 → deterministic, 1 → creative",
    ),
):
    
    typer.echo(f"Input file path: {input_oncotree_llmoutput_file}")
    typer.echo(f"Input reference file path: {input_reference_genelist}")

    if not input_oncotree_llmoutput_file.exists():
        typer.echo(f"File not found: {input_oncotree_llmoutput_file}")
        raise typer.Exit(code=1)

    with input_oncotree_llmoutput_file.open("r") as f:
        oncotree_llmoutput = json.load(f)

    if not input_reference_genelist.exists():
        typer.echo(f"File not found: {input_reference_genelist}")
        raise typer.Exit(code=1)

    tcgaset = pd.read_excel(input_reference_genelist, sheet_name="Table S1")
    tcgaset.columns = tcgaset.iloc[2].tolist()
    tcgaset = tcgaset[3:]
    tcgaset_pancan = tcgaset[tcgaset["Cancer"].str.contains("PANCAN")]
    pancan_gene_list = tcgaset_pancan["Gene"].tolist()
    pancan_set = set(pancan_gene_list)
    reference_gene_list = []
    validation_results = {}
    invalid_results = {}
    
    for item in oncotree_llmoutput:
        # Collect per-cancer results
        validation_all = {
            "valid_genes": {},
            "valid_pathways": {},
            "valid_molecular_subtypes": {}
        }
        invalid_all = {
            "invalid_genes": {},
            "invalid_pathways": {},
            "invalid_molecular_subtypes": {}
        }

        reference_gene_list = pancan_gene_list.copy()
        if item in tcgaset["Cancer"].tolist():
            tcgaset_item = tcgaset[tcgaset["Cancer"].str.contains(item)]
            item_set = set(tcgaset_item["Gene"].tolist())
            item_genes_to_add = item_set - pancan_set
            reference_gene_list.extend(list(item_genes_to_add))
        
        for gene in oncotree_llmoutput[item]["associated_genes"]:

            if gene["gene_symbol"] in reference_gene_list:
                validation_all["valid_genes"][gene["gene_symbol"]] = {
                    "validation_source": "reference_TCGA_set",
                    "valid": "yes",
                    "details": "found in gene list provided in reference input",
                    "llm_output": None,
                }
            else:
                query = f"gene AND {gene['gene_symbol']} AND {oncotree_llmoutput[item]['cancer_name']}"
                esearch_efetch_output, esearch_ids = esearch_efetch(query)
                if esearch_efetch_output == "no PMIDs found":
                    entry = {
                        "validation_source": "pubmed_llm",
                        "valid": "unknown",
                        "details": "no abstracts found in PubMed",
                        "llm_output": None,
                    }
                else:
                    llm_response = llm_to_validate_association(
                        PROMPT_TEMPLATE_GENE,
                        gene["gene_symbol"],
                        oncotree_llmoutput[item]["cancer_name"],
                        esearch_efetch_output,
                        llm_model,
                        temperature,
                    )
                    entry = {
                        "validation_source": "pubmed_llm",
                        "valid": llm_response["is_valid"],
                        "details": f"based on PMIDs: {esearch_ids}",
                        "llm_output": llm_response["explanation"],
                    }

                if entry["valid"] == "yes":
                    validation_all["valid_genes"][gene["gene_symbol"]] = entry
                else:
                    invalid_all["invalid_genes"][gene["gene_symbol"]] = entry

        for pathway, value in oncotree_llmoutput[item]["associated_pathways"].items():
            if value == "yes":
                pathway_string = " ".join(pathway.split("_"))
                query = f"{pathway_string} AND {oncotree_llmoutput[item]['cancer_name']}"
                print(query)
                esearch_efetch_output, esearch_ids = esearch_efetch(query)
                if esearch_efetch_output == "no PMIDs found":
                    entry = {
                        "validation_source": "pubmed_llm",
                        "valid": "unknown",
                        "details": "no abstracts found in PubMed",
                        "llm_output": None,
                    }
                else:
                    llm_response = llm_to_validate_association(
                        PROMPT_TEMPLATE_PATHWAY,
                        pathway_string,
                        oncotree_llmoutput[item]["cancer_name"],
                        esearch_efetch_output,
                        llm_model,
                        temperature,
                    )
                    entry = {
                        "validation_source": "pubmed_llm",
                        "valid": llm_response["is_valid"],
                        "details": f"based on PMIDs: {esearch_ids}",
                        "llm_output": llm_response["explanation"],
                    }
                
                if entry["valid"] == "yes":
                    validation_all["valid_pathways"][pathway] = entry
                else:
                    invalid_all["invalid_pathways"][pathway] = entry

        for molecular_subtype in oncotree_llmoutput[item]["molecular_subtypes"]:
            query = f"{molecular_subtype} AND {oncotree_llmoutput[item]['cancer_name']}"
            print(query)
            esearch_efetch_output, esearch_ids = esearch_efetch(query)
            if esearch_efetch_output == "no PMIDs found":
                entry = {
                    "validation_source": "pubmed_llm",
                    "valid": "unknown",
                    "details": "no abstracts found in PubMed",
                    "llm_output": None,
                }
            else:
                llm_response = llm_to_validate_association(
                    PROMPT_TEMPLATE_MOLECULAR_SUBTYPE,
                    molecular_subtype,
                    oncotree_llmoutput[item]["cancer_name"],
                    esearch_efetch_output,
                    llm_model,
                    temperature,
                )
                entry = {
                    "validation_source": "pubmed_llm",
                    "valid": llm_response["is_valid"],
                    "details": f"based on PMIDs: {esearch_ids}",
                    "llm_output": llm_response["explanation"],
                }

            if entry["valid"] == "yes":
                validation_all["valid_molecular_subtypes"][molecular_subtype] = entry
            else:
                invalid_all["invalid_molecular_subtypes"][molecular_subtype] = entry
    
        validation_results[item] = validation_all
        invalid_results[item] = invalid_all
    
    # Write valid and invalid results separately per run
    valid_output_path = f"gene_pathway_lists/VALID_genes_pathways_molecularsubtypes.json"
    invalid_output_path = f"gene_pathway_lists/INVALID_or_unknown_validation_status.json"

    with open(valid_output_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    with open(invalid_output_path, "w") as f:
        json.dump(invalid_results, f, indent=2)

if __name__ == "__main__":
    app()
