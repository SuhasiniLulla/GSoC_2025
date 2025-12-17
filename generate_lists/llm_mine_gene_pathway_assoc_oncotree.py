import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, get_args

import typer
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, ConfigDict, Field

# from utils import count_tokens_in_string

load_dotenv()

if "USE_LITELLM_PROXY" not in os.environ:
    YOUR_API_KEY = os.getenv("LLM_API_KEY")

    # Set API keys for providers
    # os.environ["OPENAI_API_KEY"] = YOUR_API_KEY
    # os.environ["ANTHROPIC_API_KEY"] = YOUR_API_KEY
    os.environ["GOOGLE_API_KEY"] = YOUR_API_KEY
    # os.environ["MISTRAL_API_KEY"] = YOUR_API_KEY


class AssociatedPathways(BaseModel):
    cell_cycle_pathway: Literal["yes", "no"]
    hippo_pathway: Literal["yes", "no"]
    myc_pathway: Literal["yes", "no"]
    notch_pathway: Literal["yes", "no"]
    nrf2_pathway: Literal["yes", "no"]
    pi3k_pathway: Literal["yes", "no"]
    tgf_β_pathway: Literal["yes", "no"]
    rtk_ras_pathway: Literal["yes", "no"]
    tp53_pathway: Literal["yes", "no"]
    wnt_pathway: Literal["yes", "no"]


class GeneInfo(BaseModel):
    association_strength: Literal[
        "very strong", "strong", "moderate", "weak", "very weak"
    ]
    reference: str
    mutations: List[str]
    mutation_origin: Literal["germline/somatic", "somatic", "germline"]
    diagnostic_implication: str
    therapeutic_relevance: str


class AssociatedGene(BaseModel):
    gene_symbol: str
    gene_info: GeneInfo


class GenerateGeneLists(BaseModel):
    cancer_name: str
    associated_genes: List[AssociatedGene] = Field(
        ..., description="List of gene symbols and their associated data"
    )

    model_config = ConfigDict(validate_by_name=True)


class GeneratePathwayLists(BaseModel):
    cancer_name: str
    associated_pathways: AssociatedPathways

    model_config = ConfigDict(validate_by_name=True)


class GenerateMolecularSubtypeLists(BaseModel):
    cancer_name: str
    molecular_subtypes: List[str]

    model_config = ConfigDict(validate_by_name=True)


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
schema_json_genes = generate_json_schema(GenerateGeneLists)
schema_json_pathways = generate_json_schema(GeneratePathwayLists)
schema_json_molecularsubtypes = generate_json_schema(GenerateMolecularSubtypeLists)

#print("Generated Schema:\n", json.dumps(schema_json_genes, indent=2))

PROMPT_TEMPLATE_GENES = """You are an expert in clinical cancer genetics, specifically in gene-disease curations (for hereditary and sporadic cancers). Based on scientific literature in PubMed, current genetic testing practices in oncology clinics, gene-disease association curations in ClinGen, OMIM, GeneReviews, and similar expert or peer reviewed resoursces, and public tumor sequencing databases such as cBioPortal, and COSMIC, list the genes, mutations in which are classically associated with {cancer_name} ({oncotree_code}). Different ontologies have different terms/codes to depict the same cancer sub-type. {oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS). Use these codes to gather as much literature/data as possible to provide a comprehensive list of genes in JSON structured format. The associated gene list should be ranked by strength and likelihood of association such that the first gene in the list has the strongest association with the cancer type and the last gene in the list has the weakest association with the cancer type. The gene list should be of high quality, accurate, and should not exceed 50 in count. The JSON should have top-level keys:
"oncotree_code",
"cancer_name" (full name of the code),
"associated_genes" (a list of dictionaries - one dictionary for every associated gene, having top level keys of 'gene_symbol' and 'gene_info'. 'gene_symbol' should be only 1 gene per key. 'gene_info' is a dictionary with keys and values formatted as follows: 1. 'association_strength', value: classified as 'very strong', 'strong', 'moderate', 'weak', or 'very weak' association of this particular gene and cancer type depending on the quality and quantity of resources used to associate the gene and cancer type, 2. 'reference', value: resource(s) used to infer the gene-cancer type association (if multiple, then separate by '|'), 3. 'mutations', value: list of types of mutations in the gene that is associated with the given cancer type (such as truncating, splice, missense gain of function, missense-loss of function, missense-neomorphic, missense-hypo-/hyper-morphic, deletion, duplication, fusion, copy number variant, structural variant, complex rearrangements, methylation, and so on relevant to the gene-cancer type association), 4. 'mutation_origin', value: MUST be either "germline/somatic" OR "somatic" where 'germline/somatic' indicates that the cancer mutation in this gene can be present in the germline as cancer predisposing or arise somatically over time (so includes both 'germline' and 'somatic' options in 1 category only), 'somatic' indicates that the cancer mutation in this gene is only of somatic origin and not seen in the germline, 5. 'diagnostic_implication', value: clinical implication of the gene as to whether it is used to diagnose the cancer type, for example, the gene KRAS is associated with PAAD: 'diagnostic: missense mutations in KRAS are associated with PAAD and used for diagnosis.' Limit to 1 sentence, 6. 'therapeutic_relevance', value: if gene mutation informs decision making for therapeutic strategy, for example, for the association of KRAS and PAAD, 'clinical trials such as NCT07020221 are actively testing inhibitors of the actionable missense mutation KRAS G12D which is frequent in PAAD. Effect on immunotherapy is ....'),
Return **strict JSON** without trailing commas, unescaped quotes, or comments. Ensure it parses with `json.loads()`."""

PROMPT_TEMPLATE_PATHWAYS = """You are an expert in clinical cancer genetics, specifically in gene-disease and pathway-disease curations (for hereditary and sporadic cancers). Based on scientific literature in PubMed, current genetic testing practices in oncology clinics, gene-disease association curations in ClinGen, OMIM, GeneReviews, and similar expert or peer reviewed resoursces, and public tumor sequencing databases such as cBioPortal, and COSMIC, list the pathways classically associated with {cancer_name} ({oncotree_code}). Different ontologies have different terms/codes to depict the same cancer sub-type. {oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS). Use these codes to gather as much literature/data as possible to provide a comprehensive list of pathways in JSON structured format. The JSON should have top-level keys:
"oncotree_code",
"cancer_name" (full name of the code),
"associated_pathways" (a dictionary with keys being each pathway name in the list: ['cell_cycle_pathway', 'hippo_pathway', 'myc_pathway', 'notch_pathway', 'nrf2_pathway', 'pi3k_pathway', 'tgf_β_pathway', 'rtk_ras_pathway', 'tp53_pathway', 'wnt_pathway'] and the value being 'yes' if associated with cancer sub-type or 'no' if pathway not associated with cancer sub-type). Return **strict JSON** without trailing commas, unescaped quotes, or comments. Ensure it parses with `json.loads()`."""

PROMPT_TEMPLATE_MOLECULARSUBTYPES = """You are an expert in clinical cancer genetics, specifically in gene-disease and pathway-disease curations (for hereditary and sporadic cancers). Based on scientific literature in PubMed, current genetic testing practices in oncology clinics, gene-disease association curations in ClinGen, OMIM, GeneReviews, and similar expert or peer reviewed resoursces, and public tumor sequencing databases such as cBioPortal, and COSMIC, list the molecular subtypes classically associated with {cancer_name} ({oncotree_code}). Different ontologies have different terms/codes to depict the same cancer sub-type. {oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS). Use these codes to gather as much literature/data as possible to provide a comprehensive list of molecular subtypes in JSON structured format. The JSON should have top-level keys:
"oncotree_code",
"cancer_name" (full name of the code),
"molecular_subtypes" (a list of expression-based, genomic, or histological molecular subtypes known to occur in {cancer_name}. These subtypes should be informative for clinical decision-making, such as guiding treatment selection or predicting prognosis. Please use descriptive names or standard nomenclature for the subtypes, combine synonymous subtypes so that it is a list of exclusive subtypes, and prioritize those with known clinical implications. Return only a list of strings and each string should be the subtype name only, without extra descriptions or nested dictionaries. The output must always include "molecular_subtypes". If no subtypes exist, return an empty list []. Never omit this field.
Return **strict JSON** without trailing commas, unescaped quotes, or comments. Ensure it parses with `json.loads()`."""


def retry_with_backoff(func, max_retries=5, base_delay=1, jitter=True):
    """Retries a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            wait_time = base_delay * (2**attempt)
            if jitter:
                wait_time += random.uniform(0, 1)
            typer.echo(f"ERROR: Attempt {attempt+1} failed: {e}. Retrying in {wait_time:.2f}s...")
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


app = typer.Typer()


@app.command()
def generate_lists(
    input_oncotree: Path = typer.Option(
        ..., "--input_oncotree_filepath", "-i", help="Path to the OncoTree JSON file"
    ),
    output_lists: Path = typer.Option(
        ..., "--output_filepath", "-o", help="Path and name for output JSON file"
    ),
    llm_model: str = typer.Option(
        "gpt-4o-mini",
        "--model_name",
        "-m",
        help="LLM model name supported by LiteLLM",
    ),
    temperature: float = typer.Option(
        0.25,
        "--input_LLM_temperature",
        "-t",
        help="Temperature setting for LLM: 0 → deterministic, 1 → creative",
    ),
    codes: List[str] = typer.Option(
        None,
        "--codes",
        "-c",
        help="Specific OncoTree codes to process (repeat '-c' flag for each code: e.g. -c BRCA -c PAAD).",
    ),
    all_codes: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="If set, process ALL OncoTree codes in the input file (overrides --codes).",
    ),
    genes_flag: bool = typer.Option(False, "--genes", "-g", help="Generate gene lists"),
    pathways_flag: bool = typer.Option(
        False, "--pathways", "-p", help="Generate pathway lists"
    ),
    molecular_flag: bool = typer.Option(
        False, "--molecular", "-ms", help="Generate molecular subtype lists"
    ),
):
    # Safety guard: require at least one generation type
    if not any([genes_flag, pathways_flag, molecular_flag]):
        typer.echo(
            "ERROR: You must specify at least one of --genes, --pathways, or --molecular"
        )
        raise typer.Exit(code=1)

    typer.echo(f"INFO: Input file path: {input_oncotree}")

    if not input_oncotree.exists():
        typer.echo(f"INFO: File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    # Determine which codes to process
    if all_codes:
        target_codes = {item["code"] for item in oncotree}
        typer.echo(f"INFO: Running for ALL {len(target_codes)} codes in the input file.")
    elif codes:
        target_codes = set(codes)
        typer.echo(f"INFO: Running for user-specified codes: {', '.join(target_codes)}")
    else:
        target_codes = {"COAD", "NSCLC", "PAAD", "DSRCT", "BRCA", "MNM"}
        typer.echo(
            f"INFO: Running for default set of (COAD, NSCLC, PAAD, DSRCT, BRCA, MNM): {', '.join(target_codes)}"
        )

    oncotree_codes_info = {}
    for item in oncotree:
        if item["code"] not in target_codes:
            continue
        code = item["code"]
        name = item["name"]
        umls = item.get("externalReferences", {}).get("UMLS", [None])[0]
        ncit = item.get("externalReferences", {}).get("NCI", [None])[0]
        oncotree_codes_info[code] = {"name": name, "NCIt": ncit, "UMLS": umls}

    if not oncotree_codes_info:
        typer.echo("ERROR: No matching OncoTree codes found in input file.")
        raise typer.Exit(code=1)

    all_results = {}  # A dictionary to store all the AI's answers

    total = len(oncotree_codes_info)
    success_count = 0
    fail_count = 0

    for idx, (oncotree_code, details) in enumerate(
        oncotree_codes_info.items(), start=1
    ):
        percent = (idx / total) * 100
        typer.echo(f"[{idx}/{total}] ({percent:.1f}%) Processing {oncotree_code}...")

        if genes_flag:
            current_prompt = PROMPT_TEMPLATE_GENES.format(
                cancer_name=details["name"],
                oncotree_code=oncotree_code,
                ncit_code=details["NCIt"],
                umls_code=details["UMLS"],
            )
            model_class = GenerateGeneLists
            try:
                response = call_llm_with_retry(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clinical cancer genetics expert. Respond only in valid JSON following the provided schema. Do not include any text outside the JSON.",
                        },
                        {"role": "user", "content": current_prompt},
                    ],
                    temperature=temperature,
                )

                raw_output = response.choices[0].message.content

                try:
                    parsed_json_data_dict = try_parse_json(raw_output)
                except Exception:
                    parsed_json_data_dict = repair_with_llm(raw_output, llm_model)

                parsed_model = model_class(**parsed_json_data_dict)
                all_results.setdefault("genes", {})[
                    oncotree_code
                ] = parsed_model.model_dump()

                safe_oncotree_code = oncotree_code.replace("/", "-")
                tmp_file = f"tmp/tmp_{safe_oncotree_code}.json"
                with open(tmp_file, "w") as f:
                    json.dump(parsed_model.model_dump(), f, indent=2)

                success_count += 1
                typer.echo(f"INFO: Success processing {oncotree_code}; genes_flag")

            except Exception as e:
                typer.echo(f"ERROR: Error processing {oncotree_code}: {e}; genes_flag")

                all_results[oncotree_code] = {
                    "error": str(e),
                    "details_provided": details,
                }
                fail_count += 1

            time.sleep(5)

        if pathways_flag:
            current_prompt = PROMPT_TEMPLATE_PATHWAYS.format(
                cancer_name=details["name"],
                oncotree_code=oncotree_code,
                ncit_code=details["NCIt"],
                umls_code=details["UMLS"],
            )
            model_class = GeneratePathwayLists
            try:
                response = call_llm_with_retry(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clinical cancer genetics expert. Respond only in valid JSON following the provided schema. Do not include any text outside the JSON.",
                        },
                        {"role": "user", "content": current_prompt},
                    ],
                    temperature=temperature,
                )

                raw_output = response.choices[0].message.content

                try:
                    parsed_json_data_dict = try_parse_json(raw_output)
                except Exception:
                    parsed_json_data_dict = repair_with_llm(raw_output, llm_model)

                parsed_model = model_class(**parsed_json_data_dict)
                all_results.setdefault("pathways", {})[
                    oncotree_code
                ] = parsed_model.model_dump()
                success_count += 1
                typer.echo(f"INFO: Success processing {oncotree_code}; pathways_flag")

            except Exception as e:
                typer.echo(f"ERROR: Error processing {oncotree_code}: {e}; pathways_flag")
                all_results[oncotree_code] = {
                    "error": str(e),
                    "details_provided": details,
                }
                fail_count += 1

            time.sleep(5)

        if molecular_flag:
            current_prompt = PROMPT_TEMPLATE_MOLECULARSUBTYPES.format(
                cancer_name=details["name"],
                oncotree_code=oncotree_code,
                ncit_code=details["NCIt"],
                umls_code=details["UMLS"],
            )
            model_class = GenerateMolecularSubtypeLists
            try:
                response = call_llm_with_retry(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clinical cancer genetics expert. Respond only in valid JSON following the provided schema. Do not include any text outside the JSON.",
                        },
                        {"role": "user", "content": current_prompt},
                    ],
                    temperature=temperature,
                )

                raw_output = response.choices[0].message.content

                try:
                    parsed_json_data_dict = try_parse_json(raw_output)
                except Exception:
                    parsed_json_data_dict = repair_with_llm(raw_output, llm_model)

                if "molecular_subtypes" not in parsed_json_data_dict:
                    typer.echo(f"{oncotree_code}: Missing molecular_subtypes, retrying...")

                    retry_prompt = (
                        current_prompt
                        + "\n\nReminder: You must include a 'molecular_subtypes' field, "
                        "even if it is an empty list []. Never omit this field."
                    )

                    retry_response = call_llm_with_retry(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": "Return only strict JSON."},
                            {"role": "user", "content": retry_prompt},
                        ],
                        temperature=temperature,
                    )

                    retry_raw_output = retry_response.choices[0].message.content
                    parsed_json_data_dict = try_parse_json(retry_raw_output)

                parsed_model = model_class(**parsed_json_data_dict)
                all_results.setdefault("molecular_subtypes", {})[
                    oncotree_code
                ] = parsed_model.model_dump()

                success_count += 1
                typer.echo(f"INFO: Success processing {oncotree_code}; molecular_flag")

            except Exception as e:
                typer.echo(f"ERROR: Error processing {oncotree_code}: {e}; molecular_flag")
                all_results[oncotree_code] = {
                    "error": str(e),
                    "details_provided": details,
                }
                fail_count += 1

            time.sleep(5)

    typer.echo(f"\nFinished: {success_count} succeeded, {fail_count} failed, total {total}.")

    with open(output_lists, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    app()
