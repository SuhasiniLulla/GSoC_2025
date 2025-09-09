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

load_dotenv()
YOUR_API_KEY = os.getenv("LLM_API_KEY")

# Set API keys for providers
os.environ["OPENAI_API_KEY"] = YOUR_API_KEY
os.environ["ANTHROPIC_API_KEY"] = YOUR_API_KEY
os.environ["GOOGLE_API_KEY"] = YOUR_API_KEY
os.environ["MISTRAL_API_KEY"] = YOUR_API_KEY


class CodeReferences(BaseModel):
    NCIt: str
    UMLS: str


class AssociatedPathways(BaseModel):
    ar_signaling: Literal["yes", "no"]
    ar_and_steroid_synthesis_enzymes: Literal["yes", "no"]
    steroid_inactivating_genes: Literal["yes", "no"]
    down_regulated_by_androgen: Literal["yes", "no"]
    rtk_ras_pi3k_akt_signaling: Literal["yes", "no"]
    rb_pathway: Literal["yes", "no"]
    cell_cycle_pathway: Literal["yes", "no"]
    hippo_pathway: Literal["yes", "no"]
    myc_pathway: Literal["yes", "no"]
    notch_pathway: Literal["yes", "no"]
    nrf2_pathway: Literal["yes", "no"]
    pi3k_pathway: Literal["yes", "no"]
    rtk_ras_pathway: Literal["yes", "no"]
    tp53_pathway: Literal["yes", "no"]
    wnt_pathway: Literal["yes", "no"]
    cell_cycle_control: Literal["yes", "no"]
    p53_signaling: Literal["yes", "no"]
    notch_signaling: Literal["yes", "no"]
    dna_damage_response: Literal["yes", "no"]
    other_growth_proliferation_signaling: Literal["yes", "no"]
    survival_cell_death_regulation_signaling: Literal["yes", "no"]
    telomere_maintenance: Literal["yes", "no"]
    rtk_signaling_family: Literal["yes", "no"]
    pi3k_akt_mtor_signaling: Literal["yes", "no"]
    ras_raf_mek_erk_jnk_signaling: Literal["yes", "no"]
    angiogenesis: Literal["yes", "no"]
    folate_transport: Literal["yes", "no"]
    invasion_and_metastasis: Literal["yes", "no"]
    tgf_β_pathway: Literal["yes", "no"]
    oncogenes_associated_with_epithelial_ovarian_cancer: Literal["yes", "no"]
    regulation_of_ribosomal_protein_synthesis_and_cell_growth: Literal["yes", "no"]


class GeneInfo(BaseModel):
    association_strength: Literal[
        "very strong", "strong", "moderate", "weak", "very weak"
    ]
    reference: str
    mutations: List[str]
    mutation_origin: Literal["germline/somatic", "somatic"]
    diagnostic_implication: str
    therapeutic_relevance: str


class AssociatedGene(BaseModel):
    gene_symbol: str
    gene_info: GeneInfo


class GenerateLists(BaseModel):
    cancer_name: str
    other_codes_used_for_data_gathering: CodeReferences
    associated_genes: List[AssociatedGene] = Field(
        ..., description="List of gene symbols and their associated data"
    )
    molecular_subtypes: List[str]
    associated_pathways: AssociatedPathways

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
schema_json = generate_json_schema(GenerateLists)

print("Generated Schema:\n", json.dumps(schema_json, indent=2))

PROMPT_TEMPLATE = """You are an expert in clinical cancer genetics, specifically in gene-disease and pathway-disease curations (for hereditary and sporadic cancers). Based on scientific literature in PubMed, current genetic testing practices in oncology clinics, gene-disease association curations in ClinGen, OMIM, GeneReviews, and similar expert or peer reviewed resoursces, and public tumor sequencing databases such as cBioPortal, and COSMIC, list the genes and pathways, mutations in which are associated with {cancer_name} ({oncotree_code}). Different ontologies have different terms/codes to depict the same cancer sub-type. {oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS). Use these codes to gather as much literature/data as possible to provide a comprehensive list of genes and pathways in JSON structured format. The associated gene list should be ranked by strength and likelihood of association such that the first gene in the list has the strongest association with the cancer type and the last gene in the list has the weakest association with the cancer type. The gene list should be of high quality, accurate, and should not exceed 50 in count. The JSON should have top-level keys: 
"oncotree_code", 
"cancer_name" (full name of the code), 
"other_codes_used_for_data_gathering" (dictionary with keys NCIt and UMLS), 
"associated_genes" (a list of dictionaries - one dictionary for every associated gene, having top level keys of 'gene_symbol' and 'gene_info'. 'gene_symbol' should be only 1 gene per key. 'gene_info' is a dictionary with keys and values formatted as follows: 1. 'association_strength', value: classified as 'very strong', 'strong', 'moderate', 'weak', or 'very weak' association of this particular gene and cancer type depending on the quality and quantity of resources used to associate the gene and cancer type, 2. 'reference', value: resource(s) used to infer the gene-cancer type association (if multiple, then separate by '|'), 3. 'mutations', value: list of types of mutations in the gene that is associated with the given cancer type (such as truncating, splice, missense gain of function, missense-loss of function, missense-neomorphic, missense-hypo-/hyper-morphic, deletion, duplication, fusion, copy number variant, structural variant, complex rearrangements, methylation, and so on relevant to the gene-cancer type association), 4. 'mutation_origin', value: MUST be either "germline/somatic" OR "somatic" where 'germline/somatic' indicates that the cancer mutation in this gene can be present in the germline as cancer predisposing or arise somatically over time (so includes both 'germline' and 'somatic' options in 1 category only), 'somatic' indicates that the cancer mutation in this gene is only of somatic origin and not seen in the germline, 5. 'diagnostic_implication', value: clinical implication of the gene as to whether it is used to diagnose the cancer type, for example, the gene KRAS is associated with PAAD: 'diagnostic: missense mutations in KRAS are associated with PAAD and used for diagnosis.' Limit to 1 sentence, 6. 'therapeutic_relevance', value: if gene mutation informs decision making for therapeutic strategy, for example, for the association of KRAS and PAAD, 'clinical trials such as NCT07020221 are actively testing inhibitors of the actionable missense mutation KRAS G12D which is frequent in PAAD. Effect on immunotherapy is ....'),
next top-level key: "molecular_subtypes", values: This should be a list of expression-based, genomic, or histological molecular subtypes known to occur in {cancer_name}. These subtypes should be informative for clinical decision-making, such as guiding treatment selection or predicting prognosis. Please use descriptive names or standard nomenclature for the subtypes, and prioritize those with known clinical implications. The output must always include "molecular_subtypes". If no subtypes exist, return an empty list []. Never omit this field.
Last top-level key: "associated_pathways" (a dictionary with keys being each pathway name in the list: ['ar_signaling', 'ar_and_steroid_synthesis_enzymes', 'steroid_inactivating_genes', 'down_regulated_by_androgen', 'rtk_ras_pi3k_akt_signaling', 'rb_pathway', 'cell_cycle_pathway', 'hippo_pathway', 'myc_pathway', 'notch_pathway', 'nrf2_pathway', 'pi3k_pathway', 'rtk_ras_pathway', 'tp53_pathway', 'wnt_pathway', 'cell_cycle_control', 'p53_signaling', 'notch_signaling', 'dna_damage_response', 'other_growth_proliferation_signaling', 'survival_cell_death_regulation_signaling', 'telomere_maintenance', 'rtk_signaling_family', 'pi3k_akt_mtor_signaling', 'ras_raf_mek_erk_jnk_signaling', 'angiogenesis', 'folate_transport', 'invasion_and_metastasis', 'tgf_β_pathway', 'oncogenes_associated_with_epithelial_ovarian_cancer', 'regulation_of_ribosomal_protein_synthesis_and_cell_growth'] and the value being 'yes' if associated with cancer sub-type or 'no' if pathway not associated with cancer sub-type). Return **strict JSON** without trailing commas, unescaped quotes, or comments. Ensure it parses with `json.loads()`."""


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
        "-model",
        help="LLM model name supported by LiteLLM",
    ),
    temperature: float = typer.Option(
        0.25,
        "--input_LLM_temperature",
        "-temp",
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
        help="If set, process ALL OncoTree codes in the input file (overrides --codes).",
    ),
):
    typer.echo(f"Input file path: {input_oncotree}")

    if not input_oncotree.exists():
        typer.echo(f"File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    # Determine which codes to process
    if all_codes:
        target_codes = {item["code"] for item in oncotree}
        typer.echo(f"Running for ALL {len(target_codes)} codes in the input file.")
    elif codes:
        target_codes = set(codes)
        typer.echo(f"Running for user-specified codes: {', '.join(target_codes)}")
    else:
        target_codes = {"COAD", "NSCLC", "PAAD", "DSRCT", "BRCA", "MNM"}
        typer.echo(
            f"Running for default set of (COAD, NSCLC, PAAD, DSRCT, BRCA, MNM): {', '.join(target_codes)}"
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
        typer.echo("No matching OncoTree codes found in input file.")
        raise typer.Exit(code=1)

    all_results = {}  # A dictionary to store all the AI's answers

    total = len(oncotree_codes_info)
    success_count = 0
    fail_count = 0

    for idx, (oncotree_code, details) in enumerate(
        oncotree_codes_info.items(), start=1
    ):
        percent = (idx / total) * 100
        print(f"[{idx}/{total}] ({percent:.1f}%) Processing {oncotree_code}...")

        current_prompt = PROMPT_TEMPLATE.format(
            cancer_name=details["name"],
            oncotree_code=oncotree_code,
            ncit_code=details["NCIt"],
            umls_code=details["UMLS"],
        )

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
                print(f"{oncotree_code}: Missing molecular_subtypes, retrying...")
                retry_prompt = (
                    current_prompt
                    + "\n\nReminder: You must include a 'molecular_subtypes' field, even if it is an empty list."
                )
                response = call_llm_with_retries(llm_model, retry_prompt, temperature)
                parsed_json_data_dict = json.loads(response)

            parsed_model = GenerateLists(**parsed_json_data_dict)
            all_results[oncotree_code] = parsed_model.model_dump()
            success_count += 1

        except Exception as e:
            print(f"  Error processing {oncotree_code}: {e}")
            all_results[oncotree_code] = {"error": str(e), "details_provided": details}
            fail_count += 1

    print(f"\nFinished: {success_count} succeeded, {fail_count} failed, total {total}.")

    with open(output_lists, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    app()
