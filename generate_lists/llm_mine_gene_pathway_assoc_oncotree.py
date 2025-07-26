import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, get_args

import google.generativeai as genai
import typer
from dotenv import load_dotenv
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel, ConfigDict, Field

TEMPERATURE = 0.25

load_dotenv()
YOUR_API_KEY = os.getenv("LLM_API_KEY")
genai.configure(api_key=YOUR_API_KEY)


class CodeReferences(BaseModel):
    NCIt: str
    UMLS: str


class AssociatedPathways(BaseModel):
    prostate_cancer_ar_signaling: Literal["yes", "no"]
    prostate_cancer_ar_and_steroid_synthesis_enzymes: Literal["yes", "no"]
    prostate_cancer_steroid_inactivating_genes: Literal["yes", "no"]
    prostate_cancer_down_regulated_by_androgen: Literal["yes", "no"]
    glioblastoma_tp53_pathway: Literal["yes", "no"]
    glioblastoma_rtk_ras_pi3k_akt_signaling: Literal["yes", "no"]
    glioblastoma_rb_pathway: Literal["yes", "no"]
    general_cell_cycle_tcga_pancan_pathways: Literal["yes", "no"]
    general_hippo_tcga_pancan_pathways: Literal["yes", "no"]
    general_myc_tcga_pancan_pathways: Literal["yes", "no"]
    general_notch_tcga_pancan_pathways: Literal["yes", "no"]
    general_nrf2_tcga_pancan_pathways: Literal["yes", "no"]
    general_pi3k_tcga_pancan_pathways: Literal["yes", "no"]
    general_tgf_beta_tcga_pancan_pathways: Literal["yes", "no"]
    general_rtk_ras_tcga_pancan_pathways: Literal["yes", "no"]
    general_tp53_tcga_pancan_pathways: Literal["yes", "no"]
    general_wnt_tcga_pancan_pathways: Literal["yes", "no"]
    general_cell_cycle_control: Literal["yes", "no"]
    general_p53_signaling: Literal["yes", "no"]
    general_notch_signaling: Literal["yes", "no"]
    general_dna_damage_response: Literal["yes", "no"]
    general_other_growth_proliferation_signaling: Literal["yes", "no"]
    general_survival_cell_death_regulation_signaling: Literal["yes", "no"]
    general_telomere_maintenance: Literal["yes", "no"]
    general_rtk_signaling_family: Literal["yes", "no"]
    general_pi3k_akt_mtor_signaling: Literal["yes", "no"]
    general_ras_raf_mek_erk_jnk_signaling: Literal["yes", "no"]
    general_angiogenesis: Literal["yes", "no"]
    general_folate_transport: Literal["yes", "no"]
    general_invasion_and_metastasis: Literal["yes", "no"]
    general_tgf_β_pathway: Literal["yes", "no"]
    ovarian_cancer_oncogenes_associated_with_epithelial_ovarian_cancer: Literal[
        "yes", "no"
    ]
    ovarian_cancer_putative_tumor_suppressor_genes_in_epithelial_ovarian_cancer: (
        Literal["yes", "no"]
    )
    general_regulation_of_ribosomal_protein_synthesis_and_cell_growth: Literal[
        "yes", "no"
    ]


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
SCHEMA_JSON = generate_gemini_compatible_schema(GenerateLists)

print("Generated Schema:\n", json.dumps(SCHEMA_JSON, indent=2))


PROMPT_TEMPLATE = """You are an expert in clinical cancer genetics, specifically in gene-disease and pathway-disease curations (for hereditary and sporadic cancers). Based on scientific literature in PubMed, current genetic testing practices in oncology clinics, gene-disease association curations in ClinGen, OMIM, GeneReviews, and similar expert or peer reviewed resoursces,
and public tumor sequencing databases such as cBioPortal, and COSMIC, list the genes
and pathways, mutations in which are associated with {cancer_name} ({oncotree_code}).
Different ontologies have different terms/codes to depict the same cancer sub-type.
{oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS).
Use these codes to gather as much literature/data as possible to provide a comprehensive list
of genes and pathways in JSON structured format. The associated gene list should be ranked by strength and likelihood of association such that the first gene in the list has the strongest association with the cancer type and the last gene in the list has the weakest association with the cancer type. The gene list should be of high quality, accurate, and should not exceed 50 in count. The JSON should have top-level keys:
"oncotree_code", "cancer_name" (full name of the code), "other_codes_used_for_data_gathering" (dictionary with keys NCIt and UMLS), "associated_genes" (a list of dictionaries - one dictionary for every associated gene, having top level keys of 'gene_symbol' and 'gene_info'. 'gene_info' is a dictionary with keys and values formatted as follows: 1. 'association_strength', value: classified as 'very strong', 'strong', 'moderate', 'weak', or 'very weak' association of this particular gene and cancer type depending on the quality and quantity of resources used to associate the gene and cancer type, 2. 'reference', value: resource(s) used to infer the gene-cancer type association (if multiple citations, then separate instances by '|'), 3. 'mutations', value: list of types of mutations in the gene that is associated with the given cancer type (such as truncating, splice, missense gain of function, missense-loss of function, missense-neomorphic, missense-hypo-/hyper-morphic, deletion, duplication, fusion, copy number variant, structural variant, complex rearrangements, methylation, and so on relevant to the gene-cancer type association), 4. 'mutation_origin', value: MUST be either "germline/somatic" OR "somatic" where 'germline/somatic' indicates that the cancer mutation in this gene can be present in the germline as cancer predisposing or arise somatically over time, 'somatic' indicates that the cancer mutation in this gene is only of somatic origin and not seen in the germline, 5. 'diagnostic_implication', value: clinical implication of the gene as to whether it is used to diagnose the cancer type, for example, the gene KRAS is associated with PAAD: 'diagnostic: missense mutations in KRAS are associated with PAAD and used for diagnosis.' Limit to 1 sentence, 6. 'therapeutic_relevance', value: if gene mutation informs decision making for therapeutic strategy, for example, for the association of KRAS and PAAD, 'clinical trials such as NCT07020221 are actively testing inhibitors of the actionable missense mutation KRAS G12D which is frequent in PAAD. Effect on immunotherapy is ....'), "molecular_subtypes", values: This should be a list of expression-based, genomic, or histological molecular subtypes known to occur in {cancer_name}. These subtypes should be informative for clinical decision-making, such as guiding treatment selection or predicting prognosis. Please use descriptive names or standard nomenclature for the subtypes, and prioritize those with known clinical implications, and "associated_pathways" (a dictionary with keys being each pathway name in the list: ['prostate_cancer_ar_signaling',
 'prostate_cancer_ar_and_steroid_synthesis_enzymes',
 'prostate_cancer_steroid_inactivating_genes',
 'prostate_cancer_down_regulated_by_androgen',
 'glioblastoma_tp53_pathway',
 'glioblastoma_rtk_ras_pi3k_akt_signaling',
 'glioblastoma_rb_pathway',
 'general_cell_cycle_tcga_pancan_pathways',
 'general_hippo_tcga_pancan_pathways',
 'general_myc_tcga_pancan_pathways',
 'general_notch_tcga_pancan_pathways',
 'general_nrf2_tcga_pancan_pathways',
 'general_pi3k_tcga_pancan_pathways',
 'general_tgf_beta_tcga_pancan_pathways',
 'general_rtk_ras_tcga_pancan_pathways',
 'general_tp53_tcga_pancan_pathways',
 'general_wnt_tcga_pancan_pathways',
 'general_cell_cycle_control',
 'general_p53_signaling',
 'general_notch_signaling',
 'general_dna_damage_response',
 'general_other_growth_proliferation_signaling',
 'general_survival_cell_death_regulation_signaling',
 'general_telomere_maintenance',
 'general_rtk_signaling_family',
 'general_pi3k_akt_mtor_signaling',
 'general_ras_raf_mek_erk_jnk_signaling',
 'general_angiogenesis',
 'general_folate_transport',
 'general_invasion_and_metastasis',
 'general_tgf_β_pathway',
 'ovarian_cancer_oncogenes_associated_with_epithelial_ovarian_cancer',
 'ovarian_cancer_putative_tumor_suppressor_genes_in_epithelial_ovarian_cancer',
 'general_regulation_of_ribosomal_protein_synthesis_and_cell_growth'] and the value being 'yes' if associated with cancer sub-type or 'no' if pathway not associated with cancer sub-type)."""


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
        "--model_name",
        "-model",
        help="enter the string name of the LLM model to be used",
    ),
):
    typer.echo(f"Input file path: {input_oncotree}")

    if not input_oncotree.exists():
        typer.echo(f"File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    # Initialize the output dictionary
    oncotree_codes_info = {}

    for item in oncotree:
        if item["code"] not in {"COAD", "NSCLC", "PAAD", "DSRCT"}:
            continue
        code = item["code"]
        name = item["name"]
        umls = (
            item["externalReferences"]["UMLS"][0]
            if "UMLS" in item["externalReferences"]
            else None
        )
        ncit = (
            item["externalReferences"]["NCI"][0]
            if "NCI" in item["externalReferences"]
            else None
        )

        # Add the extracted information to the output dictionary
        oncotree_codes_info[code] = {"name": name, "NCIt": ncit, "UMLS": umls}

    generation_config = GenerationConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",  # Ask Gemini to output JSON directly
        response_schema=SCHEMA_JSON,
    )

    model = genai.GenerativeModel(
        model_name=llm_model,
        generation_config=generation_config,
    )

    all_results = {}  # A dictionary to store all the AI's answers

    for oncotree_code, details in oncotree_codes_info.items():
        # Fill in the placeholders in the prompt template
        current_prompt = PROMPT_TEMPLATE.format(
            cancer_name=details["name"],
            oncotree_code=oncotree_code,
            ncit_code=details["NCIt"],
            umls_code=details["UMLS"],
        )

        # Send the question to the AI
        try:
            response = model.generate_content(current_prompt)
            # json_output_str = response.text

            # Convert the JSON string into a Python dictionary
            parsed_json_data_dict = json.loads(response.text)
            parsed_model = GenerateLists(**parsed_json_data_dict)

            # Store the structured data
            all_results[oncotree_code] = parsed_model.model_dump()

        except Exception as e:
            print(f"  Error processing {oncotree_code}: {e}")
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
                    print(
                        f"    Safety Ratings: {response.candidates[0].safety_ratings}"
                    )
            all_results[oncotree_code] = {"error": str(e), "details_provided": details}

    print(all_results)

    with open(output_lists, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    app()
