from google import genai
from dotenv import load_dotenv
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import List, Dict, Literal
import typer
from pathlib import Path
from pydantic import BaseModel, Field

load_dotenv()
YOUR_API_KEY = os.getenv("GENAI_API_KEY")
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
    ovarian_cancer_oncogenes_associated_with_epithelial_ovarian_cancer: Literal["yes", "no"]
    ovarian_cancer_putative_tumor_suppressor_genes_in_epithelial_ovarian_cancer: Literal["yes", "no"]
    general_regulation_of_ribosomal_protein_synthesis_and_cell_growth: Literal["yes", "no"]

class GenerateLists(BaseModel):
    cancer_name: str
    other_codes_used_for_data_gathering: CodeReferences
    associated_genes: List[str]
    associated_pathways: AssociatedPathways

    class Config:
        validate_by_name = True

def generate_gemini_compatible_schema(model: BaseModel) -> Dict[str, any]:
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    model_schema = model.model_json_schema()

    for field_name, field_info in model_schema.get("properties", {}).items():
        field_type = field_info.get("type")
        if field_type == "array":
            schema["properties"][field_name] = {
                "type": "array",
                "items": {"type": field_info["items"]["type"]}
            }
        elif field_type == "object":
            nested_props = field_info.get("properties", {})
            nested_required = field_info.get("required", [])
            schema["properties"][field_name] = {
                "type": "object",
                "properties": {
                    k: {"type": v["type"]} for k, v in nested_props.items()
                },
                "required": nested_required
            }
        else:
            schema["properties"][field_name] = {"type": field_type}

    schema["required"] = model_schema.get("required", [])
    return schema

# Auto-generate JSON schema from the Pydantic model
SCHEMA_JSON = generate_gemini_compatible_schema(GenerateLists)


PROMPT_TEMPLATE = """Based on scientific literature in PubMed, current genetic testing practices in oncology clinics,
and public tumor sequencing databases such as cBioPortal, and COSMIC, list the genes
and pathways, mutations in which are associated with {cancer_name} ({oncotree_code}).
Different ontologies have different terms/codes to depict the same cancer sub-type.
{oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS).
Use these codes to gather as much literature/data as possible to provide a comprehensive list
of genes and pathways in JSON structured format. The associated gene list should be in order of strength and likelihood of association. Gene list should be high quality and accurate and should not exceed 50 in count. The JSON should have top-level keys:
"oncotree_code", "cancer_name" (full name of the code), "other_codes_used_for_data_gathering" (dictionary with keys NCIt and UMLS), "associated_genes" (a list of gene symbols), and "associated_pathways" (a dictionary with keys being each pathway name in the list: ['prostate_cancer_ar_signaling',
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
def generate_lists(input_oncotree: Path = typer.Option(..., "--input_oncotree_filepath", "-i", help="Path to the OncoTree JSON file")):
    typer.echo(f"Input file path: {input_oncotree}")
    if not input_oncotree.exists():
        typer.echo(f"File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    # Initialize the output dictionary
    oncotree_codes_info = {}

    for item in oncotree:
        if item["code"]=="COAD" or item["code"]=="NSCLC" or item["code"]=="PAAD":
            code = item["code"]
            name = item["name"]
            umls = item["externalReferences"]["UMLS"][0] if "UMLS" in item["externalReferences"] else None
            ncit = item["externalReferences"]["NCI"][0] if "NCI" in item["externalReferences"] else None
        
            # Add the extracted information to the output dictionary
            oncotree_codes_info[code] = {
                "name": name,
                "NCIt": ncit,
                "UMLS": umls
            }

    TEMPERATURE = 0.25

    generation_config = GenerationConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",  # Ask Gemini to output JSON directly
        response_schema= SCHEMA_JSON
    )

    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash', #Knowledge cutoff date is June 2024
        generation_config=generation_config
    )

    all_results = {} # A dictionary to store all the AI's answers

    for oncotree_code, details in oncotree_codes_info.items():
        # Fill in the placeholders in the prompt template
        current_prompt = PROMPT_TEMPLATE.format(
            cancer_name=details['name'],
            oncotree_code=oncotree_code,
            ncit_code=details['NCIt'],
            umls_code=details['UMLS'],
        )
    
        # Send the question to the AI
        try:
            response = model.generate_content(current_prompt)
            #json_output_str = response.text
            
            # Convert the JSON string into a Python dictionary
            parsed_json_data_dict = json.loads(response.text)
            parsed_model = GenerateLists(**parsed_json_data_dict)

            # Store the structured data
            all_results[oncotree_code] = parsed_model.model_dump()
      
        except Exception as e:
            print(f"  Error processing {oncotree_code}: {e}")
            # Log errors, and check response for more details if available
            if 'response' in locals() and hasattr(response, 'prompt_feedback'):
                print(f"    Prompt Feedback: {response.prompt_feedback}")
            if 'response' in locals() and hasattr(response, 'candidates') and response.candidates:
                print(f"    Candidate Finish Reason: {response.candidates[0].finish_reason}")
                if response.candidates[0].finish_reason.name == 'SAFETY':
                    print(f"    Safety Ratings: {response.candidates[0].safety_ratings}")
            all_results[oncotree_code] = {"error": str(e), "details_provided": details}

    print(all_results)

    with open("export_lists_6-30-25.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    app()