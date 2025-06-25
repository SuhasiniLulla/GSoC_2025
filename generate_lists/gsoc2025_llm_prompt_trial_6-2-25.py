from google import genai
from dotenv import load_dotenv
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import List, Dict
import typer
from pathlib import Path

load_dotenv()
YOUR_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=YOUR_API_KEY)

PROMPT_TEMPLATE = """Based on scientific literature in PubMed, current genetic testing practices in oncology clinics,
and public tumor sequencing databases such as cBioPortal, and COSMIC, mutations in which genes
and pathways are associated with {cancer_name} ({oncotree_code}).
Different ontologies have different terms/codes to depict the same cancer sub-type.
{oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) and {umls_code} (UMLS).
Use these codes to gather as much literature/data as possible to provide a comprehensive list
of genes and pathways in JSON structured format. The JSON should have top-level keys:
"oncotree_code", "cancer_name" (full name of the code), "other_codes_used_for_data_gathering" (dictionary with keys NCIt and UMLS), "associated_genes" (a list of gene symbols), and "associated_pathways" (a list of pathway names)."""

response_schema_defined = {
  "type": "object",
  "properties": {
    "cancer_name": {
      "type": "string"
    },
    "other_codes_used_for_data_gathering": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "associated_genes": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "associated_pathways": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": [
    "cancer_name",
    "other_codes_used_for_data_gathering",
    "associated_genes",
    "associated_pathways"
  ]
}

app = typer.Typer()

@app.command()
def generate_lists(input_oncotree: Path):
    if not input_oncotree.exists():
        typer.echo(f"File not found: {input_oncotree}")
        raise typer.Exit(code=1)

    with input_oncotree.open("r") as f:
        oncotree = json.load(f)

    # Initialize the output dictionary
    oncotree_codes_info = {}

    #since Gemini free tier has limits: RPM=15, RPD=1,500, trying only 3 at a time while optimizing code:
    for i, item in enumerate(oncotree):
        if i >= 3:
            break
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
        response_schema= response_schema_defined
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
            json_output_str = response.text
            
            # Convert the JSON string into a Python dictionary
            parsed_json_data = json.loads(json_output_str)

            # Store the structured data
            all_results[oncotree_code] = parsed_json_data
      
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

    with open("export_lists.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    app()