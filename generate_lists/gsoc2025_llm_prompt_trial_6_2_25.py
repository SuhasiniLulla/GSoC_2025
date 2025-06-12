# -*- coding: utf-8 -*-
"""GSoC2025_LLM_prompt_trial_6-2-25.ipynb"""


#pip install -q -U google-genai

from google import genai
#from google.colab import userdata
#YOUR_API_KEY=userdata.get('GOOGLE_API_KEY')
from dotenv import load_dotenv
import os

load_dotenv()
YOUR_API_KEY = os.getenv("GENAI_API_KEY")

client = genai.Client(api_key=YOUR_API_KEY)


# Define the OncoTree codes trial for 2
oncotree_codes_info = {
    "COAD": {
        "name": "Colon Adenocarcinoma",
        "NCIt": "C4349",
        "UMLS": "C0338106",
        "ICD0_topography": "C18.9",
        "ICD0_morphology": "8140/3",
        "HemeOnc": "585"
    },
    "PPB": {
        "name": "Pleuropulmonary Blastoma",
        "NCIt": "C5669",
        "UMLS": "C1266144",
        "ICD0_topography": "C34.9",
        "ICD0_morphology": "8973/3",
    }

}
print(f"Research Assignments Loaded: {len(oncotree_codes_info)} cancer types to process.")

"""## 6-9-25: see documentaion for parameters that can be set during prompt engineering:https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values

Not only Temperature, but also:
1. Max output tokens (here no need since no need to limit the response)
2. Temperature (set to 0.25 to allow for some creativity but still be as reproducible and deterministic as possible)
3. Top-P (sounds sort of similar to temp: 'Specify a lower value for less random responses and a higher value for more random responses.')
4. Seed (preview feature: this is what we want, repeat the same response for repeated prompts. default setting is a random value, after getting prompt to work, see if can set this to a defined value here)
"""

import google.generativeai as genai
genai.configure(api_key=YOUR_API_KEY)
print("Gemini API Configured.")

import os
import json

from google.generativeai.types import GenerationConfig
#cancer_name=oncotree_code=ncit_code=umls_code=icdo_topo_code=icdo_morph_code=hemeonc_code="NaN"
PROMPT_TEMPLATE = """Based on scientific literature in PubMed, current genetic testing practices in oncology clinics,
and public tumor sequencing databases such as cBioPortal, and COSMIC, mutations in which genes
and pathways are associated with {cancer_name} ({oncotree_code}).
Different ontologies have different terms/codes to depict the same cancer sub-type.
{oncotree_code} is the OncoTree code that is the same as {ncit_code} (NCIt) {umls_code} (UMLS)
{icdo_topo_code} (ICD0_topography) {icdo_morph_code} (ICD0_morphology) {hemeonc_code} (HemeOnc).
Use these codes to gather as much literature/data as possible to provide a comprehensive list
of genes and pathways in JSON structured format. The JSON should have top-level keys:
'cancer_details' (with 'name', 'oncotree_code', 'other_codes_used_for_data_gathering'),
'associated_genes' (a list of gene symbols), and 'associated_pathways' (a list of pathway names)."""

TEMPERATURE = 0.25

generation_config = GenerationConfig(
    temperature=TEMPERATURE,
    response_mime_type="application/json"  # Ask Gemini to output JSON directly
)


model = genai.GenerativeModel(
    model_name='gemini-2.0-flash', #Knowledge cutoff date is June 2024
    generation_config=generation_config
)

all_results = {} # A dictionary to store all the AI's answers

for oncotree_code, details in oncotree_codes_info.items():
    print(f"Researching: {details['name']} ({oncotree_code})...")

    # Fill in the placeholders in the prompt template
    current_prompt = PROMPT_TEMPLATE.format(
        cancer_name=details['name'],
        oncotree_code=oncotree_code,
        ncit_code=details['NCIt'],
        umls_code=details['UMLS'],
        icdo_topo_code=details['ICD0_topography'],
        icdo_morph_code=details['ICD0_morphology'],
        hemeonc_code=details.get('HemeOnc', 'N/A') # .get in case HemeOnc is missing
    )
    print(current_prompt)

    # Send the question to the AI
    try:
      response = model.generate_content(current_prompt)
      print(response)
      # (Continuing inside the loop)
      # The AI's response text should be the JSON string
      json_output_str = response.text

      # Convert the JSON string into a Python dictionary
      parsed_json_data = json.loads(json_output_str)

      # Store the structured data
      all_results[oncotree_code] = parsed_json_data
      print(f"  Successfully received and parsed data for {oncotree_code}.")

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

