import os
import ollama
from pypdf import PdfReader
import yaml


def read_pdf(pdf_filename: str):
    reader = PdfReader(pdf_filename)
    complete_text = ""
    for page in reader.pages:
        complete_text += f"{page.extract_text()}\n"
    return complete_text

def generate_yaml(model_response: str, think):
    if think:
        return yaml.safe_load(model_response.split('</think>')[1].split('```')[1].removeprefix('yaml'))
    return yaml.safe_load(model_response.split('```')[1].removeprefix('yaml'))

def get_model_response(file: str, model='gemma3:12b'):
    prompt = lambda file: """I want you to read the following terms of reference and fill in the YAML that I provide with the information in the document.
It's quite possible that some information is missing. In this case, I want you to leave the field empty
Here's the YAML in question, this is an exemple.
Based your answer on the task description as much as possible.
For each technologies and languages estimate the experiment required between 0 and 100

# JOB ROLE: Identify the primary job role from the posting. Must be EXACTLY one of: "Data Engineer", "Data Analyst", "Data Scientist", "Scrum Master", "Finance", "Healthcare", "Marketing", or "Other" if none match. This field is CRITICAL for matching.
job_role: ""

# Years of experience required
required_experience: 0
# Languages to speak, give score between 0 and 100
languages: 
    - language_code: FR
      required: true
      level: 75
    - language_code: EN
      required: true`
      level: 75
# Name of the activity sector
activity_sector: ""
# Number of workday per week, full time would be 5 days. Put only the number here
number_of_workday_perweek: 
# How many months it lasts. Put only a number here
months: 0
# Might be unknown, in this case leave empty. Format is YYYY-MM-DD
start_date: ""
# Technological stack. Use only the names of the technology and their estimated score.Also indicate if technologies is required
technologies:
    - name: Power BI
      required: TRUE
      level: 60
# List the responsabilities given in the task description. This can NEVER be empty
responsabilites: 
  - "Program computer"
  - "Write tests"

""" + f"This is the task description : \n {read_pdf(file)}\n Remember to not add or modify any fields and to output only a yaml file"

    ollama_response = ollama.generate(model=model, prompt=prompt(file), options = {
    'temperature': 1
    })
    return ollama_response['response']
