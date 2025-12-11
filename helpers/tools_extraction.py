import json
import os
import ollama
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from .tree import TreeNode
from nltk.tokenize import RegexpTokenizer
import yaml


def extract_skills(skills_file: str):
    skills = pd.read_csv(skills_file)
    unique_skills = set()
    for _, r in skills.iterrows():
        skill_desc = r["SKILLS_DSC"]
        domain_desc = r["DOMAIN_DSC"]
        unique_skills.add((domain_desc, skill_desc))

    return list(unique_skills)

def populate_tree(skills: list[tuple[str, str]], root: TreeNode):
    # Add technologies to the tree
    for _, technology in skills:
        words = technology.split()
        words.append("<EOS/>")
        root.add_child(words, technology)

def read_pdf(pdf_filename: str):
    reader = PdfReader(pdf_filename)
    complete_text = ""
    for page in reader.pages:
        complete_text += f"{page.extract_text()}\n"
    return complete_text

def filter_pdf(mission_description: str):
    mission_description = mission_description.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(mission_description)
    return tokens

def match_skills(tree: TreeNode, tokens: list[str]):
    all_skills = set()

    PAD = "<PAD>"
    tokens = [PAD, PAD, PAD, PAD] + tokens + [PAD, PAD, PAD, PAD]

    for i in range(1, len(tokens)):
        join_with_space = lambda words: " ".join(words)

        unigram = join_with_space(tokens[i:i+1])
        bigram = join_with_space(tokens[i:i+2])
        trigram = join_with_space(tokens[i:i+3])
        fourgram = join_with_space(tokens[i:i+4])

        try:
            tree.match_term(fourgram)
        except Exception as e:
            all_skills.add(str(e))

        try:
            tree.match_term(trigram)
        except Exception as e:
            all_skills.add(str(e))
        
        try:
            tree.match_term(bigram)
        except Exception as e:
            all_skills.add(str(e))

        try:
            tree.match_term(unigram)
        except Exception as e:
            all_skills.add(str(e))

    return list(set(all_skills))

def generate_yaml(model_response: str, think):
    if think:
        return yaml.safe_load(model_response.split('</think>')[1].split('```')[1].removeprefix('yaml'))
    return yaml.safe_load(model_response.split('```')[1].removeprefix('yaml'))
    

def parse_answer(answer):
    return json.loads(answer.split('</think>')[1].replace('\n', '').replace('json', ''))

def get_model_response(file: str, model='deepseek-r1:7b'):
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

def extract_key_skills(pdf_file: str, display_skill_tree = False):
    text = read_pdf(pdf_file)
    tokenized_text = filter_pdf(text)
    skills = extract_skills(r"skills-finder-hackathon-hec-x-sbi\skillscleaned.csv")
    root = TreeNode('', '')
    populate_tree(skills, root)
    root.prune()
    if display_skill_tree:
        root.display()
    matched_skills = match_skills(root, tokenized_text) 
    return matched_skills, root

if __name__ == "__main__":
    yaml_prefix = "filtered_skills"
    os.makedirs(yaml_prefix, exist_ok=True)
    for filename in tqdm(["Data Engineer", "Data Analyst", "Scrum"]):
        model_response = get_model_response(filename+".pdf", "gemma3:12b")
        task_description = generate_yaml(model_response, think=False)
        yaml_filename=  os.path.join(yaml_prefix, filename+".yaml")
        with open(yaml_filename, 'w') as f:
            yaml.safe_dump(task_description, f)
