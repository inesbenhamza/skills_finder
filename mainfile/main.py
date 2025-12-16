import argparse
import sys
import os
from pathlib import Path
import numpy as np
import yaml

# Add parent directory to path so we can import src and helpers
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.schedule_filtering import calculate_availability_penalty
from helpers.tools_extraction import generate_yaml, get_model_response
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict
import helpers.SkillsEmbeddings as se
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from src.language_filtering import compute_language_score, apply_filters

MATCHING_CONFIGURATION_FILE = "config/search_config.yaml"

def setup_arguments():
    parser = argparse.ArgumentParser("main")
    parser.add_argument("filename", help="The name of the pdf containing the task description. Make sure that it does not contain the extension", type=str)
    args = parser.parse_args()
    return args

def linear_combination(average_skills_ratings, skills_weight, average_mission_ratings, mission_weight):
    filler_mission_value = np.mean(list(average_mission_ratings.values()))
    combinations = {}
    for consultant_id in average_skills_ratings:
        combinations[consultant_id] = average_skills_ratings[consultant_id] * skills_weight + average_mission_ratings.get(consultant_id, filler_mission_value) * mission_weight
    return combinations

if __name__ == "__main__":
    args = setup_arguments()

    with open(MATCHING_CONFIGURATION_FILE, 'r') as f:
        matching_configuration = yaml.safe_load(f)

    # print("Generating task requirements...")
    # model_response = get_model_response(args.filename+".pdf", matching_configuration["model"]["name"])
    # task_description = generate_yaml(model_response, think=matching_configuration["model"]["think"])

    with open(f"output/filtered_skills/{args.filename}.yaml", 'r') as f:
        task_requirements = yaml.safe_load(f)
    
    model = SentenceTransformer(matching_configuration["embedding_model"])
    
    skill_model = se.SkillEmbeddingModel(model)
    skills_df = pd.read_csv(matching_configuration["skills_csv"])
    # Get use_string_matching from config (defaults to False if not specified)
    use_string_matching = matching_configuration.get("skills", {}).get("use_string_matching", False)
    
    consultants_scores = se.get_consultants_skills_score(
        task_requirements, 
        skill_model, 
        matching_configuration["skills_embedding_file"], 
        matching_configuration,
        use_string_matching=use_string_matching
    )

    average_skills_rating = se.average_skills(consultants_scores)

    average_mission_rating = df_to_dict(get_consultants_mission_scores(
        task_requirements,
        percentile=matching_configuration["mission_context"]["percentile"],
        use_domain_keyword_filtering=matching_configuration.get("domain_filtering", {}).get("use_keyword_filtering", False),
        finetune_model=matching_configuration.get("mission_embedding_model", "paraphrase-multilingual-mpnet-base-v2")
    ))


    consultants_average_rating = linear_combination(average_skills_rating,
                                                    matching_configuration["skills"]["skills_weight"],
                                                    average_mission_rating, matching_configuration["mission_context"]["mission_weight"])

    language_filter = compute_language_score("data/raw/langcleaned.csv",
                                             task_requirements,
                                             acceptable_difference=matching_configuration["languages"]["acceptable_difference"],
                                             penality=matching_configuration["languages"]["penality"])

    schedule_penalties = calculate_availability_penalty('data/HCK_HEC_STAFFING.csv',
                                                        task_requirements, 
                                                        matching_configuration["disponibility"]["maximum_penalty"])
    
    filtered_score = apply_filters(consultants_average_rating, language_filter)

    filtered_score = apply_filters(filtered_score, schedule_penalties)

    sorted_consultants = se.sort_skills(filtered_score)

    print("Top 5: ")
    print(sorted_consultants[0:5])
