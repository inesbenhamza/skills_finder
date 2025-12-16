import argparse
import sys
import os
from pathlib import Path
import yaml

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
from src.schedule_filtering import calculate_availability_penalty
from helpers.tools_extraction import generate_yaml, get_model_response
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict
from helpers import SkillsEmbeddings as se
from sentence_transformers import SentenceTransformer
import pandas as pd
from src.language_filtering import compute_language_score, apply_filters

MATCHING_CONFIGURATION_FILE = "config/search_config.yaml"


DEBUG = True

def dbg(*args):
    if DEBUG:
        print("\n[DEBUG]", *args)

def linear_combination(average_skills_ratings, skills_weight, average_mission_ratings, mission_weight):
    if len(average_mission_ratings) > 0:
        filler_mission_value = np.mean(list(average_mission_ratings.values())) * 0.1  
    else:
        filler_mission_value = 0.0  
    
    combinations = {}
    for cid in average_skills_ratings:

        skill_score = average_skills_ratings[cid]
        mission_score = average_mission_ratings.get(cid, filler_mission_value)

        combined = skill_score * skills_weight + mission_score * mission_weight
        combinations[cid] = combined

        dbg(f"CID {cid}: skill={skill_score:.4f}, mission={mission_score:.4f}, combined={combined:.4f}")

    return combinations

def setup_arguments():
    parser = argparse.ArgumentParser("main")
    parser.add_argument("filename", help="Mission PDF name (without extension)", type=str)
    return parser.parse_args()


if __name__ == "__main__":

    args = setup_arguments()

    with open(MATCHING_CONFIGURATION_FILE, 'r') as f:
        config = yaml.safe_load(f)

    yaml_path = f"output/filtered_skills/{args.filename}.yaml"
    
    if os.path.exists(yaml_path):
        dbg(f"Loading mission requirements from saved YAML: {yaml_path}")
        with open(yaml_path, "r") as f:
            task_requirements = yaml.safe_load(f)
    else:
        dbg(f"YAML file not found. Generating live from PDF: {args.filename}.pdf")
        pdf_path = f"documents/{args.filename}.pdf"
        if not os.path.exists(pdf_path):
            pdf_path = f"{args.filename}.pdf"  # Try root directory
        
        if os.path.exists(pdf_path):
            model_response = get_model_response(pdf_path, config["model"]["name"])
            task_requirements = generate_yaml(model_response, think=config["model"]["think"])
            
            os.makedirs("output/filtered_skills", exist_ok=True)
            with open(yaml_path, 'w') as f:
                yaml.safe_dump(task_requirements, f)
            dbg(f"Saved generated YAML to: {yaml_path}")
        else:
            raise FileNotFoundError(f"Neither YAML ({yaml_path}) nor PDF ({pdf_path}) found!")
    
    dbg("Loaded mission task:", task_requirements)

    if 'job_role' not in task_requirements or not task_requirements.get('job_role'):
        filename = args.filename
        dbg(f"job_role not in YAML, trying to infer from filename: {filename}")
        
        filename_lower = filename.lower()
        if 'finance' in filename_lower:
            task_requirements['job_role'] = 'Finance'
            dbg(f"Inferred job_role from filename: Finance")
        elif 'dataengineer' in filename_lower or 'data_engineer' in filename_lower:
            task_requirements['job_role'] = 'Data Engineer'
            dbg(f"Inferred job_role from filename: Data Engineer")
        elif 'dataanalyst' in filename_lower or 'data_analyst' in filename_lower:
            task_requirements['job_role'] = 'Data Analyst'
            dbg(f"Inferred job_role from filename: Data Analyst")
        elif 'datascientist' in filename_lower or 'data_scientist' in filename_lower:
            task_requirements['job_role'] = 'Data Scientist'
            dbg(f"Inferred job_role from filename: Data Scientist")
        elif 'scrum' in filename_lower or 'scrummaster' in filename_lower:
            task_requirements['job_role'] = 'Scrum Master'
            dbg(f"Inferred job_role from filename: Scrum Master")
        elif 'healthcare' in filename_lower or 'health' in filename_lower:
            task_requirements['job_role'] = 'Healthcare'
            dbg(f"Inferred job_role from filename: Healthcare")
        elif 'marketing' in filename_lower:
            task_requirements['job_role'] = 'Marketing'
            dbg(f"Inferred job_role from filename: Marketing")
        else:
            dbg("Could not infer from filename, inferring from mission content...")
            responsibilities = task_requirements.get('responsabilites', [])
            technologies = [t.get('name', '').lower() for t in task_requirements.get('technologies', [])]
            all_text = ' '.join(responsibilities).lower() + ' ' + ' '.join(technologies)
            
            if any(keyword in all_text for keyword in ['healthcare', 'health care', 'medical', 'patient', 'clinical', 'nursing', 'physician', 'diagnosis', 'treatment plan', 'medical record', 'health status', 'care team', 'allied health']):
                task_requirements['job_role'] = 'Healthcare'
                dbg(f"Inferred job_role: Healthcare")
            elif any(keyword in all_text for keyword in ['scrum', 'agile', 'sprint', 'ceremonies', 'product owner', 'backlog']):
                task_requirements['job_role'] = 'Scrum Master'
                dbg(f"Inferred job_role: Scrum Master")
            elif any(keyword in all_text for keyword in ['data engineer', 'pipeline', 'etl', 'databricks', 'data factory', 'spark', 'adf']):
                task_requirements['job_role'] = 'Data Engineer'
                dbg(f"Inferred job_role: Data Engineer")
            elif any(keyword in all_text for keyword in ['finance', 'financial', 'budget', 'forecast', 'planning', 'budgétaire', 'financières', 'financier', 'consolidation', 'epm', 'cpm', 'tagetik', 'hyperion']):
                task_requirements['job_role'] = 'Finance'
                dbg(f"Inferred job_role: Finance")
            elif any(keyword in all_text for keyword in ['data analyst', 'power bi', 'tableau', 'dashboard', 'reporting', 'kpi', 'analytics']):
                task_requirements['job_role'] = 'Data Analyst'
                dbg(f"Inferred job_role: Data Analyst")
            elif any(keyword in all_text for keyword in ['data scientist', 'machine learning', 'ml', 'model', 'prediction', 'ai']):
                task_requirements['job_role'] = 'Data Scientist'
                dbg(f"Inferred job_role: Data Scientist")
            elif any(keyword in all_text for keyword in ['marketing', 'campaign', 'crm', 'google analytics']):
                task_requirements['job_role'] = 'Marketing'
                dbg(f"Inferred job_role: Marketing")
            else:
                dbg("Could not infer job_role from filename or content, domain filtering will be disabled")
    else:
        dbg(f"job_role found in YAML: {task_requirements['job_role']}")

    bi_encoder = SentenceTransformer(config["embedding_model"])
    skill_model = se.SkillEmbeddingModel(bi_encoder)
    dbg("SKILLS EMBEDDING MODEL:", config["embedding_model"])

    embedding_model_name = config.get("embedding_model")
    
    use_string_matching = config.get("skills", {}).get("use_string_matching", False)
    
    if not use_string_matching:
        dbg("Using original skill names directly (no string matching, semantic matching only)")
    else:
        dbg("Using tree-based string matching + fine-tuned embeddings")
    
    consultants_scores, debug_similarities = se.get_consultants_skills_score(
        task_requirements,
        skill_model,
        config["skills_embedding_file"],
        config,
        debug=True,
        use_string_matching=use_string_matching
    )

    dbg("SKILL SIMILARITY — top 3 per skill:")
    for skill, scores in consultants_scores.items():
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        dbg(f"  {skill} -> {top3}")
        
        if skill in debug_similarities and len(debug_similarities[skill]) > 0:
            dbg(f"  {skill} - Top raw similarities (before threshold):")
            for user_id, skill_name, sim_score, level in debug_similarities[skill][:5]:
                dbg(f"    User {user_id}: '{skill_name}' -> similarity={sim_score:.4f}, level={level}")

    average_skills = se.average_skills(consultants_scores)
    
    dbg("Average skill scores (top 10):",
        sorted(average_skills.items(), key=lambda x: x[1], reverse=True)[:10])

    use_keyword_filtering = config.get("domain_filtering", {}).get("use_keyword_filtering", False)
    mission_model = config.get("mission_embedding_model", "paraphrase-multilingual-mpnet-base-v2")
    dbg("MISSION EMBEDDING MODEL:", mission_model)
    consultant_missions_df = get_consultants_mission_scores(
        task_requirements,
        percentile=config["mission_context"]["percentile"],
        use_domain_keyword_filtering=use_keyword_filtering,
        finetune_model=mission_model
    )
    average_mission = df_to_dict(consultant_missions_df)

    dbg("Mission scores (top 10):",
        sorted(average_mission.items(), key=lambda x: x[1], reverse=True)[:10])

    combined_scores = linear_combination(
        average_skills,
        config["skills"]["skills_weight"],
        average_mission,
        config["mission_context"]["mission_weight"]
    )

    dbg("Combined scores (top 10):",
        sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10])

    language_filter = compute_language_score(
        "data/HCK_HEC_LANG.csv",
        task_requirements,
        acceptable_difference=config["languages"]["acceptable_difference"],
        penality=config["languages"]["penality"]
    )
    dbg("Language penalties (sample):", list(language_filter.items())[:10])

    schedule_penalties = calculate_availability_penalty(
        "data/HCK_HEC_STAFFING.csv",
        task_requirements,
        config["disponibility"]["maximum_penalty"]
    )
    dbg("Availability penalties (sample):", list(schedule_penalties.items())[:10])

    filtered = apply_filters(combined_scores, language_filter)
    dbg("After language filter (top 10):",
        sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:10])

    filtered = apply_filters(filtered, schedule_penalties)
    dbg("After availability filter (top 10):",
        sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:10])

  



    sorted_consultants = se.sort_skills(filtered)
    final_top5 = sorted_consultants[:5]

    full_df = pd.read_csv("full_df.csv")
    
    expected_job_role = task_requirements.get("job_role", "Unknown")

    print("\nFINAL TOP 5 CANDIDATES (from BI-ENCODER ranking)")
    print(f"Expected Job Role: {expected_job_role}")
    print()
    
    correct_count = 0
    for i, (cid, score) in enumerate(final_top5, 1):
        consultant_row = full_df[full_df["USER_ID"] == cid]
        if len(consultant_row) > 0:
            ground_truth = consultant_row.iloc[0]["JOB_RULE"]
            is_correct = str(ground_truth).strip() == str(expected_job_role).strip()
            if is_correct:
                correct_count += 1
            print(f"{i}. Consultant {cid}: score={score:.4f}, ground_truth='{ground_truth}'")
        else:
            print(f"{i}. Consultant {cid}: score={score:.4f}, ground_truth='NOT FOUND'")
    
    print()

    print(f"ACCURACY: {correct_count}/5 = {correct_count/5*100:.1f}%")
