import yaml
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import json
import os
import warnings
from tqdm import tqdm

def load_offline_missions_embeddings(finetune_model: str=None,
                                     encoded_mission_filename: str="skills-finder-hackathon-hec-x-sbi/mission_encoded.csv",
                                     task_requirements: dict = {}):
    # Use default if not provided (should be provided from config in practice)
    if finetune_model is None:
        finetune_model = "paraphrase-multilingual-mpnet-base-v2"
    
    # Print which model is being used (for verification)
    print(f"📦 Loading mission embedding model: {finetune_model}")
    if "modelfinetuned_domain" in finetune_model:
        print("   ⚠️  WARNING: Using fine-tuned model (trained with ground truth labels)")
    else:
        print("   ✅ Using base model (no fine-tuning, fair for evaluation)")
    
    # Suppress tokenizer warnings (Mistral regex pattern warning is harmless)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tokenizer.*regex.*", category=UserWarning)
        # Load model with fix for Mistral tokenizer warning
        try:
            model = SentenceTransformer(finetune_model, tokenizer_kwargs={"fix_mistral_regex": True})
        except (TypeError, ValueError):
            # Fallback if the parameter is not supported
            model = SentenceTransformer(finetune_model)
    
    # Check if file exists, if not we need to create it
    if not os.path.exists(encoded_mission_filename):
        print(f"⚠️  Mission embeddings file not found: {encoded_mission_filename}")
        print(f"   Will create it by encoding missions from HCK_HEC_XP.csv")
        # Load missions from source
        missions_source = "data/HCK_HEC_XP.csv"
        if os.path.exists(missions_source):
            dfmission = pd.read_csv(missions_source)
            dfmission = dfmission.drop_duplicates(subset=['USER_ID', 'MISSION_DSC']).reset_index(drop=True)
        else:
            raise FileNotFoundError(f"Neither {encoded_mission_filename} nor {missions_source} found!")
    else:
        dfmission = pd.read_csv(encoded_mission_filename)
        # Load stored embeddings
        dfmission["embedding"] = [json.loads(emb) for emb in dfmission["embedding"].values]
    
    # Check embedding dimensions
    stored_emb_dim = len(dfmission["embedding"].iloc[0]) if "embedding" in dfmission.columns else 0
    model_emb_dim = len(model.encode("test", convert_to_tensor=False))
    
    # Check if missions were encoded with a different model
    # Check if _encoding_model column exists in the stored file
    stored_model = None
    if "_encoding_model" in dfmission.columns and len(dfmission) > 0:
        stored_model = dfmission["_encoding_model"].iloc[0]
    # If column doesn't exist, assume it was encoded with fine-tuned model (old format)
    # This will trigger re-encoding when switching to base model
    elif stored_emb_dim > 0:  # File exists and has embeddings
        # If switching to base model but no metadata, assume old embeddings were fine-tuned
        if finetune_model == "paraphrase-multilingual-mpnet-base-v2":
            stored_model = "models/modelfinetuned_domain"  # Assume old embeddings were fine-tuned
            print(f"⚠️  No model metadata found. Assuming stored embeddings were from fine-tuned model.")
            print(f"   Will re-encode with base model: {finetune_model}")
    
    # Force re-encode if model changed (even if dimensions match)
    model_changed = stored_model is not None and stored_model != finetune_model
    dimensions_dont_match = stored_emb_dim != model_emb_dim and stored_emb_dim > 0
    
    if model_changed:
        print(f"🔄 Model changed: stored='{stored_model}' -> current='{finetune_model}'")
        print(f"   Re-encoding missions with new model...")
    
    # If dimensions don't match OR model changed, re-encode with current model and save
    if dimensions_dont_match or model_changed:
        if dimensions_dont_match:
            print(f"⚠️  Warning: Stored embeddings dimension ({stored_emb_dim}) doesn't match model dimension ({model_emb_dim})")
        if model_changed:
            print(f"🔄 Model changed: Re-encoding missions with current model: {finetune_model}")
        print("   This may take a moment...")
        
        # Need to reload from source if we're switching models (MISSION_DSC might not be in encoded file)
        if ("MISSION_DSC" not in dfmission.columns) or model_changed:
            print(f"   Loading missions from source to re-encode...")
            missions_source = "data/HCK_HEC_XP.csv"
            if os.path.exists(missions_source):
                dfmission = pd.read_csv(missions_source)
                dfmission = dfmission.drop_duplicates(subset=['USER_ID', 'MISSION_DSC']).reset_index(drop=True)
            else:
                raise FileNotFoundError(f"Source file {missions_source} not found for re-encoding!")
        
        # Re-encode with progress bar
        embeddings = []
        for mission in tqdm(dfmission["MISSION_DSC"], desc="Encoding missions"):
            emb = model.encode(mission, convert_to_tensor=False).tolist()
            embeddings.append(emb)
        dfmission["embedding"] = embeddings
        
        # Store which model was used for encoding
        dfmission["_encoding_model"] = finetune_model
        
        # Save the re-encoded embeddings back to CSV
        print(f"   Saving re-encoded embeddings to {encoded_mission_filename}...")
        dfmission["embedding"] = dfmission["embedding"].apply(json.dumps)
        dfmission.to_csv(encoded_mission_filename, index=False)
        print("   ✓ Embeddings saved! Future runs will be faster.")
        
        # Convert back to list format for processing
        dfmission["embedding"] = [json.loads(emb) for emb in dfmission["embedding"].values]

    job_responsibilities = task_requirements.get("responsabilites", [])

    # Encode job responsibilities
    job_responsibility_embeddings = model.encode(job_responsibilities, convert_to_tensor=True).cpu()

    mission_tensor = torch.stack(dfmission["embedding"].apply(torch.tensor).tolist()).cpu()

    sim_matrix = cos_sim(mission_tensor, job_responsibility_embeddings)

    return sim_matrix, dfmission

def get_responsability_thresholds(sim_matrix, percentile=99):
    # Compute dynamic threshold 
    responsibility_thresholds = []
    for i in range(sim_matrix.shape[1]):
        scores = sim_matrix[:, i].detach().numpy()
        threshold = np.percentile(scores, percentile)
        responsibility_thresholds.append(threshold)
    return responsibility_thresholds

def count_top_matches_per_responsibility(mission_similarities, thresholds):
    return sum(sim > threshold for sim, threshold in zip(mission_similarities, thresholds))

def extract_best_similarity(mission_similarities):
    return max(mission_similarities)

def get_consultants_mission_scores(task_requirements, percentile, use_domain_keyword_filtering=False, finetune_model=None):
    """
    Get mission similarity scores for consultants.
    
    Args:
        task_requirements: Mission requirements dict
        percentile: Percentile threshold for similarity (e.g., 90)
        use_domain_keyword_filtering: If True, use keyword-based domain filtering (for production).
                                       If False, rely purely on embeddings (for fair evaluation).
    """
    matches = []

    # Use provided model or default to base model (no fine-tuning)
    model_to_use = finetune_model if finetune_model else "paraphrase-multilingual-mpnet-base-v2"
    sim_matrix, dfmission = load_offline_missions_embeddings(finetune_model=model_to_use, task_requirements=task_requirements)
    responsibility_thresholds = get_responsability_thresholds(sim_matrix, percentile=percentile)
    
    # Domain keyword filtering (OPTIONAL - only for production, not for evaluation)
    # NOTE: Using keyword filtering is "cheating" for evaluation because it's rule-based
    # For fair evaluation, set use_domain_keyword_filtering=False and let embeddings do the work
    job_role = task_requirements.get('job_role', '').strip() if task_requirements.get('job_role') else None
    
    if use_domain_keyword_filtering:
        # Domain keywords to identify cross-domain matches (using specific, non-generic terms)
        domain_keywords = {
            'Healthcare': ['patient', 'clinical', 'medical', 'healthcare', 'physician', 'nursing', 'diagnosis', 'treatment', 'care team', 'medical record'],
            'Scrum Master': ['scrum', 'agile', 'sprint', 'ceremonies', 'backlog', 'product owner', 'velocity', 'sprint planning', 'daily standup'],
            'Data Analyst': ['data analyst', 'dashboard', 'reporting', 'kpi', 'analytics', 'power bi', 'tableau', 'business intelligence'],
            'Data Engineer': ['data engineer', 'pipeline', 'etl', 'databricks', 'data factory', 'spark', 'data warehouse', 'data lake', 'azure data factory', 'azure databricks', 'apache spark', 'big data', 'streaming data', 'data ingestion', 'data transformation', 'data orchestration'],
            'Finance': ['finance', 'financial', 'budget', 'forecast', 'accounting', 'financial planning', 'financial close']
        }

    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            score = sim_matrix[i][j].item()
            if score > responsibility_thresholds[j]:
                # Domain keyword filtering (only if enabled)
                if use_domain_keyword_filtering and job_role and job_role in domain_keywords:
                    mission_text = dfmission.iloc[i]["MISSION_DSC"].lower()
                    required_keywords = domain_keywords[job_role]
                    mission_has_domain_keywords = any(kw in mission_text for kw in required_keywords)
                    
                    # Check for wrong domain keywords
                    wrong_domain_keywords = []
                    for domain, keywords in domain_keywords.items():
                        if domain != job_role:
                            wrong_domain_keywords.extend(keywords)
                    
                    mission_has_wrong_domain = any(kw in mission_text for kw in wrong_domain_keywords)
                    
                    # STRICT FILTERING: For specific roles, require domain keywords
                    if job_role in ['Scrum Master', 'Data Engineer', 'Data Analyst', 'Healthcare']:
                        # For these roles, require at least one domain keyword
                        if not mission_has_domain_keywords:
                            continue  # Skip - no required domain keywords
                    elif mission_has_wrong_domain and not mission_has_domain_keywords:
                        continue  # Skip this match - wrong domain
                
                matches.append({
                    "USER_ID": dfmission.iloc[i]["USER_ID"],
                    "MISSION_DSC": dfmission.iloc[i]["MISSION_DSC"],
                    "similarity": score
                })

    match_df = pd.DataFrame(matches)

    consultant_similarity_scores = (
        match_df.groupby("USER_ID")
        .agg(
            total_matches=("similarity", "count"),
            avg_score=("similarity", "mean"),
            best_score=("similarity", "max")
        )
    )

    consultant_similarity_scores["final_score"] = (
        consultant_similarity_scores["avg_score"] * np.log1p(consultant_similarity_scores["total_matches"])
    )

    consultant_similarity_scores = consultant_similarity_scores.sort_values(
        by=["final_score", "total_matches", "avg_score", "best_score"], ascending=False
    )

    consultant_similarity_scores["final_score"] = consultant_similarity_scores["final_score"] / consultant_similarity_scores["final_score"].max()

    return consultant_similarity_scores

def df_to_dict(consultant_missions_scores: pd.DataFrame):
    scores = {}
    for idx, row in consultant_missions_scores.iterrows():
        scores[idx] = row["final_score"]
    return scores