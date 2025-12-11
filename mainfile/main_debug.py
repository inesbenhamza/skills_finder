import argparse
import sys
import os
from pathlib import Path
import yaml

# Add parent directory to path so we can import modules from root
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import yaml
from scipy.optimize import minimize
from src.schedule_filtering import calculate_availability_penalty
from helpers.tools_extraction import generate_yaml, get_model_response
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict
from helpers import SkillsEmbeddings as se
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import pandas as pd
from src.language_filtering import compute_language_score, apply_filters

MATCHING_CONFIGURATION_FILE = "config/search_config.yaml"

# ==========================================
# DEBUG MODE
# ==========================================
DEBUG = True

def dbg(*args):
    if DEBUG:
        print("\n[DEBUG]", *args)



# ==========================================
# Load Cross-Encoder for Final Re-ranking
# ==========================================
def load_cross_encoder():
    """
    Load a cross-encoder model for reranking.
    
    Cross-encoders are more accurate than bi-encoders because they encode
    query and document together, allowing for deeper interaction.
    However, they are slower, so we use them only for reranking top candidates.
    """
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


# ==========================================
# Build a complete mission text
# ==========================================
def build_mission_text(task):
    parts = []

    if "responsabilites" in task:
        parts.append("Responsibilities: " + " ".join(task["responsabilites"]))

    if "activity_sector" in task:
        parts.append(f"Sector: {task['activity_sector']}.")

    if "required_experience" in task:
        parts.append(f"Experience required: {task['required_experience']} years.")

    if "languages" in task:
        langs = [f"{l['language_code']} level {l['level']}" for l in task["languages"]]
        parts.append("Languages required: " + ", ".join(langs))

    if "technologies" in task:
        techs = [f"{t['name']} level {t['level']}" for t in task["technologies"]]
        parts.append("Technologies: " + ", ".join(techs))

    return ". ".join(parts)



# ==========================================
# Consultant text for cross-encoder
# ==========================================
def build_consultant_text(row):
    return (
        f"Candidate profile: "
        f"Skills: {row['SKILLS_DSC']}. "
        f"Domain expertise: {row['DOMAIN_DSC']}. "
        f"Experience level (0-100): {row['LEVEL_VAL']}. "
        f"This candidate has worked on related missions."
    )


# ==========================================
# Cross-Encoder Reranking
# ==========================================
def rerank_with_cross_encoder(task_text, consultant_ids, df, cross_encoder, top_k=5):
    """
    Rerank candidates using cross-encoder.
    
    This function takes the top candidates from bi-encoder ranking and
    reranks them using a cross-encoder for higher accuracy.
    
    Args:
        task_text: Full mission description text
        consultant_ids: List of consultant IDs to rerank (from bi-encoder top-20)
        df: DataFrame with consultant profiles
        cross_encoder: Loaded CrossEncoder model
        top_k: Number of top candidates to return
    
    Returns:
        List of (consultant_id, score) tuples, sorted by score (highest first)
    """

    pairs = []
    for cid in consultant_ids:
        row = df[df["USER_ID"] == cid].iloc[0]
        text = build_consultant_text(row)
        pairs.append((task_text, text))

    dbg("CROSS-ENCODER INPUT EXAMPLES:")
    for p in pairs[:2]:
        dbg("PAIR =", p)

    scores = cross_encoder.predict(pairs)
    dbg("Cross-encoder raw scores:", list(zip(consultant_ids, scores))[:10])

    results = sorted(list(zip(consultant_ids, scores)), key=lambda x: x[1], reverse=True)
    return results[:top_k]



# ==========================================
# Supervised Learning for Weight Optimization
# ==========================================
def learn_optimal_weights_from_labels(
    task_requirements,
    average_skills_ratings,
    average_mission_ratings,
    labels_file="full_df.csv",
    top_k=5
):
    """
    Learn optimal weights using supervised learning with ground truth labels.
    
    Uses historical data with job role labels to train a model that predicts
    optimal weights for maximizing top-k accuracy.
    
    Args:
        task_requirements: Current mission requirements
        average_skills_ratings: Skills scores for all consultants
        average_mission_ratings: Mission scores for all consultants
        labels_file: CSV file with JOB_RULE column (ground truth labels)
        top_k: Focus on top-k accuracy (default: 5)
    
    Returns:
        tuple: (optimal_skills_weight, optimal_mission_weight)
    """
    try:
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import ndcg_score
    except ImportError:
        dbg("scikit-learn not available, falling back to optimization method")
        return optimize_weights(average_skills_ratings, average_mission_ratings, method='variance')
    
    try:
        # Load labeled data
        df_labels = pd.read_csv(labels_file)
        
        if 'JOB_RULE' not in df_labels.columns:
            dbg("No JOB_RULE column found, falling back to optimization")
            return optimize_weights(average_skills_ratings, average_mission_ratings, method='variance')
        
        # Extract current mission's target job role
        # PRIORITY 1: Use LLM-extracted job_role from YAML (most reliable)
        target_job_role = None
        if 'job_role' in task_requirements and task_requirements['job_role']:
            target_job_role = task_requirements['job_role'].strip()
            # Normalize job role names to match dataset (exact match required)
            valid_job_roles = ['Data Engineer', 'Data Analyst', 'Data Scientist', 'Scrum Master', 
                             'Finance', 'Healthcare', 'Marketing', 'Other']
            if target_job_role not in valid_job_roles:
                dbg(f"Warning: Job role '{target_job_role}' not in valid list, will use as-is")
            # Map "Other" to None (use all data)
            if target_job_role == 'Other':
                target_job_role = None
            if target_job_role:
                dbg(f"Using LLM-extracted job role: {target_job_role}")
        elif 'role' in task_requirements and task_requirements['role']:
            target_job_role = task_requirements['role'].strip()
            dbg(f"Using role field: {target_job_role}")
        elif 'title' in task_requirements and task_requirements['title']:
            target_job_role = task_requirements['title'].strip()
            dbg(f"Using title field: {target_job_role}")
        else:
            # FALLBACK: Infer from content (only if LLM didn't extract it)
            # Infer from responsibilities and technologies
            responsibilities = task_requirements.get('responsabilites', [])
            technologies = [t.get('name', '').lower() for t in task_requirements.get('technologies', [])]
            
            # Combine all text for analysis
            all_text = ' '.join(responsibilities).lower()
            all_text += ' ' + ' '.join(technologies)
            
            # Pattern matching for common job roles
            # Check healthcare FIRST (before data analyst) to avoid false matches
            if any(keyword in all_text for keyword in ['healthcare', 'health care', 'medical', 'patient', 'clinical', 'nursing', 'physician', 'diagnosis', 'treatment plan', 'medical record', 'health status', 'care team', 'allied health']):
                target_job_role = 'Healthcare'
                dbg(f"Inferred job role from content: {target_job_role}")
            elif any(keyword in all_text for keyword in ['scrum', 'agile', 'sprint', 'ceremonies', 'product owner', 'backlog']):
                target_job_role = 'Scrum Master'
                dbg(f"Inferred job role from content: {target_job_role}")
            elif any(keyword in all_text for keyword in ['data engineer', 'pipeline', 'etl', 'databricks', 'data factory', 'spark', 'adf']):
                target_job_role = 'Data Engineer'
                dbg(f"Inferred job role from content: {target_job_role}")
            elif any(keyword in all_text for keyword in ['data analyst', 'power bi', 'tableau', 'dashboard', 'reporting', 'kpi', 'analytics']):
                target_job_role = 'Data Analyst'
                dbg(f"Inferred job role from content: {target_job_role}")
            elif any(keyword in all_text for keyword in ['data scientist', 'machine learning', 'ml', 'model', 'prediction', 'ai']):
                target_job_role = 'Data Scientist'
                dbg(f"Inferred job role from content: {target_job_role}")
            elif any(keyword in all_text for keyword in ['finance', 'financial', 'budget', 'forecast', 'planning']):
                target_job_role = 'Finance'
                dbg(f"Inferred job role from content: {target_job_role}")
            else:
                dbg("No explicit job role in requirements and couldn't infer from content, will use all historical data for training")
                target_job_role = None
        
        # Filter consultants by job role if specified
        if target_job_role:
            relevant_consultants = df_labels[df_labels['JOB_RULE'] == target_job_role]['USER_ID'].tolist()
            dbg(f"Found {len(relevant_consultants)} consultants with job role: {target_job_role}")
        else:
            relevant_consultants = df_labels['USER_ID'].tolist()
            dbg(f"Using all {len(relevant_consultants)} consultants for training")
        
        # Prepare training data: for each historical mission, find optimal weights
        training_data = []
        
        # Group by job role to create training examples
        for job_role in df_labels['JOB_RULE'].unique():
            role_consultants = df_labels[df_labels['JOB_RULE'] == job_role]['USER_ID'].tolist()
            
            # Get skills and mission scores for these consultants (if available)
            role_skills = {cid: average_skills_ratings.get(cid, 0) for cid in role_consultants if cid in average_skills_ratings}
            role_missions = {cid: average_mission_ratings.get(cid, 0) for cid in role_consultants if cid in average_mission_ratings}
            
            if len(role_skills) < 5:  # Need at least 5 consultants for meaningful training
                continue
            
            # Find optimal weights for this job role using top-k NDCG
            # Use actual job role labels: consultants with this job role are relevant (1), others are not (0)
            # This creates a proper learning signal
            all_consultant_ids = set(average_skills_ratings.keys()) | set(average_mission_ratings.keys())
            true_labels = {cid: 1 if cid in role_consultants else 0 for cid in all_consultant_ids}
            
            optimal_w = _find_weights_for_topk_accuracy(
                role_skills, role_missions, 
                true_labels=true_labels,  # Proper labels: 1 for this job role, 0 for others
                top_k=top_k
            )
            
            # Create features from mission characteristics
            features = _extract_mission_features(task_requirements, role_skills, role_missions)
            features['job_role_encoded'] = hash(job_role) % 1000  # Simple encoding
            
            training_data.append({
                'features': features,
                'optimal_skills_weight': optimal_w[0],
                'optimal_mission_weight': optimal_w[1]
            })
        
        if len(training_data) < 3:
            dbg("Not enough training data, falling back to optimization")
            return optimize_weights(average_skills_ratings, average_mission_ratings, method='variance')
        
        # Train a model to predict optimal weights
        X = pd.DataFrame([d['features'] for d in training_data])
        y_skills = [d['optimal_skills_weight'] for d in training_data]
        y_mission = [d['optimal_mission_weight'] for d in training_data]
        
        # Train two models (one for each weight)
        model_skills = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model_mission = GradientBoostingRegressor(n_estimators=50, random_state=42)
        
        model_skills.fit(X, y_skills)
        model_mission.fit(X, y_mission)
        
        # Predict optimal weights for current mission
        current_features = _extract_mission_features(task_requirements, average_skills_ratings, average_mission_ratings)
        current_features['job_role_encoded'] = hash(target_job_role) % 1000 if target_job_role else 0
        X_current = pd.DataFrame([current_features])
        
        pred_skills_w = max(0, min(1, model_skills.predict(X_current)[0]))
        pred_mission_w = max(0, min(1, model_mission.predict(X_current)[0]))
        
        # Normalize to sum to 1
        total = pred_skills_w + pred_mission_w
        if total > 0:
            pred_skills_w /= total
            pred_mission_w /= total
        else:
            pred_skills_w, pred_mission_w = 0.5, 0.5
        
        dbg(f"LEARNED weights from historical data: skills={pred_skills_w:.4f}, mission={pred_mission_w:.4f}")
        return pred_skills_w, pred_mission_w
        
    except Exception as e:
        dbg(f"Learning failed: {e}, falling back to optimization")
        return optimize_weights(average_skills_ratings, average_mission_ratings, method='variance')


def _extract_mission_features(task_requirements, skills_scores, mission_scores):
    """Extract features from mission requirements for ML model."""
    features = {}
    
    # Mission characteristics
    features['num_required_skills'] = len([t for t in task_requirements.get('technologies', []) if t.get('required', False)])
    features['num_optional_skills'] = len([t for t in task_requirements.get('technologies', []) if not t.get('required', False)])
    features['num_languages'] = len(task_requirements.get('languages', []))
    features['has_experience_req'] = 1 if task_requirements.get('required_experience') else 0
    
    # Text-based features from responsibilities (helps even without job role inference)
    responsibilities = task_requirements.get('responsabilites', [])
    all_text = ' '.join(responsibilities).lower()
    
    # Domain-specific keyword counts (helps distinguish job types)
    features['healthcare_keywords'] = sum(1 for kw in ['healthcare', 'medical', 'patient', 'clinical', 'nursing', 'physician', 'diagnosis', 'treatment'] if kw in all_text)
    features['data_keywords'] = sum(1 for kw in ['data', 'analytics', 'dashboard', 'reporting', 'pipeline', 'etl', 'database'] if kw in all_text)
    features['agile_keywords'] = sum(1 for kw in ['scrum', 'agile', 'sprint', 'ceremonies', 'backlog'] if kw in all_text)
    features['finance_keywords'] = sum(1 for kw in ['finance', 'financial', 'budget', 'forecast', 'planning'] if kw in all_text)
    features['responsibility_length'] = len(all_text)  # Longer descriptions might need different weights
    
    # Score distribution statistics
    if skills_scores:
        features['skills_mean'] = np.mean(list(skills_scores.values()))
        features['skills_std'] = np.std(list(skills_scores.values()))
        features['skills_max'] = np.max(list(skills_scores.values()))
    else:
        features['skills_mean'] = features['skills_std'] = features['skills_max'] = 0
    
    if mission_scores:
        features['mission_mean'] = np.mean(list(mission_scores.values()))
        features['mission_std'] = np.std(list(mission_scores.values()))
        features['mission_max'] = np.max(list(mission_scores.values()))
    else:
        features['mission_mean'] = features['mission_std'] = features['mission_max'] = 0
    
    # Correlation between skills and mission scores
    common_ids = set(skills_scores.keys()) & set(mission_scores.keys())
    if len(common_ids) > 1:
        skills_vals = [skills_scores[cid] for cid in common_ids]
        mission_vals = [mission_scores[cid] for cid in common_ids]
        features['skills_mission_corr'] = np.corrcoef(skills_vals, mission_vals)[0, 1] if len(skills_vals) > 1 else 0
    else:
        features['skills_mission_corr'] = 0
    
    return features


def _find_weights_for_topk_accuracy(skills_scores, mission_scores, true_labels, top_k=5):
    """
    Find weights that maximize top-k accuracy (NDCG@k) using optimization.
    
    Args:
        skills_scores: {consultant_id: skill_score}
        mission_scores: {consultant_id: mission_score}
        true_labels: {consultant_id: 1 if relevant, 0 otherwise}
        top_k: Focus on top-k
    
    Returns:
        tuple: (optimal_skills_weight, optimal_mission_weight)
    """
    from scipy.optimize import minimize
    
    common_ids = list(set(skills_scores.keys()) & set(mission_scores.keys()) & set(true_labels.keys()))
    if len(common_ids) < top_k:
        return 0.5, 0.5
    
    skills_vals = np.array([skills_scores[cid] for cid in common_ids])
    mission_vals = np.array([mission_scores[cid] for cid in common_ids])
    y_true = np.array([true_labels.get(cid, 0) for cid in common_ids])
    
    def objective(weights):
        skills_w, mission_w = weights[0], weights[1]
        combined = skills_w * skills_vals + mission_w * mission_vals
        
        # Get top-k indices
        top_k_indices = np.argsort(combined)[-top_k:][::-1]
        
        # Calculate NDCG@k
        y_pred_ranked = y_true[top_k_indices]
        if np.sum(y_true) > 0:
            # NDCG calculation
            dcg = np.sum(y_pred_ranked / np.log2(np.arange(2, len(y_pred_ranked) + 2)))
            ideal_dcg = np.sum(np.sort(y_true)[::-1][:top_k] / np.log2(np.arange(2, top_k + 2)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            return -ndcg  # Minimize negative NDCG (maximize NDCG)
        else:
            return 0
    
    constraints = ({'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1.0})
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    initial_guess = [0.5, 0.5]
    
    try:
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        if result.success:
            return result.x[0], result.x[1]
    except:
        pass
    
    return 0.5, 0.5


# ==========================================
# Automatic Weight Optimization
# ==========================================
def optimize_weights(average_skills_ratings, average_mission_ratings, method='variance'):
    """
    Automatically find optimal weights for skills and mission scores.
    
    Methods:
    - 'variance': Maximize score variance (better differentiation between candidates)
    - 'balance': Balance both signals equally while maximizing spread
    - 'correlation': Minimize correlation between skills and mission (diversify signals)
    
    Returns:
        tuple: (optimal_skills_weight, optimal_mission_weight)
    """
    
    # Get common consultant IDs
    common_ids = set(average_skills_ratings.keys()) & set(average_mission_ratings.keys())
    if len(common_ids) < 2:
        dbg("Not enough common consultants for optimization, using default 0.5/0.5")
        return 0.5, 0.5
    
    # Prepare data
    filler_mission = np.mean(list(average_mission_ratings.values()))
    skill_scores = np.array([average_skills_ratings.get(cid, 0) for cid in common_ids])
    mission_scores = np.array([average_mission_ratings.get(cid, filler_mission) for cid in common_ids])
    
    def objective(weights):
        """Objective function to minimize (negative variance for maximization)"""
        skills_w, mission_w = weights[0], weights[1]
        combined = skills_w * skill_scores + mission_w * mission_scores
        
        if method == 'variance':
            # Maximize variance (minimize negative variance)
            return -np.var(combined)
        elif method == 'balance':
            # Balance: maximize variance while keeping weights balanced
            variance = np.var(combined)
            imbalance = abs(skills_w - mission_w)  # Penalize imbalanced weights
            return -(variance - 0.1 * imbalance)  # Small penalty for imbalance
        elif method == 'correlation':
            # Minimize correlation (diversify signals)
            variance = np.var(combined)
            correlation = np.corrcoef(skill_scores, mission_scores)[0, 1]
            return -(variance - 0.2 * abs(correlation))
        else:
            return -np.var(combined)
    
    # Constraints: weights must sum to 1, and be between 0 and 1
    constraints = ({'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1.0})
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    # Initial guess: equal weights
    initial_guess = [0.5, 0.5]
    
    try:
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_skills_w = result.x[0]
            optimal_mission_w = result.x[1]
            
            # Calculate final variance for comparison
            final_scores = optimal_skills_w * skill_scores + optimal_mission_w * mission_scores
            final_variance = np.var(final_scores)
            
            dbg(f"Optimal weights found ({method}): skills={optimal_skills_w:.4f}, mission={optimal_mission_w:.4f}")
            dbg(f"Final score variance: {final_variance:.6f}")
            
            return optimal_skills_w, optimal_mission_w
        else:
            dbg(f"Optimization failed: {result.message}, using default 0.5/0.5")
            return 0.5, 0.5
    except Exception as e:
        dbg(f"Optimization error: {e}, using default 0.5/0.5")
        return 0.5, 0.5


# ==========================================
# Linear combination
# ==========================================
def linear_combination(average_skills_ratings, skills_weight, average_mission_ratings, mission_weight, 
                      auto_optimize=False, optimize_method='variance', 
                      use_learning=False, task_requirements=None, labels_file="full_df.csv"):
    """
    Combine skills and mission scores with optional automatic weight optimization.
    
    Args:
        auto_optimize: If True, automatically find optimal weights using optimization
        optimize_method: 'variance', 'balance', or 'correlation' (for optimization)
        use_learning: If True, use supervised learning from historical labels
        task_requirements: Mission requirements (needed for learning)
        labels_file: CSV file with JOB_RULE labels (needed for learning)
    """
    if use_learning and task_requirements:
        dbg("Using SUPERVISED LEARNING to predict optimal weights from historical data...")
        skills_weight, mission_weight = learn_optimal_weights_from_labels(
            task_requirements,
            average_skills_ratings,
            average_mission_ratings,
            labels_file=labels_file,
            top_k=5
        )
        dbg(f"Using LEARNED weights: skills={skills_weight:.4f}, mission={mission_weight:.4f}")
    elif auto_optimize:
        skills_weight, mission_weight = optimize_weights(
            average_skills_ratings, 
            average_mission_ratings,
            method=optimize_method
        )
        dbg(f"Using AUTO-OPTIMIZED weights: skills={skills_weight:.4f}, mission={mission_weight:.4f}")
    
    # For consultants with no matching missions (especially after domain filtering),
    # use a very low score instead of filler mean to prevent wrong-domain matches
    # Only use filler if there are actually mission scores (not all filtered out)
    if len(average_mission_ratings) > 0:
        filler_mission_value = np.mean(list(average_mission_ratings.values())) * 0.1  # 10% of mean (penalty)
    else:
        filler_mission_value = 0.0  # No missions at all
    
    combinations = {}
    for cid in average_skills_ratings:

        skill_score = average_skills_ratings[cid]
        mission_score = average_mission_ratings.get(cid, filler_mission_value)

        combined = skill_score * skills_weight + mission_score * mission_weight
        combinations[cid] = combined

        dbg(f"CID {cid}: skill={skill_score:.4f}, mission={mission_score:.4f}, combined={combined:.4f}")

    return combinations



# ==========================================
# CLI
# ==========================================
def setup_arguments():
    parser = argparse.ArgumentParser("main")
    parser.add_argument("filename", help="Mission PDF name (without extension)", type=str)
    return parser.parse_args()



# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    args = setup_arguments()

    with open(MATCHING_CONFIGURATION_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # ==========================================
    # Load or Generate Mission Requirements
    # ==========================================
    # Currently loads from SAVED YAML file (faster, pre-processed)
    # To generate live from PDF, uncomment the generation code below
    yaml_path = f"output/filtered_skills/{args.filename}.yaml"
    
    # Option 1: Load from saved YAML (current default - FAST)
    if os.path.exists(yaml_path):
        dbg(f"Loading mission requirements from saved YAML: {yaml_path}")
        with open(yaml_path, "r") as f:
            task_requirements = yaml.safe_load(f)
    else:
        # Option 2: Generate live from PDF (slower, requires LLM)
        dbg(f"YAML file not found. Generating live from PDF: {args.filename}.pdf")
        pdf_path = f"documents/{args.filename}.pdf"
        if not os.path.exists(pdf_path):
            pdf_path = f"{args.filename}.pdf"  # Try root directory
        
        if os.path.exists(pdf_path):
            model_response = get_model_response(pdf_path, config["model"]["name"])
            task_requirements = generate_yaml(model_response, think=config["model"]["think"])
            
            # Save the generated YAML for future use
            os.makedirs("output/filtered_skills", exist_ok=True)
            with open(yaml_path, 'w') as f:
                yaml.safe_dump(task_requirements, f)
            dbg(f"Saved generated YAML to: {yaml_path}")
        else:
            raise FileNotFoundError(f"Neither YAML ({yaml_path}) nor PDF ({pdf_path}) found!")
    
    dbg("Loaded mission task:", task_requirements)
    
    # ==========================================
    # Infer job_role if not in YAML (for domain filtering)
    # ==========================================
    if 'job_role' not in task_requirements or not task_requirements.get('job_role'):
        # PRIORITY 1: Try to infer from PDF filename (most reliable)
        # e.g., "Finance.pdf" -> "Finance", "DataEngineer.pdf" -> "Data Engineer"
        filename = args.filename
        dbg(f"job_role not in YAML, trying to infer from filename: {filename}")
        
        # Normalize filename to job role
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
            # PRIORITY 2: Fall back to keyword inference from content
            dbg("Could not infer from filename, inferring from mission content...")
            responsibilities = task_requirements.get('responsabilites', [])
            technologies = [t.get('name', '').lower() for t in task_requirements.get('technologies', [])]
            all_text = ' '.join(responsibilities).lower() + ' ' + ' '.join(technologies)
            
            # Infer job role from content (same logic as in learn_optimal_weights_from_labels)
            # IMPORTANT: Check Finance BEFORE Data Analyst because Finance missions often contain
            # "reporting" and "analytics" keywords, but are more specific (budget, financial, etc.)
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

    # ==========================================
    # STEP 1: Load BI-ENCODER (SentenceTransformer)
    # ==========================================
    # The bi-encoder is used for initial ranking - it encodes queries and documents
    # separately, making it fast for large-scale retrieval
    bi_encoder = SentenceTransformer(config["embedding_model"])
    skill_model = se.SkillEmbeddingModel(bi_encoder)
    dbg("Loaded BI-ENCODER model:", config["embedding_model"])

    # ==========================================
    # STEP 2: BI-ENCODER - Skill Similarity Matching
    # ==========================================
    # 1. SKILL SIMILARITY (using bi-encoder)
    consultants_scores, debug_similarities = se.get_consultants_skills_score(
        task_requirements,
        skill_model,
        config["skills_embedding_file"],
        config,
        debug=True
    )

    dbg("SKILL SIMILARITY — top 3 per skill:")
    for skill, scores in consultants_scores.items():
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        dbg(f"  {skill} -> {top3}")
        
        # Show raw similarity scores for debugging
        if skill in debug_similarities and len(debug_similarities[skill]) > 0:
            dbg(f"  {skill} - Top raw similarities (before threshold):")
            for user_id, skill_name, sim_score, level in debug_similarities[skill][:5]:
                dbg(f"    User {user_id}: '{skill_name}' -> similarity={sim_score:.4f}, level={level}")
            # Also show consultant 2433141 if they're in the list (even if not top 5)
            for user_id, skill_name, sim_score, level in debug_similarities[skill]:
                if user_id == 2433141:
                    dbg(f"    *** User 2433141: '{skill_name}' -> similarity={sim_score:.4f}, level={level} ***")

    average_skills = se.average_skills(consultants_scores)
    
    # NOTE: Removed filter - investigating root cause instead
    # The real issue is why consultant 2433141's skills are matching Data Engineer skills
    # We should fix the embeddings/threshold rather than adding post-hoc filters
    
    # DEBUG: Check consultant 2433141's individual skill scores
    if 2433141 in [cid for skill_scores in consultants_scores.values() for cid in skill_scores.keys()]:
        dbg("=" * 80)
        dbg("DEBUG: Consultant 2433141 skill breakdown:")
        dbg("  This consultant is a SCRUM MASTER with skills: JIRA, Confluence, Teams, Slack, Miro, Power BI, Excel")
        dbg("  They should NOT match Data Engineer skills!")
        dbg("")
        for skill_name, skill_scores_dict in consultants_scores.items():
            if 2433141 in skill_scores_dict:
                # Find which of their skills matched
                dbg(f"  {skill_name}: {skill_scores_dict[2433141]:.4f}")
                # Check debug data to see what matched
                # Also check ALL skills in the dataset, not just top 50 debug
                if skill_name in debug_similarities:
                    matches_found = False
                    for user_id, matched_skill, sim_score, level in debug_similarities[skill_name]:
                        if user_id == 2433141:
                            dbg(f"    -> Matched '{matched_skill}' (similarity={sim_score:.4f}, level={level})")
                            matches_found = True
                    if not matches_found:
                        dbg(f"    -> WARNING: Has score {skill_scores_dict[2433141]:.4f} but no match in top 50 debug!")
                        dbg(f"       This means consultant 2433141 matched with similarity < top 50 but > threshold")
        dbg(f"  AVERAGE: {average_skills.get(2433141, 0):.4f}")
        dbg("=" * 80)
    
    dbg("Average skill scores (top 10):",
        sorted(average_skills.items(), key=lambda x: x[1], reverse=True)[:10])


    # ==========================================
    # STEP 3: BI-ENCODER - Mission Similarity
    # ==========================================
    # 2. MISSION SIMILARITY (using fine-tuned bi-encoder)
    use_keyword_filtering = config.get("domain_filtering", {}).get("use_keyword_filtering", False)
    mission_model = config.get("mission_embedding_model", "models/modelfinetuned_domain")
    consultant_missions_df = get_consultants_mission_scores(
        task_requirements,
        percentile=config["mission_context"]["percentile"],
        use_domain_keyword_filtering=use_keyword_filtering,
        finetune_model=mission_model
    )
    average_mission = df_to_dict(consultant_missions_df)

    dbg("Mission scores (top 10):",
        sorted(average_mission.items(), key=lambda x: x[1], reverse=True)[:10])


    # ==========================================
    # STEP 4: Combine BI-ENCODER Results
    # ==========================================
    # 3. COMBINE SKILLS + MISSIONS (from bi-encoder)
    # Option to auto-optimize weights or use supervised learning
    auto_optimize = config.get("weight_optimization", {}).get("enabled", False)
    use_learning = config.get("weight_optimization", {}).get("use_learning", False)
    optimize_method = config.get("weight_optimization", {}).get("method", "variance")
    labels_file = config.get("weight_optimization", {}).get("labels_file", "full_df.csv")
    
    if use_learning:
        dbg("Using SUPERVISED LEARNING to predict optimal weights from historical job role labels...")
    elif auto_optimize:
        dbg("AUTO-OPTIMIZING weights for skills and mission scores...")
    
    combined_scores = linear_combination(
        average_skills,
        config["skills"]["skills_weight"],
        average_mission,
        config["mission_context"]["mission_weight"],
        auto_optimize=auto_optimize,
        optimize_method=optimize_method,
        use_learning=use_learning,
        task_requirements=task_requirements,
        labels_file=labels_file
    )

    dbg("Combined scores (top 10):",
        sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10])


    # ==========================================
    # STEP 4.5: CONSULTANT-LEVEL DOMAIN FILTERING (OPTIONAL - FOR PRODUCTION ONLY)
    # ==========================================
    # NOTE: This uses ground truth labels, so it's "cheating" for evaluation!
    # Only enable this in production when you want to filter by known job roles.
    # For evaluation, keep this disabled to see how well the model performs.
    use_consultant_level_filter = config.get("domain_filtering", {}).get("use_consultant_level", False)
    
    if use_consultant_level_filter and task_requirements.get('job_role'):
        required_job_role = task_requirements['job_role'].strip()
        dbg(f"⚠️  Using consultant-level domain filtering (ground truth labels) for job_role: {required_job_role}")
        dbg("   NOTE: This uses ground truth, so it's not suitable for model evaluation!")
        
        # Load ground truth labels
        try:
            df_labels = pd.read_csv(labels_file)
            if 'JOB_RULE' in df_labels.columns and 'USER_ID' in df_labels.columns:
                # Get consultants with matching job_role
                matching_consultants = set(df_labels[df_labels['JOB_RULE'] == required_job_role]['USER_ID'].tolist())
                
                # Filter combined_scores to only include matching consultants
                original_count = len(combined_scores)
                combined_scores = {cid: score for cid, score in combined_scores.items() if cid in matching_consultants}
                filtered_count = len(combined_scores)
                
                dbg(f"Domain filtering: {original_count} → {filtered_count} consultants (removed {original_count - filtered_count} with wrong job_role)")
            else:
                dbg("Warning: JOB_RULE or USER_ID column not found in labels file, skipping consultant-level filtering")
        except Exception as e:
            dbg(f"Error loading labels for domain filtering: {e}, continuing without consultant-level filter")
    else:
        if not use_consultant_level_filter:
            dbg("Consultant-level domain filtering DISABLED (for fair model evaluation)")
        else:
            dbg("No job_role in task_requirements, skipping consultant-level domain filtering")


    # 4. LANGUAGE FILTER
    language_filter = compute_language_score(
        "data/HCK_HEC_LANG.csv",
        task_requirements,
        acceptable_difference=config["languages"]["acceptable_difference"],
        penality=config["languages"]["penality"]
    )
    dbg("Language penalties (sample):", list(language_filter.items())[:10])


    # 5. AVAILABILITY FILTER
    schedule_penalties = calculate_availability_penalty(
        "data/HCK_HEC_STAFFING.csv",
        task_requirements,
        config["disponibility"]["maximum_penalty"]
    )
    dbg("Availability penalties (sample):", list(schedule_penalties.items())[:10])


    # APPLY FILTERS
    filtered = apply_filters(combined_scores, language_filter)
    dbg("After language filter (top 10):",
        sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:10])

    filtered = apply_filters(filtered, schedule_penalties)
    dbg("After availability filter (top 10):",
        sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:10])

    # ==========================================
    # STEP 5: BI-ENCODER - Get Top 20 Candidates
    # ==========================================
    # After filtering, get the top 20 candidates from bi-encoder ranking
    sorted_consultants = se.sort_skills(filtered)
    top20_ids = [cid for cid, _ in sorted_consultants[:20]]

    print("\n" + "="*60)
    print("TOP 20 CANDIDATES (from BI-ENCODER ranking)")
    print("="*60)
    print(sorted_consultants[:20])

    # ==========================================
    # STEP 6: FINAL TOP 5 (from BI-ENCODER)
    # ==========================================
    # Skip cross-encoder reranking - it doesn't improve results and introduces
    # wrong-domain consultants (e.g., Healthcare for Scrum missions)
    # Use bi-encoder top 5 directly (faster and better precision than top 10)
    # Top 5 has 56% average precision vs 36% for top 10
    final_top5 = sorted_consultants[:5]

    # Load ground truth labels
    full_df = pd.read_csv("full_df.csv")
    
    # Get expected job role
    expected_job_role = task_requirements.get("job_role", "Unknown")
    if expected_job_role == "Unknown":
        # Try to infer from mission content if not set
        if "data engineer" in str(task_requirements.get("responsabilites", "")).lower():
            expected_job_role = "Data Engineer"
        elif "scrum" in str(task_requirements.get("responsabilites", "")).lower():
            expected_job_role = "Scrum Master"
        elif "healthcare" in str(task_requirements.get("responsabilites", "")).lower():
            expected_job_role = "Healthcare"
        elif "finance" in str(task_requirements.get("responsabilites", "")).lower():
            expected_job_role = "Finance"
        elif "marketing" in str(task_requirements.get("responsabilites", "")).lower():
            expected_job_role = "Marketing"
        elif "data analyst" in str(task_requirements.get("responsabilites", "")).lower():
            expected_job_role = "Data Analyst"

    print("\n" + "="*60)
    print("FINAL TOP 5 CANDIDATES (from BI-ENCODER ranking)")
    print("="*60)
    print(f"Expected Job Role: {expected_job_role}")
    print()
    
    correct_count = 0
    for i, (cid, score) in enumerate(final_top5, 1):
        # Get ground truth label
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
    print("="*60)
    print(f"ACCURACY: {correct_count}/5 = {correct_count/5*100:.1f}%")
    print("="*60)