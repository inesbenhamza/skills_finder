#!/usr/bin/env python3
"""
Comparison script to evaluate the effect of fine-tuning on ranking.

Tests all 4 combinations:
1. Skills: base + Mission: base
2. Skills: base + Mission: fine-tuned
3. Skills: fine-tuned + Mission: base
4. Skills: fine-tuned + Mission: fine-tuned
"""

import yaml
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer
import helpers.SkillsEmbeddings as se
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict
from src.language_filtering import compute_language_score, apply_filters
from src.schedule_filtering import calculate_availability_penalty
from mainfile.main_debug import linear_combination

# Test configurations
CONFIGURATIONS = [
    {
        "name": "Skills: Base | Mission: Base",
        "skills_model": "BAAI/bge-m3",
        "mission_model": "paraphrase-multilingual-mpnet-base-v2",
        "skills_embedding_file": "skills-finder-hackathon-hec-x-sbi/skills_encoded_base.npz"  # Will need to be generated
    },
    {
        "name": "Skills: Base | Mission: Fine-tuned",
        "skills_model": "BAAI/bge-m3",
        "mission_model": "models/modelfinetuned_domain",
        "skills_embedding_file": "skills-finder-hackathon-hec-x-sbi/skills_encoded_base.npz"
    },
    {
        "name": "Skills: Fine-tuned | Mission: Base",
        "skills_model": "skills_finetuned_hierarchy",
        "mission_model": "paraphrase-multilingual-mpnet-base-v2",
        "skills_embedding_file": "skills-finder-hackathon-hec-x-sbi/skills_encoded.npz"  # Existing fine-tuned embeddings
    },
    {
        "name": "Skills: Fine-tuned | Mission: Fine-tuned",
        "skills_model": "skills_finetuned_hierarchy",
        "mission_model": "models/modelfinetuned_domain",
        "skills_embedding_file": "skills-finder-hackathon-hec-x-sbi/skills_encoded.npz"
    }
]


def generate_base_skills_embeddings():
    """Generate embeddings using base BGE-M3 model if they don't exist."""
    base_emb_file = "skills-finder-hackathon-hec-x-sbi/skills_encoded_base.npz"
    
    if os.path.exists(base_emb_file):
        print(f"✓ Base skills embeddings already exist: {base_emb_file}")
        return
    
    print("📦 Generating base skills embeddings with BAAI/bge-m3...")
    from notebooks.reencode_skills_with_hierarchy import reencode_skills_with_hierarchical_model
    
    reencode_skills_with_hierarchical_model(
        input_csv="skills-finder-hackathon-hec-x-sbi/skillscleaned.csv",
        output_npz=base_emb_file,
        model_path="BAAI/bge-m3"  # Base model
    )
    print(f"✓ Generated base skills embeddings: {base_emb_file}")


def run_comparison(filename: str, output_dir: str = "output/comparison"):
    """
    Run matching pipeline with all 4 configurations and compare results.
    
    Args:
        filename: Name of the PDF file (without .pdf extension)
        output_dir: Directory to save comparison results
    """
    # Load base config
    with open("config/search_config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Generate base skills embeddings if needed
    generate_base_skills_embeddings()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load task requirements
    yaml_path = f"output/filtered_skills/{filename}.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ Error: YAML file not found: {yaml_path}")
        print("   Please run the main pipeline first to generate the YAML file.")
        return
    
    with open(yaml_path, 'r') as f:
        task_requirements = yaml.safe_load(f)
    
    # Load ground truth labels for accuracy calculation
    labels_file = base_config.get("weight_optimization", {}).get("labels_file", "full_df.csv")
    if not os.path.exists(labels_file):
        print(f"⚠️  Warning: Labels file not found: {labels_file}")
        print("   Accuracy calculation will be skipped.")
        full_df = None
    else:
        full_df = pd.read_csv(labels_file)
        if 'JOB_RULE' not in full_df.columns or 'USER_ID' not in full_df.columns:
            print(f"⚠️  Warning: Labels file missing JOB_RULE or USER_ID columns")
            full_df = None
        else:
            print(f"✓ Loaded ground truth labels from: {labels_file}")
    
    # Get expected job role
    expected_job_role = task_requirements.get("job_role", "Unknown")
    if expected_job_role == "Unknown":
        # Try to infer from filename or content
        if "data engineer" in filename.lower():
            expected_job_role = "Data Engineer"
        elif "scrum" in filename.lower():
            expected_job_role = "Scrum Master"
        elif "healthcare" in filename.lower():
            expected_job_role = "Healthcare"
        elif "finance" in filename.lower():
            expected_job_role = "Finance"
        elif "marketing" in filename.lower():
            expected_job_role = "Marketing"
        elif "data analyst" in filename.lower():
            expected_job_role = "Data Analyst"
    
    print(f"Expected Job Role: {expected_job_role}")
    
    print("=" * 80)
    print(f"COMPARING FINE-TUNING EFFECTS FOR: {filename}")
    print("=" * 80)
    
    all_results = {}
    
    # Run each configuration
    for i, config in enumerate(CONFIGURATIONS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/4] {config['name']}")
        print(f"{'='*80}")
        
        # Create modified config
        test_config = base_config.copy()
        test_config["embedding_model"] = config["skills_model"]
        test_config["mission_embedding_model"] = config["mission_model"]
        test_config["skills_embedding_file"] = config["skills_embedding_file"]
        
        # Run matching pipeline
        try:
            # Load skills model
            print(f"  Loading skills model: {config['skills_model']}")
            skills_bi_encoder = SentenceTransformer(config["skills_model"])
            skill_model = se.SkillEmbeddingModel(skills_bi_encoder)
            
            # Get skill scores
            consultants_scores = se.get_consultants_skills_score(
                task_requirements,
                skill_model,
                config["skills_embedding_file"],
                test_config,
                debug=False
            )
            average_skills_rating = se.average_skills(consultants_scores)
            
            # Get mission scores
            print(f"  Loading mission model: {config['mission_model']}")
            mission_scores_df = get_consultants_mission_scores(
                task_requirements,
                percentile=test_config["mission_context"]["percentile"],
                use_domain_keyword_filtering=False,
                finetune_model=config["mission_model"]
            )
            
            # Convert mission scores to dict
            average_mission_rating = df_to_dict(mission_scores_df)
            
            # Combine scores
            combined_scores = linear_combination(
                average_skills_rating,
                test_config["skills"]["skills_weight"],
                average_mission_rating,
                test_config["mission_context"]["mission_weight"]
            )
            
            # Apply filters
            language_filter = compute_language_score(
                "data/HCK_HEC_LANG.csv",
                task_requirements,
                acceptable_difference=test_config["languages"]["acceptable_difference"],
                penality=test_config["languages"]["penality"]
            )
            
            schedule_penalties = calculate_availability_penalty(
                "data/HCK_HEC_STAFFING.csv",
                task_requirements,
                test_config["disponibility"]["maximum_penalty"]
            )
            
            filtered = apply_filters(combined_scores, language_filter)
            filtered = apply_filters(filtered, schedule_penalties)
            
            # Get top 5
            sorted_consultants = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            top5 = []
            for cid, score in sorted_consultants[:5]:
                top5.append({
                    "user_id": cid,
                    "score": score,
                    "skill_score": average_skills_rating.get(cid, 0),
                    "mission_score": average_mission_rating.get(cid, 0)
                })
            
            # Calculate accuracy if ground truth available
            accuracy = None
            correct_count = 0
            accuracy_details = []
            
            if full_df is not None and expected_job_role != "Unknown":
                for consultant in top5:
                    cid = consultant['user_id']
                    consultant_row = full_df[full_df["USER_ID"] == cid]
                    if len(consultant_row) > 0:
                        ground_truth = consultant_row.iloc[0]["JOB_RULE"]
                        is_correct = str(ground_truth).strip() == str(expected_job_role).strip()
                        if is_correct:
                            correct_count += 1
                        accuracy_details.append({
                            "user_id": cid,
                            "ground_truth": ground_truth,
                            "is_correct": is_correct
                        })
                    else:
                        accuracy_details.append({
                            "user_id": cid,
                            "ground_truth": "NOT FOUND",
                            "is_correct": False
                        })
                accuracy = (correct_count / len(top5)) * 100 if top5 else 0
            
            # Store results
            all_results[config['name']] = {
                "top5": top5,
                "scores": filtered,
                "config": config,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "accuracy_details": accuracy_details
            }
            
            print(f"\n✓ Completed: {config['name']}")
            print(f"  Top 5 consultants: {[c['user_id'] for c in top5]}")
            if accuracy is not None:
                print(f"  Accuracy: {correct_count}/5 = {accuracy:.1f}%")
            
        except Exception as e:
            print(f"❌ Error with {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results[config['name']] = {"error": str(e)}
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Create comparison DataFrame
    comparison_data = []
    for config_name, results in all_results.items():
        if "error" in results:
            continue
        
        top5 = results.get("top5", [])
        accuracy_details = results.get("accuracy_details", [])
        
        for rank, consultant in enumerate(top5[:5], 1):
            detail = accuracy_details[rank - 1] if rank <= len(accuracy_details) else {}
            comparison_data.append({
                "Configuration": config_name,
                "Rank": rank,
                "User ID": consultant.get("user_id", "N/A"),
                "Score": consultant.get("score", 0),
                "Skill Score": consultant.get("skill_score", 0),
                "Mission Score": consultant.get("mission_score", 0),
                "Ground Truth": detail.get("ground_truth", "N/A"),
                "Correct": "✓" if detail.get("is_correct", False) else "✗"
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save to CSV
        output_file = f"{output_dir}/{filename}_comparison.csv"
        df_comparison.to_csv(output_file, index=False)
        print(f"\n✓ Comparison saved to: {output_file}")
        
        # Display accuracy comparison table
        print("\n" + "=" * 80)
        print("ACCURACY COMPARISON (Top-5 Accuracy)")
        print("=" * 80)
        print(f"Expected Job Role: {expected_job_role}")
        print("-" * 80)
        
        accuracy_summary = []
        for config_name, results in all_results.items():
            if "error" in results:
                continue
            accuracy = results.get("accuracy")
            correct_count = results.get("correct_count", 0)
            if accuracy is not None:
                accuracy_summary.append({
                    "Configuration": config_name,
                    "Accuracy": f"{accuracy:.1f}%",
                    "Correct": f"{correct_count}/5"
                })
        
        if accuracy_summary:
            df_accuracy = pd.DataFrame(accuracy_summary)
            print(df_accuracy.to_string(index=False))
            
            # Find best configuration
            best_config = max(accuracy_summary, key=lambda x: float(x["Accuracy"].replace("%", "")))
            print(f"\n🏆 Best Configuration: {best_config['Configuration']} ({best_config['Accuracy']})")
        
        # Display top 5 by configuration
        print("\n" + "=" * 80)
        print("TOP 5 CONSULTANTS BY CONFIGURATION")
        print("=" * 80)
        for config_name in all_results.keys():
            if "error" in all_results[config_name]:
                continue
            top5 = all_results[config_name].get("top5", [])
            accuracy_details = all_results[config_name].get("accuracy_details", [])
            accuracy = all_results[config_name].get("accuracy")
            
            user_ids = [str(c.get("user_id", "N/A")) for c in top5[:5]]
            accuracy_str = f" ({accuracy:.1f}%)" if accuracy is not None else ""
            print(f"\n{config_name}{accuracy_str}:")
            for i, (cid, detail) in enumerate(zip(user_ids, accuracy_details[:5]), 1):
                gt = detail.get("ground_truth", "N/A")
                correct = "✓" if detail.get("is_correct", False) else "✗"
                print(f"  {i}. Consultant {cid}: {gt} {correct}")
        
        # Check for differences and accuracy impact
        print("\n" + "=" * 80)
        print("ANALYSIS: Fine-tuning Impact on Accuracy")
        print("=" * 80)
        
        # Compare base vs fine-tuned for skills
        base_skills_base_mission = all_results.get("Skills: Base | Mission: Base", {})
        ft_skills_base_mission = all_results.get("Skills: Fine-tuned | Mission: Base", {})
        
        if base_skills_base_mission.get("top5") and ft_skills_base_mission.get("top5"):
            base_acc = base_skills_base_mission.get("accuracy")
            ft_acc = ft_skills_base_mission.get("accuracy")
            base_top5_ids = [c.get("user_id") for c in base_skills_base_mission["top5"][:5]]
            ft_top5_ids = [c.get("user_id") for c in ft_skills_base_mission["top5"][:5]]
            
            print("\n1. Skills Fine-tuning Impact (with base mission model):")
            if base_acc is not None and ft_acc is not None:
                acc_diff = ft_acc - base_acc
                print(f"   Base model accuracy:    {base_acc:.1f}%")
                print(f"   Fine-tuned accuracy:     {ft_acc:.1f}%")
                print(f"   Improvement:            {acc_diff:+.1f}%")
            print(f"   Base model top 5:        {base_top5_ids}")
            print(f"   Fine-tuned top 5:        {ft_top5_ids}")
            overlap = len(set(base_top5_ids) & set(ft_top5_ids))
            print(f"   Overlap:                 {overlap}/5 consultants")
        
        # Compare base vs fine-tuned for missions
        base_skills_base_mission = all_results.get("Skills: Base | Mission: Base", {})
        base_skills_ft_mission = all_results.get("Skills: Base | Mission: Fine-tuned", {})
        
        if base_skills_base_mission.get("top5") and base_skills_ft_mission.get("top5"):
            base_acc = base_skills_base_mission.get("accuracy")
            ft_acc = base_skills_ft_mission.get("accuracy")
            base_top5_ids = [c.get("user_id") for c in base_skills_base_mission["top5"][:5]]
            ft_top5_ids = [c.get("user_id") for c in base_skills_ft_mission["top5"][:5]]
            
            print("\n2. Mission Fine-tuning Impact (with base skills model):")
            if base_acc is not None and ft_acc is not None:
                acc_diff = ft_acc - base_acc
                print(f"   Base model accuracy:    {base_acc:.1f}%")
                print(f"   Fine-tuned accuracy:    {ft_acc:.1f}%")
                print(f"   Improvement:           {acc_diff:+.1f}%")
            print(f"   Base model top 5:        {base_top5_ids}")
            print(f"   Fine-tuned top 5:        {ft_top5_ids}")
            overlap = len(set(base_top5_ids) & set(ft_top5_ids))
            print(f"   Overlap:                 {overlap}/5 consultants")
        
        # Compare all fine-tuned vs all base
        all_base = all_results.get("Skills: Base | Mission: Base", {})
        all_ft = all_results.get("Skills: Fine-tuned | Mission: Fine-tuned", {})
        
        if all_base.get("top5") and all_ft.get("top5"):
            base_acc = all_base.get("accuracy")
            ft_acc = all_ft.get("accuracy")
            base_top5_ids = [c.get("user_id") for c in all_base["top5"][:5]]
            ft_top5_ids = [c.get("user_id") for c in all_ft["top5"][:5]]
            
            print("\n3. Combined Fine-tuning Impact:")
            if base_acc is not None and ft_acc is not None:
                acc_diff = ft_acc - base_acc
                print(f"   All base accuracy:      {base_acc:.1f}%")
                print(f"   All fine-tuned accuracy: {ft_acc:.1f}%")
                print(f"   Improvement:            {acc_diff:+.1f}%")
            print(f"   All base top 5:          {base_top5_ids}")
            print(f"   All fine-tuned top 5:    {ft_top5_ids}")
            overlap = len(set(base_top5_ids) & set(ft_top5_ids))
            print(f"   Overlap:                 {overlap}/5 consultants")
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_finetuning.py <filename>")
        print("Example: python compare_finetuning.py Healthcare")
        sys.exit(1)
    
    filename = sys.argv[1].replace(".yaml", "").replace(".pdf", "")
    run_comparison(filename)

