#!/usr/bin/env python3
"""
Fine-tune skill embeddings based on ROLE-BASED relationships (not vendor hierarchy).

This script:
1. Loads ground truth data (full_df.csv) to identify which skills belong to which roles
2. Creates positive pairs: skills that appear together in the same role
3. Creates negative pairs: skills from different roles
4. Fine-tunes embeddings to make role-based skills similar
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
import ast
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def extract_skills_from_full_df(full_df_path: str, skills_csv: str) -> Dict[str, List[str]]:
    """
    Extract skills per role from full_df.csv.
    
    Returns:
        Dict mapping role -> list of skills that appear in that role
    """
    print(f"\n[1/6] Loading data...")
    full_df = pd.read_csv(full_df_path)
    skills_df = pd.read_csv(skills_csv)
    
    all_skills = set(skills_df['SKILLS_DSC'].unique())
    
    role_skills = {}
    
    for idx, row in full_df.iterrows():
        role = str(row['JOB_RULE']).strip()
        if role == 'nan' or role == '':
            continue
        
        skills_in_role = set()
        
        for col in ['Data Analytics', 'Data Integration', 'Data Management', 
                   'Data Platform', 'Data Science', 'Programming Language']:
            if pd.notna(row[col]) and row[col] != '':
                try:
                    skills_dict = ast.literal_eval(str(row[col]))
                    if isinstance(skills_dict, dict):
                        for skill_name in skills_dict.keys():
                            skill_name = str(skill_name).strip()
                            matching_skill = None
                            for s in all_skills:
                                if skill_name.lower() == s.lower() or skill_name.lower() in s.lower() or s.lower() in skill_name.lower():
                                    matching_skill = s
                                    break
                            if matching_skill:
                                skills_in_role.add(matching_skill)
                except:
                    pass
        
        if skills_in_role:
            if role not in role_skills:
                role_skills[role] = set()
            role_skills[role].update(skills_in_role)
    
    role_skills = {role: list(skills) for role, skills in role_skills.items()}
    
    print(f"✓ Found {len(role_skills)} roles with skills")
    for role, skills in role_skills.items():
        print(f"  {role}: {len(skills)} skills")
    
    return role_skills


def build_role_based_pairs(role_skills: Dict[str, List[str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Build positive and negative pairs based on role relationships.
    
    Positive pairs: Skills that appear together in the same role
    Negative pairs: Skills from different roles
    """
    print(f"\n[2/6] Building role-based training pairs...")
    
    positive_pairs = []
    negative_pairs = []
    
    for role, skills in role_skills.items():
        skills_list = list(skills)
        for i, skill1 in enumerate(skills_list):
            for skill2 in skills_list[i+1:]:
                positive_pairs.append((skill1, skill2))
    
    roles_list = list(role_skills.keys())
    for i, role1 in enumerate(roles_list):
        for role2 in roles_list[i+1:]:
            skills1 = role_skills[role1]
            skills2 = role_skills[role2]
            for skill1 in skills1:
                for skill2 in skills2:
                    negative_pairs.append((skill1, skill2))
    
    print(f"✓ Created {len(positive_pairs)} positive pairs (skills in same role)")
    print(f"✓ Created {len(negative_pairs)} negative pairs (skills in different roles)")
    
    max_negatives = len(positive_pairs) * 5
    if len(negative_pairs) > max_negatives:
        np.random.seed(42)
        negative_pairs = np.random.choice(len(negative_pairs), max_negatives, replace=False)
        negative_pairs = [negative_pairs[i] for i in negative_pairs]
        print(f"✓ Limited to {len(negative_pairs)} negative pairs (5:1 ratio)")
    
    return positive_pairs, negative_pairs


def create_training_examples(positive_pairs: List[Tuple[str, str]], 
                            negative_pairs: List[Tuple[str, str]]) -> List[InputExample]:
    """Create InputExample objects for training."""
    examples = []
    
    for skill1, skill2 in positive_pairs:
        examples.append(InputExample(texts=[skill1, skill2], label=1.0))
    
    for skill1, skill2 in negative_pairs:
        examples.append(InputExample(texts=[skill1, skill2], label=0.0))
    
    return examples


def main():
    """Main fine-tuning function."""
    print("FINE-TUNING SKILL EMBEDDINGS FOR ROLE-BASED RELATIONSHIPS")
    
    config_path = os.path.join(project_root, "config/search_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    full_df_path = os.path.join(project_root, "full_df.csv")
    skills_csv = config["skills_csv"]
    
    role_skills = extract_skills_from_full_df(full_df_path, skills_csv)
    
    positive_pairs, negative_pairs = build_role_based_pairs(role_skills)
    print(f"\n[3/6] Creating training examples...")
    train_examples = create_training_examples(positive_pairs, negative_pairs)
    print(f"✓ Created {len(train_examples)} training examples")
    
    print("\n📋 Sample positive pairs (same role):")
    for i, (skill1, skill2) in enumerate(positive_pairs[:5]):
        print(f"   {i+1}. '{skill1}' <-> '{skill2}'")
    
    print("\n📋 Sample negative pairs (different roles):")
    for i, (skill1, skill2) in enumerate(negative_pairs[:5]):
        print(f"   {i+1}. '{skill1}' <-> '{skill2}'")
    
    print(f"\n[4/6] Loading base embedding model...")
    base_model = config.get("embedding_model", "paraphrase-multilingual-mpnet-base-v2")
    model = SentenceTransformer(base_model)
    print(f"✓ Loaded: {base_model}")
    
    print(f"\n[5/6] Fine-tuning model...")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    train_loss = losses.ContrastiveLoss(
        model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        margin=1.0
    )
    
    print(f"Training with {len(train_examples)} examples")
    print(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")
    print(f"Using ContrastiveLoss with margin=1.0")
    print("This will make skills from the same role close together in embedding space!")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        optimizer_params={'lr': 2e-5},
        show_progress_bar=True,
        use_amp=False
    )
    
    output_path = os.path.join(project_root, "skills_finetuned_roles")
    model.save(output_path)
    print(f"\n[6/6] ✓ Saved fine-tuned model to: {output_path}")
    
    print("\nROLE-BASED FINE-TUNING COMPLETE!")
    print("\nNext steps:")
    print("   1. Update config/search_config.yaml:")
    print(f"      embedding_model: skills_finetuned_roles")
    print("   2. Re-encode skills with the new model")
    print("   3. Test the matching to see role-based relationships working!")


if __name__ == "__main__":
    main()

