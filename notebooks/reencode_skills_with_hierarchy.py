#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from helpers.SkillsEmbeddings import SkillEmbeddingModel


def reencode_skills_with_hierarchical_model(
    input_csv: str,
    output_npz: str,
    model_path: str = "skills_finetuned_hierarchy",
    config_path: str = "config/search_config.yaml"
):
    print("RE-ENCODING SKILLS WITH HIERARCHICAL MODEL")
    
    print(f"\n[1/4] Loading fine-tuned hierarchical model: {model_path}...")
    try:
        base_model = SentenceTransformer(model_path)
        skill_model = SkillEmbeddingModel(base_model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've run finetune_skill_hierarchy.py first!")
        return False
    
    print(f"\n[2/4] Loading skills from: {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} skill records")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    print(f"\n[3/4] Re-encoding skills with hierarchical model...")
    
    skill_embeddings = []
    domain_embeddings = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        skill_name = str(row['SKILLS_DSC'])
        domain_name = str(row['DOMAIN_DSC'])
        
        skill_emb = skill_model.encode([skill_name])[0]
        domain_emb = skill_model.encode([domain_name])[0]
        
        skill_embeddings.append(skill_emb)
        domain_embeddings.append(domain_emb)
    
    print(f"Encoded {len(skill_embeddings)} skills")
    
    print(f"\n[4/4] Saving embeddings to: {output_npz}...")
    skill_embs = np.stack(skill_embeddings)
    domain_embs = np.stack(domain_embeddings)
    metadata = df[['USER_ID', 'LEVEL_VAL', 'SKILLS_DSC']].values
    
    np.savez(output_npz, skill_embs=skill_embs, domain_embs=domain_embs, metadata=metadata)
    print(f"Saved embeddings to {output_npz}")
    
    return True


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Re-encode skills with hierarchical model")
    parser.add_argument("--model", default="skills_finetuned_hierarchy",
                       help="Path to fine-tuned hierarchical model")
    parser.add_argument("--input", default=None,
                       help="Path to input skills CSV (default: from config)")
    parser.add_argument("--output", default=None,
                       help="Path to output .npz file (default: from config)")
    parser.add_argument("--config", default="config/search_config.yaml",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        input_csv = args.input or config.get("skills_csv", "data/HCK_HEC_SKILLS.csv")
        output_npz = args.output or config.get("skills_embedding_file", "data/skills_encoded.npz")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        input_csv = args.input or "data/HCK_HEC_SKILLS.csv"
        output_npz = args.output or "data/skills_encoded.npz"
    
    if not os.path.isabs(input_csv):
        input_csv = os.path.join(project_root, input_csv)
    if not os.path.isabs(output_npz):
        output_npz = os.path.join(project_root, output_npz)
    
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    
    success = reencode_skills_with_hierarchical_model(
        input_csv=input_csv,
        output_npz=output_npz,
        model_path=args.model,
        config_path=args.config
    )
    
    sys.exit(0 if success else 1)

