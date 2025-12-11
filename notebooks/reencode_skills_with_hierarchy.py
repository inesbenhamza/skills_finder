#!/usr/bin/env python3
"""
Re-encode skills using the fine-tuned hierarchical model.

This script:
1. Loads the fine-tuned hierarchical model
2. Re-encodes all skills from the CSV
3. Saves new embeddings that respect the hierarchical structure
"""

import os
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from helpers.SkillsEmbeddings import SkillEmbeddingModel


def reencode_skills_with_hierarchical_model(
    input_csv: str,
    output_npz: str,
    model_path: str = "skills_finetuned_hierarchy",
    config_path: str = "config/search_config.yaml"
):
    """
    Re-encode skills using the fine-tuned hierarchical model.
    
    Args:
        input_csv: Path to skills CSV (e.g., "data/HCK_HEC_SKILLS.csv")
        output_npz: Path to output .npz file
        model_path: Path to fine-tuned hierarchical model
        config_path: Path to config file (to get original paths)
    """
    print("=" * 80)
    print("🔄 RE-ENCODING SKILLS WITH HIERARCHICAL MODEL")
    print("=" * 80)
    
    # Step 1: Load fine-tuned model
    print(f"\n[1/4] Loading fine-tuned hierarchical model: {model_path}...")
    try:
        base_model = SentenceTransformer(model_path)
        skill_model = SkillEmbeddingModel(base_model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Make sure you've run finetune_skill_hierarchy.py first!")
        return False
    
    # Step 2: Load skills CSV
    print(f"\n[2/4] Loading skills from: {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df)} skill records")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Step 3: Re-encode skills
    print(f"\n[3/4] Re-encoding skills with hierarchical model...")
    print("  This may take a few minutes...")
    
    # Encode skills
    skill_embeddings = []
    domain_embeddings = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        skill_name = str(row['SKILLS_DSC'])
        domain_name = str(row['DOMAIN_DSC'])
        
        # Encode using hierarchical model
        skill_emb = skill_model.encode([skill_name])[0]
        domain_emb = skill_model.encode([domain_name])[0]
        
        skill_embeddings.append(skill_emb)
        domain_embeddings.append(domain_emb)
    
    print(f"✓ Encoded {len(skill_embeddings)} skills")
    
    # Step 4: Save embeddings
    print(f"\n[4/4] Saving embeddings to: {output_npz}...")
    skill_embs = np.stack(skill_embeddings)
    domain_embs = np.stack(domain_embeddings)
    metadata = df[['USER_ID', 'LEVEL_VAL', 'SKILLS_DSC']].values
    
    np.savez(output_npz, skill_embs=skill_embs, domain_embs=domain_embs, metadata=metadata)
    print(f"✓ Saved embeddings to {output_npz}")
    
    print("\n" + "=" * 80)
    print("✅ RE-ENCODING COMPLETE!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. Update config/search_config.yaml:")
    print(f"      embedding_model: {model_path}")
    print(f"      skills_embedding_file: {output_npz}")
    print("   2. Test the matching to see hierarchical relationships working!")
    
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
    
    # Load config to get default paths
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        input_csv = args.input or config.get("skills_csv", "data/HCK_HEC_SKILLS.csv")
        output_npz = args.output or config.get("skills_embedding_file", "data/skills_encoded.npz")
    except Exception as e:
        print(f"⚠️  Warning: Could not load config: {e}")
        input_csv = args.input or "data/HCK_HEC_SKILLS.csv"
        output_npz = args.output or "data/skills_encoded.npz"
    
    # Convert relative paths to absolute
    if not os.path.isabs(input_csv):
        input_csv = os.path.join(project_root, input_csv)
    if not os.path.isabs(output_npz):
        output_npz = os.path.join(project_root, output_npz)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    
    success = reencode_skills_with_hierarchical_model(
        input_csv=input_csv,
        output_npz=output_npz,
        model_path=args.model,
        config_path=args.config
    )
    
    sys.exit(0 if success else 1)

