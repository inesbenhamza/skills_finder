
"""
Re-encode skills using BAAI/bge-m3 model.
"""

import os
import sys
import yaml
from sentence_transformers import SentenceTransformer
from helpers.SkillsEmbeddings import SkillEmbeddingModel, save_skills_emb

def main():

    config_path = "config/search_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    input_csv = config["skills_csv"]
    output_npz = "skills/skills_encoded_bge-m3.npz"
    model_name = "BAAI/bge-m3"
    
    print("RE-ENCODING SKILLS WITH BAAI/bge-m3 MODEL")
    print(f"\nModel: {model_name}")
    print(f"Input CSV: {input_csv}")
    print(f"Output NPZ: {output_npz}")
    
    # Load model
    print(f"\n[1/3] Loading model: {model_name}...")
    try:
        base_model = SentenceTransformer(model_name)
        skill_model = SkillEmbeddingModel(base_model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return False
    
    print(f"\n[2/3] Re-encoding skills...")
    print("  This may take a few minutes...")
    try:
        save_skills_emb(input_csv, output_npz, skill_model)
        print("[OK] Skills re-encoded successfully")
    except Exception as e:
        print(f"[ERROR] Error re-encoding: {e}")
        return False
    
    print(f"\nSkills are now encoded with: {model_name}")
    print(f"   Output file: {output_npz}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

