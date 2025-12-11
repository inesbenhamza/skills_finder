#!/usr/bin/env python3
"""
Compare mission scores between base and fine-tuned models
"""
import sys
import yaml
import pandas as pd
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict

# Load mission requirements
mission_file = "output/filtered_skills/DataAnalyst.yaml"
with open(mission_file, 'r') as f:
    task_requirements = yaml.safe_load(f)

print("="*80)
print("COMPARING MISSION SCORES: Base vs Fine-tuned Model")
print("="*80)

# Test with base model
print("\n1. BASE MODEL (paraphrase-multilingual-mpnet-base-v2):")
base_scores_df = get_consultants_mission_scores(
    task_requirements,
    percentile=90,
    use_domain_keyword_filtering=False,
    finetune_model="paraphrase-multilingual-mpnet-base-v2"
)
base_scores = df_to_dict(base_scores_df)
base_top10 = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:10]
print("   Top 10 mission scores:")
for cid, score in base_top10:
    print(f"      {cid}: {score:.4f}")

# Test with fine-tuned model
print("\n2. FINE-TUNED MODEL (models/modelfinetuned_domain):")
finetuned_scores_df = get_consultants_mission_scores(
    task_requirements,
    percentile=90,
    use_domain_keyword_filtering=False,
    finetune_model="models/modelfinetuned_domain"
)
finetuned_scores = df_to_dict(finetuned_scores_df)
finetuned_top10 = sorted(finetuned_scores.items(), key=lambda x: x[1], reverse=True)[:10]
print("   Top 10 mission scores:")
for cid, score in finetuned_top10:
    print(f"      {cid}: {score:.4f}")

# Compare
print("\n3. COMPARISON:")
print("   Top 10 consultant IDs (base model):", [cid for cid, _ in base_top10])
print("   Top 10 consultant IDs (fine-tuned):", [cid for cid, _ in finetuned_top10])

if [cid for cid, _ in base_top10] == [cid for cid, _ in finetuned_top10]:
    print("\n   ✅ Same top 10 consultants (same ranking)")
else:
    print("\n   ⚠️  Different top 10 consultants (different ranking)")

# Check score differences
print("\n4. SCORE DIFFERENCES (for top 10 base model consultants):")
for cid, base_score in base_top10:
    finetuned_score = finetuned_scores.get(cid, 0.0)
    diff = finetuned_score - base_score
    print(f"   {cid}: base={base_score:.4f}, finetuned={finetuned_score:.4f}, diff={diff:+.4f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if [cid for cid, _ in base_top10] == [cid for cid, _ in finetuned_top10]:
    print("Fine-tuned model produces SAME ranking as base model.")
    print("This means fine-tuning didn't improve mission similarity matching.")
    print("Possible reasons:")
    print("  - Skills dominate the final ranking")
    print("  - Fine-tuning didn't help for these specific missions")
    print("  - Base model already works well for domain matching")
else:
    print("Fine-tuned model produces DIFFERENT ranking.")
    print("But if final results are the same, skills must be dominating.")
print("="*80)
