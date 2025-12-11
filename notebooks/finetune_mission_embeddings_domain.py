"""
Fine-tune Mission Embeddings with Domain-Specific Examples

This script fine-tunes mission embeddings to better distinguish between different job roles:
- Scrum Master vs Data Engineer vs Healthcare vs Finance vs Data Analyst vs Marketing

Uses contrastive learning to make same-domain missions similar and different-domain missions dissimilar.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

print("=" * 80)
print("🚀 FINE-TUNING MISSION EMBEDDINGS FOR DOMAIN DISTINCTION")
print("=" * 80)
print()

# ==========================================
# 1. Load Base Model
# ==========================================
print("📦 Loading base model...")
# Use modelfinetuned_domain as base (or fallback to paraphrase-multilingual-mpnet-base-v2)
import os
if os.path.exists("models/modelfinetuned_domain"):
    base_model = "models/modelfinetuned_domain"  # Continue from existing domain model
    print("   Using existing modelfinetuned_domain as base")
else:
    base_model = "paraphrase-multilingual-mpnet-base-v2"  # Fallback to base model
    print("   Using paraphrase-multilingual-mpnet-base-v2 as base")

# Force CPU to avoid MPS memory issues
import torch
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Explicitly set device to CPU
device = 'cpu'
print(f"   Using device: {device} (to avoid MPS memory issues)")

model = SentenceTransformer(base_model, device=device)
print(f"✅ Loaded: {base_model}")
print(f"   Model dimension: {model.get_sentence_embedding_dimension()}")
print()

# ==========================================
# 2. Create Domain-Specific Training Examples
# ==========================================

print("📝 Creating training examples from REAL mission data...")

# Load real mission data
df_full = pd.read_csv(os.path.join(project_root, 'full_df.csv'))
missions_df = pd.read_csv(os.path.join(project_root, 'data/HCK_HEC_XP.csv'))

# Get missions by job role
scrum_ids = df_full[df_full['JOB_RULE'] == 'Scrum Master']['USER_ID'].tolist()
de_ids = df_full[df_full['JOB_RULE'] == 'Data Engineer']['USER_ID'].tolist()
healthcare_ids = df_full[df_full['JOB_RULE'] == 'Healthcare']['USER_ID'].tolist()
finance_ids = df_full[df_full['JOB_RULE'] == 'Finance']['USER_ID'].tolist()
analyst_ids = df_full[df_full['JOB_RULE'] == 'Data Analyst']['USER_ID'].tolist()
marketing_ids = df_full[df_full['JOB_RULE'] == 'Marketing']['USER_ID'].tolist()

scrum_missions = missions_df[missions_df['USER_ID'].isin(scrum_ids)]['MISSION_DSC'].dropna().tolist()
de_missions = missions_df[missions_df['USER_ID'].isin(de_ids)]['MISSION_DSC'].dropna().tolist()
healthcare_missions = missions_df[missions_df['USER_ID'].isin(healthcare_ids)]['MISSION_DSC'].dropna().tolist()
finance_missions = missions_df[missions_df['USER_ID'].isin(finance_ids)]['MISSION_DSC'].dropna().tolist()
analyst_missions = missions_df[missions_df['USER_ID'].isin(analyst_ids)]['MISSION_DSC'].dropna().tolist()
marketing_missions = missions_df[missions_df['USER_ID'].isin(marketing_ids)]['MISSION_DSC'].dropna().tolist()

print(f"   Found {len(scrum_missions)} Scrum missions")
print(f"   Found {len(de_missions)} Data Engineer missions")
print(f"   Found {len(healthcare_missions)} Healthcare missions")
print(f"   Found {len(finance_missions)} Finance missions")
print(f"   Found {len(analyst_missions)} Data Analyst missions")
print(f"   Found {len(marketing_missions)} Marketing missions")
print()

# Positive pairs: Same domain missions should be similar
# Use REAL missions from your dataset!
positive_pairs = []

# Helper function to create pairs from mission list
def create_pairs_from_missions(mission_list, max_pairs=50):
    """Create pairs from real missions (same domain)"""
    pairs = []
    # Sample missions and pair them
    import random
    random.seed(42)
    # Limit to reasonable number to avoid memory issues
    max_samples = min(len(mission_list), max_pairs * 2, 100)  # Cap at 100 missions
    sampled = random.sample(mission_list, max_samples)
    for i in range(0, len(sampled) - 1, 2):
        if i + 1 < len(sampled):
            # Clean mission text (remove very long missions)
            m1, m2 = sampled[i], sampled[i + 1]
            if len(m1) < 500 and len(m2) < 500:  # Skip very long missions
                pairs.append((m1, m2))
    return pairs[:max_pairs]  # Limit pairs

# Create positive pairs from real missions (reduced for memory)
if len(scrum_missions) >= 2:
    scrum_pairs = create_pairs_from_missions(scrum_missions, max_pairs=15)  # Reduced
    positive_pairs.extend(scrum_pairs)
    print(f"   ✅ Created {len(scrum_pairs)} Scrum positive pairs")

if len(de_missions) >= 2:
    de_pairs = create_pairs_from_missions(de_missions, max_pairs=15)  # Reduced
    positive_pairs.extend(de_pairs)
    print(f"   ✅ Created {len(de_pairs)} Data Engineer positive pairs")

if len(healthcare_missions) >= 2:
    healthcare_pairs = create_pairs_from_missions(healthcare_missions, max_pairs=10)  # Reduced
    positive_pairs.extend(healthcare_pairs)
    print(f"   ✅ Created {len(healthcare_pairs)} Healthcare positive pairs")

if len(finance_missions) >= 2:
    finance_pairs = create_pairs_from_missions(finance_missions, max_pairs=10)  # Reduced
    positive_pairs.extend(finance_pairs)
    print(f"   ✅ Created {len(finance_pairs)} Finance positive pairs")

if len(analyst_missions) >= 2:
    analyst_pairs = create_pairs_from_missions(analyst_missions, max_pairs=10)  # Reduced
    positive_pairs.extend(analyst_pairs)
    print(f"   ✅ Created {len(analyst_pairs)} Data Analyst positive pairs")

if len(marketing_missions) >= 2:
    marketing_pairs = create_pairs_from_missions(marketing_missions, max_pairs=10)  # Reduced
    positive_pairs.extend(marketing_pairs)
    print(f"   ✅ Created {len(marketing_pairs)} Marketing positive pairs")

# Also add some synthetic examples for clarity
synthetic_positives = [
    ("Facilitated Scrum ceremonies and sprint planning", "Led daily standups and sprint retrospectives"),
    ("Built data pipelines using Azure Data Factory", "Developed ETL processes for data warehouse"),
    ("Created Power BI dashboards for reporting", "Developed Tableau visualizations for KPIs"),
]
positive_pairs.extend(synthetic_positives)

# Negative pairs: Different domain missions should be dissimilar
# Use REAL missions from different domains!
negative_pairs = []

# Helper function to create cross-domain pairs
def create_cross_domain_pairs(list1, list2, max_pairs=20):
    """Create pairs from different domains (should be dissimilar)"""
    pairs = []
    import random
    random.seed(42)
    sampled1 = random.sample(list1, min(len(list1), max_pairs))
    sampled2 = random.sample(list2, min(len(list2), max_pairs))
    for m1, m2 in zip(sampled1, sampled2):
        pairs.append((m1, m2))
    return pairs

# Create negative pairs from real missions (cross-domain) - reduced for memory
if len(scrum_missions) > 0 and len(healthcare_missions) > 0:
    pairs = create_cross_domain_pairs(scrum_missions, healthcare_missions, max_pairs=8)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Scrum vs Healthcare negative pairs")

if len(scrum_missions) > 0 and len(de_missions) > 0:
    pairs = create_cross_domain_pairs(scrum_missions, de_missions, max_pairs=8)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Scrum vs Data Engineer negative pairs")

if len(scrum_missions) > 0 and len(finance_missions) > 0:
    pairs = create_cross_domain_pairs(scrum_missions, finance_missions, max_pairs=5)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Scrum vs Finance negative pairs")

if len(de_missions) > 0 and len(healthcare_missions) > 0:
    pairs = create_cross_domain_pairs(de_missions, healthcare_missions, max_pairs=8)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Data Engineer vs Healthcare negative pairs")

if len(de_missions) > 0 and len(analyst_missions) > 0:
    pairs = create_cross_domain_pairs(de_missions, analyst_missions, max_pairs=8)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Data Engineer vs Data Analyst negative pairs")

if len(healthcare_missions) > 0 and len(finance_missions) > 0:
    pairs = create_cross_domain_pairs(healthcare_missions, finance_missions, max_pairs=5)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Healthcare vs Finance negative pairs")

# Add Marketing cross-domain pairs
if len(marketing_missions) > 0 and len(scrum_missions) > 0:
    pairs = create_cross_domain_pairs(marketing_missions, scrum_missions, max_pairs=5)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Marketing vs Scrum negative pairs")

if len(marketing_missions) > 0 and len(de_missions) > 0:
    pairs = create_cross_domain_pairs(marketing_missions, de_missions, max_pairs=5)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Marketing vs Data Engineer negative pairs")

if len(marketing_missions) > 0 and len(healthcare_missions) > 0:
    pairs = create_cross_domain_pairs(marketing_missions, healthcare_missions, max_pairs=5)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Marketing vs Healthcare negative pairs")

if len(marketing_missions) > 0 and len(finance_missions) > 0:
    pairs = create_cross_domain_pairs(marketing_missions, finance_missions, max_pairs=5)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Marketing vs Finance negative pairs")

if len(marketing_missions) > 0 and len(analyst_missions) > 0:
    pairs = create_cross_domain_pairs(marketing_missions, analyst_missions, max_pairs=5)  # Reduced
    negative_pairs.extend(pairs)
    print(f"   ✅ Created {len(pairs)} Marketing vs Data Analyst negative pairs")

# Also add some synthetic examples for clarity
synthetic_negatives = [
    ("Facilitated Scrum ceremonies", "Coordinated patient care and treatment"),
    ("Built data pipelines", "Prepared financial reports"),
    ("Created Power BI dashboards", "Managed clinical documentation"),
]
negative_pairs.extend(synthetic_negatives)

print(f"✅ Created {len(positive_pairs)} positive pairs (same domain)")
print(f"✅ Created {len(negative_pairs)} negative pairs (different domains)")
print()

# ==========================================
# 3. Create Training Examples
# ==========================================
print("🔄 Preparing training data...")

train_examples = [
    InputExample(texts=[t1, t2], label=1.0) for t1, t2 in positive_pairs
] + [
    InputExample(texts=[t1, t2], label=0.0) for t1, t2 in negative_pairs
]

print(f"✅ Total training examples: {len(train_examples)}")
print(f"   - Positive (label=1.0): {len(positive_pairs)}")
print(f"   - Negative (label=0.0): {len(negative_pairs)}")
print()

# ==========================================
# 4. Fine-Tune Model
# ==========================================
print("🎯 Fine-tuning model with ContrastiveLoss...")
print()

# Use smaller batch size to avoid memory issues
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)  # Reduced to 4 for CPU
train_loss = losses.ContrastiveLoss(model, margin=0.5)

# Training parameters
epochs = 3  # Reduced from 5 to 3 to save time and memory
warmup_steps = 30  # Reduced from 100
learning_rate = 2e-5

print(f"Training parameters:")
print(f"   - Epochs: {epochs}")
print(f"   - Batch size: 16")
print(f"   - Warmup steps: {warmup_steps}")
print(f"   - Learning rate: {learning_rate}")
print(f"   - Loss: ContrastiveLoss (margin=0.5)")
print()
print("⏳ Training... (this may take a few minutes)")
print()

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': learning_rate},
    show_progress_bar=True
)

# ==========================================
# 5. Save Fine-Tuned Model
# ==========================================
output_model_name = "models/modelfinetuned_domain"
print()
print(f"💾 Saving fine-tuned model to: {output_model_name}")
model.save(output_model_name)
print("✅ Model saved!")
print()

# ==========================================
# 6. Test the Fine-Tuned Model
# ==========================================
print("🧪 Testing fine-tuned model...")
print()

from sentence_transformers.util import cos_sim

# Test cases
test_cases = [
    # Same domain (should be similar)
    ("Facilitated Scrum ceremonies", "Led sprint planning meetings", "Scrum vs Scrum", True),
    ("Built data pipelines", "Developed ETL processes", "Data Engineer vs Data Engineer", True),
    ("Created Power BI dashboards", "Built reporting dashboards", "Data Analyst vs Data Analyst", True),
    
    # Different domains (should be dissimilar)
    ("Facilitated Scrum ceremonies", "Coordinated patient care", "Scrum vs Healthcare", False),
    ("Built data pipelines", "Facilitated Scrum ceremonies", "Data Engineer vs Scrum", False),
    ("Created Power BI dashboards", "Prepared financial reports", "Data Analyst vs Finance", False),
    ("Coordinated marketing campaigns", "Built data pipelines", "Marketing vs Data Engineer", False),
    ("Developed social media strategies", "Facilitated Scrum ceremonies", "Marketing vs Scrum", False),
]

print("Similarity scores (higher = more similar):")
print("-" * 80)
for text1, text2, description, should_be_similar in test_cases:
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = cos_sim(emb1, emb2).item()
    
    status = "✅" if (should_be_similar and similarity > 0.5) or (not should_be_similar and similarity < 0.5) else "⚠️"
    print(f"{status} {description:40s} Similarity: {similarity:.4f}")

print()
print("=" * 80)
print("✅ FINE-TUNING COMPLETE!")
print("=" * 80)
print()
print("📋 NEXT STEPS:")
print(f"   1. Update config/search_config.yaml:")
print(f"      Change 'finetune_model' in mission_embeddings.py to: {output_model_name}")
print()
print("   2. Re-encode missions with new model:")
print("      - Delete or rename: skills-finder-hackathon-hec-x-sbi/mission_encoded.csv")
print("      - Run main_debug.py - it will auto-re-encode with new model")
print()
print("   3. Test the improvements:")
print("      python3 mainfile/main_debug.py Scrum")
print("      python3 mainfile/main_debug.py DataEngineer")
print()
print("   4. Compare Precision@10 before/after fine-tuning")
print()
print("=" * 80)

