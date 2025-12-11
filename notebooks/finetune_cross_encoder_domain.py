#!/usr/bin/env python3
"""
Fine-tune Cross-Encoder for domain-aware reranking.

This script fine-tunes the cross-encoder to better distinguish between
different job roles (Scrum Master, Data Engineer, Data Analyst, Healthcare, etc.)
by training it on real mission-consultant pairs with ground truth labels.
"""

import os
import sys
import pandas as pd
import random
from pathlib import Path
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import helper function to build consultant text
from mainfile.main_debug import build_consultant_text

print("=" * 80)
print("🚀 FINE-TUNING CROSS-ENCODER FOR DOMAIN DISTINCTION")
print("=" * 80)
print()

# ==========================================
# 1. Load Base Model
# ==========================================
print("📦 Loading base cross-encoder model...")
base_model = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
model = CrossEncoder(base_model)
print(f"✅ Loaded: {base_model}")
print()

# ==========================================
# 2. Load Data
# ==========================================
print("📝 Loading data...")
full_df_path = project_root / "full_df.csv"
missions_df_path = project_root / "data" / "HCK_HEC_XP.csv"
skills_df_path = project_root / "data" / "HCK_HEC_SKILLS.csv"

full_df = pd.read_csv(full_df_path)
missions_df = pd.read_csv(missions_df_path)
skills_df = pd.read_csv(skills_df_path)

print(f"   ✅ Loaded {len(full_df)} consultants from full_df.csv")
print(f"   ✅ Loaded {len(missions_df)} missions from HCK_HEC_XP.csv")
print(f"   ✅ Loaded {len(skills_df)} skill records from HCK_HEC_SKILLS.csv")
print()

# ==========================================
# 3. Create Training Examples
# ==========================================
print("🔄 Creating training examples from real data...")
print()

# Merge data to get job roles for missions
missions_with_roles = missions_df.merge(
    full_df[['USER_ID', 'JOB_RULE']].drop_duplicates(),
    on='USER_ID',
    how='left'
)
missions_with_roles = missions_with_roles.dropna(subset=['JOB_RULE', 'MISSION_DSC'])

# Group missions by job role
domain_missions = missions_with_roles.groupby('JOB_RULE')['MISSION_DSC'].apply(list).to_dict()

print("   Found missions by domain:")
for domain, missions in domain_missions.items():
    print(f"      - {domain}: {len(missions)} missions")

print()

# Create training examples
train_examples = []

# Helper to build mission text from description
def build_mission_text_from_desc(mission_desc):
    """Convert mission description to full mission text format"""
    # Format similar to what build_mission_text creates
    # The cross-encoder sees: "Responsibilities: ... Sector: ... Technologies: ..."
    # For training, we'll use the mission description as the main content
    return f"Responsibilities: {mission_desc}"

# Helper to get consultant profile text
def get_consultant_profile_text(user_id):
    """Get consultant profile text for cross-encoder"""
    consultant_skills = skills_df[skills_df['USER_ID'] == user_id]
    if len(consultant_skills) == 0:
        return None
    
    # Get first row for basic info
    row = consultant_skills.iloc[0]
    
    # Build skills text
    skills_list = consultant_skills['SKILLS_DSC'].unique().tolist()
    skills_text = ", ".join(skills_list[:10])  # Limit to 10 skills
    
    # Get domain
    domain = row.get('DOMAIN_DSC', 'General')
    
    # Get experience level (average)
    avg_level = consultant_skills['LEVEL_VAL'].mean() if 'LEVEL_VAL' in consultant_skills.columns else 50
    
    profile_text = (
        f"Skills: {skills_text}. "
        f"Domain expertise: {domain}. "
        f"Experience level (0-100): {avg_level:.0f}. "
        f"This candidate has worked on related missions."
    )
    
    return profile_text

# Create positive pairs (same domain)
print("   Creating positive pairs (same domain)...")
positive_count = 0
for job_role, missions in domain_missions.items():
    if len(missions) < 2:
        continue
    
    # Get consultants with this job role
    consultants = full_df[full_df['JOB_RULE'] == job_role]['USER_ID'].unique().tolist()
    
    if len(consultants) == 0:
        continue
    
    # Create pairs: mission from this domain + consultant from this domain
    for _ in range(min(len(missions) * 2, 30)):  # Up to 30 pairs per domain
        mission = random.choice(missions)
        consultant_id = random.choice(consultants)
        
        profile_text = get_consultant_profile_text(consultant_id)
        if profile_text:
            mission_text = build_mission_text_from_desc(mission)
            train_examples.append({
                'texts': [mission_text, profile_text],
                'label': 1.0  # Positive match
            })
            positive_count += 1

print(f"   ✅ Created {positive_count} positive pairs")
print()

# Create negative pairs (different domains)
print("   Creating negative pairs (different domains)...")
negative_count = 0
all_job_roles = list(domain_missions.keys())

for _ in range(min(positive_count, 100)):  # Match number of positive pairs
    # Pick two different domains
    role1, role2 = random.sample(all_job_roles, 2)
    
    if len(domain_missions[role1]) == 0 or len(domain_missions[role2]) == 0:
        continue
    
    # Mission from role1, consultant from role2 (wrong match)
    mission = random.choice(domain_missions[role1])
    consultants = full_df[full_df['JOB_RULE'] == role2]['USER_ID'].unique().tolist()
    
    if len(consultants) == 0:
        continue
    
    consultant_id = random.choice(consultants)
    profile_text = get_consultant_profile_text(consultant_id)
    
    if profile_text:
        mission_text = build_mission_text_from_desc(mission)
        train_examples.append({
            'texts': [mission_text, profile_text],
            'label': 0.0  # Negative match
        })
        negative_count += 1

print(f"   ✅ Created {negative_count} negative pairs")
print()

print(f"✅ Total training examples: {len(train_examples)}")
print(f"   - Positive (label=1.0): {positive_count}")
print(f"   - Negative (label=0.0): {negative_count}")
print()

# ==========================================
# 4. Prepare DataLoader
# ==========================================
print("🔄 Preparing training data...")
from sentence_transformers import InputExample

# Convert to InputExample format
train_input_examples = [
    InputExample(texts=ex['texts'], label=ex['label'])
    for ex in train_examples
]

# Shuffle
random.shuffle(train_input_examples)

# Split into train/validation (90/10)
split_idx = int(len(train_input_examples) * 0.9)
train_data = train_input_examples[:split_idx]
val_data = train_input_examples[split_idx:]

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=16)

print(f"   ✅ Training examples: {len(train_data)}")
print(f"   ✅ Validation examples: {len(val_data)}")
print()

# ==========================================
# 5. Fine-Tune Model
# ==========================================
print("🎯 Fine-tuning cross-encoder...")
print()

# Training parameters
epochs = 3
warmup_steps = 50
learning_rate = 2e-5

print(f"Training parameters:")
print(f"   - Epochs: {epochs}")
print(f"   - Batch size: 16")
print(f"   - Warmup steps: {warmup_steps}")
print(f"   - Learning rate: {learning_rate}")
print()

print("⏳ Training... (this may take a few minutes)")
print()

# Fine-tune the model
# CrossEncoder.fit() signature:
# fit(train_dataloader, evaluator=None, epochs=1, warmup_steps=10000, 
#     optimizer_params={'lr': 2e-05}, show_progress_bar=True, ...)
model.fit(
    train_dataloader=train_dataloader,
    epochs=epochs,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': learning_rate},
    show_progress_bar=True
)

# ==========================================
# 6. Save Fine-Tuned Model
# ==========================================
output_model_name = "models/crossencoder_finetuned_domain"
os.makedirs(output_model_name, exist_ok=True)

print()
print(f"💾 Saving fine-tuned model to: {output_model_name}")
model.save(output_model_name)
print("✅ Model saved!")
print()

# ==========================================
# 7. Test Fine-Tuned Model
# ==========================================
print("🧪 Testing fine-tuned model...")
print()

# Test with a few examples
test_cases = [
    # Positive: Scrum mission + Scrum consultant
    ("Mission: Facilitate Scrum ceremonies and sprint planning", 
     "Skills: Atlassian JIRA Software, Agile. Domain expertise: Agile Delivery. Experience level (0-100): 90. This candidate has worked on related missions.",
     "Scrum + Scrum (should be high)"),
    
    # Negative: Scrum mission + Data Engineer consultant
    ("Mission: Facilitate Scrum ceremonies and sprint planning",
     "Skills: Apache Spark, Databricks, Python. Domain expertise: Data Engineering. Experience level (0-100): 85. This candidate has worked on related missions.",
     "Scrum + Data Engineer (should be low)"),
    
    # Positive: Data Engineer mission + Data Engineer consultant
    ("Mission: Build data pipelines using Apache Spark and Databricks",
     "Skills: Apache Spark, Databricks, Python. Domain expertise: Data Engineering. Experience level (0-100): 85. This candidate has worked on related missions.",
     "Data Engineer + Data Engineer (should be high)"),
    
    # Negative: Healthcare mission + Data Analyst consultant
    ("Mission: Evaluate patient health status and develop treatment plans",
     "Skills: Power BI, SQL, Excel. Domain expertise: Business Intelligence. Experience level (0-100): 75. This candidate has worked on related missions.",
     "Healthcare + Data Analyst (should be low)"),
]

print("Similarity scores (higher = better match):")
print("-" * 80)

for mission_text, profile_text, description in test_cases:
    score = model.predict([(mission_text, profile_text)])[0]
    print(f"{description:<50} Score: {score:.4f}")

print()
print("=" * 80)
print("✅ FINE-TUNING COMPLETE!")
print("=" * 80)
print()
print("📋 NEXT STEPS:")
print(f"   1. Update mainfile/main_debug.py:")
print(f"      Change 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' to '{output_model_name}'")
print()
print("   2. Test the improvements:")
print("      python3 mainfile/main_debug.py Scrum")
print("      python3 mainfile/main_debug.py DataEngineer")
print()
print("   3. Compare Precision@10 before/after fine-tuning")
print()
print("=" * 80)

