"""
Fine-tune Mission Embeddings with Domain-Specific Examples

This script fine-tunes mission embeddings to better distinguish between different job roles:
- Scrum Master vs Data Engineer vs Healthcare vs Finance vs Data Analyst

Uses contrastive learning to make same-domain missions similar and different-domain missions dissimilar.
"""

import sys
import os
import ast
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'



project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


if os.path.exists("notebooks/modelfinetuned1"):
    base_model = "notebooks/modelfinetuned1"
    print("   Using language-fine-tuned model (modelfinetuned1) as base")
elif os.path.exists("models/modelfinetuned_domain"):
    base_model = "models/modelfinetuned_domain"
    print("   Using existing modelfinetuned_domain as base")
else:
    base_model = "paraphrase-multilingual-mpnet-base-v2"
    print("   Using paraphrase-multilingual-mpnet-base-v2 as base")


import torch
torch.set_default_device('cpu')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

device = torch.device('cpu')
print(f"   Using device: {device} (CPU only, MPS disabled)")

model = SentenceTransformer(base_model, device=device)
model = model.to(device)
print(f"Loaded: {base_model}")
print(f"   Model dimension: {model.get_sentence_embedding_dimension()}")
print()



df_full = pd.read_csv(os.path.join(project_root, 'full_df.csv'))

def extract_missions_from_list(missions_str):
    """Extract individual missions from MISSIONS_LIST string"""
    if pd.isna(missions_str) or missions_str == '':
        return []
    try:
        missions = ast.literal_eval(missions_str)
        if isinstance(missions, list):
            return [m for m in missions if m and isinstance(m, str)]
        return []
    except:
        return []

def get_missions_by_role(df, role):
    """Extract all missions for a given job role"""
    role_df = df[df['JOB_RULE'] == role]
    all_missions = []
    for _, row in role_df.iterrows():
        missions = extract_missions_from_list(row.get('MISSIONS_LIST', ''))
        all_missions.extend(missions)
    return all_missions

scrum_missions = get_missions_by_role(df_full, 'Scrum Master')
de_missions = get_missions_by_role(df_full, 'Data Engineer')
healthcare_missions = get_missions_by_role(df_full, 'Healthcare')
finance_missions = get_missions_by_role(df_full, 'Finance')
analyst_missions = get_missions_by_role(df_full, 'Data Analyst')

print(f"   Found {len(scrum_missions)} Scrum missions")
print(f"   Found {len(de_missions)} Data Engineer missions")
print(f"   Found {len(healthcare_missions)} Healthcare missions")
print(f"   Found {len(finance_missions)} Finance missions")
print(f"   Found {len(analyst_missions)} Data Analyst missions")
print()

positive_pairs = []

def create_pairs_from_missions(mission_list, max_pairs=50):
    """Create pairs from real missions (same domain)"""
    pairs = []
    import random
    random.seed(42)
    max_samples = min(len(mission_list), max_pairs * 2, 100)  
    sampled = random.sample(mission_list, max_samples)
    for i in range(0, len(sampled) - 1, 2):
        if i + 1 < len(sampled):
            m1, m2 = sampled[i], sampled[i + 1]
            if len(m1) < 500 and len(m2) < 500:
                pairs.append((m1, m2))
    return pairs[:max_pairs]

if len(scrum_missions) >= 2:
    scrum_pairs = create_pairs_from_missions(scrum_missions, max_pairs=15)
    positive_pairs.extend(scrum_pairs)
    print(f"   Created {len(scrum_pairs)} Scrum positive pairs")

if len(de_missions) >= 2:
    de_pairs = create_pairs_from_missions(de_missions, max_pairs=15)
    positive_pairs.extend(de_pairs)
    print(f"   Created {len(de_pairs)} Data Engineer positive pairs")

if len(healthcare_missions) >= 2:
    healthcare_pairs = create_pairs_from_missions(healthcare_missions, max_pairs=10)
    positive_pairs.extend(healthcare_pairs)
    print(f"   Created {len(healthcare_pairs)} Healthcare positive pairs")

if len(finance_missions) >= 2:
    finance_pairs = create_pairs_from_missions(finance_missions, max_pairs=10)
    positive_pairs.extend(finance_pairs)
    print(f"   Created {len(finance_pairs)} Finance positive pairs")

if len(analyst_missions) >= 2:
    analyst_pairs = create_pairs_from_missions(analyst_missions, max_pairs=10)
    positive_pairs.extend(analyst_pairs)
    print(f"   Created {len(analyst_pairs)} Data Analyst positive pairs")

synthetic_positives = [
    ("Facilitated Scrum ceremonies and sprint planning", "Led daily standups and sprint retrospectives"),
    ("Built data pipelines using Azure Data Factory", "Developed ETL processes for data warehouse"),
    ("Created Power BI dashboards for reporting", "Developed Tableau visualizations for KPIs"),
]
positive_pairs.extend(synthetic_positives)

negative_pairs = []

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

if len(scrum_missions) > 0 and len(healthcare_missions) > 0:
    pairs = create_cross_domain_pairs(scrum_missions, healthcare_missions, max_pairs=8)
    negative_pairs.extend(pairs)
    print(f"   Created {len(pairs)} Scrum vs Healthcare negative pairs")

if len(scrum_missions) > 0 and len(de_missions) > 0:
    pairs = create_cross_domain_pairs(scrum_missions, de_missions, max_pairs=8)
    negative_pairs.extend(pairs)
    print(f"   Created {len(pairs)} Scrum vs Data Engineer negative pairs")

if len(scrum_missions) > 0 and len(finance_missions) > 0:
    pairs = create_cross_domain_pairs(scrum_missions, finance_missions, max_pairs=5)
    negative_pairs.extend(pairs)
    print(f"   Created {len(pairs)} Scrum vs Finance negative pairs")

if len(de_missions) > 0 and len(healthcare_missions) > 0:
    pairs = create_cross_domain_pairs(de_missions, healthcare_missions, max_pairs=8)
    negative_pairs.extend(pairs)
    print(f"   Created {len(pairs)} Data Engineer vs Healthcare negative pairs")

if len(de_missions) > 0 and len(analyst_missions) > 0:
    pairs = create_cross_domain_pairs(de_missions, analyst_missions, max_pairs=8)
    negative_pairs.extend(pairs)
    print(f"   Created {len(pairs)} Data Engineer vs Data Analyst negative pairs")

if len(healthcare_missions) > 0 and len(finance_missions) > 0:
    pairs = create_cross_domain_pairs(healthcare_missions, finance_missions, max_pairs=5)
    negative_pairs.extend(pairs)
    print(f"   Created {len(pairs)} Healthcare vs Finance negative pairs")


synthetic_negatives = [
    ("Facilitated Scrum ceremonies", "Coordinated patient care and treatment"),
    ("Built data pipelines", "Prepared financial reports"),
    ("Created Power BI dashboards", "Managed clinical documentation"),
]
negative_pairs.extend(synthetic_negatives)

print(f"Created {len(positive_pairs)} positive pairs (same domain)")
print(f"Created {len(negative_pairs)} negative pairs (different domains)")
print()



train_examples = [
    InputExample(texts=[t1, t2], label=1.0) for t1, t2 in positive_pairs
] + [
    InputExample(texts=[t1, t2], label=0.0) for t1, t2 in negative_pairs
]

print(f"Total training examples: {len(train_examples)}")
print(f"   - Positive (label=1.0): {len(positive_pairs)}")
print(f"   - Negative (label=0.0): {len(negative_pairs)}")
print()




train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
train_loss = losses.ContrastiveLoss(model, margin=0.5)


epochs = 3 
warmup_steps = 30  
learning_rate = 2e-5

print(f"Training parameters:")
print(f"   - Epochs: {epochs}")
print(f"   - Batch size: 2")
print(f"   - Warmup steps: {warmup_steps}")
print(f"   - Learning rate: {learning_rate}")
print(f"   - Loss: ContrastiveLoss (margin=0.5)")
print(f"   - Device: CPU (MPS disabled)")

model = model.to(torch.device('cpu'))

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': learning_rate},
    show_progress_bar=True,
    use_amp=False
)


output_model_name = "models/modelfinetuned_domain"
print()
print(f"Saving fine-tuned model to: {output_model_name}")
model.save(output_model_name)
print("Model saved!")
print()



from sentence_transformers.util import cos_sim


if os.path.exists("notebooks/modelfinetuned1"):
    base_model_name = "notebooks/modelfinetuned1"
elif os.path.exists("models/modelfinetuned_domain"):
    base_model_name = "paraphrase-multilingual-mpnet-base-v2" 
else:
    base_model_name = "paraphrase-multilingual-mpnet-base-v2"

base_model = SentenceTransformer(base_model_name, device=device)
base_model = base_model.to(device)
print(f"Base model: {base_model_name}")
print()

test_cases = [

    ("Facilitated Scrum ceremonies", "Led sprint planning meetings", "Scrum vs Scrum", True),
    ("Built data pipelines", "Developed ETL processes", "Data Engineer vs Data Engineer", True),
    ("Created Power BI dashboards", "Built reporting dashboards", "Data Analyst vs Data Analyst", True),
    ("Analyzed patient medical records", "Coordinated care with medical team", "Healthcare vs Healthcare", True),
    ("Implemented budget simulation", "Prepared financial reports", "Finance vs Finance", True),
    
    ("Facilitated Scrum ceremonies", "Coordinated patient care", "Scrum vs Healthcare", False),
    ("Built data pipelines", "Facilitated Scrum ceremonies", "Data Engineer vs Scrum", False),
    ("Created Power BI dashboards", "Prepared financial reports", "Data Analyst vs Finance", False),
    ("Analyzed patient medical records", "Built data pipelines", "Healthcare vs Data Engineer", False),
    ("Implemented budget simulation", "Facilitated Scrum ceremonies", "Finance vs Scrum", False),
]

print("COMPARISON: Base Model vs Fine-Tuned Model")

print(f"{'Test Case':<45} {'Base':<8} {'Fine-Tuned':<10} {'Change':<10} {'Status':<8}")


base_similarities = []
ft_similarities = []
same_domain_base = []
same_domain_ft = []
diff_domain_base = []
diff_domain_ft = []

for text1, text2, description, should_be_similar in test_cases:
    base_emb1 = base_model.encode(text1, convert_to_tensor=True)
    base_emb2 = base_model.encode(text2, convert_to_tensor=True)
    base_sim = cos_sim(base_emb1, base_emb2).item()
    
    ft_emb1 = model.encode(text1, convert_to_tensor=True)
    ft_emb2 = model.encode(text2, convert_to_tensor=True)
    ft_sim = cos_sim(ft_emb1, ft_emb2).item()
    
    change = ft_sim - base_sim
    change_pct = (change / base_sim * 100) if base_sim > 0 else 0
    
    if should_be_similar:
        status = "BETTER" if ft_sim > base_sim else "WORSE" if ft_sim < base_sim else "SAME"
        same_domain_base.append(base_sim)
        same_domain_ft.append(ft_sim)
    else:
        status = "BETTER" if ft_sim < base_sim else "WORSE" if ft_sim > base_sim else "SAME"
        diff_domain_base.append(base_sim)
        diff_domain_ft.append(ft_sim)
    
    base_similarities.append(base_sim)
    ft_similarities.append(ft_sim)
    
    print(f"{description:<45} {base_sim:<8.4f} {ft_sim:<10.4f} {change:+.4f} ({change_pct:+.1f}%) {status:<8}")

print("\nSUMMARY STATISTICS:")

if same_domain_base:
    avg_same_base = sum(same_domain_base) / len(same_domain_base)
    avg_same_ft = sum(same_domain_ft) / len(same_domain_ft)
    print(f"Same Domain (should be HIGH):")
    print(f"  Base Model Average:     {avg_same_base:.4f}")
    print(f"  Fine-Tuned Average:      {avg_same_ft:.4f}")
    print(f"  Improvement:             {avg_same_ft - avg_same_base:+.4f} ({(avg_same_ft - avg_same_base) / avg_same_base * 100:+.1f}%)")

if diff_domain_base:
    avg_diff_base = sum(diff_domain_base) / len(diff_domain_base)
    avg_diff_ft = sum(diff_domain_ft) / len(diff_domain_ft)
    print(f"\nDifferent Domains (should be LOW):")
    print(f"  Base Model Average:     {avg_diff_base:.4f}")
    print(f"  Fine-Tuned Average:      {avg_diff_ft:.4f}")
    print(f"  Improvement:             {avg_diff_base - avg_diff_ft:+.4f} ({(avg_diff_base - avg_diff_ft) / avg_diff_base * 100:+.1f}%)")

if same_domain_base and diff_domain_base:
    gap_base = avg_same_base - avg_diff_base
    gap_ft = avg_same_ft - avg_diff_ft
    print(f"\nGap (Same - Different):")
    print(f"  Base Model Gap:         {gap_base:.4f}")
    print(f"  Fine-Tuned Gap:          {gap_ft:.4f}")
    print(f"  Gap Improvement:         {gap_ft - gap_base:+.4f}")



