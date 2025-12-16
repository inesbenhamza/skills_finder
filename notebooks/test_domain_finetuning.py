import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch

device = 'cpu'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("COMPARING BASE MODEL vs FINE-TUNED DOMAIN MODEL")
print()

if os.path.exists("notebooks/modelfinetuned1"):
    base_model_name = "notebooks/modelfinetuned1"
    print(f"Base model: {base_model_name} (language fine-tuned)")
elif os.path.exists("models/modelfinetuned_domain"):
    base_model_name = "paraphrase-multilingual-mpnet-base-v2"
    print(f"Base model: {base_model_name} (original)")
else:
    base_model_name = "paraphrase-multilingual-mpnet-base-v2"
    print(f"Base model: {base_model_name} (original)")

print("Loading models...")
base_model = SentenceTransformer(base_model_name, device=device)

if os.path.exists("models/modelfinetuned_domain"):
    ft_model = SentenceTransformer("models/modelfinetuned_domain", device=device)
    print("Fine-tuned model: models/modelfinetuned_domain")
else:
    print("WARNING: Fine-tuned model not found. Run finetune_mission_embeddings_domain.py first.")
    sys.exit(1)

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

print("COMPARISON RESULTS:")
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
    
    ft_emb1 = ft_model.encode(text1, convert_to_tensor=True)
    ft_emb2 = ft_model.encode(text2, convert_to_tensor=True)
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


