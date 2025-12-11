#!/usr/bin/env python3
"""
Test script to verify hierarchical skill embeddings are working correctly.

This script checks if parent-child skills are close together in embedding space.
"""

import os
import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_hierarchical_embeddings(model_path: str = "skills_finetuned_hierarchy"):
    """
    Test if hierarchical relationships are preserved in embeddings.
    """
    print("=" * 80)
    print("🧪 TESTING HIERARCHICAL SKILL EMBEDDINGS")
    print("=" * 80)
    
    # Load model
    print(f"\n[1/3] Loading model: {model_path}...")
    try:
        model = SentenceTransformer(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure you've run finetune_skill_hierarchy.py first!")
        return
    
    # Test cases: parent-child pairs that should be close
    test_cases = [
        # Microsoft hierarchy
        ("Microsoft", "Microsoft Power BI", "Parent -> Child"),
        ("Microsoft", "Microsoft SQL Server Reporting Services", "Parent -> Child"),
        ("Microsoft Power BI", "Microsoft SQL Server Reporting Services", "Siblings"),
        ("Microsoft SQL Server", "Microsoft SQL Server Reporting Services", "Parent -> Child"),
        ("Microsoft SQL Server", "Microsoft SQL Server Integration Services", "Parent -> Child"),
        
        # Atlassian hierarchy
        ("Atlassian", "Atlassian JIRA Software", "Parent -> Child"),
        ("Atlassian", "Atlassian Bitbucket", "Parent -> Child"),
        ("Atlassian JIRA Software", "Atlassian Bitbucket", "Siblings"),
        
        # Oracle hierarchy
        ("Oracle", "Oracle Hyperion Planning", "Parent -> Child"),
        ("Oracle Hyperion", "Oracle Hyperion Essbase", "Parent -> Child"),
        
        # SAP hierarchy
        ("SAP", "SAP BusinessObjects BI", "Parent -> Child"),
        ("SAP", "SAP BW", "Parent -> Child"),
        
        # Unrelated skills (should be far)
        ("Microsoft Power BI", "Oracle Database", "Unrelated"),
        ("Atlassian JIRA Software", "SAP BW", "Unrelated"),
        ("Python", "Java", "Unrelated (but both programming languages)"),
    ]
    
    print(f"\n[2/3] Testing {len(test_cases)} skill pairs...")
    print("\n" + "-" * 80)
    print(f"{'Skill 1':<40} {'Skill 2':<40} {'Similarity':<12} {'Status'}")
    print("-" * 80)
    
    results = []
    for skill1, skill2, relationship in test_cases:
        # Encode both skills
        emb1 = model.encode(skill1, convert_to_tensor=True)
        emb2 = model.encode(skill2, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        
        # Determine if result is good
        if "Parent -> Child" in relationship or "Siblings" in relationship:
            status = "✅ GOOD" if similarity > 0.6 else "⚠️  LOW"
        elif "Unrelated" in relationship:
            status = "✅ GOOD" if similarity < 0.5 else "⚠️  HIGH"
        else:
            status = "❓"
        
        print(f"{skill1:<40} {skill2:<40} {similarity:<12.4f} {status}")
        results.append((skill1, skill2, relationship, similarity, status))
    
    print("-" * 80)
    
    # Summary
    print(f"\n[3/3] Summary:")
    parent_child_pairs = [r for r in results if "Parent -> Child" in r[2]]
    sibling_pairs = [r for r in results if "Siblings" in r[2]]
    unrelated_pairs = [r for r in results if "Unrelated" in r[2]]
    
    if parent_child_pairs:
        avg_sim = np.mean([r[3] for r in parent_child_pairs])
        print(f"  Parent-Child pairs: {len(parent_child_pairs)} pairs, avg similarity: {avg_sim:.4f}")
        print(f"    → Should be > 0.6: {'✅' if avg_sim > 0.6 else '❌'}")
    
    if sibling_pairs:
        avg_sim = np.mean([r[3] for r in sibling_pairs])
        print(f"  Sibling pairs: {len(sibling_pairs)} pairs, avg similarity: {avg_sim:.4f}")
        print(f"    → Should be > 0.5: {'✅' if avg_sim > 0.5 else '❌'}")
    
    if unrelated_pairs:
        avg_sim = np.mean([r[3] for r in unrelated_pairs])
        print(f"  Unrelated pairs: {len(unrelated_pairs)} pairs, avg similarity: {avg_sim:.4f}")
        print(f"    → Should be < 0.5: {'✅' if avg_sim < 0.5 else '❌'}")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test hierarchical skill embeddings")
    parser.add_argument("--model", default="skills_finetuned_hierarchy", 
                       help="Path to fine-tuned model")
    args = parser.parse_args()
    
    test_hierarchical_embeddings(args.model)

