#!/usr/bin/env python3
"""
Interactive Consultant Matching Agent

This agent allows you to:
1. Upload a PDF mission description
2. Have a conversation to refine requirements
3. Get updated predictions based on your feedback

Example:
    agent = InteractiveMatchingAgent()
    agent.load_mission("documents/DataEngineer.pdf")
    agent.chat("I don't mind if they don't speak English")
    results = agent.get_top_candidates(5)
"""

import sys
import os
from pathlib import Path
import yaml
import json
import re

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from helpers import SkillsEmbeddings as se
from helpers.tools_extraction import generate_yaml, get_model_response
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict
from src.language_filtering import compute_language_score, apply_filters
from src.schedule_filtering import calculate_availability_penalty
from mainfile.main_debug import linear_combination

class InteractiveMatchingAgent:
    """
    Interactive agent for consultant matching with conversational feedback.
    """
    
    def __init__(self, config_path: str = "config/search_config.yaml"):
        """Initialize the agent."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Mission requirements (will be updated based on feedback)
        self.task_requirements = None
        
        # Original requirements (before modifications)
        self.original_requirements = None
        
        # Feedback history
        self.feedback_history = []
        
        # Results cache
        self.last_results = None
        
        # Load models (lazy)
        self.skill_model = None
        self._load_models()
        
        print("🤖 Interactive Matching Agent initialized!")
        print("   You can now load a mission and chat with me to refine requirements.")
    
    def _load_models(self):
        """Load embedding models."""
        print("📦 Loading models...")
        model = SentenceTransformer(self.config["embedding_model"])
        self.skill_model = se.SkillEmbeddingModel(model)
        print("✅ Models loaded")
    
    def load_mission(self, pdf_path: str):
        """
        Load a mission from PDF file.
        
        Args:
            pdf_path: Path to PDF file
        """
        print(f"\n📄 Loading mission from: {pdf_path}")
        
        # Extract requirements from PDF
        model_response = get_model_response(pdf_path, self.config["model"]["name"])
        self.task_requirements = generate_yaml(model_response, think=self.config["model"]["think"])
        
        # Save original for reference
        self.original_requirements = json.loads(json.dumps(self.task_requirements))  # Deep copy
        
        # Infer job role from filename if not in YAML
        filename = Path(pdf_path).stem
        if 'job_role' not in self.task_requirements or not self.task_requirements.get('job_role'):
            filename_lower = filename.lower()
            if 'finance' in filename_lower:
                self.task_requirements['job_role'] = 'Finance'
            elif 'dataengineer' in filename_lower or 'data_engineer' in filename_lower:
                self.task_requirements['job_role'] = 'Data Engineer'
            elif 'dataanalyst' in filename_lower or 'data_analyst' in filename_lower:
                self.task_requirements['job_role'] = 'Data Analyst'
            elif 'scrum' in filename_lower:
                self.task_requirements['job_role'] = 'Scrum Master'
            elif 'healthcare' in filename_lower:
                self.task_requirements['job_role'] = 'Healthcare'
            elif 'marketing' in filename_lower:
                self.task_requirements['job_role'] = 'Marketing'
        
        print(f"✅ Mission loaded: {self.task_requirements.get('job_role', 'Unknown')}")
        print(f"   Technologies: {len(self.task_requirements.get('technologies', []))} skills")
        print(f"   Languages: {len(self.task_requirements.get('languages', []))} languages")
        
        return self.task_requirements
    
    def chat(self, message: str):
        """
        Process user feedback and update requirements.
        
        Args:
            message: User's feedback/instruction (e.g., "I don't mind if they don't speak English")
        
        Returns:
            Updated requirements
        """
        print(f"\n💬 Processing: '{message}'")
        
        message_lower = message.lower()
        
        # Language-related feedback
        if any(phrase in message_lower for phrase in ["don't mind", "don't care", "not important", "not required", "optional"]) and \
           any(word in message_lower for word in ["english", "en", "language", "languages"]):
            print("   → Making English optional...")
            self._make_language_optional("EN")
            self.feedback_history.append({"type": "language", "action": "made_english_optional", "message": message})
        
        elif any(phrase in message_lower for phrase in ["don't mind", "don't care", "not important", "not required", "optional"]) and \
             any(word in message_lower for word in ["french", "fr", "language", "languages"]):
            print("   → Making French optional...")
            self._make_language_optional("FR")
            self.feedback_history.append({"type": "language", "action": "made_french_optional", "message": message})
        
        # Availability feedback
        elif any(phrase in message_lower for phrase in ["don't mind", "don't care", "not important", "availability", "schedule"]):
            print("   → Reducing availability penalty...")
            self.config["disponibility"]["maximum_penalty"] = 0.1  # Very low penalty
            self.feedback_history.append({"type": "availability", "action": "reduced_penalty", "message": message})
        
        # Skill-related feedback
        elif any(phrase in message_lower for phrase in ["don't need", "not required", "optional"]) and \
             any(word in message_lower for word in ["skill", "technology", "tech"]):
            # Try to extract skill name
            for tech in self.task_requirements.get('technologies', []):
                tech_name = tech.get('name', '').lower()
                if tech_name in message_lower:
                    print(f"   → Making {tech.get('name')} optional...")
                    tech['required'] = False
                    self.feedback_history.append({"type": "skill", "action": "made_optional", "skill": tech.get('name'), "message": message})
        
        # Weight adjustment feedback
        elif "skills" in message_lower and "more important" in message_lower:
            print("   → Increasing skills weight...")
            self.config["skills"]["skills_weight"] = 0.7
            self.config["mission_context"]["mission_weight"] = 0.3
            self.feedback_history.append({"type": "weights", "action": "increased_skills", "message": message})
        
        elif "mission" in message_lower and "more important" in message_lower or \
             "experience" in message_lower and "more important" in message_lower:
            print("   → Increasing mission weight...")
            self.config["skills"]["skills_weight"] = 0.3
            self.config["mission_context"]["mission_weight"] = 0.7
            self.feedback_history.append({"type": "weights", "action": "increased_mission", "message": message})
        
        else:
            print("   ⚠️  Could not parse feedback. Please be more specific.")
            print("   Examples:")
            print("     - 'I don't mind if they don't speak English'")
            print("     - 'Availability is not important'")
            print("     - 'Skills are more important than experience'")
            return None
        
        print("✅ Requirements updated!")
        return self.task_requirements
    
    def _make_language_optional(self, lang_code: str):
        """Make a language optional (not required)."""
        if 'languages' not in self.task_requirements:
            return
        
        for lang in self.task_requirements['languages']:
            if lang.get('language_code') == lang_code:
                lang['required'] = False
                print(f"      {lang_code} is now optional")
    
    def get_top_candidates(self, top_k: int = 5, apply_feedback: bool = True):
        """
        Get top candidates based on current requirements.
        
        Args:
            top_k: Number of top candidates to return
            apply_feedback: Whether to apply feedback modifications
        
        Returns:
            List of (consultant_id, score, details) tuples
        """
        if self.task_requirements is None:
            print("❌ No mission loaded. Please load a mission first.")
            return None
        
        print(f"\n🔍 Computing top {top_k} candidates...")
        
        # Compute skill scores
        result = se.get_consultants_skills_score(
            self.task_requirements,
            self.skill_model,
            self.config["skills_embedding_file"],
            self.config,
            debug=False
        )
        # Handle both return types (with/without debug)
        if isinstance(result, tuple):
            consultants_scores, _ = result
        else:
            consultants_scores = result
        average_skills = se.average_skills(consultants_scores)
        
        # Compute mission scores
        consultant_missions_df = get_consultants_mission_scores(
            self.task_requirements,
            percentile=self.config["mission_context"]["percentile"],
            use_domain_keyword_filtering=self.config.get("domain_filtering", {}).get("use_keyword_filtering", False),
            finetune_model=self.config.get("mission_embedding_model", "paraphrase-multilingual-mpnet-base-v2")
        )
        average_mission = df_to_dict(consultant_missions_df)
        
        # Combine scores
        combined_scores = linear_combination(
            average_skills,
            self.config["skills"]["skills_weight"],
            average_mission,
            self.config["mission_context"]["mission_weight"],
            auto_optimize=False,
            use_learning=False
        )
        
        # Apply language filter
        language_filter = compute_language_score(
            "data/HCK_HEC_LANG.csv",
            self.task_requirements,
            acceptable_difference=self.config["languages"]["acceptable_difference"],
            penality=self.config["languages"]["penality"]
        )
        filtered = apply_filters(combined_scores, language_filter)
        
        # Apply availability filter
        schedule_penalties = calculate_availability_penalty(
            "data/HCK_HEC_STAFFING.csv",
            self.task_requirements,
            self.config["disponibility"]["maximum_penalty"]
        )
        filtered = apply_filters(filtered, schedule_penalties)
        
        # Sort and get top K
        sorted_consultants = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_consultants[:top_k]
        
        # Load ground truth labels
        full_df = pd.read_csv("full_df.csv")
        expected_job_role = self.task_requirements.get("job_role", "Unknown")
        
        results = []
        for cid, score in top_candidates:
            consultant_row = full_df[full_df["USER_ID"] == cid]
            ground_truth = consultant_row.iloc[0]["JOB_RULE"] if len(consultant_row) > 0 else "Unknown"
            is_correct = str(ground_truth).strip() == str(expected_job_role).strip()
            
            results.append({
                "consultant_id": int(cid),
                "score": float(score),
                "ground_truth": str(ground_truth),
                "is_correct": is_correct
            })
        
        self.last_results = results
        return results
    
    def show_results(self):
        """Display the last results in a nice format."""
        if self.last_results is None:
            print("❌ No results yet. Run get_top_candidates() first.")
            return
        
        expected_job_role = self.task_requirements.get("job_role", "Unknown")
        correct_count = sum(1 for r in self.last_results if r["is_correct"])
        
        print("\n" + "="*60)
        print(f"TOP {len(self.last_results)} CANDIDATES")
        print("="*60)
        print(f"Expected Job Role: {expected_job_role}\n")
        
        for i, result in enumerate(self.last_results, 1):
            status = "✅" if result["is_correct"] else "❌"
            print(f"{i}. Consultant {result['consultant_id']}: "
                  f"score={result['score']:.4f}, "
                  f"ground_truth='{result['ground_truth']}' {status}")
        
        print()
        print("="*60)
        print(f"ACCURACY: {correct_count}/{len(self.last_results)} = {correct_count/len(self.last_results)*100:.1f}%")
        print("="*60)
    
    def show_feedback_history(self):
        """Show all feedback given so far."""
        if not self.feedback_history:
            print("No feedback given yet.")
            return
        
        print("\n📝 Feedback History:")
        for i, feedback in enumerate(self.feedback_history, 1):
            print(f"  {i}. [{feedback['type']}] {feedback.get('action', 'unknown')}: {feedback.get('message', '')}")


def main():
    """Interactive CLI interface."""
    print("="*60)
    print("🤖 INTERACTIVE CONSULTANT MATCHING AGENT")
    print("="*60)
    print()
    print("Commands:")
    print("  load <pdf_path>     - Load a mission from PDF")
    print("  chat <message>      - Give feedback (e.g., 'I don't mind if they don't speak English')")
    print("  search [top_k]      - Get top candidates (default: 5)")
    print("  show                - Show last results")
    print("  feedback            - Show feedback history")
    print("  reset               - Reset to original requirements")
    print("  quit                - Exit")
    print()
    
    agent = InteractiveMatchingAgent()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'show':
                agent.show_results()
            
            elif user_input.lower() == 'feedback':
                agent.show_feedback_history()
            
            elif user_input.lower() == 'reset':
                if agent.original_requirements:
                    agent.task_requirements = json.loads(json.dumps(agent.original_requirements))
                    agent.feedback_history = []
                    print("✅ Reset to original requirements")
                else:
                    print("❌ No original requirements to reset to")
            
            elif user_input.startswith('load '):
                pdf_path = user_input[5:].strip()
                if not os.path.exists(pdf_path):
                    print(f"❌ File not found: {pdf_path}")
                else:
                    agent.load_mission(pdf_path)
            
            elif user_input.startswith('chat '):
                message = user_input[5:].strip()
                agent.chat(message)
            
            elif user_input.startswith('search'):
                parts = user_input.split()
                top_k = int(parts[1]) if len(parts) > 1 else 5
                results = agent.get_top_candidates(top_k)
                if results:
                    agent.show_results()
            
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

