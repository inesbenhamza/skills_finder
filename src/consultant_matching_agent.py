"""
Autonomous Consultant Matching Agent

This agent autonomously handles the entire consultant matching pipeline:
- Processes mission requirements
- Makes intelligent decisions about weights and parameters
- Learns from feedback and improves over time
- Handles errors and adapts
- Explains its reasoning
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder
from helpers import SkillsEmbeddings as se
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict
from src.language_filtering import compute_language_score, apply_filters
from src.schedule_filtering import calculate_availability_penalty
from helpers.tools_extraction import generate_yaml, get_model_response


@dataclass
class AgentState:
    """Agent's internal state and memory"""
    mission_history: List[Dict] = None
    performance_metrics: Dict[str, float] = None
    learned_weights: Dict[str, Tuple[float, float]] = None  # job_role -> (skills_w, mission_w)
    error_history: List[Dict] = None
    feedback_history: List[Dict] = None
    
    def __post_init__(self):
        if self.mission_history is None:
            self.mission_history = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.learned_weights is None:
            self.learned_weights = {}
        if self.error_history is None:
            self.error_history = []
        if self.feedback_history is None:
            self.feedback_history = []


class ConsultantMatchingAgent:
    """
    Autonomous agent for consultant matching.
    
    Capabilities:
    - Autonomous pipeline execution
    - Adaptive decision-making
    - Learning from feedback
    - Error handling and recovery
    - Performance tracking
    - Explainable decisions
    """
    
    def __init__(self, config_path: str = "config/search_config.yaml", 
                 state_file: str = "agent_state.pkl",
                 verbose: bool = True):
        """
        Initialize the agent.
        
        Args:
            config_path: Path to configuration file
            state_file: Path to save/load agent state
            verbose: Print detailed logs
        """
        self.config_path = config_path
        self.state_file = state_file
        self.verbose = verbose
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load or initialize state
        self.state = self._load_state() or AgentState()
        
        # Initialize models (lazy loading)
        self.bi_encoder = None
        self.skill_model = None
        self.cross_encoder = None
        
        # Performance tracking
        self.current_mission_id = None
        self.current_results = None
        
    def _log(self, message: str, level: str = "INFO"):
        """Logging utility"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def _load_state(self) -> Optional[AgentState]:
        """Load agent state from disk"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state_dict = pickle.load(f)
                    return AgentState(**state_dict)
            except Exception as e:
                self._log(f"Failed to load state: {e}", "WARNING")
        return None
    
    def _save_state(self):
        """Save agent state to disk"""
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(asdict(self.state), f)
            self._log("State saved successfully")
        except Exception as e:
            self._log(f"Failed to save state: {e}", "ERROR")
    
    def _load_models(self):
        """Lazy load models when needed"""
        if self.bi_encoder is None:
            self._log("Loading bi-encoder model...")
            self.bi_encoder = SentenceTransformer(self.config["embedding_model"])
            self.skill_model = se.SkillEmbeddingModel(self.bi_encoder)
            self._log(f"✓ Loaded bi-encoder: {self.config['embedding_model']}")
        
        if self.cross_encoder is None:
            self._log("Loading cross-encoder model...")
            self.cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
            self._log("✓ Loaded cross-encoder")
    
    def _decide_weights(self, task_requirements: Dict, 
                       average_skills: Dict, 
                       average_mission: Dict) -> Tuple[float, float]:
        """
        Autonomous decision-making for optimal weights.
        
        Uses multiple strategies:
        1. Check learned weights from similar missions
        2. Use supervised learning if labels available
        3. Fall back to optimization
        4. Use config defaults as last resort
        """
        self._log("🤔 Deciding optimal weights...")
        
        # Strategy 1: Check learned weights from similar missions
        job_role = self._extract_job_role(task_requirements)
        if job_role and job_role in self.state.learned_weights:
            weights = self.state.learned_weights[job_role]
            self._log(f"✓ Using learned weights for {job_role}: {weights}")
            return weights
        
        # Strategy 2: Try supervised learning
        if self.config.get("weight_optimization", {}).get("use_learning", False):
            try:
                import sys
                from pathlib import Path
                # Add parent directory to path for imports
                parent_dir = Path(__file__).parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))
                from mainfile.main_debug import learn_optimal_weights_from_labels
                weights = learn_optimal_weights_from_labels(
                    task_requirements, average_skills, average_mission,
                    labels_file=self.config.get("weight_optimization", {}).get("labels_file", "full_df.csv"),
                    top_k=5
                )
                # Store learned weights
                if job_role:
                    self.state.learned_weights[job_role] = weights
                self._log(f"✓ Learned optimal weights: {weights}")
                return weights
            except Exception as e:
                self._log(f"Learning failed: {e}, trying optimization...", "WARNING")
        
        # Strategy 3: Use optimization
        if self.config.get("weight_optimization", {}).get("enabled", False):
            try:
                import sys
                from pathlib import Path
                parent_dir = Path(__file__).parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))
                from mainfile.main_debug import optimize_weights
                method = self.config.get("weight_optimization", {}).get("method", "variance")
                weights = optimize_weights(average_skills, average_mission, method=method)
                if job_role:
                    self.state.learned_weights[job_role] = weights
                self._log(f"✓ Optimized weights: {weights}")
                return weights
            except Exception as e:
                self._log(f"Optimization failed: {e}, using defaults...", "WARNING")
        
        # Strategy 4: Use config defaults
        skills_w = self.config["skills"]["skills_weight"]
        mission_w = self.config["mission_context"]["mission_weight"]
        self._log(f"✓ Using default weights: ({skills_w}, {mission_w})")
        return skills_w, mission_w
    
    def _extract_job_role(self, task_requirements: Dict) -> Optional[str]:
        """
        Extract job role from requirements.
        First tries explicit fields, then infers from content.
        """
        # Try explicit fields first
        for key in ['job_role', 'role', 'title', 'position']:
            if key in task_requirements and task_requirements[key]:
                return str(task_requirements[key]).strip()
        
        # Infer from responsibilities and technologies
        responsibilities = task_requirements.get('responsabilites', [])
        technologies = [t.get('name', '').lower() for t in task_requirements.get('technologies', [])]
        
        # Combine all text for analysis
        all_text = ' '.join(responsibilities).lower()
        all_text += ' ' + ' '.join(technologies)
        
        # Pattern matching for common job roles
        if any(keyword in all_text for keyword in ['scrum', 'agile', 'sprint', 'ceremonies', 'product owner', 'backlog']):
            return 'Scrum Master'
        elif any(keyword in all_text for keyword in ['data engineer', 'pipeline', 'etl', 'databricks', 'data factory', 'spark']):
            return 'Data Engineer'
        elif any(keyword in all_text for keyword in ['data analyst', 'power bi', 'tableau', 'dashboard', 'reporting', 'kpi']):
            return 'Data Analyst'
        elif any(keyword in all_text for keyword in ['data scientist', 'machine learning', 'ml', 'model', 'prediction']):
            return 'Data Scientist'
        elif any(keyword in all_text for keyword in ['finance', 'financial', 'budget', 'forecast']):
            return 'Finance'
        
        return None
    
    def _build_mission_text(self, task: Dict) -> str:
        """Build mission text for cross-encoder"""
        parts = []
        if "responsabilites" in task:
            parts.append("Responsibilities: " + " ".join(task["responsabilites"]))
        if "activity_sector" in task:
            parts.append(f"Sector: {task['activity_sector']}.")
        if "required_experience" in task:
            parts.append(f"Experience required: {task['required_experience']} years.")
        if "languages" in task:
            langs = [f"{l['language_code']} level {l['level']}" for l in task["languages"]]
            parts.append("Languages required: " + ", ".join(langs))
        if "technologies" in task:
            techs = [f"{t['name']} level {t['level']}" for t in task["technologies"]]
            parts.append("Technologies: " + ", ".join(techs))
        return ". ".join(parts)
    
    def _build_consultant_text(self, row: pd.Series) -> str:
        """Build consultant text for cross-encoder"""
        return (
            f"Candidate profile: "
            f"Skills: {row['SKILLS_DSC']}. "
            f"Domain expertise: {row['DOMAIN_DSC']}. "
            f"Experience level (0-100): {row['LEVEL_VAL']}. "
            f"This candidate has worked on related missions."
        )
    
    def execute_matching(self, mission_input: Any, 
                        mission_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main agent execution: autonomously execute the entire matching pipeline.
        
        Args:
            mission_input: Can be:
                - Path to PDF file
                - Path to YAML file
                - Dict with task requirements
            mission_id: Optional mission identifier
        
        Returns:
            Dict with results, explanations, and metadata
        """
        self.current_mission_id = mission_id or f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._log(f"🚀 Starting autonomous matching for mission: {self.current_mission_id}")
        
        try:
            # Step 1: Load/process mission requirements
            task_requirements = self._process_mission_input(mission_input)
            self._log("✓ Mission requirements processed")
            
            # Step 2: Load models
            self._load_models()
            
            # Step 3: Compute skill scores
            self._log("📊 Computing skill similarity scores...")
            consultants_scores, debug_similarities = se.get_consultants_skills_score(
                task_requirements,
                self.skill_model,
                self.config["skills_embedding_file"],
                self.config,
                debug=True
            )
            average_skills = se.average_skills(consultants_scores)
            self._log(f"✓ Computed skill scores for {len(average_skills)} consultants")
            
            # Step 4: Compute mission scores
            self._log("📊 Computing mission similarity scores...")
            consultant_missions_df = get_consultants_mission_scores(
                task_requirements,
                percentile=self.config["mission_context"]["percentile"],
                use_domain_keyword_filtering=self.config.get("domain_filtering", {}).get("use_keyword_filtering", False),
                finetune_model=self.config.get("mission_embedding_model", "paraphrase-multilingual-mpnet-base-v2")
            )
            average_mission = df_to_dict(consultant_missions_df)
            self._log(f"✓ Computed mission scores for {len(average_mission)} consultants")
            
            # Step 5: Autonomous weight decision
            skills_weight, mission_weight = self._decide_weights(
                task_requirements, average_skills, average_mission
            )
            
            # Step 6: Combine scores
            self._log("📊 Combining scores...")
            # Simple linear combination (weights already decided)
            filler_mission_value = np.mean(list(average_mission.values())) if average_mission else 0
            combined_scores = {}
            for cid in average_skills:
                skill_score = average_skills[cid]
                mission_score = average_mission.get(cid, filler_mission_value)
                combined_scores[cid] = skill_score * skills_weight + mission_score * mission_weight
            
            # Step 7: Apply filters
            self._log("🔍 Applying language and availability filters...")
            language_filter = compute_language_score(
                "data/HCK_HEC_LANG.csv",
                task_requirements,
                acceptable_difference=self.config["languages"]["acceptable_difference"],
                penality=self.config["languages"]["penality"]
            )
            
            schedule_penalties = calculate_availability_penalty(
                "data/HCK_HEC_STAFFING.csv",
                task_requirements,
                self.config["disponibility"]["maximum_penalty"]
            )
            
            filtered = apply_filters(combined_scores, language_filter)
            filtered = apply_filters(filtered, schedule_penalties)
            
            # Step 8: Get top 20 from bi-encoder
            sorted_consultants = se.sort_skills(filtered)
            top20_ids = [cid for cid, _ in sorted_consultants[:20]]
            self._log(f"✓ Top 20 candidates selected")
            
            # Step 9: Cross-encoder reranking
            self._log("🎯 Reranking with cross-encoder...")
            task_text = self._build_mission_text(task_requirements)
            df_profiles = pd.read_csv(self.config["skills_csv"])
            
            pairs = []
            for cid in top20_ids:
                row = df_profiles[df_profiles["USER_ID"] == cid].iloc[0]
                text = self._build_consultant_text(row)
                pairs.append((task_text, text))
            
            scores = self.cross_encoder.predict(pairs)
            final_top5 = sorted(
                list(zip(top20_ids, scores)), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            self._log(f"✓ Final top 5 candidates selected")
            
            # Step 10: Build results
            results = {
                "mission_id": self.current_mission_id,
                "top_5_candidates": [
                    {"user_id": cid, "score": float(score)} 
                    for cid, score in final_top5
                ],
                "top_20_candidates": [
                    {"user_id": cid, "score": float(score)} 
                    for cid, score in sorted_consultants[:20]
                ],
                "weights_used": {
                    "skills_weight": float(skills_weight),
                    "mission_weight": float(mission_weight),
                    "decision_method": "learned" if self._extract_job_role(task_requirements) in self.state.learned_weights else "optimized"
                },
                "statistics": {
                    "total_consultants_evaluated": len(average_skills),
                    "consultants_after_filters": len(filtered),
                    "job_role": self._extract_job_role(task_requirements)
                },
                "explanation": self._generate_explanation(
                    task_requirements, final_top5, skills_weight, mission_weight
                )
            }
            
            self.current_results = results
            
            # Step 11: Record in history
            self._record_mission(task_requirements, results)
            
            # Step 12: Save state
            self._save_state()
            
            self._log("✅ Matching complete!")
            return results
            
        except Exception as e:
            self._log(f"❌ Error during matching: {e}", "ERROR")
            self._record_error(e, mission_input)
            raise
    
    def _process_mission_input(self, mission_input: Any) -> Dict:
        """Process different types of mission input"""
        if isinstance(mission_input, dict):
            return mission_input
        elif isinstance(mission_input, str):
            path = Path(mission_input)
            if path.suffix == '.yaml' or path.suffix == '.yml':
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            elif path.suffix == '.pdf':
                # Extract from PDF
                model_response = get_model_response(str(path), self.config["model"]["name"])
                return generate_yaml(model_response, think=self.config["model"]["think"])
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        else:
            raise ValueError(f"Unsupported input type: {type(mission_input)}")
    
    def _generate_explanation(self, task_requirements: Dict, 
                             top5: List[Tuple], 
                             skills_w: float, 
                             mission_w: float) -> str:
        """Generate human-readable explanation of decisions"""
        job_role = self._extract_job_role(task_requirements) or "Unknown"
        explanation = f"""
        Matching Explanation:
        - Job Role: {job_role}
        - Weights: Skills={skills_w:.2%}, Mission={mission_w:.2%}
        - Top candidate score: {top5[0][1]:.4f}
        - Selected {len(top5)} candidates from {len(task_requirements.get('technologies', []))} required skills
        """
        return explanation.strip()
    
    def _record_mission(self, task_requirements: Dict, results: Dict):
        """Record mission in history"""
        self.state.mission_history.append({
            "mission_id": self.current_mission_id,
            "timestamp": datetime.now().isoformat(),
            "job_role": self._extract_job_role(task_requirements),
            "top_5": results["top_5_candidates"],
            "weights": results["weights_used"],
            "statistics": results["statistics"]
        })
    
    def _record_error(self, error: Exception, mission_input: Any):
        """Record error for learning"""
        self.state.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "mission_id": self.current_mission_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "mission_input": str(mission_input)[:200]  # Truncate
        })
    
    def learn_from_feedback(self, mission_id: str, 
                           feedback: Dict[str, Any]):
        """
        Learn from human feedback to improve future matches.
        
        Args:
            mission_id: Mission identifier
            feedback: Dict with:
                - "selected_candidates": List of user_ids that were actually selected
                - "rejected_candidates": List of user_ids that were rejected
                - "rating": Overall rating (1-5)
        """
        self._log(f"📚 Learning from feedback for mission: {mission_id}")
        
        # Find mission in history
        mission = next(
            (m for m in self.state.mission_history if m["mission_id"] == mission_id),
            None
        )
        
        if not mission:
            self._log(f"Mission {mission_id} not found in history", "WARNING")
            return
        
        # Record feedback
        self.state.feedback_history.append({
            "mission_id": mission_id,
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        })
        
        # Update learned weights based on feedback
        if "selected_candidates" in feedback and mission["job_role"]:
            # Analyze which candidates were selected vs rejected
            top5_ids = [c["user_id"] for c in mission["top_5"]]
            selected_ids = feedback["selected_candidates"]
            rejected_ids = feedback.get("rejected_candidates", [])
            
            overlap = len(set(top5_ids) & set(selected_ids))
            total_selected = len(selected_ids)
            
            # Calculate overlap percentage
            overlap_pct = overlap / total_selected if total_selected > 0 else 0
            
            self._log(f"Feedback analysis: {overlap}/{total_selected} selected in top-5 ({overlap_pct:.1%} overlap)")
            
            # Get current weights
            current_weights = self.state.learned_weights.get(mission["job_role"], (0.5, 0.5))
            skills_w, mission_w = current_weights[0], current_weights[1]
            
            # Adjust weights based on feedback
            if overlap < total_selected * 0.5:  # Less than 50% overlap - poor match
                self._log(f"⚠️  Poor match detected - adjusting weights for {mission['job_role']}")
                # If selected candidates weren't in top-5, we need to rebalance
                # Try to find better weights by slightly adjusting
                # Increase mission weight if skills-based matching missed good candidates
                if rejected_ids and rejected_ids[0] in top5_ids[:2]:  # Top candidate was rejected
                    # Top candidate was wrong - adjust more aggressively
                    mission_w = min(1.0, mission_w + 0.1)
                    skills_w = max(0.0, skills_w - 0.1)
                    self._log(f"  Top candidate rejected - increasing mission weight")
                else:
                    # General poor match - slight adjustment
                    mission_w = min(1.0, mission_w + 0.05)
                    skills_w = max(0.0, skills_w - 0.05)
                    self._log(f"  Adjusting: skills={skills_w:.4f}, mission={mission_w:.4f}")
            elif overlap_pct >= 0.8:  # Good match (80%+ overlap)
                self._log(f"✅ Good match - current weights work well for {mission['job_role']}")
                # Keep current weights, they're working
            else:  # Moderate match (50-80% overlap)
                self._log(f"⚠️  Moderate match - fine-tuning weights for {mission['job_role']}")
                # Fine-tune: if selected candidates are lower in ranking, adjust slightly
                selected_positions = [top5_ids.index(sid) for sid in selected_ids if sid in top5_ids]
                if selected_positions and max(selected_positions) > 2:  # Selected candidates ranked 3-5
                    # Selected candidates were lower ranked - slightly favor mission
                    mission_w = min(1.0, mission_w + 0.03)
                    skills_w = max(0.0, skills_w - 0.03)
                    self._log(f"  Fine-tuning: skills={skills_w:.4f}, mission={mission_w:.4f}")
            
            # Normalize weights to sum to 1
            total = skills_w + mission_w
            if total > 0:
                skills_w /= total
                mission_w /= total
            else:
                skills_w, mission_w = 0.5, 0.5
            
            # Update learned weights
            self.state.learned_weights[mission["job_role"]] = (float(skills_w), float(mission_w))
            self._log(f"✓ Updated weights for {mission['job_role']}: skills={skills_w:.4f}, mission={mission_w:.4f}")
        
        self._save_state()
        self._log("✓ Feedback incorporated")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        return {
            "total_missions": len(self.state.mission_history),
            "total_errors": len(self.state.error_history),
            "total_feedback": len(self.state.feedback_history),
            "learned_weights": self.state.learned_weights,
            "recent_missions": self.state.mission_history[-5:] if self.state.mission_history else [],
            "recent_errors": self.state.error_history[-5:] if self.state.error_history else []
        }
    
    def explain_decision(self, mission_id: Optional[str] = None) -> str:
        """Generate explanation for a decision"""
        if mission_id:
            mission = next(
                (m for m in self.state.mission_history if m["mission_id"] == mission_id),
                None
            )
            if mission:
                return f"""
                Decision Explanation for {mission_id}:
                - Job Role: {mission.get('job_role', 'Unknown')}
                - Weights Used: {mission.get('weights', {})}
                - Top 5 Candidates: {mission.get('top_5', [])}
                """
        
        if self.current_results:
            return self.current_results.get("explanation", "No explanation available")
        
        return "No mission results available"


# ==========================================
# Convenience function for easy usage
# ==========================================
def create_agent(config_path: str = "config/search_config.yaml", 
                 verbose: bool = True) -> ConsultantMatchingAgent:
    """Create and return a configured agent"""
    return ConsultantMatchingAgent(config_path=config_path, verbose=verbose)


if __name__ == "__main__":
    # Example usage
    agent = create_agent()
    
    # Execute matching
    results = agent.execute_matching("output/filtered_skills/Scrum.yaml")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Get performance report
    print("\n" + "="*60)
    print("PERFORMANCE REPORT")
    print("="*60)
    print(json.dumps(agent.get_performance_report(), indent=2))

