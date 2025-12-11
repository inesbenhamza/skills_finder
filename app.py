import streamlit as st
import yaml
import tempfile
import pandas as pd
import numpy as np
import torch
import sys
import os
import copy
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# --- IMPORT YOUR EXISTING PIPELINE ---
from sentence_transformers import SentenceTransformer
from helpers.SkillsEmbeddings import SkillEmbeddingModel, get_consultants_skills_score, average_skills, sort_skills
from src.mission_embeddings import get_consultants_mission_scores, df_to_dict
from src.language_filtering import compute_language_score, apply_filters
from src.schedule_filtering import calculate_availability_penalty
from helpers.tools_extraction import get_model_response, generate_yaml
from mainfile.main_debug import linear_combination

MATCHING_CONFIGURATION_FILE = "config/search_config.yaml"

# Custom CSS for better styling
st.set_page_config(
    page_title="AI Consultant Matching Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .consultant-card {
        background-color: transparent;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        border: 1px solid #4a5568;
    }
    .skill-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        margin: 0.25rem;
        font-size: 0.85rem;
    }
    .mission-text {
        background-color: transparent;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #4a5568;
        margin-top: 0.5rem;
    }
    .language-badge {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        margin: 0.25rem;
        font-size: 0.85rem;
    }
    .availability-good {
        color: #2e7d32;
        font-weight: bold;
    }
    .availability-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .availability-poor {
        color: #d32f2f;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# Load data files (cached)
# ----------------------------------------------------
@st.cache_data
def load_consultant_data():
    """Load all consultant data files."""
    skills_df = pd.read_csv("data/HCK_HEC_SKILLS.csv")
    missions_df = pd.read_csv("data/HCK_HEC_XP.csv")
    languages_df = pd.read_csv("data/HCK_HEC_LANG.csv")
    staffing_df = pd.read_csv("data/HCK_HEC_STAFFING.csv")
    return skills_df, missions_df, languages_df, staffing_df


def get_consultant_skills(user_id: int, skills_df: pd.DataFrame) -> List[Dict]:
    """Get all skills for a consultant with their levels."""
    consultant_skills = skills_df[skills_df['USER_ID'] == user_id]
    skills_list = []
    for _, row in consultant_skills.iterrows():
        skills_list.append({
            'skill': row['SKILLS_DSC'],
            'domain': row['DOMAIN_DSC'],
            'level': row['LEVEL_VAL']
        })
    return skills_list


def get_consultant_missions(user_id: int, missions_df: pd.DataFrame, max_missions: int = 3) -> List[str]:
    """Get recent missions for a consultant."""
    consultant_missions = missions_df[missions_df['USER_ID'] == user_id]['MISSION_DSC'].tolist()
    # Return most recent missions (last N)
    return consultant_missions[-max_missions:] if len(consultant_missions) > max_missions else consultant_missions


def get_consultant_languages(user_id: int, languages_df: pd.DataFrame) -> List[Dict]:
    """Get languages for a consultant."""
    consultant_langs = languages_df[languages_df['USER_ID'] == user_id]
    languages_list = []
    for _, row in consultant_langs.iterrows():
        languages_list.append({
            'language': row['LANGUAGE_SKILL_DSC'],
            'level': row['LANGUAGE_SKILL_LVL']
        })
    return languages_list


def get_consultant_availability(user_id: int, staffing_df: pd.DataFrame) -> Dict:
    """Get availability for a consultant (12 months)."""
    consultant_staffing = staffing_df[staffing_df['USER_ID'] == user_id]
    if consultant_staffing.empty:
        return {'months': [0] * 12, 'avg_availability': 0, 'status': 'Unknown'}
    
    row = consultant_staffing.iloc[0]
    months = [row[f'MONTH_{i}'] for i in range(1, 13)]
    avg_availability = np.mean(months)
    
    if avg_availability >= 50:
        status = 'Good'
    elif avg_availability >= 25:
        status = 'Medium'
    else:
        status = 'Limited'
    
    return {
        'months': months,
        'avg_availability': avg_availability,
        'status': status
    }


def display_consultant_profile(user_id: int, rank: int, score: float, 
                               skills_df: pd.DataFrame, missions_df: pd.DataFrame,
                               languages_df: pd.DataFrame, staffing_df: pd.DataFrame,
                               consultant_skill_scores: Dict):
    """Display a detailed consultant profile card."""
    st.markdown(f"""
    <div class="consultant-card">
        <h3>#{rank} - Consultant ID: {user_id} | Score: {score:.4f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Skills Section
        st.subheader("🛠️ Skills")
        skills = get_consultant_skills(user_id, skills_df)
        
        # Get skill scores for this consultant
        skill_scores_dict = {}
        for skill_name, scores_dict in consultant_skill_scores.items():
            if user_id in scores_dict:
                skill_scores_dict[skill_name] = scores_dict[user_id]
        
        if skills:
            # Group by domain
            domains = {}
            for skill_info in skills:
                domain = skill_info['domain']
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append(skill_info)
            
            for domain, domain_skills in domains.items():
                st.markdown(f"**{domain}:**")
                skill_text = ""
                for skill_info in domain_skills[:10]:  # Show top 10 skills per domain
                    skill_name = skill_info['skill']
                    level = skill_info['level']
                    # Check if this skill matches any required skill
                    match_score = ""
                    for req_skill, score in skill_scores_dict.items():
                        if skill_name.lower() in req_skill.lower() or req_skill.lower() in skill_name.lower():
                            match_score = f" (Match: {score:.3f})"
                            break
                    
                    skill_text += f'<span class="skill-badge">{skill_name} (Level: {level}/100){match_score}</span>'
                st.markdown(skill_text, unsafe_allow_html=True)
                if len(domain_skills) > 10:
                    st.caption(f"... and {len(domain_skills) - 10} more skills")
        else:
            st.info("No skills data available")
        
        # Missions Section
        st.subheader("📋 Recent Missions")
        missions = get_consultant_missions(user_id, missions_df, max_missions=3)
        if missions:
            for i, mission in enumerate(missions, 1):
                st.markdown(f"""
                <div class="mission-text">
                    <strong>Mission {i}:</strong><br>
                    {mission[:500]}{'...' if len(mission) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No mission history available")
    
    with col2:
        # Languages Section
        st.subheader("🌍 Languages")
        languages = get_consultant_languages(user_id, languages_df)
        if languages:
            lang_text = ""
            for lang_info in languages:
                lang_text += f'<span class="language-badge">{lang_info["language"]}: {lang_info["level"]}/100</span>'
            st.markdown(lang_text, unsafe_allow_html=True)
        else:
            st.info("No language data")
        
        # Availability Section
        st.subheader("📅 Availability (Next 12 Months)")
        availability = get_consultant_availability(user_id, staffing_df)
        
        if availability['status'] == 'Good':
            status_class = 'availability-good'
            status_icon = '✅'
        elif availability['status'] == 'Medium':
            status_class = 'availability-medium'
            status_icon = '⚠️'
        else:
            status_class = 'availability-poor'
            status_icon = '❌'
        
        st.markdown(f"""
        <div>
            <p><strong>{status_icon} Status: <span class="{status_class}">{availability['status']}</span></strong></p>
            <p>Average Availability: <strong>{availability['avg_availability']:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show monthly availability as a simple bar
        months_data = pd.DataFrame({
            'Month': [f'M{i}' for i in range(1, 13)],
            'Availability': availability['months']
        })
        st.bar_chart(months_data.set_index('Month'))
    
    st.markdown("---")


# ----------------------------------------------------
# Load configuration + embedding model once
# ----------------------------------------------------
@st.cache_resource
def load_config_and_models():
    with open(MATCHING_CONFIGURATION_FILE, "r") as f:
        config = yaml.safe_load(f)

    # Load embedding model
    embedding_model = SentenceTransformer(config["embedding_model"])
    skill_model = SkillEmbeddingModel(embedding_model)

    return config, skill_model


# ----------------------------------------------------
# Step 1 — Extract YAML from PDF using LLM
# ----------------------------------------------------
def extract_requirements_from_pdf(uploaded_file, config):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Run LLM extraction
    model_name = config["model"]["name"]
    think_flag = config["model"]["think"]

    st.info("📄 Extracting skills & requirements from PDF using LLM…")

    model_response = get_model_response(pdf_path, model_name)
    task_requirements = generate_yaml(model_response, think=think_flag)

    return task_requirements


# ----------------------------------------------------
# Step 2 — Full ranking pipeline
# ----------------------------------------------------
def compute_best_consultants(task_requirements, config, skill_model):
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🔍 Scoring consultant skills…")
    progress_bar.progress(20)
    consultant_skill_scores = get_consultants_skills_score(
        task_requirements,
        skill_model,
        config["skills_embedding_file"],
        config
    )
    avg_skill_scores = average_skills(consultant_skill_scores)

    status_text.text("📊 Scoring mission similarity…")
    progress_bar.progress(40)
    mission_scores = df_to_dict(get_consultants_mission_scores(
        task_requirements,
        percentile=config["mission_context"]["percentile"],
        use_domain_keyword_filtering=config.get("domain_filtering", {}).get("use_keyword_filtering", False),
        finetune_model=config.get("mission_embedding_model", "paraphrase-multilingual-mpnet-base-v2")
    ))

    status_text.text("⚖️ Combining scores with adaptive weights…")
    progress_bar.progress(60)
    # Use adaptive weight optimization if enabled
    auto_optimize = config.get("weight_optimization", {}).get("enabled", False)
    use_learning = config.get("weight_optimization", {}).get("use_learning", False)
    optimize_method = config.get("weight_optimization", {}).get("method", "variance")
    labels_file = config.get("weight_optimization", {}).get("labels_file", "full_df.csv")
    
    if use_learning:
        status_text.text("🤖 Using supervised learning to predict optimal weights...")
    elif auto_optimize:
        status_text.text("⚙️ Auto-optimizing weights...")
    else:
        status_text.text("📊 Using fixed weights from config...")
    
    final_scores = linear_combination(
        avg_skill_scores,
        config["skills"]["skills_weight"],
        mission_scores,
        config["mission_context"]["mission_weight"],
        auto_optimize=auto_optimize,
        optimize_method=optimize_method,
        use_learning=use_learning,
        task_requirements=task_requirements,
        labels_file=labels_file
    )

    status_text.text("🔍 Applying language & availability filters...")
    progress_bar.progress(80)
    language_filter = compute_language_score(
        "data/HCK_HEC_LANG.csv",
        task_requirements,
        acceptable_difference=config["languages"]["acceptable_difference"],
        penality=config["languages"]["penality"]
    )
    availability_penalty = calculate_availability_penalty(
        "data/HCK_HEC_STAFFING.csv",
        task_requirements,
        config["disponibility"]["maximum_penalty"]
    )

    filtered = apply_filters(final_scores, language_filter)
    filtered = apply_filters(filtered, availability_penalty)

    progress_bar.progress(100)
    status_text.text("✅ Ranking complete!")
    
    return sort_skills(filtered), consultant_skill_scores


# ----------------------------------------------------
# Process feedback and update requirements
# ----------------------------------------------------
def process_feedback(feedback: str, task_requirements: Dict, config: Dict):
    """Process user feedback and update requirements/config."""
    feedback_lower = feedback.lower()
    updates = []
    
    # Language feedback - Check French FIRST (before English) to avoid "french" matching "en"
    if any(phrase in feedback_lower for phrase in ["don't mind", "don't care", "not important", "not required", "optional"]) and \
       any(word in feedback_lower for word in ["french", "fr"]) and "english" not in feedback_lower:
        for lang_req in task_requirements.get('languages', []):
            if lang_req['language_code'].upper() == 'FR':
                lang_req['required'] = False
                updates.append("French is now optional")
                break
    
    elif any(phrase in feedback_lower for phrase in ["don't mind", "don't care", "not important", "not required", "optional"]) and \
         ("english" in feedback_lower or (" en " in feedback_lower or feedback_lower.startswith("en ") or feedback_lower.endswith(" en"))):
        for lang_req in task_requirements.get('languages', []):
            if lang_req['language_code'].upper() == 'EN':
                lang_req['required'] = False
                updates.append("English is now optional")
                break
    
    # Availability feedback
    elif any(phrase in feedback_lower for phrase in ["don't mind", "don't care", "not important", "availability", "schedule"]):
        config["disponibility"]["maximum_penalty"] = 0.1
        updates.append("Availability penalty reduced")
    
    # Weight adjustments (handle typos like "skilss" -> "skills")
    elif any(phrase in feedback_lower for phrase in ["skills are more important", "skills more important", "skill are more important", "skill more important"]) or \
         (any(word in feedback_lower for word in ["skill", "skil"]) and "more important" in feedback_lower and "experience" in feedback_lower):
        config["skills"]["skills_weight"] = 0.7
        config["mission_context"]["mission_weight"] = 0.3
        config["weight_optimization"]["enabled"] = False
        config["weight_optimization"]["use_learning"] = False
        updates.append("Skills weight increased to 0.7, mission weight decreased to 0.3")
    
    elif "mission experience is more important" in feedback_lower or \
         ("experience" in feedback_lower and "more important" in feedback_lower and "mission" in feedback_lower):
        config["skills"]["skills_weight"] = 0.3
        config["mission_context"]["mission_weight"] = 0.7
        config["weight_optimization"]["enabled"] = False
        config["weight_optimization"]["use_learning"] = False
        updates.append("Mission weight increased to 0.7, skills weight decreased to 0.3")
    
    return updates


# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------
def main():
    st.markdown('<div class="main-header">AI Consultant Matching Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        Upload a mission PDF, get matches, and refine with conversational feedback!
    </div>
    """, unsafe_allow_html=True)

    config, skill_model = load_config_and_models()
    
    # Initialize session state
    if 'task_requirements' not in st.session_state:
        st.session_state.task_requirements = None
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'consultant_skill_scores' not in st.session_state:
        st.session_state.consultant_skill_scores = None
    if 'config' not in st.session_state:
        st.session_state.config = copy.deepcopy(config)
    
    # Sidebar with configuration info
    with st.sidebar:
        st.header("Configuration")
        st.info(f"**Model:** {config['embedding_model']}")
        
        weight_method = "Supervised Learning" if config.get("weight_optimization", {}).get("use_learning", False) else \
                       "Auto-Optimization" if config.get("weight_optimization", {}).get("enabled", False) else \
                       "Fixed Weights"
        st.info(f"**Weight Method:** {weight_method}")
        
        st.markdown("---")
        st.markdown("### Feedback Examples:")
        st.markdown("""
        - "I don't mind if they don't speak English"
        - "I don't mind if they don't speak French"
        - "Availability is not important"
        - "Skills are more important than experience"
        - "Mission experience is more important"
        """)
        
        st.markdown("---")
        st.markdown("### 📊 How it works:")
        st.markdown("""
        1. **Upload PDF** - Mission description
        2. **Extract Requirements** - LLM extracts skills & requirements
        3. **Match Consultants** - Uses hierarchical embeddings
        4. **Give Feedback** - Refine requirements conversationally
        5. **Re-run Matching** - Get updated results
        6. **View Accuracy** - See ground truth labels
        """)

    # Main content area
    tab1, tab2 = st.tabs(["Upload & Match", "Interactive Feedback"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload Mission PDF", type=["pdf"], help="Upload a PDF file containing the mission description", key="pdf_uploader")

        if uploaded_file:
            # Check if this is a new file (by comparing file name)
            current_file_name = uploaded_file.name
            if 'current_file_name' not in st.session_state or st.session_state.current_file_name != current_file_name:
                # New file uploaded - reset state
                st.session_state.current_file_name = current_file_name
                st.session_state.task_requirements = None
                st.session_state.feedback_history = []
                st.session_state.results = None
                st.session_state.consultant_skill_scores = None
                st.session_state.config = copy.deepcopy(config)
            
            if st.session_state.task_requirements is None:
                st.info("Processing uploaded file…")
                # Step 1: Extract requirements using LLM
                with st.expander("Extracted Requirements", expanded=True):
                    task_requirements = extract_requirements_from_pdf(uploaded_file, config)
                    
                    # Infer job role from filename
                    filename = uploaded_file.name.lower()
                    if 'job_role' not in task_requirements or not task_requirements.get('job_role'):
                        if 'finance' in filename:
                            task_requirements['job_role'] = 'Finance'
                        elif 'dataengineer' in filename or 'data_engineer' in filename:
                            task_requirements['job_role'] = 'Data Engineer'
                        elif 'dataanalyst' in filename or 'data_analyst' in filename:
                            task_requirements['job_role'] = 'Data Analyst'
                        elif 'scrum' in filename:
                            task_requirements['job_role'] = 'Scrum Master'
                        elif 'healthcare' in filename:
                            task_requirements['job_role'] = 'Healthcare'
                        elif 'marketing' in filename:
                            task_requirements['job_role'] = 'Marketing'
                    
                    st.json(task_requirements)
                    st.session_state.task_requirements = task_requirements
            else:
                with st.expander("Extracted Requirements", expanded=False):
                    st.json(st.session_state.task_requirements)

            if st.button("Compute Best Matches", type="primary", use_container_width=True):
                # Load data
                skills_df, missions_df, languages_df, staffing_df = load_consultant_data()
                
                # Compute matches
                results, consultant_skill_scores = compute_best_consultants(
                    st.session_state.task_requirements, 
                    st.session_state.config, 
                    skill_model
                )
                
                st.session_state.results = results
                st.session_state.consultant_skill_scores = consultant_skill_scores

                # Display summary table
                st.markdown("## Top 5 Consultant Matches (Summary)")
                df = pd.DataFrame(results[:5], columns=["User ID", "Score"])
                df["Score"] = df["Score"].round(4)
                df.index = range(1, len(df) + 1)
                df.index.name = "Rank"
                st.dataframe(df, use_container_width=True)
                
                # Show accuracy with ground truth
                if st.session_state.task_requirements.get('job_role'):
                    expected_job_role = st.session_state.task_requirements['job_role']
                    full_df = pd.read_csv("full_df.csv")
                    
                    correct = 0
                    accuracy_data = []
                    for i, (user_id, score) in enumerate(results[:5], 1):
                        ground_truth = full_df[full_df['USER_ID'] == user_id]['JOB_RULE'].iloc[0] if user_id in full_df['USER_ID'].values else 'N/A'
                        is_correct = ground_truth == expected_job_role
                        if is_correct:
                            correct += 1
                        accuracy_data.append({
                            "Rank": i,
                            "Consultant ID": user_id,
                            "Score": f"{score:.4f}",
                            "Ground Truth": ground_truth,
                            "Match": "✅" if is_correct else "❌"
                        })
                    
                    accuracy = (correct / len(results[:5])) * 100 if results else 0
                    st.markdown(f"### Accuracy: {correct}/5 = **{accuracy:.1f}%**")
                    st.dataframe(pd.DataFrame(accuracy_data), use_container_width=True)
                
                # Show weight information
                if st.session_state.config.get("weight_optimization", {}).get("use_learning", False):
                    st.success("Used supervised learning to predict optimal weights from historical data")
                elif st.session_state.config.get("weight_optimization", {}).get("enabled", False):
                    st.success("Used auto-optimization to find optimal weights")
                
                st.markdown("---")
                
                # Display detailed profiles for top 5
                st.markdown("## Detailed Profiles: Top 5 Consultants")
                
                for rank, (user_id, score) in enumerate(results[:5], 1):
                    display_consultant_profile(
                        user_id, rank, score,
                        skills_df, missions_df, languages_df, staffing_df,
                        consultant_skill_scores
                    )

                st.success("Matching complete! Go to 'Interactive Feedback' tab to refine results.")
    
    with tab2:
        if st.session_state.task_requirements is None:
            st.warning("Please upload a PDF and compute matches first in the 'Upload & Match' tab.")
        else:
            st.markdown("### Provide Feedback to Refine Requirements")
            st.info("Example: 'I don't mind if they don't speak English' or 'Skills are more important than experience'")
            
            # Chat interface
            feedback = st.text_input("Your feedback:", placeholder="Type your feedback here...")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Send Feedback", type="primary"):
                    if feedback:
                        updates = process_feedback(feedback, st.session_state.task_requirements, st.session_state.config)
                        st.session_state.feedback_history.append({
                            "feedback": feedback,
                            "updates": updates,
                            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                        })
                        if updates:
                            for update in updates:
                                st.success(update)
                        else:
                            st.warning("Could not understand feedback. Try: 'I don't mind if they don't speak English'")
            
            # Feedback history
            if st.session_state.feedback_history:
                st.markdown("### Feedback History")
                for i, entry in enumerate(reversed(st.session_state.feedback_history), 1):
                    with st.expander(f"💬 {entry['feedback']} ({entry['timestamp']})"):
                        if entry['updates']:
                            for update in entry['updates']:
                                st.write(update)
                        else:
                            st.write("No changes applied")
            
            # Re-run matching button
            if st.session_state.feedback_history:
                st.markdown("---")
                if st.button("Re-run Matching with Updated Requirements", type="primary", use_container_width=True):
                    # Load data
                    skills_df, missions_df, languages_df, staffing_df = load_consultant_data()
                    
                    # Compute matches with updated requirements
                    results, consultant_skill_scores = compute_best_consultants(
                        st.session_state.task_requirements, 
                        st.session_state.config, 
                        skill_model
                    )
                    
                    st.session_state.results = results
                    st.session_state.consultant_skill_scores = consultant_skill_scores
                    
                    st.success("Matching updated! Scroll up to see new results.")
                    
                    # Show updated results
                    st.markdown("## Updated Top 5 Consultant Matches")
                    df = pd.DataFrame(results[:5], columns=["User ID", "Score"])
                    df["Score"] = df["Score"].round(4)
                    df.index = range(1, len(df) + 1)
                    df.index.name = "Rank"
                    st.dataframe(df, use_container_width=True)
                    
                    # Show updated accuracy
                    if st.session_state.task_requirements.get('job_role'):
                        expected_job_role = st.session_state.task_requirements['job_role']
                        full_df = pd.read_csv("full_df.csv")
                        
                        correct = 0
                        accuracy_data = []
                        for i, (user_id, score) in enumerate(results[:5], 1):
                            ground_truth = full_df[full_df['USER_ID'] == user_id]['JOB_RULE'].iloc[0] if user_id in full_df['USER_ID'].values else 'N/A'
                            is_correct = ground_truth == expected_job_role
                            if is_correct:
                                correct += 1
                            accuracy_data.append({
                                "Rank": i,
                                "Consultant ID": user_id,
                                "Score": f"{score:.4f}",
                                "Ground Truth": ground_truth,
                                "Match": "✅" if is_correct else "❌"
                            })
                        
                        accuracy = (correct / len(results[:5])) * 100 if results else 0
                        st.markdown(f"### Updated Accuracy: {correct}/5 = **{accuracy:.1f}%**")
                        st.dataframe(pd.DataFrame(accuracy_data), use_container_width=True)


if __name__ == "__main__":
    main()
