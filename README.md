# Skills Finder Project

## Project Structure

```
.
├── src/                          # Core Python modules
│   ├── __init__.py
│   ├── mission_embeddings.py    # Mission similarity calculations
│   ├── language_filtering.py    # Language matching and filtering
│   └── schedule_filtering.py    # Availability/schedule filtering
│
├── helpers/                      # Helper utilities
│   ├── __init__.py
│   ├── SkillsEmbeddings.py      # Skill matching and ranking
│   ├── tools_extraction.py      # PDF extraction and YAML generation
│   └── tree.py                  # Tree structure for skill matching
│
├── mainfile/                     # Main execution scripts
│   ├── main.py                  # Basic matching script
│   ├── main2.py                 # With cross-encoder reranking
│   └── main_debug.py            # Debug version with detailed output
│
├── notebooks/                    # Jupyter notebooks
│   ├── finetuningembeddings.ipynb  # Model fine-tuning (KEEP THIS)
│   ├── data_analysis.ipynb
│   └── LanguageEmbeddings.ipynb
│
├── models/                       # Trained models
│   ├── modelfinetuned/          # Fine-tuned model v1
│   ├── modelfinetuned2/         # Fine-tuned model v2
│   ├── checkpoints/             # Training checkpoints
│   └── skill_model_embeddings_best.pt
│
├── config/                       # Configuration files
│   └── search_config.yaml       # Main configuration
│
├── data/                         # Data files
│   ├── HCK_HEC_*.csv           # Original data files
│   └── raw/                     # Raw/processed data
│       ├── skillscleaned.csv
│       ├── langcleaned.csv
│       └── skills-finder-hackathon-hec-x-sbi/
│
├── documents/                    # PDF documents
│   ├── Data Analyst.pdf
│   ├── Data Engineer.pdf
│   └── Scrum.pdf
│
├── output/                       # Generated output files
│   ├── filtered_skills/          # Extracted skill YAML files
│   └── filtered_skills_without_level/
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file

```

## Quick Start

### Running the main script:
```bash
python mainfile/main_debug.py Scrum
```

### Running the Streamlit app:
```bash
streamlit run app.py
```

## Key Features

- **Automatic Skill Matching**: Matches partial skill names (e.g., "Jira") to full names (e.g., "Atlassian JIRA Software")
- **Multi-stage Ranking**: Skills → Missions → Language → Availability filtering
- **Cross-Encoder Reranking**: Final top-5 using cross-encoder for better accuracy

## Configuration

Edit `config/search_config.yaml` to adjust:
- Embedding model
- Skill matching thresholds
- Weighting factors
- Filter parameters

