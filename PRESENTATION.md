# Consultant Matching System: Technical Deep Dive
## Deep Learning-Based Semantic Matching for Talent Acquisition

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Embedding Models & Fine-Tuning](#4-embedding-models--fine-tuning)
5. [Matching Algorithm](#5-matching-algorithm)
6. [Evaluation & Results](#6-evaluation--results)
7. [Technical Implementation Details](#7-technical-implementation-details)

---

## 1. Project Overview

### Problem Statement
Match consultants to job missions based on:
- **Skills**: Technical competencies and experience levels
- **Mission History**: Past project experience and domain expertise
- **Constraints**: Language requirements, availability, domain fit

### Solution Approach
**Bi-encoder architecture** using Sentence Transformers for semantic similarity matching:
- **Skills Matching**: Embedding-based skill similarity with hierarchical relationships
- **Mission Matching**: Semantic similarity between mission requirements and consultant history
- **Score Combination**: Weighted linear combination with optional optimization
- **Note**: Cross-encoder reranking was tested but removed (didn't improve results, introduced wrong-domain matches)

### Key Technologies
- **Sentence Transformers**: `paraphrase-multilingual-mpnet-base-v2` (base), fine-tuned variants
- **Fine-tuning**: Contrastive learning for domain distinction and skill hierarchy
- **Similarity Metric**: Cosine similarity on normalized embeddings
- **Language**: Python 3.12, PyTorch, Sentence Transformers library

---

## 2. System Architecture

### High-Level Pipeline

```
PDF Mission Description
    ↓
[LLM Extraction] → YAML (skills, responsibilities, languages, etc.)
    ↓
[Skill Matching] → Cosine similarity with consultant skills
    ↓
[Mission Matching] → Cosine similarity with consultant mission history
    ↓
[Score Combination] → Weighted linear combination (α·skills + β·missions)
    ↓
[Filtering] → Language & availability constraints
    ↓
[Ranking] → Top-K candidates
```

### Component Breakdown

#### 2.1 Input Processing
- **PDF Parsing**: Extract mission requirements using LLM (Gemma3:12b)
- **YAML Generation**: Structured output with:
  - Job role, required experience
  - Technologies (skills) with levels and required flags
  - Responsibilities (text descriptions)
  - Language requirements
  - Availability constraints

#### 2.2 Dual Matching System
1. **Skills Matching Module**
   - Embeds mission skill requirements
   - Compares against consultant skill profiles
   - Considers experience levels and required/optional flags

2. **Mission Matching Module**
   - Embeds mission responsibilities
   - Compares against consultant past mission descriptions
   - Uses percentile-based thresholding

#### 2.3 Score Aggregation
- **Linear Combination**: `score = α·skill_score + β·mission_score`
- **Default Weights**: α = 0.5, β = 0.5 (configurable)
- **Optional Optimization**: Variance-based or supervised learning

#### 2.4 Filtering & Ranking
- **Language Filtering**: Penalize consultants below required language levels
- **Availability Filtering**: Penalize fully booked consultants
- **Final Ranking**: Sort by combined score, return top-K

---

## 3. Data Pipeline

### 3.1 Data Sources

#### Consultant Skills (`HCK_HEC_SKILLS.csv`)
- **Format**: `USER_ID, SKILLS_DSC, DOMAIN_DSC, LEVEL_VAL`
- **Size**: 3,532 skill records across 113 consultants
- **Example**: `(29209385, "Dossier Médical Électronique (DMÉ)", "Healthcare Systems", 60.0)`

#### Consultant Missions (`HCK_HEC_XP.csv`)
- **Format**: `USER_ID, MISSION_DSC`
- **Size**: 1,704 unique mission descriptions
- **Purpose**: Historical project experience for semantic matching

#### Consultant Languages (`HCK_HEC_LANG.csv`)
- **Format**: Language proficiency scores (0-100)
- **Usage**: Filter consultants below required levels

#### Consultant Availability (`HCK_HEC_STAFFING.csv`)
- **Format**: Monthly availability percentages
- **Note**: 100% = fully booked (not available), 0% = fully available

#### Ground Truth Labels (`full_df.csv`)
- **Format**: `USER_ID, JOB_RULE`
- **Purpose**: Evaluation only (not used in matching pipeline)
- **Domains**: Data Engineer, Data Analyst, Data Scientist, Scrum Master, Finance, Healthcare, Marketing

### 3.2 Data Preprocessing

#### Skill Normalization
- **Automatic Matching**: "Jira" → "Atlassian JIRA Software"
- **Tree-based Matching**: Handles partial matches and synonyms
- **Exact Match Detection**: Case-insensitive exact matching

#### Mission Deduplication
- Remove duplicate `(USER_ID, MISSION_DSC)` pairs
- Preserve unique mission experiences per consultant

#### Embedding Caching
- Pre-encode all skills → `skills_encoded.npz`
- Pre-encode all missions → `mission_encoded.csv`
- Store model metadata to detect re-encoding needs

---

## 4. Embedding Models & Fine-Tuning

### 4.1 Base Models

#### Skills Embedding Model
- **Base**: `BAAI/bge-m3` (multilingual, 1024 dimensions)
- **Fine-tuned**: `skills_finetuned_hierarchy`
- **Purpose**: Hierarchical skill relationships (e.g., "Jira" ↔ "Atlassian JIRA Software")

#### Mission Embedding Model
- **Base**: `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- **Fine-tuned**: `models/modelfinetuned_domain` (optional, for production)
- **Purpose**: Domain-specific mission similarity

### 4.2 Fine-Tuning Process

#### Skills Hierarchy Fine-Tuning (`notebooks/finetune_skill_hierarchy.py`)

**Objective**: Learn hierarchical relationships between skills so that related skills (e.g., "Jira" and "Atlassian JIRA Software") are close in embedding space, while unrelated skills (e.g., "JIRA" and "Scala") are far apart.

**Step 1: Skill Tree Definition**
The hierarchy is defined as an indented text tree:
```
microsoft
  power bi
  sql server
    reporting services
      ssrs
atlassian
  jira software
scala
```

**Step 2: Tree Parsing**
- Parses indentation to build parent-child relationships
- Creates full paths: "microsoft sql server reporting services"
- Builds hierarchy dictionary: `{parent: [children]}`

**Step 3: Positive Pair Generation**
Creates pairs that should be **similar** in embedding space:

a) **Direct Parent-Child**:
   - "microsoft" ↔ "microsoft power bi"
   - "microsoft" ↔ "microsoft sql server"

b) **Grandparent-Grandchild**:
   - "microsoft" ↔ "microsoft sql server reporting services"
   - "microsoft sql server" ↔ "microsoft sql server reporting services"

c) **Siblings** (children of same parent):
   - "microsoft power bi" ↔ "microsoft sql server"
   - "microsoft sql server reporting services" ↔ "microsoft sql server integration services"

**Step 4: Negative Pair Generation**
Creates pairs that should be **dissimilar** in embedding space:

a) **Root-Level Skills** (CRITICAL FIX):
   - All root-level skills (not children) are explicitly negative pairs
   - Example: "atlassian" ↔ "scala" (completely unrelated)
   - Example: "microsoft" ↔ "oracle" (different vendors)
   - **This prevents spurious matches** like JIRA ↔ Scala

b) **Unrelated Skills**:
   - Skills from different branches with no common words
   - Negative ratio: **5:1** (5 negatives per positive)
   - Ensures model learns to distinguish unrelated skills

**Step 5: Training with Contrastive Loss**

**Loss Function**:
```
L = max(0, margin - sim(positive_pair) + sim(negative_pair))
```

**Parameters**:
- **Margin**: 1.0 (increased from 0.5 for stronger separation)
- **Distance Metric**: Cosine distance
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Epochs**: 5
- **Batch Size**: 16

**How It Works**:
1. For positive pairs: Minimizes distance (makes them similar)
2. For negative pairs: Maximizes distance (makes them dissimilar)
3. Margin ensures negative pairs are at least `margin` distance apart

**Example Training Pairs**:

**Positive (label=1.0)**:
- "Atlassian" ↔ "Atlassian JIRA Software"
- "Microsoft" ↔ "Microsoft Power BI"
- "Microsoft SQL Server" ↔ "Microsoft SQL Server Reporting Services"

**Negative (label=0.0)**:
- "Atlassian" ↔ "Scala"
- "Microsoft" ↔ "Oracle"
- "JIRA Software" ↔ "Scala"

**Step 6: Key Improvements Applied**

1. **Root-Level Negative Pairs**:
   - Ensures all root skills are explicitly negative
   - Prevents unrelated skills from becoming similar
   - Fixed bug where JIRA ↔ Scala had 0.99 similarity

2. **Higher Negative Ratio** (5:1):
   - More negative examples than positive
   - Forces model to learn better discrimination
   - Prevents overfitting to positive relationships

3. **Stronger Margin** (1.0):
   - Ensures unrelated skills are at least distance 1.0 apart
   - Creates clearer separation in embedding space

**Result**: 
- Related skills are close: "Jira" ↔ "Atlassian JIRA Software" (high similarity)
- Unrelated skills are far: "JIRA" ↔ "Scala" (similarity < 0.6, down from 0.99)
- Prevents spurious matches in skill matching

#### Mission Domain Fine-Tuning (`notebooks/finetune_mission_embeddings_domain.py`)

**Objective**: Distinguish between domains (Healthcare, Data Engineer, etc.)

**Training Data**:
- **Positive Pairs**: Same-domain missions
- **Negative Pairs**: Different-domain missions
- **Source**: Ground truth labels (`JOB_RULE` from `full_df.csv`)

**Loss Function**: Contrastive Loss with domain distinction

**Note**: Uses ground truth labels → "cheating" for evaluation, but improves production performance

**Result**: Better domain separation, but base model often sufficient when skills dominate

### 4.3 Embedding Normalization

All embeddings are **L2-normalized**:
```python
emb_normalized = emb / ||emb||
```

**Why?**
- Cosine similarity becomes dot product: `cos_sim(a, b) = a · b` (for normalized vectors)
- Computationally efficient
- Scale-invariant (magnitude doesn't affect similarity)

---

## 5. Matching Algorithm

### 5.1 Skills Matching

#### Step 1: Skill Name Matching
```python
def find_best_skill_match(query_skill, skill_tree, all_skills_list):
    # Strategy 1: Exact match (case-insensitive)
    # Strategy 2: Contains match ("Jira" → "Atlassian JIRA Software")
    # Strategy 3: Tree-based partial matching
```

#### Step 2: Embedding Similarity
```python
query_embedding = model.encode(matched_skill)
for consultant_skill in consultant_skills:
    similarity = cosine_sim(consultant_skill_embedding, query_embedding)
```

#### Step 3: Score Calculation
```python
# Normalize experience level
level_factor = min(level_xp, top_level) / top_level

# Normalize similarity (above threshold)
similarity_factor = max(0, similarity - min_level) / (1 - min_level)

# Combine
score = level_factor * similarity_factor

# Apply multipliers
if is_required:
    score *= required_multiplier  # Default: 2.0

if exact_match:
    score *= 1.5  # 50% boost for exact matches
```

**Example**: Healthcare consultant with "Dossier Médical Électronique (DMÉ)"
- Similarity: 1.0 (exact match)
- Level: 60/60 = 1.0
- Required: ×2.0
- Exact match: ×1.5
- **Final score**: 1.0 × 1.0 × 2.0 × 1.5 = **3.0**

#### Step 4: Average Across Skills
```python
average_skills[consultant_id] = mean(scores for all required skills)
```

### 5.2 Mission Matching

#### Step 1: Embed Mission Requirements
```python
job_responsibilities = task_requirements["responsabilites"]
job_embeddings = model.encode(job_responsibilities)
```

#### Step 2: Compute Similarity Matrix
```python
# All consultant missions (pre-encoded)
mission_tensor = torch.stack([emb for emb in mission_embeddings])

# Mission requirements
job_embeddings = model.encode(responsibilities)

# Cosine similarity matrix
sim_matrix = cos_sim(mission_tensor, job_embeddings)
# Shape: (num_missions, num_responsibilities)
```

#### Step 3: Thresholding
```python
# Dynamic threshold per responsibility (90th percentile)
thresholds = [percentile(sim_matrix[:, i], 90) for i in range(num_responsibilities)]

# Count matches above threshold
matches = sum(score > threshold for score, threshold in zip(similarities, thresholds))
```

#### Step 4: Score Aggregation
```python
# Per consultant: aggregate across all their missions
consultant_scores = {
    "total_matches": count(matches > threshold),
    "avg_score": mean(similarities),
    "best_score": max(similarities)
}

# Final score with logarithmic boost for multiple matches
final_score = avg_score * log(1 + total_matches)
final_score = final_score / max(final_score)  # Normalize to [0, 1]
```

### 5.3 Score Combination

#### Linear Combination
```python
def linear_combination(skills_scores, skills_weight, mission_scores, mission_weight):
    combined = {}
    for consultant_id in skills_scores:
        skill_score = skills_scores[consultant_id]
        mission_score = mission_scores.get(consultant_id, filler_value)
        combined[consultant_id] = (skill_score * skills_weight + 
                                   mission_score * mission_weight)
    return combined
```

**Default Weights**: `skills_weight = 0.5`, `mission_weight = 0.5`

**Optional Optimization**:
- **Variance-based**: Maximize score variance to separate candidates
- **Supervised Learning**: Learn from historical matches (uses `JOB_RULE` labels)

### 5.4 Filtering

#### Language Filtering
```python
def compute_language_score(consultant_lang_level, required_level, acceptable_difference=40):
    if consultant_lang_level >= required_level:
        return 1.0  # No penalty
    elif (required_level - consultant_lang_level) <= acceptable_difference:
        return 0.8  # Small penalty
    else:
        return 1.0 / penalty_factor  # Large penalty (default: 10x reduction)
```

#### Availability Filtering
```python
def calculate_availability_penalty(availability_percentage, maximum_penalty=0.5):
    # Note: 100% = fully booked (not available)
    booking_ratio = availability_percentage / 100.0
    
    # Penalty ranges from 1.0 (fully available) to maximum_penalty (fully booked)
    penalty = 1.0 - (1.0 - maximum_penalty) * booking_ratio
    
    return penalty  # Applied as multiplier to combined score
```

**Example**: Consultant with 80% booking
- Penalty: 1.0 - (1.0 - 0.5) × 0.8 = 0.6
- Final score: combined_score × 0.6

---

## 6. Evaluation & Results

### 6.1 Evaluation Metrics

#### Precision@K
- **Definition**: Fraction of top-K candidates with correct `JOB_RULE`
- **K**: 5 (top 5 candidates)
- **Ground Truth**: `JOB_RULE` column in `full_df.csv`

#### Accuracy Calculation
```python
correct_predictions = sum(
    ground_truth[cid] == expected_job_role 
    for cid, _ in top_5_candidates
)
accuracy = correct_predictions / 5
```

### 6.2 Results by Domain

#### Healthcare: 100% Precision@5
**Why?**
- Exact skill match: "Dossier Médical Électronique (DMÉ)"
- Only 5 consultants in dataset have this skill
- Perfect similarity (1.0) + required multiplier (2.0) + exact match boost (1.5) = **3.0**
- Skills dominate ranking → mission model irrelevant

#### Data Analyst: 20% Precision@5
**Challenges**:
- Skills less specific (Power BI, Tableau, etc.)
- More consultants share these skills
- Mission similarity becomes more important
- Fine-tuned model might help here

#### Data Engineer: Varies
- Depends on specific technologies (Databricks, Spark, etc.)
- Mission history important for domain distinction

### 6.3 Fair Evaluation Setup

**No "Cheating"**:
- ✅ Base model for missions (no fine-tuning with ground truth)
- ✅ No consultant-level filtering by `JOB_RULE`
- ✅ No supervised weight learning from labels
- ✅ No keyword-based domain filtering
- ✅ `JOB_RULE` only used for accuracy calculation (evaluation, not matching)

**What IS Used**:
- ✅ Skill embeddings (fine-tuned for hierarchy, not domain)
- ✅ Mission embeddings (base model, no ground truth)
- ✅ Cosine similarity (standard metric)
- ✅ Language/availability constraints (from mission requirements)

---

## 7. Technical Implementation Details

### 7.1 Code Structure

```
project/
├── mainfile/
│   └── main_debug.py          # Main execution pipeline
├── helpers/
│   ├── SkillsEmbeddings.py    # Skill matching logic
│   └── tools_extraction.py    # PDF → YAML extraction
├── src/
│   ├── mission_embeddings.py  # Mission matching logic
│   ├── language_filtering.py  # Language constraints
│   └── schedule_filtering.py  # Availability constraints
├── notebooks/
│   ├── finetune_skill_hierarchy.py      # Skills fine-tuning
│   └── finetune_mission_embeddings_domain.py  # Mission fine-tuning
├── config/
│   └── search_config.yaml     # Configuration
└── data/
    ├── HCK_HEC_SKILLS.csv
    ├── HCK_HEC_XP.csv
    ├── HCK_HEC_LANG.csv
    └── HCK_HEC_STAFFING.csv
```

### 7.2 Key Functions

#### Skill Matching
```python
def get_consultants_skills_score(job_profile, skill_model, skills_embeddings_file, config):
    # 1. Extract skills from job profile
    # 2. Match skill names (exact/contains/tree)
    # 3. Rank consultants per skill (cosine similarity)
    # 4. Average across all skills
    return {consultant_id: average_score}
```

#### Mission Matching
```python
def get_consultants_mission_scores(task_requirements, percentile, finetune_model):
    # 1. Load pre-encoded mission embeddings
    # 2. Encode mission requirements
    # 3. Compute similarity matrix (cosine)
    # 4. Apply percentile thresholding
    # 5. Aggregate per consultant
    return DataFrame with consultant scores
```

#### Score Combination
```python
def linear_combination(skills_scores, skills_weight, mission_scores, mission_weight):
    # Weighted sum: α·skills + β·missions
    return {consultant_id: combined_score}
```

### 7.3 Configuration

**Key Parameters** (`config/search_config.yaml`):
```yaml
skills:
  required_multiplier: 2.0      # Boost for required skills
  min_level: 0.6                # Minimum similarity threshold
  skills_weight: 0.5            # Weight in combination

mission_context:
  percentile: 90                # Threshold for mission matches
  mission_weight: 0.5           # Weight in combination

languages:
  acceptable_difference: 40     # Max language gap before penalty
  penality: 10                  # Penalty factor

disponibility:
  maximum_penalty: 0.5          # Max penalty for fully booked
```

### 7.4 Performance Optimizations

1. **Pre-encoding**: All skills and missions pre-encoded and cached
2. **Batch Processing**: Mission embeddings computed in batches
3. **Normalized Embeddings**: Dot product instead of full cosine calculation
4. **Metadata Tracking**: Model metadata prevents unnecessary re-encoding

### 7.5 Model Switching

**Automatic Re-encoding**:
- System detects model change via `_encoding_model` metadata
- Automatically re-encodes missions when model differs
- Saves new embeddings with updated metadata

**Example**:
```python
# Config: mission_embedding_model: models/modelfinetuned_domain
# Stored: paraphrase-multilingual-mpnet-base-v2
# → Detects mismatch → Re-encodes all missions → Saves with new model name
```

---

## 8. Key Insights & Lessons Learned

### 8.1 Skills vs. Missions

**Finding**: Skills often dominate final ranking
- Healthcare: Skill score = 3.0, Mission score = 0.9
- Combined: (3.0 × 0.5) + (0.9 × 0.5) = 1.95
- Even if mission score changes to 0.7: (3.0 × 0.5) + (0.7 × 0.5) = 1.85
- **Ranking unchanged** → Skills dominate

**Implication**: Mission model fine-tuning may not always improve results

### 8.2 Fine-Tuning Trade-offs

**Skills Fine-Tuning**: ✅ Essential
- Prevents spurious matches (JIRA ↔ Scala)
- Improves hierarchical relationships
- No ground truth needed (uses skill relationships)

**Mission Fine-Tuning**: ⚠️ Optional
- Uses ground truth labels (not fair for evaluation)
- May not help if skills dominate
- Useful for production when skills are ambiguous

### 8.3 Evaluation Fairness

**Critical**: No ground truth labels in matching pipeline
- Base model for missions (no fine-tuning)
- No consultant-level filtering
- No supervised weight learning
- Pure embedding-based matching

**Result**: Fair evaluation, but production can use fine-tuned models

---

## 9. Future Improvements

1. **Cross-Encoder Reranking**: Tested but removed - didn't improve results and introduced wrong-domain consultants (e.g., Healthcare consultants for Scrum missions). System uses bi-encoder results directly.
2. **Adaptive Weights**: Learn optimal α, β per domain
3. **Multi-objective Optimization**: Balance precision, recall, diversity
4. **Explainability**: Show why each consultant was ranked (skill matches, mission matches)
5. **Interactive Feedback**: Allow users to refine matches (implemented in Streamlit app)

---

## 10. Conclusion

### Summary
- **Architecture**: Bi-encoder with dual matching (skills + missions)
- **Similarity**: Cosine similarity on normalized embeddings
- **Fine-tuning**: Skills hierarchy (essential), Mission domain (optional)
- **Evaluation**: Fair (no ground truth in pipeline)
- **Results**: Domain-dependent (Healthcare: 100%, Data Analyst: 20%)

### Key Takeaways
1. **Skills matching is powerful** - Often dominates ranking
2. **Fine-tuning helps** - But not always necessary
3. **Fair evaluation matters** - No ground truth in matching pipeline
4. **Cosine similarity works well** - Standard for semantic matching
5. **System is robust** - Works with base models, fine-tuning optional

---

## Appendix: Mathematical Formulations

### Cosine Similarity
```
cos(θ) = (A · B) / (||A|| × ||B||)
```
For normalized vectors: `cos(θ) = A · B`

### Contrastive Loss
```
L = max(0, margin - sim(positive_pair) + sim(negative_pair))
```

### Score Combination
```
score = α·skill_score + β·mission_score
where α + β = 1.0 (typically α = β = 0.5)
```

### Skill Score
```
score = (level_factor × similarity_factor) × multipliers
where:
  level_factor = min(level, top_level) / top_level
  similarity_factor = max(0, similarity - min_level) / (1 - min_level)
  multipliers = required_multiplier × exact_match_boost
```

---

**End of Presentation**

