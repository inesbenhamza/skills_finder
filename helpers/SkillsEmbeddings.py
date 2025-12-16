import numpy as np
import pandas as pd


class SkillEmbeddingModel:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def encode(self, text):
        emb = self.embedding_model.encode(text, convert_to_tensor=False)
        emb = np.array(emb, dtype=np.float32)
        
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    else:
        return float(np.dot(a, b) / denom)


def load_embeddings(npz_file: str):
    data = np.load(npz_file, allow_pickle=True)
    return data["skill_embs"], data["domain_embs"], data["metadata"]

def build_skill_tree(skills_embeddings_file):
    from .tree import TreeNode
    
    _, _, metadata = load_embeddings(skills_embeddings_file)
    
    unique_skills = {}
    for _, _, skill_name in metadata:
        if skill_name not in unique_skills:
            unique_skills[skill_name] = skill_name
    
    root = TreeNode('', '')
    for skill_name in unique_skills.values():
        words = skill_name.split()
        words.append("<EOS/>")
        root.add_child(words, skill_name)
    
    root.prune()
    return root


def get_all_unique_skills(skills_embeddings_file):
    _, _, metadata = load_embeddings(skills_embeddings_file)
    unique_skills = set()
    for _, _, skill_name in metadata:
        unique_skills.add(skill_name)
    return sorted(list(unique_skills))


def find_best_skill_match(query_skill, skill_tree, all_skills_list):
    query_lower = query_skill.lower().strip()
    
    for skill in all_skills_list:
        if skill.lower() == query_lower:
            return skill, 'exact'
    
    query_words = set(query_lower.split())
    best_match = None
    best_score = 0
    
    for skill in all_skills_list:
        skill_lower = skill.lower()
        skill_words = set(skill_lower.split())
        
        if query_words.issubset(skill_words):
            score = len(query_words) / len(skill_words)
            
            if query_lower in skill_lower:
                score += 1.0
            
            query_phrase = " ".join(query_lower.split())
            if query_phrase in skill_lower:
                score += 0.5
            
            if len(skill_words) > len(query_words) * 2:
                score *= 0.7
            
            if score > best_score:
                best_score = score
                best_match = skill
    
    if best_match and best_score > 0.3:
        return best_match, 'contains'
    
    if len(query_words) > 1:
        try:
            query_words_list = query_skill.lower().split()
            matched_skills = {}
            
            for ngram_len in range(min(len(query_words_list), 3), 1, -1):
                for i in range(len(query_words_list) - ngram_len + 1):
                    ngram = " ".join(query_words_list[i:i+ngram_len])
                    try:
                        skill_tree.match_term(ngram)
                    except Exception as e:
                        matched_skill = str(e)
                        matched_skill_words = set(matched_skill.lower().split())
                        if len(query_words.intersection(matched_skill_words)) >= len(query_words) * 0.7:
                            if matched_skill not in matched_skills:
                                matched_skills[matched_skill] = ngram_len
            
            if matched_skills:
                best_tree_match = max(matched_skills.items(), key=lambda x: x[1])[0]
                return best_tree_match, 'partial'
        except:
            pass
    
    return query_skill, None

def rank_consultants(npz_file: str,
                     query_skill: str,
                     model,
                     top_level=100,
                     is_required=True,
                     score_multiplier=2,
                     min_level=0.55,
                     debug=False,
                     exact_match_boost=False):
    skill_embs, _, metadata = load_embeddings(npz_file)
    query_embedding = model.encode([query_skill])[0]
    
    consultant_scores = {}
    debug_info = []
    
    for i, (user_id, level_xp, skill_name) in enumerate(metadata):
        similarity = cosine_sim(skill_embs[i], query_embedding)
        
        if debug and similarity > 0.2:
            debug_info.append((user_id, skill_name, similarity, level_xp))
        
        level_factor = min(level_xp, top_level) / top_level
        
        similarity_factor = max(0, similarity - min_level) / (1 - min_level)
        
        score = level_factor * similarity_factor
        
        if is_required:
            score *= score_multiplier
        
        if exact_match_boost:
            skill_name_str = str(skill_name).lower()
            matched_skill_lower = query_skill.lower()
            if skill_name_str == matched_skill_lower:
                score *= 1.5
        
        consultant_scores[user_id] = max(consultant_scores.get(user_id, 0), score)
    
    sorted_consultants = sorted(consultant_scores.items(),
                                key=lambda x: x[1],
                                reverse=True)
    
    result = {c: s for c, s in sorted_consultants}
    
    if debug:
        debug_info.sort(key=lambda x: x[2], reverse=True)
        return result, debug_info[:50]
    
    return result

def get_consultants_skills_score(job_profile, skill_model, skills_embeddings_file, configuration, debug=False, use_string_matching=True):
    scores = {}
    debug_data = {}
    skills = job_profile["technologies"]
    
    if use_string_matching:
        skill_tree = build_skill_tree(skills_embeddings_file)
        all_skills = get_all_unique_skills(skills_embeddings_file)
    
    for skill_req in skills:
        if not skill_req:  # empty dict or missing
            continue
        
        skill_name = skill_req["name"]
        skill_level = skill_req.get("level", 100)
        is_required = skill_req["required"]
        
        if use_string_matching:
            matched_skill, match_type = find_best_skill_match(skill_name, skill_tree, all_skills)
            
            if debug and match_type:
                print(f"[DEBUG] Skill matching: '{skill_name}' -> '{matched_skill}' (match type: {match_type})")
        else:
            matched_skill = skill_name
            match_type = None
            
            if debug:
                print(f"[DEBUG] Base model: Using original skill name '{skill_name}' directly (no string matching)")
        
        top_level = skill_level if skill_level and skill_level > 0 else 100
        
        result = rank_consultants(
            skills_embeddings_file,
            matched_skill,
            skill_model,
            top_level=top_level,
            is_required=is_required,
            score_multiplier=configuration["skills"]["required_multiplier"],
            min_level=configuration["skills"]["min_level"],
            debug=debug,
            exact_match_boost=(match_type == 'exact' if use_string_matching else False)
        )
        
        if debug:
            scores[skill_name], debug_data[skill_name] = result
        else:
            scores[skill_name] = result
    
    if debug:
        return scores, debug_data
    return scores


def average_skills(consultant_scores: dict):
    means = {}
    all_skills = list(consultant_scores.keys())
    
    if not all_skills:
        return {}
    
    all_consultant_ids = set()
    for skill in all_skills:
        all_consultant_ids.update(consultant_scores[skill].keys())
    
    for user_id in all_consultant_ids:
        consultant_mean_score = [
            consultant_scores[skill].get(user_id, 0.0)
            for skill in all_skills
        ]
        means[user_id] = np.mean(consultant_mean_score)
    
    return means


def sort_skills(final_scores):
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)



def save_skills_emb(input_csv: str, output_npz: str, model):
    df = pd.read_csv(input_csv)
    
    df['SKILLS_EMB'] = df['SKILLS_DSC'].apply(lambda x: model.encode([x])[0])
    df['DOMAIN_EMB'] = df['DOMAIN_DSC'].apply(lambda x: model.encode([x])[0])
    
    skill_embs = np.stack(df['SKILLS_EMB'].values)
    domain_embs = np.stack(df['DOMAIN_EMB'].values)
    metadata = df[['USER_ID', 'LEVEL_VAL', 'SKILLS_DSC']].values
    
    np.savez(output_npz, skill_embs=skill_embs, domain_embs=domain_embs, metadata=metadata)
