import pandas as pd
import yaml

LANG_MAP = {
    "FR": "French",
    "EN": "English"
}

def apply_filters(users_scores, filter_score):
    filtered_score = {}
    for user_id in users_scores:
        filtered_score[user_id] = filter_score.get(user_id, 1) * users_scores[user_id]
    return filtered_score

def compute_language_score(csv_file: str, requirements, acceptable_difference, penality) -> dict:
    df = pd.read_csv(csv_file)
    
    required_languages = {lang['language_code']: lang for lang in requirements['languages']}
    
    user_scores = {}
    for user_id, group in df.groupby('USER_ID'):
        score = 1.0
        
        for lang, req in required_languages.items():
            required_level = req['level']

            user_level = group.loc[group['LANGUAGE_SKILL_DSC'] == LANG_MAP[lang], 'LANGUAGE_SKILL_LVL'].max() if LANG_MAP[lang] in group['LANGUAGE_SKILL_DSC'].values else 0
            
            if req['required'] and user_level < required_level - acceptable_difference:
                score /= penality
                break
            else:
                score *= min(1, user_level / required_level)
        
        user_scores[user_id] = score
    
    return user_scores