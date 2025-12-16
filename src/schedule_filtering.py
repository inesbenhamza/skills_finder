import pandas as pd
import yaml
from datetime import datetime

def calculate_availability_penalty(csv_file: str, config, maximum_penalty) -> dict:
    df = pd.read_csv(csv_file)
    
    months_to_consider = config['months']
    start_date = config.get('start_date', '')
    
    if start_date:
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            start_month = start_date_obj.month
        except ValueError:
            start_month = 1
    else:
        start_month = 1
    
    user_penalties = {}
    for _, row in df.iterrows():
        user_id = row['USER_ID']
        availability_scores = row.iloc[start_month:start_month + months_to_consider]
        
        # NOTE: In the dataset, 100% means fully booked (NOT available), 0% means fully available
        # Calculate booking ratio (0 to 1, where 1 = 100% booked = not available)
        booking_ratio = min(1.0, max(0.0, availability_scores.mean() / 100))
        
        # Calculate penalty using maximum_penalty:
        # - Fully available (0% booked) → penalty = 1.0 (no reduction)
        # - Fully booked (100% booked) → penalty = maximum_penalty (maximum reduction)
        # If maximum_penalty < 1.0: penalty ranges from 1.0 to maximum_penalty (reduces score)
        if maximum_penalty <= 1.0:
            # Penalty is a multiplier < 1.0 (reduces score)
            penalty = 1.0 - (1.0 - maximum_penalty) * booking_ratio
        else:
            # If maximum_penalty > 1.0, convert to a multiplier < 1.0
            # Example: maximum_penalty = 2.0 means fully booked consultants get 1/2 = 0.5 of their score
            penalty = 1.0 / (1.0 + (maximum_penalty - 1.0) * booking_ratio)
        
        user_penalties[user_id] = penalty
    
    return user_penalties