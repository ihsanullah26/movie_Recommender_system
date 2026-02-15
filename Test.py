import numpy as np # Make sure to import numpy!
import pickle
try:
    model = pickle.load(open('lightfm_model.pkl','rb'))
    dataset = pickle.load(open('dataset.pkl','rb'))
    movie_title = pickle.load(open('movie_titles.pkl','rb'))
except FileNotFoundError:
    print("Please RUn app.py first to train the Model")
    exit()

def map_age_and_gender(age, gender):
    gender_clean = str(gender).strip().upper()[0]
    gender_tag = f"Gender_{gender_clean}" if gender_clean in ['M','F'] else "Gender_M"
    
    age = int(age)
    if age < 18: age_code = "1"
    elif 18 <= age <= 24: age_code = "18"
    elif 24 < age <= 34: age_code = "25"
    elif 34 < age <= 44: age_code = "35"
    elif 44 < age <= 49: age_code = "45"
    elif 49 < age <= 55: age_code = "50" # Fixed to 50 to match README
    else: age_code = "56"

    # IMPORTANT: Added "Age_" prefix to match your training script labels
    return [gender_tag, f"Age_{age_code}"]

def Test_Model():
    print("Recommendation System\n")
    user_id_input = input("Enter User ID (Leave Blank if New User): ").strip()
    gender = input("Enter Gender (M/F): ")
    real_age = input("Enter Age: ")

    user_id_map, _, item_id_map, _ = dataset.mapping()
    n_items = len(item_id_map)

    # 1. Handle User Identity
    # If the ID is valid, use it. If not, use the first user in our map (ID: 1) as a placeholder.
    if user_id_input and int(user_id_input) in user_id_map:
        u_id_to_use = int(user_id_input)
        u_index = user_id_map[u_id_to_use]
    else:
        # We use a placeholder ID (the first key in our map) to avoid the "None" error
        u_id_to_use = list(user_id_map.keys())[0] 
        u_index = 0 

    # 2. Get the specific tags (e.g., ['Gender_M', 'Age_25'])
    user_tags = map_age_and_gender(real_age, gender)

    # 3. FIX: Use the placeholder ID instead of None
    query_feature = dataset.build_user_features([(u_id_to_use, user_tags)])

    # 4. Predict
    scores = model.predict(u_index, np.arange(n_items), user_features=query_feature)
    top_indices = np.argsort(-scores)[:10]

    print(f"\n--- Top 10 Recommendations for Gender: {gender}, Age: {real_age} ---")
    
    movie_ids_list = list(item_id_map.keys())
    for i, idx in enumerate(top_indices):
        m_id = movie_ids_list[idx]
        # Use movie_title (the variable name you loaded via pickle)
        mov_name = movie_title.get(m_id, "Unknown Movie")
        print(f"{i+1}. {mov_name}")
    print("-" * 30)
    
Test_Model()