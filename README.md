# 🎬 Hybrid Movie Recommendation System

A **Hybrid Recommender System** built with Python and the **LightFM** library for personalized movie recommendations.  
This system tackles the **Cold-Start Problem** by combining:

- **Collaborative Filtering:** User-item interactions using **LightFM**  
- **Content-Based Filtering:** User demographics & movie metadata  

---

## 🌟 Features

- **Hybrid Learning with LightFM:** Optimizes movie rankings using **WARP (Weighted Approximate-Rank Pairwise)** loss.  
- **Cold-Start Ready:** Supports new users with no prior history using **Age** and **Gender**.  
- **Demographic Mapping:** Converts real-world ages into **MovieLens category buckets**.  
- **Fast Inference:** Model and mappings serialized with `pickle` for quick loading.  

---

## 🛠️ Environment Setup

### 1. Clone the Project

```bash
git clone https://github.com/ihsanullah26/movie_Recommender_system.git
cd movie_Recommender_system
```

### 2. Create and Activate Virtual Environment

```bash
# Create the environment
python3 -m venv project_env

# Activate (Linux/macOS)
source project_env/bin/activate

# Activate (Windows)
# project_env\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install pandas numpy lightfm
```

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `train.py` | Processes data, trains the model using **LightFM**, and saves `.pkl` files. |
| `Test.py` | Interactive script for getting recommendations. |
| `movies.dat` | MovieLens dataset file (movies info). |
| `users.dat` | MovieLens dataset file (user info). |
| `ratings.dat` | MovieLens dataset file (ratings info). |
| `lightfm_model.pkl` | Trained **LightFM** model weights (generated after training). |
| `dataset.pkl` | LightFM dataset mappings (generated after training). |

---

## 🚀 Usage

### 1. Train the Model

```bash
python3 train.py
```

### 2. Test Recommendations

```bash
python3 Test.py
```

---

## 🧠 Technical Workflow

### Age Mapping Logic

The system automatically categorizes user ages into **MovieLens dataset bins**:

| Category | Age Range |
|----------|-----------|
| 1        | Under 18  |
| 18       | 18-24     |
| 25       | 25-34     |
| 35       | 35-44     |
| 45       | 45-49     |
| 50       | 50-55     |
| 56       | 56+       |

This ensures consistency with the training data and improves recommendations for new users.  

---

✅ **Notes:**  
- Ensure your MovieLens dataset files are in the project root.  
- Serialized `.pkl` files allow **fast inference** with **LightFM** without retraining.
