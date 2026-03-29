# ⚽ Expected Goals (xG) Model & Shot Analysis

## 📌 Overview
This project builds a machine learning model to estimate the probability of a shot resulting in a goal — commonly known as **Expected Goals (xG)**.

The model uses shot location and contextual features to assign a probability value to every shot, helping quantify chance quality in football matches.

---

## 🎯 Objective
To predict whether a shot will result in a goal by estimating its probability using historical shot data.

---

## 📊 Dataset
The dataset contains event-level shot data with the following features:

- `x`, `y` → Shot location on the pitch  
- `shot_type` → Type of shot  
- `situation` → Open play, set piece, etc.  
- `body_part` → Foot, head, etc.  
- `minute` → Time of the shot  
- `is_goal` → Target variable (1 = goal, 0 = no goal)

Additional engineered features:
- `shot_distance` → Distance from goal  

---

## 🧹 Data Preprocessing
- Handled missing values in categorical and binary features  
- Converted relevant columns to boolean  
- Encoded categorical variables using one-hot encoding  
- Removed irrelevant features (e.g. own goals)  
- Created new features such as shot distance  

---

## 🤖 Model
A **Logistic Regression** model was used as a baseline to predict the probability of a goal.

Why Logistic Regression?
- Interpretable  
- Suitable for probability estimation  
- Common baseline in xG modeling  

---

## 📈 Model Performance

| Metric        | Score |
|--------------|------|
| ROC AUC      | 0.81 |
| Log Loss     | 0.269 |
| Brier Score  | 0.077 |

### Interpretation:
- **ROC AUC (0.81)** → Good ability to distinguish between goals and non-goals  
- **Log Loss (0.269)** → Predictions are reasonably accurate  
- **Brier Score (0.077)** → Well-calibrated probability estimates  

---

## 📍 Visualization

The shot map below shows each shot colored by its xG value:

- 🔴 Low probability shots (long distance / wide angles)  
- 🟡 Medium probability shots  
- 🟢 High probability shots (close to goal, central areas)  

> This demonstrates how shot location strongly influences scoring probability.

---

## 💡 Key Insights
- Most shots are taken from low-probability areas  
- High-quality chances are relatively rare but critical  
- Shot distance is one of the most important predictors  
- Central positions yield higher xG compared to wide areas  

---

## ⚠️ Limitations
- No data on defensive pressure  
- No goalkeeper positioning information  
- Limited contextual features (e.g., pass type, game state)  
- Model can be improved with more advanced algorithms  

---

## 🚀 Future Improvements
- Implement **XGBoost** for better performance  
- Add more contextual features (pressure, assist type)  
- Build an interactive dashboard using Streamlit  
- Apply the model to player and team performance analysis  

---

## 🛠️ Tools & Libraries
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 📂 Project Structure
