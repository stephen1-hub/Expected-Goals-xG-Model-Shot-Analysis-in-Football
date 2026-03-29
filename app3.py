import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from mplsoccer import Pitch
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# TITLE
# -----------------------------
st.title("⚽ xG Model Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("shots3.csv")

df = load_data()

st.write("Dataset Preview:", df.head())

# -----------------------------
# MODEL
# -----------------------------
X = df.drop(columns=['is_goal'])
y = df['is_goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict xG
df['xg'] = model.predict_proba(X)[:, 1]

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

show_goals = st.sidebar.checkbox("Show Goals Only")
show_big_chances = st.sidebar.checkbox("Show Big Chances Only")

plot_df = df.copy()

if show_goals:
    plot_df = plot_df[plot_df['is_goal'] == True]

if show_big_chances and 'BigChance' in df.columns:
    plot_df = plot_df[plot_df['BigChance'] == True]

# -----------------------------
# SHOT MAP
# -----------------------------
st.subheader("📍 Shot Map (xG)")

pitch = Pitch(pitch_type='statsbomb')
fig, ax = pitch.draw(figsize=(10, 8))

colors = ['red', 'yellow', 'green']
cmap = LinearSegmentedColormap.from_list("xg_cmap", colors)

sc = pitch.scatter(
    plot_df['x'],
    plot_df['y'],
    c=plot_df['xg'],
    cmap=cmap,
    edgecolors='black',
    linewidth=0.5,
    s=50,
    alpha=0.7,
    ax=ax
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("xG")

st.pyplot(fig)

# -----------------------------
# METRICS
# -----------------------------
st.subheader("📊 Model Performance")

y_pred_proba = model.predict_proba(X_test)[:, 1]

st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
st.write(f"Log Loss: {log_loss(y_test, y_pred_proba):.3f}")
st.write(f"Brier Score: {brier_score_loss(y_test, y_pred_proba):.3f}")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📈 Feature Importance")

importance = pd.Series(model.coef_[0], index=X.columns)
importance = importance.sort_values(ascending=False)

st.bar_chart(importance)

# -----------------------------
# SINGLE SHOT PREDICTION
# -----------------------------
st.subheader("🎯 Predict xG for a Shot")

x_input = st.slider("X Position", 0.0, 100.0, 80.0)
y_input = st.slider("Y Position", 0.0, 100.0, 50.0)

distance = np.sqrt((100 - x_input)**2 + (50 - y_input)**2)

input_data = pd.DataFrame({
    'x': [x_input],
    'y': [y_input],
    'shot_distance': [distance]
})

# Fill missing columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

pred = model.predict_proba(input_data)[:, 1][0]

st.success(f"Predicted xG: {pred:.3f}")