import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Reels Reach Predictor", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# --- Load historical data ---
uploaded_data = st.file_uploader("Upload your reels_model_output.csv", type="csv")
if uploaded_data:
    df = pd.read_csv(uploaded_data)

    # --- Summary ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Actual Reach", f"{int(df['reach'].mean()):,}")
    col2.metric("Avg Predicted Reach", f"{int(df['predicted_reach'].mean()):,}")
    col3.metric("Mean % Error", f"{np.mean(np.abs((df['reach'] - df['predicted_reach']) / df['predicted_reach']) * 100):.2f}%")

    # --- Category Counts ---
    st.subheader("📌 Performance Distribution")
    category_counts = df['performance'].value_counts()
    st.bar_chart(category_counts)

    # --- Scatter Plot ---
    st.subheader("📈 Actual vs Predicted Reach")
    fig, ax = plt.subplots()
    ax.scatter(df['predicted_reach'], df['reach'], alpha=0.7, color='mediumslateblue')
    ax.plot([0, df['predicted_reach'].max()], [0, df['predicted_reach'].max()], 'r--')
    ax.set_xlabel("Predicted Reach")
    ax.set_ylabel("Actual Reach")
    st.pyplot(fig)

# --- Prediction Tool ---
st.subheader("🎯 Predict Reach for a New Reel")

with st.form("predict_form"):
    shares = st.number_input("Shares", min_value=0, value=0)
    saves = st.number_input("Saves", min_value=0, value=0)
    comments = st.number_input("Comments", min_value=0, value=0)
    likes = st.number_input("Likes", min_value=0, value=0)
    submit = st.form_submit_button("Predict Reach")

if submit:
    # Apply strategic weights
    X_new = pd.DataFrame([{
        'shares': shares * 10,
        'saves': saves * 5,
        'comments': comments * 3,
        'likes': likes * 2
    }])

    # Use hardcoded coefficients from your trained model
    coef = np.array([model.coef_ for model in [LinearRegression()]])[0] if not uploaded_data else None

    # Optional fallback linear model if file not uploaded
    if uploaded_data:
        # Train inline for demo (normally you'd load model)
        X = df[['shares', 'saves', 'comments', 'likes']] * [10, 5, 3, 2]
        y = df['reach']
        model = LinearRegression().fit(X, y)
        prediction = model.predict(X_new)[0]
        st.success(f"📢 Predicted Reach: {int(prediction):,}")
    else:
        st.warning("Please upload your CSV to activate predictions.")
