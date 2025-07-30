import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="Reels Reach Predictor", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# --- Load historical data ---
uploaded_data = st.file_uploader("Upload your reels_model_output.csv", type="csv")
df = None
model = None

if uploaded_data:
    df = pd.read_csv(uploaded_data)

    # --- Train model from uploaded data without multiplying weights ---
    if 'predicted_reach' in df.columns:
        df = df.drop(columns=['predicted_reach'])
    if 'performance' in df.columns:
        df = df.drop(columns=['performance'])

    X = df[['shares', 'saves', 'comments', 'likes']]
    y = df['reach']
    model = LinearRegression().fit(X, y)
    df['predicted_reach'] = model.predict(X)

    # --- Categorize performance ---
    def categorize(row):
        ratio = row['reach'] / row['predicted_reach'] if row['predicted_reach'] else 0
        if ratio > 2.0:
            return "Viral"
        elif ratio > 1.5:
            return "Excellent"
        elif ratio > 1.0:
            return "Good"
        elif ratio > 0.5:
            return "Average"
        else:
            return "Poor"

    df['performance'] = df.apply(categorize, axis=1)

    # --- Summary ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Actual Reach", f"{int(df['reach'].mean()):,}")
    col2.metric("Avg Predicted Reach", f"{int(df['predicted_reach'].mean()):,}")
    col3.metric("Mean % Error", f"{np.mean(np.abs((df['reach'] - df['predicted_reach']) / df['predicted_reach']) * 100):.2f}%")

    # --- Category Counts ---
    st.subheader("📌 Performance Distribution")
    category_counts = df['performance'].value_counts()
    st.bar_chart(category_counts)

    # --- Viral Reels Table ---
    st.subheader("🔥 Viral & Excellent Reels")
    st.dataframe(df[df['performance'].isin(['Viral', 'Excellent'])].sort_values(by='reach', ascending=False))

    # --- Time Filter ---
    st.subheader("📅 Filter by Post Date")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
        # Set timezone to match date column
        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize("Asia/Kolkata")
        else:
            df['date'] = df['date'].dt.tz_convert("Asia/Kolkata")
    
        min_date, max_date = df['date'].min().date(), df['date'].max().date()
        start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])
    
        # Convert selected dates to timezone-aware Timestamps
        start_date = pd.Timestamp(start_date).tz_localize("Asia/Kolkata")
        end_date = pd.Timestamp(end_date).tz_localize("Asia/Kolkata")
    
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # --- Export to CSV ---
    st.subheader("⬇️ Download Full Categorized Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='reels_with_predictions.csv',
        mime='text/csv'
    )

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
    if model is not None:
        X_new = pd.DataFrame([{
            'shares': shares,
            'saves': saves,
            'comments': comments,
            'likes': likes
        }])
        prediction = model.predict(X_new)[0]
        st.success(f"📢 Predicted Reach: {int(prediction):,}")
    else:
        st.warning("Please upload your CSV to activate predictions.")
