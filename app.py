import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="Reels Reach Predictor", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# --- Load historical data ---
df = pd.read_csv("posts_zero1byzerodha.csv")
model = None

# --- Clean unexpected symbols and convert ---
for col in ['reach', 'likes', 'comments', 'shares', 'saves']:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .replace('', np.nan)
        .astype(float)
    )

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

# --- Format in K/M ---
def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))

# --- Time Filter ---
st.subheader("📅 Filter by Post Date")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
        except (TypeError, AttributeError):
            try:
                df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')
            except Exception:
                pass

    min_date, max_date = df['date'].min().date(), df['date'].max().date()
    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])

    start_date = pd.Timestamp(start_date).tz_localize("Asia/Kolkata")
    end_date = pd.Timestamp(end_date).tz_localize("Asia/Kolkata")

    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# --- Summary Metrics ---
st.subheader("📈 Summary Insights")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Actual Reach", format_number(df['reach'].mean()))
col2.metric("Avg Predicted Reach", format_number(df['predicted_reach'].mean()))
error = np.mean(np.abs((df['reach'] - df['predicted_reach']) / df['predicted_reach']) * 100)
col3.metric("Mean % Error", f"{error:.2f}%")

with st.expander("🧠 Why is the error high?"):
    st.markdown("- Viral outliers (e.g., collab posts) disproportionately inflate error.")
    st.markdown("- Linear regression assumes linearity. Real reach may depend on more complex, nonlinear factors.")
    st.markdown("- Some reels may benefit from timing, trending audio, or influencer effects not captured by likes/saves alone.")

# --- Performance Table ---
st.subheader("🔥 Viral & Excellent Reels")
display_df = df[df['performance'].isin(['Viral', 'Excellent'])].copy()
display_df['reach'] = display_df['reach'].apply(format_number)
display_df['predicted_reach'] = display_df['predicted_reach'].apply(format_number)
st.dataframe(display_df.sort_values(by='reach', ascending=False)
             [['date', 'title', 'reach', 'predicted_reach', 'performance']])

# --- Key Takeaways ---
st.subheader("📌 Quick Insights")
most_saved = df.sort_values(by='saves', ascending=False).iloc[0]
st.markdown(f"✅ Reel with highest saves: **{most_saved['title']}** with {int(most_saved['saves'])} saves")
most_shared = df.sort_values(by='shares', ascending=False).iloc[0]
st.markdown(f"✅ Reel with highest shares: **{most_shared['title']}** with {int(most_shared['shares'])} shares")
viral_count = df[df['performance'] == 'Viral'].shape[0]
st.markdown(f"✅ Number of Viral Reels: **{viral_count}**")

# --- Export to CSV ---
st.subheader("⬇️ Download Full Categorized Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='reels_with_predictions.csv',
    mime='text/csv'
)

# --- Prediction Tool ---
st.subheader("🎯 Predict Reach for a New Reel")
st.markdown("⚠️ Note: Predictions are accurate only **after your reel has gone live**, since `likes`, `shares`, etc. are required inputs.")

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
        st.success(f"📢 Predicted Reach: {format_number(prediction)}")
    else:
        st.warning("Model not available.")
