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
df = df[df['type'].str.lower() == 'reel']  # Filter for Reels only
df.columns = df.columns.str.strip().str.lower()  # Standardize column names
st.write("Column names:", df.columns.tolist())  # Debug
model = None

# --- Clean unexpected symbols and convert ---
for col in ['reach', 'likes', 'comments', 'shares', 'saved']:
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

X = df[['shares', 'saved', 'comments', 'likes']]
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
             [[col for col in [col for col in ['date', 'caption', 'reach', 'predicted_reach', 'performance'] if col in display_df.columns] if col in display_df.columns]])

# --- Content-Based Analysis (Pre-Reel Launch) ---
import openai

openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

if 'caption' in df.columns:
    st.subheader("🧠 Content Intelligence (using NLP)")
    sample_titles = df['caption'].dropna().sample(min(5, len(df))).tolist()
    insights_prompt = f"""
You are an Instagram Reels strategist. Based on these 5 reel titles:

{chr(10).join(['- ' + t for t in sample_titles])}

Give 3 bullet points:
1. Common content patterns or themes
2. Predicted tone or emotion (e.g. fun, serious, educational)
3. One idea to improve virality based on this sample
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": insights_prompt}]
        )
        nlp_summary = response['choices'][0]['message']['content']
        st.markdown(nlp_summary)
    except Exception as e:
        st.warning("Could not fetch content-based insights.")

# --- Key Takeaways ---
st.subheader("📌 Quick Insights")
most_saved = df.loc[df['caption'].str.contains('save', case=False, na=False)].iloc[0] if 'caption' in df.columns else None
if most_saved is not None:
    st.markdown(f"✅ Reel with highest saves-like caption: **{most_saved['caption']}**")
most_shared = df.sort_values(by='shares', ascending=False).iloc[0]
st.markdown(f"✅ Reel with highest shares: **{most_shared['caption']}** with {int(most_shared['shares'])} shares")
viral_count = df[df['performance'] == 'Viral'].shape[0]
st.markdown(f"✅ Number of Viral Reels: **{viral_count}**")

# --- Collab vs Non-Collab Insights ---
if 'collab' in df.columns:
    df['is_collab'] = df['collab'].str.lower() == 'yes'
    collab_viral_count = df[(df['performance'] == 'Viral') & (df['is_collab'])].shape[0]
    noncollab_viral_count = df[(df['performance'] == 'Viral') & (~df['is_collab'])].shape[0]
    st.markdown(f"🤝 Viral Collab Reels: **{collab_viral_count}**, Non-Collab: **{noncollab_viral_count}**")
    avg_collab_reach = df[df['is_collab']]['reach'].mean()
    avg_noncollab_reach = df[~df['is_collab']]['reach'].mean()
    st.markdown(f"📊 Avg Reach - Collab: **{format_number(avg_collab_reach)}**, Non-Collab: **{format_number(avg_noncollab_reach)}**")
    st.markdown("🔍 Collab reels may benefit from network effects, hence typically show higher reach. Use this insight to plan future collaborations.")

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
            'saved': saves,
            'comments': comments,
            'likes': likes
        }])
        prediction = model.predict(X_new)[0]
        st.success(f"📢 Predicted Reach: {format_number(prediction)}")
    else:
        st.warning("Model not available.")
