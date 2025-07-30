import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import openai

st.set_page_config(page_title="Reels Reach Predictor", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# --- Load historical data ---
df = pd.read_csv("posts_zero1byzerodha.csv")
df.columns = df.columns.str.strip().str.lower()
df = df[df['type'].str.lower() == 'reel']
model = None

# --- Clean numeric columns ---
for col in ['reach', 'likes', 'comments', 'shares', 'saved']:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r'[^\d.]', '', regex=True)
            .replace('', np.nan)
            .astype(float)
        )

# --- Parse and filter by date ---
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    try:
        df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
    except TypeError:
        df['date'] = df['date'].dt.tz_localize('Asia/Kolkata', ambiguous='NaT', nonexistent='NaT')

    min_date, max_date = df['date'].min().date(), df['date'].max().date()
    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])

    start_date = pd.Timestamp(start_date).tz_localize("Asia/Kolkata")
    end_date = pd.Timestamp(end_date).tz_localize("Asia/Kolkata")
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# --- Train model ---
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

# --- Format numbers ---
def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))

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

# --- Top Reels ---
st.subheader("🔥 Viral & Excellent Reels")
display_df = df[df['performance'].isin(['Viral', 'Excellent'])].copy()
display_df['reach'] = display_df['reach'].apply(format_number)
display_df['predicted_reach'] = display_df['predicted_reach'].apply(format_number)
if 'date' in display_df.columns:
    st.dataframe(display_df[['date', 'caption', 'reach', 'predicted_reach', 'performance']].sort_values(by='reach', ascending=False))

# --- Content-Based Insights ---
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
if 'caption' in df.columns and len(df) > 0:
    st.subheader("🧠 Content Intelligence")
    sample_titles = df['caption'].dropna().sample(min(5, len(df))).tolist()
    prompt = f"""
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
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown(response['choices'][0]['message']['content'])
    except Exception as e:
        st.warning("NLP analysis unavailable.")

# --- Virality Score ---
df['virality_score'] = df['reach'] / df['predicted_reach']
df['virality_score'] = df['virality_score'].replace([np.inf, -np.inf], np.nan).fillna(0)

st.subheader("📊 Top Content Ideas by Virality")
top_ideas = df.sort_values(by='virality_score', ascending=False).head(5)
top_ideas['reach'] = top_ideas['reach'].apply(format_number)
top_ideas['predicted_reach'] = top_ideas['predicted_reach'].apply(format_number)
top_ideas['virality_score'] = top_ideas['virality_score'].apply(lambda x: f"{x:.2f}x")
st.dataframe(top_ideas[['caption', 'reach', 'predicted_reach', 'virality_score']])

# --- Export CSV ---
st.subheader("⬇️ Download Categorized Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "reels_with_predictions.csv", "text/csv")

# --- Predict New Reel ---
st.subheader("🎯 Predict Reach for New Reel")
st.markdown("⚠️ Predictions require inputs after reel is live.")
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
        st.warning("Model not trained.")
