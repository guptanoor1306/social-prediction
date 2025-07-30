import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import openai

# Page config
st.set_page_config(page_title="Reels Reach Predictor", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# --- Load and preprocess data ---
df = pd.read_csv("posts_zero1byzerodha.csv")
df.columns = df.columns.str.strip().str.lower()

# Filter only Reels
if 'type' in df.columns:
    df = df[df['type'].str.lower() == 'reel']

# Clean numeric fields
for col in ['reach', 'likes', 'comments', 'shares', 'saved']:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                   .str.replace(r'[^\d.]', '', regex=True)
                   .replace('', np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Identify and filter by date column
date_col = next((c for c in df.columns if 'date' in c), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['post_date'] = df[date_col].dt.date
    st.sidebar.subheader("📅 Filter by Post Date")
    start_date, end_date = st.sidebar.date_input(
        "Select date range", [df['post_date'].min(), df['post_date'].max()]
    )
    df = df[df['post_date'].between(start_date, end_date)]

# Features for modeling
features = [f for f in ['shares','saved','comments','likes'] if f in df.columns]

# Train Linear Regression model
y = df['reach'].fillna(0)
X = df[features].fillna(0)
model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# Categorize performance
def categorize(r, p):
    if p == 0:
        return 'Uncategorized'
    ratio = r / p
    if ratio > 2.0:
        return 'Viral'
    if ratio > 1.5:
        return 'Excellent'
    if ratio > 1.0:
        return 'Good'
    if ratio > 0.5:
        return 'Average'
    return 'Poor'

df['performance'] = df.apply(lambda row: categorize(row['reach'], row['predicted_reach']), axis=1)

# Number formatting
def fmt(n):
    if pd.isna(n):
        return "-"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(int(n))

# --- Summary Metrics ---
st.subheader("📈 Summary Insights")
col1, col2, col3 = st.columns(3)
avg_act = df['reach'].mean()
avg_pred = df['predicted_reach'].mean()
mean_err = np.mean(np.abs((df['reach'] - df['predicted_reach']) / np.where(df['predicted_reach']==0,1,df['predicted_reach']))) * 100
col1.metric("Avg Actual Reach", fmt(avg_act))
col2.metric("Avg Predicted Reach", fmt(avg_pred))
col3.metric("Mean % Error", f"{mean_err:.2f}%")

with st.expander("🧠 Why is the error high?"):
    st.markdown("- Collab or outlier posts can skew averages.")
    st.markdown("- Linear model may not capture nonlinear trends.")
    st.markdown("- Factors like audio, timing, and hashtags are not included.")

# --- Viral & Excellent Reels ---
st.subheader("🔥 Viral & Excellent Reels")
ve = df[df['performance'].isin(['Viral','Excellent'])]
if not ve.empty:
    ve_display = ve[['post_date'] + features + ['reach','predicted_reach','performance']].copy()
    ve_display['reach'] = ve_display['reach'].apply(fmt)
    ve_display['predicted_reach'] = ve_display['predicted_reach'].apply(fmt)
    ve_display = ve_display.rename(columns={
        'post_date':'Date', 'reach':'Reach', 'predicted_reach':'Predicted Reach', 'performance':'Performance'
    })
    st.dataframe(ve_display.sort_values('Reach', ascending=False))
else:
    st.write("No Viral or Excellent reels in this range.")

# --- Content Intelligence (NLP) ---
# Debug: show first few rows and columns to verify caption field
st.subheader("🔎 Data Preview")
st.write("Columns:", df.columns.tolist())
st.write(df.head(3))

# Load API key from secrets (supporting both root and [general] table)
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general", {}).get("OPENAI_API_KEY")
if not api_key:
    st.warning("OpenAI API key not found. Please add OPENAI_API_KEY to Streamlit secrets at root level.")
else:
    openai.api_key = api_key
    st.success("✅ OpenAI API key loaded.")
    # Determine text column
    text_col = 'caption' if 'caption' in df.columns else ('title' if 'title' in df.columns else None)
    if not text_col:
        st.info("No caption or title column available for NLP analysis.")
    else:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            st.info(f"No data in {text_col} column for NLP analysis.")
        else:
            st.subheader(f"🧠 Content Intelligence using '{text_col}'")
            sample_texts = texts.sample(min(5, len(texts))).tolist()
            prompt = (
                f"You are an Instagram strategist. Analyze these {text_col}s for patterns, themes, and tone:
"
                + "
".join(sample_texts)
            )
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role":"user","content":prompt}]
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"🛑 NLP analysis error: {e}")

# --- Virality Score ---
st.subheader("📊 Top Content by Virality Score")
df['virality_score'] = df['reach']/np.where(df['predicted_reach']==0, np.nan, df['predicted_reach'])
df['virality_score'] = df['virality_score'].replace([np.inf,-np.inf], np.nan).fillna(0)
top5 = df.sort_values('virality_score', ascending=False).head(5)
top5_display = top5[['caption','reach','predicted_reach','virality_score']].copy()
top5_display['reach'] = top5_display['reach'].apply(fmt)
top5_display['predicted_reach'] = top5_display['predicted_reach'].apply(fmt)
top5_display['virality_score'] = top5_display['virality_score'].apply(lambda x: f"{x:.2f}x")
top5_display = top5_display.rename(columns={
    'caption':'Caption','reach':'Reach','predicted_reach':'Predicted Reach','virality_score':'Virality Score'
})
st.dataframe(top5_display)

# --- Download Data ---
st.subheader("⬇️ Download Data")
st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "reels_data.csv", "text/csv")

# --- Predict New Reel Reach ---
st.subheader("🎯 Predict Reach for New Reel")
with st.form("predict_form"):
    inputs = {f: st.number_input(f.capitalize(), 0, value=0) for f in features}
    submit = st.form_submit_button("Predict Reach")
if submit:
    Xn = pd.DataFrame([inputs])
    pred = model.predict(Xn)[0]
    st.success(f"Predicted Reach: {fmt(pred)}")
