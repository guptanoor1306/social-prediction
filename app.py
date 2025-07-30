import streamlit as st
import pandas as pd
import numpy as np
import openai

# Page configuration
st.set_page_config(page_title="Instagram Reels Performance Dashboard", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# --- Load & preprocess data ---
df = pd.read_csv("posts_zero1byzerodha.csv")
df.columns = df.columns.str.strip().str.lower()

# Filter only Reels
if 'type' in df.columns:
    df = df[df['type'].str.lower() == 'reel']

# Clean numeric fields
for col in ['reach', 'likes', 'comments', 'shares', 'saved']:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r'[^\d.]', '', regex=True)
            .replace('', np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Date detection & filtering ---
date_col = next((c for c in df.columns if 'date' in c), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['post_date'] = df[date_col].dt.date

    st.sidebar.subheader("📅 Filter by Post Date")
    dates = st.sidebar.date_input(
        "Select date range",
        [df['post_date'].min(), df['post_date'].max()]
    )
    if isinstance(dates, (list, tuple)) and len(dates) == 2:
        start_date, end_date = dates
    else:
        start_date = end_date = dates
    df = df[df['post_date'].between(start_date, end_date)]
else:
    st.sidebar.info("No date column found for filtering.")

# --- Prediction via normalized impact weights ---
# Raw impact weights: Shares=10, Saves=5, Comments=3, Likes=2
raw_weights = {'shares': 10, 'saved': 5, 'comments': 3, 'likes': 2}
total_w = sum(raw_weights.values())
weights = {k: v / total_w for k, v in raw_weights.items()}

# Ensure columns exist
for col in weights:
    if col not in df.columns:
        df[col] = 0

# Compute weighted-sum reach prediction
df['predicted_reach'] = (
    df['shares']   * weights['shares'] +
    df['saved']    * weights['saved'] +
    df['comments'] * weights['comments'] +
    df['likes']    * weights['likes']
)

# --- Categorize performance ---
def categorize(actual, pred):
    if pred == 0:
        return 'Uncategorized'
    ratio = actual / pred
    if ratio > 2.0:
        return 'Viral'
    if ratio > 1.5:
        return 'Excellent'
    if ratio > 1.0:
        return 'Good'
    if ratio > 0.5:
        return 'Average'
    return 'Poor'

df['performance'] = df.apply(lambda r: categorize(r['reach'], r['predicted_reach']), axis=1)

# --- Number formatting helper ---
def fmt(x):
    if pd.isna(x):
        return "-"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{x/1e3:.1f}K"
    return str(int(x))

# --- Viral & Excellent subset for summaries ---
ve = df[df['performance'].isin(['Viral', 'Excellent'])]

# --- Summary Metrics (Viral & Excellent Totals) ---
st.subheader("📈 Summary Insights (Viral & Excellent Totals)")
c1, c2, c3 = st.columns(3)
total_act = ve['reach'].sum()
total_pred = ve['predicted_reach'].sum()
deviation = (abs(total_act - total_pred) / total_pred * 100) if total_pred else 0

c1.metric("Total Actual Reach", fmt(total_act))
c2.metric("Total Predicted Reach", fmt(total_pred))
c3.metric("Deviation %", f"{deviation:.2f}%")

with st.expander("🧠 Why is the deviation high?"):
    st.markdown("- Weights are fixed, not trained on this dataset.")
    st.markdown("- Collab/outlier reels can skew totals.")
    st.markdown("- Only Viral & Excellent subset is summarized here.")

# --- Viral & Excellent Reels Table ---
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    disp = ve.copy()
    columns = [
        'post_date', 'caption', 'shares', 'saved',
        'comments', 'likes', 'reach', 'predicted_reach', 'performance'
    ]
    disp = disp[[c for c in columns if c in disp.columns]]
    disp['reach'] = disp['reach'].apply(fmt)
    disp['predicted_reach'] = disp['predicted_reach'].apply(fmt)
    disp = disp.rename(columns={
        'post_date': 'Date',
        'caption': 'Caption',
        'shares': 'Shares',
        'saved': 'Saves',
        'comments': 'Comments',
        'likes': 'Likes',
        'reach': 'Reach',
        'predicted_reach': 'Predicted Reach',
        'performance': 'Performance'
    })
    try:
        styled = disp.style.set_properties(
            subset=['Caption'], **{'white-space': 'pre-wrap'}
        )
        st.dataframe(styled, use_container_width=True)
    except:
        st.dataframe(disp, use_container_width=True)
else:
    st.write("No Viral or Excellent reels in this range.")

# --- Content Intelligence (NLP) ---
st.subheader("🧠 Content Intelligence (NLP)")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general", {}).get("OPENAI_API_KEY")
if not api_key:
    st.warning("🛑 OPENAI_API_KEY not found in Streamlit secrets.")
else:
    client = openai.OpenAI(api_key=api_key)
    text_col = 'caption' if 'caption' in df.columns else ('title' if 'title' in df.columns else None)
    if not text_col:
        st.info("No 'caption' or 'title' column available for NLP analysis.")
    else:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            st.info(f"No data in '{text_col}'.")
        else:
            sample = texts.sample(min(5, len(texts))).tolist()
            prompt = (
                f"You are an Instagram strategist. Analyze these {text_col}s for patterns, themes, and tone:\n"
                + "\n".join(sample)
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"🛑 NLP analysis error: {e}")

# --- Virality Score Ranking ---
st.subheader("📊 Top Content by Virality Score")
df['virality_score'] = df['reach'] / np.where(df['predicted_reach'] == 0, np.nan, df['predicted_reach'])
df['virality_score'] = df['virality_score'].replace([np.inf, -np.inf], np.nan).fillna(0)
top5 = df.nlargest(5, 'virality_score')
if 'caption' in top5.columns:
    t5 = top5[['caption', 'reach', 'predicted_reach', 'virality_score']].copy()
    t5['reach'] = t5['reach'].apply(fmt)
    t5['predicted_reach'] = t5['predicted_reach'].apply(fmt)
    t5['virality_score'] = t5['virality_score'].apply(lambda x: f"{x:.2f}x")
    t5 = t5.rename(columns={
        'caption': 'Caption',
        'reach': 'Reach',
        'predicted_reach': 'Predicted Reach',
        'virality_score': 'Virality Score'
    })
    st.dataframe(t5, use_container_width=True)
else:
    st.write("No captions available for virality ranking.")

# --- Strategic Takeaways ---
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
- **Shares carry the most weight (10/20)** in our model, so maximize shareable hooks.
- **Saves (5/20)** indicate evergreen value—create content worth revisiting.
- **Comments (3/20)** drive conversation—use questions or prompts.
- **Likes (2/20)** still matter for social proof.
""")

# --- Download Full Data ---
st.subheader("⬇️ Download Full Data")
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode('utf-8'),
    "reels_with_predictions.csv",
    mime="text/csv"
)

# --- Predict Reach for a New Reel ---
st.subheader("🎯 Predict Reach for New Reel (Pre‑launch)")
st.markdown("Inputs use the same 10:5:3:2 weighting model.")
with st.form("predict_form"):
    s = st.number_input("Shares",   min_value=0, value=0)
    sv = st.number_input("Saves",   min_value=0, value=0)
    c = st.number_input("Comments", min_value=0, value=0)
    l = st.number_input("Likes",    min_value=0, value=0)
    submitted = st.form_submit_button("Predict")
if submitted:
    pred = (
        s * weights['shares'] +
        sv * weights['saved'] +
        c * weights['comments'] +
        l * weights['likes']
    )
    st.success(f"📢 Predicted Reach: {fmt(pred)}")
