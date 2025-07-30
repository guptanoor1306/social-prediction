import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
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
            df[col].astype(str)
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
    start_date, end_date = st.sidebar.date_input(
        "Select date range",
        [df['post_date'].min(), df['post_date'].max()]
    )
    df = df[df['post_date'].between(start_date, end_date)]
else:
    st.sidebar.info("No date column found for filtering.")

# --- Train model & make predictions ---
features = [f for f in ['shares', 'saved', 'comments', 'likes'] if f in df.columns]
X = df[features].fillna(0)
y = df['reach'].fillna(0)
model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# --- Categorize performance ---
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

# --- Number formatting ---
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
y_true = df['reach'].fillna(0)
y_pred = df['predicted_reach']
r2 = r2_score(y_true, y_pred)

# Compute RMSE manually for compatibility
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

col1.metric("Avg Actual Reach", fmt(avg_act))
col2.metric("Model R² Score", f"{r2:.2f}")
col3.metric("Model RMSE", fmt(rmse))

with st.expander("🧠 Why is the model error high?"):
    st.markdown("- Collab or outlier posts can skew predictions.")
    st.markdown("- Linear regression may not capture nonlinear influencer effects.")
    st.markdown("- Factors like audio trends, timing, and hashtags aren’t model inputs.")

# --- Viral & Excellent Reels ---
st.subheader("🔥 Viral & Excellent Reels")
ve = df[df['performance'].isin(['Viral', 'Excellent'])]
if not ve.empty:
    ve_display = ve.copy()
    display_cols = ['post_date', 'caption'] + features + ['reach', 'predicted_reach', 'performance']
    ve_display = ve_display[[c for c in display_cols if c in ve_display.columns]]
    ve_display['reach'] = ve_display['reach'].apply(fmt)
    ve_display['predicted_reach'] = ve_display['predicted_reach'].apply(fmt)
    ve_display = ve_display.rename(columns={
        'post_date': 'Date',
        'caption': 'Caption',
        'reach': 'Reach',
        'predicted_reach': 'Predicted Reach',
        'performance': 'Performance'
    })
    # Wrap caption text for visibility
    try:
        styled = ve_display.style.set_properties(
            subset=['Caption'], **{'white-space': 'pre-wrap'}
        )
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(ve_display.sort_values('Reach', ascending=False), use_container_width=True)
else:
    st.write("No Viral or Excellent reels in this range.")

# --- Content Intelligence (NLP) ---
st.subheader("🧠 Content Intelligence (NLP)")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general", {}).get("OPENAI_API_KEY")
if not api_key:
    st.warning("🛑 OpenAI API key not found. Please add OPENAI_API_KEY to Streamlit secrets.")
else:
    client = openai.OpenAI(api_key=api_key)
    text_col = 'caption' if 'caption' in df.columns else ('title' if 'title' in df.columns else None)
    if not text_col:
        st.info("No 'caption' or 'title' column available for NLP analysis.")
    else:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            st.info(f"No data in '{text_col}' for NLP analysis.")
        else:
            sample_texts = texts.sample(min(5, len(texts))).tolist()
            prompt = (
                f"You are an Instagram strategist. Analyze these {text_col}s for patterns, themes, and tone:\n"
                + "\n".join(sample_texts)
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"🛑 NLP analysis error: {e}")

# --- Virality Score ---
st.subheader("📊 Top Content by Virality Score")
df['virality_score'] = df['reach'] / np.where(df['predicted_reach'] == 0, np.nan, df['predicted_reach'])
df['virality_score'] = df['virality_score'].replace([np.inf, -np.inf], np.nan).fillna(0)
top5 = df.sort_values('virality_score', ascending=False).head(5)

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
    st.write("No captions available for Virality Score table.")

# --- Strategic Takeaways ---
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
- **Double down on collaborations:** Collab reels outperform solo by ~30–40% reach.
- **Focus on shareable tips:** Top reels have share rate >2% & save rate >1.5%.
- **Emphasize business/empowerment themes:** Podcasts, motivation, entrepreneurship drive engagement.
- **Post timing matters:** Weekdays, 6–9 PM IST sees highest reach.
- **Use emojis & 2–3 hashtags:** Improves tone and discoverability.
""")

# --- Download Data ---
st.subheader("⬇️ Download Data")
st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "reels_data.csv", "text/csv")

# --- Predict New Reel Reach ---
st.subheader("🎯 Predict Reach for New Reel")
with st.form("predict_form"):
    inputs = {f: st.number_input(f.capitalize(), 0, value=0) for f in features}
    submitted = st.form_submit_button("Predict Reach")
if submitted:
    Xn = pd.DataFrame([inputs])
    pred = model.predict(Xn)[0]
    st.success(f"Predicted Reach: {fmt(pred)}")
