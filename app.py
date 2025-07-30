import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import openai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Instagram Reels Performance Dashboard", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# ── Load & preprocess data ────────────────────────────────────────────────────
df = pd.read_csv("posts_zero1byzerodha.csv")
df.columns = df.columns.str.strip().str.lower()

# Filter to Reels
if 'type' in df.columns:
    df = df[df['type'].str.lower() == 'reel']

# Clean numeric columns
for col in ['reach','likes','comments','shares','saved']:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                 .str.replace(r'[^\d.]', '', regex=True)
                 .replace('', np.nan)
                 .astype(float)
        )

# ── Date filter ───────────────────────────────────────────────────────────────
date_col = next((c for c in df.columns if 'date' in c), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df['post_date'] = df[date_col].dt.date

    st.sidebar.subheader("📅 Filter by Post Date")
    dates = st.sidebar.date_input(
        "Select date range",
        [df['post_date'].min(), df['post_date'].max()]
    )
    if isinstance(dates, (list, tuple)) and len(dates)==2:
        start_date, end_date = dates
    else:
        start_date = end_date = dates
    df = df[df['post_date'].between(start_date, end_date)]
else:
    st.sidebar.info("No date column found for filtering.")

# ── Train & predict (linear regression) ───────────────────────────────────────
features = [f for f in ['shares','saved','comments','likes'] if f in df.columns]
X = df[features].fillna(0)
y = df['reach'].fillna(0)

model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# ── Categorize performance ────────────────────────────────────────────────────
def categorize(a, p):
    if p <= 0: return 'Uncategorized'
    r = a/p
    if r > 2.0:    return 'Viral'
    if r > 1.5:    return 'Excellent'
    if r > 1.0:    return 'Good'
    if r > 0.5:    return 'Average'
    return 'Poor'

df['performance'] = df.apply(lambda row: categorize(row['reach'], row['predicted_reach']), axis=1)

# ── Formatting helper ──────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n): return "-"
    if n >= 1e6:   return f"{n/1e6:.2f}M"
    if n >= 1e3:   return f"{n/1e3:.1f}K"
    return str(int(n))

# ── Viral & Excellent subset ─────────────────────────────────────────────────
ve = df[df['performance'].isin(['Viral','Excellent'])]

# ── Summary Insights ──────────────────────────────────────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent Totals)")
c1, c2, c3 = st.columns(3)
total_act  = ve['reach'].sum()
total_pred = ve['predicted_reach'].sum()
deviation  = (abs(total_act-total_pred)/total_pred*100) if total_pred else 0

c1.metric("Total Actual Reach",      fmt(total_act))
c2.metric("Total Predicted Reach",   fmt(total_pred))
c3.metric("Deviation %",             f"{deviation:.2f}%")

with st.expander("🧠 Why is the deviation high?"):
    st.markdown("- Collab or outlier reels heavily influence totals.")
    st.markdown("- Linear regression cannot perfectly capture outliers.")
    st.markdown("- We're only summing the Viral & Excellent group.")

# ── Viral & Excellent Reels Table ─────────────────────────────────────────────
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    disp = ve.copy()
    cols = ['post_date','caption'] + features + ['reach','predicted_reach','performance']
    disp = disp[[c for c in cols if c in disp.columns]]
    disp['reach']           = disp['reach'].apply(fmt)
    disp['predicted_reach'] = disp['predicted_reach'].apply(fmt)
    disp = disp.rename(columns={
        'post_date':       'Date',
        'caption':         'Caption',
        'shares':          'Shares',
        'saved':           'Saves',
        'comments':        'Comments',
        'likes':           'Likes',
        'reach':           'Reach',
        'predicted_reach': 'Predicted Reach',
        'performance':     'Performance'
    })
    try:
        styled = disp.style.set_properties(
            subset=['Caption'], **{'white-space':'pre-wrap'}
        )
        st.dataframe(styled, use_container_width=True)
    except:
        st.dataframe(disp, use_container_width=True)
else:
    st.write("No Viral or Excellent reels in this range.")

# ── Performance Distribution Chart ────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
perf_counts = df['performance'].value_counts().reindex(
    ['Viral','Excellent','Good','Average','Poor','Uncategorized'], fill_value=0
)
st.bar_chart(perf_counts)

# ── Content Intelligence (NLP) ─────────────────────────────────────────────────
st.subheader("🧠 Content Intelligence (NLP)")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general",{}).get("OPENAI_API_KEY")
if not api_key:
    st.warning("🛑 Please add OPENAI_API_KEY to Streamlit secrets.")
else:
    client = openai.OpenAI(api_key=api_key)
    text_col = 'caption' if 'caption' in df.columns else ('title' if 'title' in df.columns else None)
    if not text_col:
        st.info("No caption/title column found for NLP.")
    else:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            st.info(f"No data in '{text_col}'.")
        else:
            sample = texts.sample(min(5,len(texts))).tolist()
            prompt = (
                f"You are an Instagram strategist. Analyze these {text_col}s "
                "for patterns, themes, and tone:\n" + "\n".join(sample)
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4", messages=[{"role":"user","content":prompt}]
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"🛑 NLP analysis error: {e}")

# ── Top 5 by Virality Score ───────────────────────────────────────────────────
st.subheader("📈 Top Content by Virality Score")
df['virality_score'] = df['reach'] / np.where(df['predicted_reach']==0, np.nan, df['predicted_reach'])
df['virality_score'] = df['virality_score'].replace([np.inf,-np.inf],np.nan).fillna(0)
top5 = df.nlargest(5, 'virality_score')
if 'caption' in top5.columns:
    t5 = top5[['caption','reach','predicted_reach','virality_score']].copy()
    t5['reach']           = t5['reach'].apply(fmt)
    t5['predicted_reach'] = t5['predicted_reach'].apply(fmt)
    t5['virality_score']  = t5['virality_score'].apply(lambda x: f"{x:.2f}x")
    t5 = t5.rename(columns={
        'caption':'Caption','reach':'Reach',
        'predicted_reach':'Predicted Reach','virality_score':'Virality Score'
    })
    st.dataframe(t5, use_container_width=True)
else:
    st.write("No captions for virality ranking.")

# ── Strategic Takeaways ───────────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
1. **Double‑down on saveable “how‑to” tips** – high saves = strong long‑term reach.  
2. **Increase share prompts** – “tag a friend” or “share this with….” boosts non‑follower reach.  
3. **Leverage trending audio** – align with current sounds/transitions for algorithmic lift.  
4. **Post mid‑week evenings** (Wed/Thu, 6–9 PM IST) for peak engagement windows.  
5. **Ramp up collaborations** – co‑tags and mentions double your viral odds.  
6. **Optimize captions** with 2–3 niche hashtags + emojis (🔖,⚡️) for tone & discoverability.
""")

# ── Download & Predict ─────────────────────────────────────────────────────────
st.subheader("⬇️ Download Full Data")
st.download_button(
    "Download CSV", df.to_csv(index=False).encode('utf-8'),
    "reels_with_predictions.csv", mime="text/csv"
)

st.subheader("🎯 Predict Reach for a New Reel (Post‑launch)")
st.markdown("⚠️ Enter live engagement values; model scales from your history.")
with st.form("predict"):
    vals = {f: st.number_input(f.capitalize(), 0, value=0) for f in features}
    go = st.form_submit_button("Predict")
if go:
    pred = model.predict(pd.DataFrame([vals]))[0]
    st.success(f"📢 Predicted Reach: {fmt(pred)}")
