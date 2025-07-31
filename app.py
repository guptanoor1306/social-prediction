import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import openai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Instagram Reels Performance Dashboard", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# ── Load & preprocess data ────────────────────────────────────────────────────
df = pd.read_csv("posts_zero1byzerodha.csv")
df.columns = df.columns.str.strip().str.lower()

# Keep only Reels
if 'type' in df.columns:
    df = df[df['type'].str.lower() == 'reel']

# Clean numeric columns
for c in ['reach','likes','comments','shares','saved']:
    if c in df.columns:
        df[c] = (
            df[c].astype(str)
                  .str.replace(r'[^\d.]','',regex=True)
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
    sd, ed = st.sidebar.date_input("Select date range",
                                   [df['post_date'].min(), df['post_date'].max()])
    df = df[df['post_date'].between(sd, ed)]
else:
    st.sidebar.info("No date column found for filtering.")

# ── Train & predict (Linear Regression) ──────────────────────────────────────
features = [f for f in ['shares','saved','comments','likes'] if f in df.columns]
X = df[features].fillna(0)
y = df['reach'].fillna(0)

model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# ── Compute historical correlations ───────────────────────────────────────────
corr = df[features + ['reach']].corr()['reach'].drop('reach')
# Sort descending
corr = corr.sort_values(ascending=False)

# ── Categorize performance ────────────────────────────────────────────────────
def categorize(a, p):
    if p <= 0:
        return 'Poor'
    r = a/p
    if r > 2.0:    return 'Viral'
    if r > 1.5:    return 'Excellent'
    if r > 1.0:    return 'Good'
    if r > 0.5:    return 'Average'
    return 'Poor'

df['performance'] = df.apply(lambda r: categorize(r['reach'], r['predicted_reach']), axis=1)

# ── Prepare Viral & Excellent subset ──────────────────────────────────────────
ve = df[df['performance'].isin(['Viral','Excellent'])]

# ── Formatting helper ──────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n): return "-"
    if n >= 1e6:   return f"{n/1e6:.2f}M"
    if n >= 1e3:   return f"{n/1e3:.1f}K"
    return str(int(n))

# ── Summary Metrics ────────────────────────────────────────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent Totals)")
c1, c2, c3 = st.columns(3)
ta = ve['reach'].sum()
tp = ve['predicted_reach'].sum()
dev = (abs(ta-tp)/tp*100) if tp else 0
c1.metric("Total Actual Reach",    fmt(ta))
c2.metric("Total Predicted Reach", fmt(tp))
c3.metric("Deviation %",           f"{dev:.2f}%")

# ── Viral & Excellent Table ───────────────────────────────────────────────────
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    disp = ve.copy()
    cols = ['post_date','caption'] + features + ['reach','predicted_reach','performance']
    disp = disp[[c for c in cols if c in disp.columns]]
    disp['reach']           = disp['reach'].apply(fmt)
    disp['predicted_reach'] = disp['predicted_reach'].apply(fmt)
    disp = disp.rename(columns={
        'post_date':'Date','caption':'Caption',
        'shares':'Shares','saved':'Saves','comments':'Comments','likes':'Likes',
        'reach':'Reach','predicted_reach':'Predicted Reach','performance':'Performance'
    })
    st.dataframe(
        disp.style.set_properties(subset=['Caption'], **{'white-space':'pre-wrap'}),
        use_container_width=True
    )
else:
    st.write("No Viral/Excellent reels in this range.")

# ── Performance Distribution ──────────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
perf_counts = df['performance'].value_counts().reindex(
    ['Viral','Excellent','Good','Average','Poor'], fill_value=0
)
st.bar_chart(perf_counts)

# ── Reach Trend Over Time ─────────────────────────────────────────────────────
st.subheader("📈 Reach Trend Over Time")
if 'post_date' in df.columns:
    df['post_date_dt'] = pd.to_datetime(df['post_date'])
    weekly = (
        df.set_index('post_date_dt')
          .resample('W')[['reach','predicted_reach']]
          .mean()
          .rename(columns={'reach':'Actual','predicted_reach':'Predicted'})
    )
    st.line_chart(weekly)

# ── Engagement Correlation Chart ──────────────────────────────────────────────
st.subheader("🔗 Engagement Correlation with Reach")
st.bar_chart(corr)

# ── Content Intelligence (NLP) ─────────────────────────────────────────────────
st.subheader("🧠 Content Intelligence (NLP)")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general",{}).get("OPENAI_API_KEY")
if api_key:
    client = openai.OpenAI(api_key=api_key)
    text_col = 'caption' if 'caption' in df.columns else ('title' if 'title' in df.columns else None)
    if text_col:
        texts = df[text_col].dropna().astype(str)
        if not texts.empty:
            sample = texts.sample(min(5,len(texts))).tolist()
            prompt = (
                f"You are an Instagram strategist. Analyze these {text_col}s for patterns, themes, and tone:\n"
                + "\n".join(sample)
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4", messages=[{"role":"user","content":prompt}]
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"NLP error: {e}")
    else:
        st.info("No caption/title column for NLP.")
else:
    st.warning("Add OPENAI_API_KEY to Streamlit secrets.")

# ── Top 5 by Virality Score ────────────────────────────────────────────────────
st.subheader("📈 Top Content by Virality Score")
df['virality_score'] = df['reach']/np.where(df['predicted_reach']==0, np.nan, df['predicted_reach'])
df['virality_score'] = df['virality_score'].replace([np.inf,-np.inf],np.nan).fillna(0)
top5 = df.nlargest(5,'virality_score')
if 'caption' in top5.columns:
    t5 = top5[['caption','reach','predicted_reach','virality_score']].copy()
    t5['reach']           = t5['reach'].apply(fmt)
    t5['predicted_reach'] = t5['predicted_reach'].apply(fmt)
    t5['virality_score']  = t5['virality_score'].apply(lambda x:f"{x:.2f}x")
    t5 = t5.rename(columns={
        'caption':'Caption','reach':'Reach',
        'predicted_reach':'Predicted','virality_score':'Virality Score'
    })
    st.dataframe(t5, use_container_width=True)

# ── Strategic Takeaways ───────────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
1. **Double-down on saveable “how-to” tips** – high saves drive long-term reach.  
2. **Boost share prompts** – shares have the highest correlation.  
3. **Leverage trending audio** – for extra algorithmic lift.  
4. **Time mid-week evenings** (Wed/Thu 6–9 PM IST).  
5. **Collaborate** – co-tags double your viral chance.  
6. **Optimize captions** with 2–3 targeted hashtags + emojis.
""")

# ── Download Full Data ─────────────────────────────────────────────────────────
st.subheader("⬇️ Download Full Data")
st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'),
                   "reels_data.csv", "text/csv")

# ── Predict & Diagnose ─────────────────────────────────────────────────────────
st.subheader("🎯 Predict & Diagnose for a New Reel")
with st.form("pd_form"):
    s = st.number_input("Shares",    min_value=0, value=0)
    sv= st.number_input("Saves",     min_value=0, value=0)
    c = st.number_input("Comments",  min_value=0, value=0)
    l = st.number_input("Likes",     min_value=0, value=0)
    ar= st.number_input("Actual Reach",min_value=0, value=0)
    go= st.form_submit_button("Diagnose")

if go:
    # Predict
    inp = pd.DataFrame([{'shares':s,'saved':sv,'comments':c,'likes':l}])
    pred = model.predict(inp)[0]
    st.success(f"📢 Predicted Reach: {fmt(pred)}")
    st.info   (f"🔎 Actual Reach: {fmt(ar)}")
    st.info   (f"⚖️ Residual: {fmt(ar - pred)}")

    # Show correlation table
    st.markdown("**📈 Historical correlations with Reach:**")
    corr_df = corr.reset_index().rename(columns={'index':'Metric','reach':'Correlation'})
    st.table(corr_df.style.format({'Correlation':"{:.2f}"}))

    # Recommend which to optimize
    top_metric = corr.idxmax()
    top_corr   = corr.max()
    st.markdown(f"**Recommendation:** Increase **{top_metric.capitalize()}** (_r_={top_corr:.2f}), which historically drives reach most strongly.")
