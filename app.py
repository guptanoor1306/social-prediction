import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
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

# Clean numeric fields
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
    df['post_date_dt'] = pd.to_datetime(df['post_date'])

    st.sidebar.subheader("📅 Filter by Post Date")
    sd, ed = st.sidebar.date_input(
        "Select date range",
        [df['post_date'].min(), df['post_date'].max()]
    )
    df = df[df['post_date'].between(sd, ed)]
else:
    st.sidebar.info("No date column found for filtering.")

# ── Train & Predict (Linear Regression) ───────────────────────────────────────
features = [f for f in ['shares','saved','comments','likes'] if f in df.columns]
X = df[features].fillna(0)
y = df['reach'].fillna(0)

model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# ── Compute historical correlations ────────────────────────────────────────────
corr = df[features + ['reach']].corr()['reach'].drop('reach').sort_values(ascending=False)

# ── Categorize performance ────────────────────────────────────────────────────
def categorize(a, p):
    if p <= 0:      return 'Poor'
    r = a / p
    if r > 2.0:     return 'Viral'
    if r > 1.5:     return 'Excellent'
    if r > 1.0:     return 'Good'
    if r > 0.5:     return 'Average'
    return 'Poor'

df['performance'] = df.apply(lambda r: categorize(r['reach'], r['predicted_reach']), axis=1)
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
dev= (abs(ta-tp)/tp*100) if tp else 0

c1.metric("Total Actual Reach",    fmt(ta))
c2.metric("Total Predicted Reach", fmt(tp))
c3.metric("Deviation %",           f"{dev:.2f}%")

with st.expander("🧠 Why high deviation?"):
    st.markdown("- Collab/outlier reels skew totals.")
    st.markdown("- Linear model under/over estimates extremes.")
    st.markdown("- Summarizes only Viral & Excellent.")

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
        'reach':'Reach','predicted_reach':'Predicted','performance':'Performance'
    })
    st.dataframe(disp.style.set_properties(
        subset=['Caption'], **{'white-space':'pre-wrap'}),
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
weekly = (
    df.set_index('post_date_dt').resample('W')[['reach','predicted_reach']]
      .mean()
      .rename(columns={'reach':'Actual','predicted_reach':'Predicted'})
)
st.line_chart(weekly)

# ── Engagement Correlation Chart ──────────────────────────────────────────────
st.subheader("🔗 Engagement Correlation with Reach")
st.bar_chart(corr)

# ── 💡 Intelligent Insights ───────────────────────────────────────────────────
st.subheader("💡 Intelligent Insights")
# 1) Average reach by weekday
if 'post_date_dt' in df.columns:
    df['weekday'] = df['post_date_dt'].dt.day_name()
    avg_by_day = df.groupby('weekday')['reach'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    ).dropna()
    st.bar_chart(avg_by_day)
    best = avg_by_day.idxmax()
    st.markdown(f"- 📅 Highest avg reach on **{best}** ({fmt(avg_by_day.max())}).")

# 2) Reach distribution quartiles
q1,q2,q3 = df['reach'].quantile([0.25,0.50,0.75])
st.markdown(
    f"- 📊 Reach quartiles: 25% < {fmt(q1)}, median {fmt(q2)}, 75% > {fmt(q3)}."
)

# ── Content Intelligence (NLP) ─────────────────────────────────────────────────
st.subheader("🧠 Content Intelligence (NLP)")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general",{}).get("OPENAI_API_KEY")
if not api_key:
    st.warning("🛑 Add OPENAI_API_KEY to Streamlit secrets.")
else:
    client = openai.OpenAI(api_key=api_key)
    text_col = 'caption' if 'caption' in df.columns else ('title' if 'title' in df.columns else None)
    if text_col:
        texts = df[text_col].dropna().astype(str)
        if len(texts):
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
        'predicted_reach':'Predicted Reach','virality_score':'Virality Score'
    })
    st.dataframe(t5, use_container_width=True)

# ── Strategic Takeaways ───────────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
1. **Double-down on saveable ‘how-to’ tips** – high saves drive long-term reach.  
2. **Boost share prompts** – shares correlate most strongly (_r_≈{corr['shares']:.2f}).  
3. **Leverage trending audio** formats & sounds.  
4. **Post mid-week evenings** (Wed/Thu 6–9 PM IST).  
5. **Collaborate** – co-tags double your viral chances.  
6. **Optimize captions** with 2–3 targeted hashtags + emojis.
""".format(corr=corr))

# ── Download Full Data ─────────────────────────────────────────────────────────
st.subheader("⬇️ Download Full Data")
st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'),
                   "reels_with_predictions.csv", mime="text/csv")

# ── Predict & Diagnose ─────────────────────────────────────────────────────────
st.subheader("🎯 Predict & Diagnose for a New Reel")
with st.form("pd_form"):
    s  = st.number_input("Shares",    min_value=0, value=0)
    sv = st.number_input("Saves",     min_value=0, value=0)
    c  = st.number_input("Comments",  min_value=0, value=0)
    l  = st.number_input("Likes",     min_value=0, value=0)
    ar = st.number_input("Actual Reach", min_value=0, value=0)
    go = st.form_submit_button("Diagnose")

if go:
    inp = pd.DataFrame([{'shares':s,'saved':sv,'comments':c,'likes':l}])
    pred = model.predict(inp)[0]
    st.success(f"📢 Predicted Reach: {fmt(pred)}")
    st.info(f"🔎 Actual Reach: {fmt(ar)}")
    st.info(f"⚖️ Residual: {fmt(ar - pred)}")

    # Show correlation table
    st.markdown("**📈 Historical correlations with Reach:**")
    corr_df = corr.reset_index().rename(columns={'index':'Metric','reach':'Correlation'})
    st.table(corr_df.style.format({'Correlation':"{:.2f}"}))

    # Recommend top metric to optimize
    top_m = corr.idxmax()
    top_r = corr.max()
    st.markdown(f"**Recommendation:** Focus on **{top_m.capitalize()}** (r={top_r:.2f}), " +
                "which historically drives reach most strongly.")
