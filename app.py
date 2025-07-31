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

# Clean numeric fields
for col in ['reach','shares','saved','comments','likes']:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                   .str.replace(r'[^\d.]','', regex=True)
                   .replace('', np.nan)
                   .astype(float)
        )

# ── Date filter ───────────────────────────────────────────────────────────────
date_col = next((c for c in df.columns if 'date' in c), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df['post_date']    = df[date_col].dt.date
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

# ── Remove super‐outliers for correlation analysis ─────────────────────────────
df_no = df.copy()
for f in features + ['reach']:
    if f in df_no.columns:
        low, high = df_no[f].quantile([0.01, 0.99])
        df_no = df_no[df_no[f].between(low, high)]

# ── Compute correlations ───────────────────────────────────────────────────────
corr_overall = df_no[features + ['reach']].corr()['reach'].drop('reach').sort_values(ascending=False)

cats = ['Poor','Average','Good','Excellent','Viral']
corr_by_cat = pd.DataFrame(index=features, columns=cats)
for cat in cats:
    sub = df_no[df_no['performance'] == cat] if 'performance' in df_no.columns else pd.DataFrame()
    if len(sub) > len(features):
        corr_by_cat[cat] = sub[features + ['reach']].corr().loc['reach', features]
    else:
        corr_by_cat[cat] = np.nan

# ── Categorize performance ────────────────────────────────────────────────────
def categorize(actual, pred):
    if pred <= 0:
        return 'Poor'
    r = actual / pred
    if r > 2.0:    return 'Viral'
    if r > 1.5:    return 'Excellent'
    if r > 1.0:    return 'Good'
    if r > 0.5:    return 'Average'
    return 'Poor'

df['performance'] = df.apply(lambda r: categorize(r['reach'], r['predicted_reach']), axis=1)
ve = df[df['performance'].isin(['Viral','Excellent'])]

# ── Formatting helper ──────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n): return "-"
    if n >= 1e6:   return f"{n/1e6:.2f}M"
    if n >= 1e3:   return f"{n/1e3:.1f}K"
    return str(int(n))

# ── Summary Metrics (Viral & Excellent Totals) ────────────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent Totals)")
c1, c2, c3 = st.columns(3)
ta = ve['reach'].sum()
tp = ve['predicted_reach'].sum()
dev = (abs(ta - tp) / tp * 100) if tp else 0

c1.metric("Total Actual Reach",    fmt(ta))
c2.metric("Total Predicted Reach", fmt(tp))
c3.metric("Deviation %",           f"{dev:.2f}%")

with st.expander("🧠 Why is the deviation high?"):
    st.markdown("- Collaborative or outlier posts skew totals.")
    st.markdown("- Linear model may under/over estimate extreme cases.")
    st.markdown("- Summary limited to Viral & Excellent posts.")

# ── Viral & Excellent Reels Table ─────────────────────────────────────────────
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    out = ve.copy()
    cols = ['post_date','caption'] + features + ['reach','predicted_reach','performance']
    out = out[[c for c in cols if c in out.columns]]
    out['reach']           = out['reach'].apply(fmt)
    out['predicted_reach'] = out['predicted_reach'].apply(fmt)
    out = out.rename(columns={
        'post_date':'Date','caption':'Caption',
        'shares':'Shares','saved':'Saves','comments':'Comments','likes':'Likes',
        'reach':'Reach','predicted_reach':'Predicted Reach','performance':'Performance'
    })
    st.dataframe(
        out.style.set_properties(subset=['Caption'], **{'white-space':'pre-wrap'}),
        use_container_width=True
    )
else:
    st.write("No Viral & Excellent reels in this range.")

# ── Performance Distribution ──────────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
perf_counts = df['performance'].value_counts().reindex(cats, fill_value=0)
st.bar_chart(perf_counts)

# ── Reach Trend Over Time ─────────────────────────────────────────────────────
st.subheader("📈 Reach Trend Over Time")
if 'post_date_dt' in df.columns:
    weekly = (df.set_index('post_date_dt')
                .resample('W')[['reach','predicted_reach']]
                .mean()
                .rename(columns={'reach':'Actual Reach','predicted_reach':'Predicted Reach'}))
    st.line_chart(weekly)

# ── Engagement Correlation by Category ────────────────────────────────────────
st.subheader("🔗 Engagement Correlation by Category")
st.bar_chart(corr_by_cat)

# ── Intelligent Insights ──────────────────────────────────────────────────────
st.subheader("💡 Intelligent Insights")
if 'post_date_dt' in df.columns:
    df['weekday'] = df['post_date_dt'].dt.day_name()
    avg_day = df.groupby('weekday')['reach'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    ).dropna()
    st.bar_chart(avg_day)
    best_day = avg_day.idxmax()
    st.markdown(f"- 📅 Highest average reach on **{best_day}** ({fmt(avg_day.max())}).")
q1, q2, q3 = df['reach'].quantile([0.25,0.5,0.75])
st.markdown(f"- 📊 Reach quartiles: 25% < {fmt(q1)}, median {fmt(q2)}, 75% > {fmt(q3)}.")

# ── Content Intelligence (NLP) ─────────────────────────────────────────────────
st.subheader("🧠 Content Intelligence (NLP)")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general",{}).get("OPENAI_API_KEY")
if api_key and any(c in df.columns for c in ['caption','title']):
    client = openai.OpenAI(api_key=api_key)
    tc = 'caption' if 'caption' in df.columns else 'title'
    texts = df[tc].dropna().astype(str)
    if not texts.empty:
        sample = texts.sample(min(5,len(texts))).tolist()
        prompt = (f"You are an Instagram strategist. Analyze these {tc}s for patterns, themes, and tone:\n"
                  + "\n".join(sample))
        try:
            resp = client.chat.completions.create(
                model="gpt-4", messages=[{"role":"user","content":prompt}]
            )
            st.markdown(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"NLP error: {e}")
else:
    st.warning("Add OPENAI_API_KEY to secrets and include a caption/title column for NLP.")

# ── Top 5 by Virality Score ────────────────────────────────────────────────────
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

# ── Strategic Takeaways ───────────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown(f"""
1. **Shares** drive reach most strongly overall (r={corr_overall['shares']:.2f}).  
2. **Correlations vary by category**—see chart above to tailor your strategy.  
3. **If all engagement levers are optimized yet reach lags**, revisit your hook/creative.
""")

# ── Download & Predict & Diagnose ─────────────────────────────────────────────
st.subheader("⬇️ Download Full Data")
st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'),
                   "reels_with_predictions.csv", "text/csv")

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
    perf = categorize(ar, pred)
    st.success(f"📢 Predicted Reach: {fmt(pred)}")
    st.info(f"🔎 Actual Reach: {fmt(ar)}")
    st.info(f"🎯 Category: **{perf}**")

    # Show per-category correlation table
    st.markdown("**📈 Engagement Correlations by Category:**")
    st.table(corr_by_cat)

    # Recommend next metric to optimize
    if perf in ['Poor','Average','Good','Excellent']:
        next_bucket = {'Poor':'Average','Average':'Good','Good':'Excellent','Excellent':'Viral'}[perf]
        best_metric = corr_by_cat[next_bucket].idxmax()
        best_val    = corr_by_cat[next_bucket].max()
        st.markdown(f"🔜 To move from **{perf}** → **{next_bucket}**, focus on **{best_metric.capitalize()}** (r={best_val:.2f}).")
    else:
        st.markdown("✅ You’re already in the Viral category—keep up the great work!")
