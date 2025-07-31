import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import altair as alt
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

# ── Categorize performance ────────────────────────────────────────────────────
def categorize(actual, pred):
    if pred <= 0:       return 'Poor'
    r = actual / pred
    if r > 2.0:         return 'Viral'
    if r > 1.5:         return 'Excellent'
    if r > 1.0:         return 'Good'
    if r > 0.5:         return 'Average'
    return 'Poor'

df['performance'] = df.apply(lambda r: categorize(r['reach'], r['predicted_reach']), axis=1)
ve = df[df['performance'].isin(['Viral','Excellent'])]

# ── Remove super-outliers ─────────────────────────────────────────────────────
df_no = df.copy()
for f in features + ['reach']:
    if f in df_no.columns:
        low, high = df_no[f].quantile([0.01, 0.99])
        df_no = df_no[df_no[f].between(low, high)]

# ── Compute overall & per-category correlations ───────────────────────────────
corr_overall = df_no[features + ['reach']].corr()['reach'].drop('reach').sort_values(ascending=False)

cats = ['Viral','Excellent','Good','Average','Poor']
corr_by_cat = pd.DataFrame(index=features, columns=cats)
for cat in cats:
    subset = df_no[df_no['performance'] == cat]
    if len(subset) >= len(features):
        corr_by_cat[cat] = subset[features + ['reach']].corr().loc['reach', features]
    else:
        corr_by_cat[cat] = np.nan

# ── Formatter ─────────────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n): return "-"
    if n >= 1e6:   return f"{n/1e6:.2f}M"
    if n >= 1e3:   return f"{n/1e3:.1f}K"
    return str(int(n))

# ── Summary Metrics ────────────────────────────────────────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent Totals)")
col1, col2, col3 = st.columns(3)
total_act = ve['reach'].sum()
total_pr  = ve['predicted_reach'].sum()
dev_pct   = (abs(total_act - total_pr) / total_pr * 100) if total_pr else 0

col1.metric("Total Actual Reach",    fmt(total_act))
col2.metric("Total Predicted Reach", fmt(total_pr))
col3.metric("Deviation %",           f"{dev_pct:.2f}%")

with st.expander("🧠 Why is the deviation high?"):
    st.markdown("- Collaborative/outlier posts skew totals")  
    st.markdown("- Linear model under/over estimates extremes")  
    st.markdown("- Only Viral & Excellent posts counted")

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
    st.write("No Viral or Excellent reels in this range.")

# ── Performance Distribution ──────────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
perf_counts = df['performance'].value_counts().reindex(cats, fill_value=0)
st.bar_chart(perf_counts)

# ── Reach Trend Over Time ─────────────────────────────────────────────────────
st.subheader("📈 Reach Trend Over Time")
if 'post_date_dt' in df.columns:
    weekly = (
        df.set_index('post_date_dt')
          .resample('W')[['reach','predicted_reach']]
          .mean()
          .rename(columns={'reach':'Actual Reach','predicted_reach':'Predicted Reach'})
    )
    st.line_chart(weekly)

# ── Engagement Correlation by Category ────────────────────────────────────────
st.subheader("🔗 Engagement Correlation by Category")
# Prepare for grouped bar chart
source = (
    corr_by_cat
    .reset_index()
    .melt('index', var_name='Category', value_name='Correlation')
    .rename(columns={'index':'Engagement'})
)
chart = (
    alt.Chart(source)
       .mark_bar()
       .encode(
           x=alt.X('Engagement:N', title='Engagement Metric'),
           xOffset='Category:N',
           y=alt.Y('Correlation:Q'),
           color='Category:N'
       )
       .properties(width='container', height=300)
)
st.altair_chart(chart, use_container_width=True)

# ── Content Intelligence ───────────────────────────────────────────────────────
st.subheader("🧠 Content Intelligence")
api_key = (st.secrets.get("OPENAI_API_KEY")
           or st.secrets.get("general",{}).get("OPENAI_API_KEY"))
if api_key and 'caption' in df.columns:
    client = openai.OpenAI(api_key=api_key)
    sample = df['caption'].dropna().astype(str).sample(min(5, len(df))).tolist()
    prompt = (
        "You are an Instagram strategist. Based on these 5 reel captions:\n\n" +
        "\n".join(f"- {t}" for t in sample) +
        "\n\nGive:\n"
        "1. Common content patterns or themes\n"
        "2. Predicted tone or emotion\n"
        "3. One idea to improve virality\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4", messages=[{"role":"user","content":prompt}]
        )
        st.markdown(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"NLP error: {e}")
else:
    st.info("Add OPENAI_API_KEY in Streamlit secrets to enable Content Intelligence.")

# ── Strategic Takeaways ───────────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
1. **Double-down on saveable “how-to” tips** – high saves drive long-term reach.  
2. **Drive share prompts** (“tag a friend”) for algorithmic uplift.  
3. **Leverage trending audio** formats & sounds.  
4. **Post Wed/Thu evenings (6–9 PM IST)** for peak visibility.  
5. **Collaborate** with peers/influencers – co-tags double viral odds.  
6. **Optimize captions** with 2–3 targeted hashtags + emojis.
""")

# ── Download & Predict & Diagnose ─────────────────────────────────────────────
st.subheader("⬇️ Download Full Data")
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode('utf-8'),
    "reels_with_predictions.csv",
    "text/csv"
)

st.subheader("🎯 Predict & Diagnose for a New Reel")
with st.form("diagnose_form"):
    s  = st.number_input("Shares",      0, value=0)
    sv = st.number_input("Saves",       0, value=0)
    c  = st.number_input("Comments",    0, value=0)
    l  = st.number_input("Likes",       0, value=0)
    ar = st.number_input("Actual Reach",0, value=0)
    go = st.form_submit_button("Diagnose")

if go:
    inp  = pd.DataFrame([{'shares':s,'saved':sv,'comments':c,'likes':l}])
    pred = model.predict(inp)[0]
    perf = categorize(ar, pred)

    st.success(f"📢 Predicted Reach: {fmt(pred)}")
    st.info   (f"🔎 Actual Reach: {fmt(ar)}")
    st.info   (f"🎯 Category: **{perf}**")

    st.markdown("**📈 Engagement Correlations by Category:**")
    st.table(corr_by_cat)

    # Promotion mapping
    promotion = {'Poor':'Average','Average':'Good','Good':'Excellent','Excellent':'Viral'}
    if perf in promotion:
        target = promotion[perf]
        rcat   = corr_by_cat[target]
        if rcat.notna().any():
            best_m = rcat.idxmax()
            best_v = rcat.max()
            st.markdown(f"🔜 To move **{perf}** → **{target}**, focus on **{best_m.capitalize()}** (r={best_v:.2f}).")
        else:
            st.write("🔍 Not enough data to recommend the next step.")
    else:
        st.markdown("✅ You’re already in **Viral** territory—awesome work!")
