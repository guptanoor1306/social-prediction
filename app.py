import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import altair as alt
import openai

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Instagram Reels Performance Dashboard", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# ── Load & Preprocess Full Dataset ────────────────────────────────────────────
df_full = pd.read_csv("posts_zero1byzerodha.csv")
df_full.columns = df_full.columns.str.strip().str.lower()

# Keep only Reels
if 'type' in df_full.columns:
    df_full = df_full[df_full['type'].str.lower() == 'reel']

# Clean numeric fields
for col in ['reach','shares','saved','comments','likes']:
    if col in df_full.columns:
        df_full[col] = (
            df_full[col].astype(str)
                    .str.replace(r'[^\d.]','', regex=True)
                    .replace('', np.nan)
                    .astype(float)
        )

# ── One-Time: Train on Full Data & Categorize ────────────────────────────────
features = [c for c in ['shares','saved','comments','likes'] if c in df_full.columns]

# Fit a regression on the full set
X_full = df_full[features].fillna(0)
y_full = df_full['reach'].fillna(0)
full_model = LinearRegression().fit(X_full, y_full)
df_full['predicted_reach'] = full_model.predict(X_full)

# Categorization function
def categorize(actual, pred):
    if pred <= 0:
        return 'Poor'
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

df_full['performance'] = df_full.apply(
    lambda r: categorize(r['reach'], r['predicted_reach']), axis=1
)

# ── One-Time: Compute Correlations by Category (Full Data) ──────────────────
cats = ['Viral','Excellent','Good','Average','Poor']
corr_by_cat = pd.DataFrame(index=features, columns=cats)

for cat in cats:
    sub = df_full[df_full['performance'] == cat]
    # require at least 3 posts for stable correlation
    if len(sub) >= 3:
        corr_by_cat[cat] = sub[features + ['reach']].corr().loc['reach', features]
    else:
        corr_by_cat[cat] = np.nan

# ── Copy df_full for Filtering & Modeling ────────────────────────────────────
df = df_full.copy()

# ── Sidebar Date Filter ───────────────────────────────────────────────────────
date_col = next((c for c in df.columns if 'date' in c), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df['post_date']    = df[date_col].dt.date
    df['post_date_dt'] = pd.to_datetime(df['post_date'])
    st.sidebar.subheader("📅 Filter by Post Date")
    start_date, end_date = st.sidebar.date_input(
        "Select date range",
        [df['post_date'].min(), df['post_date'].max()]
    )
    df = df[df['post_date'].between(start_date, end_date)]
else:
    st.sidebar.info("No date column found for filtering.")

# ── Retrain on Filtered Data ─────────────────────────────────────────────────
X = df[features].fillna(0)
y = df['reach'].fillna(0)
model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# ── Formatting Helper ────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n):
        return "-"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(int(n))

# ── Summary Metrics ──────────────────────────────────────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent Totals)")
ve = df[df['performance'].isin(['Viral','Excellent'])]
total_act = ve['reach'].sum()
total_pr  = ve['predicted_reach'].sum()
dev_pct    = (abs(total_act - total_pr) / total_pr * 100) if total_pr else 0
c1, c2, c3 = st.columns(3)
c1.metric("Total Actual Reach",    fmt(total_act))
c2.metric("Total Predicted Reach", fmt(total_pr))
c3.metric("Deviation %",           f"{dev_pct:.2f}%")

# ── Viral & Excellent Reels Table ─────────────────────────────────────────────
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    table = ve[['post_date','caption'] + features + ['reach','predicted_reach','performance']].copy()
    table['reach']           = table['reach'].apply(fmt)
    table['predicted_reach'] = table['predicted_reach'].apply(fmt)
    table = table.rename(columns={
        'post_date':'Date','caption':'Caption',
        'shares':'Shares','saved':'Saves','comments':'Comments','likes':'Likes',
        'reach':'Reach','predicted_reach':'Predicted Reach','performance':'Performance'
    })
    st.dataframe(
        table.style.set_properties(subset=['Caption'], **{'white-space':'pre-wrap'}),
        use_container_width=True
    )
else:
    st.write("No Viral/Excellent reels in this range.")

# ── Performance Distribution ──────────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
perf_counts = df['performance'].value_counts().reindex(cats, fill_value=0)
st.bar_chart(perf_counts)

# ── Reach Trend Over Time ─────────────────────────────────────────────────────
st.subheader("📈 Reach Trend Over Time")
if 'post_date_dt' in df.columns and not df.empty:
    ts = df.set_index('post_date_dt')[['reach','predicted_reach']].rename(
        columns={'reach':'Actual Reach','predicted_reach':'Predicted Reach'}
    )
    st.line_chart(ts)
else:
    st.write("No date-indexed data to plot trend.")

# ── Engagement Correlation by Category ────────────────────────────────────────
st.subheader("🔗 Engagement Correlation by Category")
chart_data = (
    corr_by_cat
    .reset_index()
    .melt('index', var_name='Category', value_name='Correlation')
    .rename(columns={'index':'Engagement'})
)
chart = (
    alt.Chart(chart_data)
       .mark_bar()
       .encode(
           x=alt.X('Engagement:N', title='Engagement Metric'),
           xOffset='Category:N',
           y='Correlation:Q',
           color='Category:N'
       )
       .properties(width='container', height=300)
)
st.altair_chart(chart, use_container_width=True)

# ── Content Intelligence ───────────────────────────────────────────────────────
st.subheader("🧠 Content Intelligence")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general",{}).get("OPENAI_API_KEY")
if api_key and 'caption' in df_full.columns:
    client = openai.OpenAI(api_key=api_key)
    sample = df_full['caption'].dropna().astype(str).sample(min(5, len(df_full))).tolist()
    prompt = (
        "You are an Instagram strategist. Based on these 5 reel captions:\n\n"
        + "\n".join(f"- {t}" for t in sample)
        + "\n\nGive:\n"
          "1. Common content patterns or themes\n"
          "2. Predicted tone or emotion\n"
          "3. One idea to improve virality\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}]
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
3. **Leverage trending audio** for extra boost.  
4. **Post Wed/Thu evenings (6–9 PM IST)** for peak engagement.  
5. **Collaborate** – co-tags double your viral odds.  
6. **Optimize captions** with targeted hashtags + emojis.
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
    st.info(f"🔎 Actual Reach: {fmt(ar)}")
    st.info(f"🎯 Category: **{perf}**")

    st.markdown("**📈 Engagement Correlations by Category (full data):**")
    st.table(corr_by_cat)

    promotion = {'Poor':'Average','Average':'Good','Good':'Excellent','Excellent':'Viral'}
    if perf in promotion:
        tgt  = promotion[perf]
        rcat = corr_by_cat[tgt]
        if rcat.notna().any():
            best, val = rcat.idxmax(), rcat.max()
            st.markdown(f"🔜 To move **{perf}** → **{tgt}**, focus on **{best.capitalize()}** (r={val:.2f}).")
        else:
            st.write("🔍 Not enough data to recommend next steps.")
    else:
        st.markdown("✅ You’re already in **Viral** territory—awesome work!")
