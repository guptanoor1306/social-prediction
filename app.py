import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import altair as alt
import openai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Instagram Reels Performance Dashboard", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# ── Load & preprocess the full dataset ────────────────────────────────────────
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

# ── Categorize performance for the full set ───────────────────────────────────
def categorize(actual, predicted):
    if predicted <= 0:
        return 'Poor'
    ratio = actual / predicted
    if ratio > 2.0:     return 'Viral'
    if ratio > 1.5:     return 'Excellent'
    if ratio > 1.0:     return 'Good'
    if ratio > 0.5:     return 'Average'
    return 'Poor'

# We need predicted reach to categorize, so train a quick model on the full set:
features = [f for f in ['shares','saved','comments','likes'] if f in df_full.columns]
X_full = df_full[features].fillna(0)
y_full = df_full['reach'].fillna(0)
full_model = LinearRegression().fit(X_full, y_full)
df_full['predicted_reach'] = full_model.predict(X_full)
df_full['performance'] = df_full.apply(
    lambda r: categorize(r['reach'], r['predicted_reach']), axis=1
)

# ── Outlier removal for correlation analysis ─────────────────────────────────
df_no = df_full.copy()
for f in features + ['reach']:
    low, high = df_no[f].quantile([0.01, 0.99])
    df_no = df_no[df_no[f].between(low, high)]

# ── Compute one‐time correlations overall & by category ──────────────────────
corr_overall = df_no[features + ['reach']].corr()['reach'].drop('reach').sort_values(ascending=False)

cats = ['Viral','Excellent','Good','Average','Poor']
corr_by_cat = pd.DataFrame(index=features, columns=cats)
for cat in cats:
    subset = df_no[df_no['performance'] == cat]
    if len(subset) >= 3:   # require at least 3 samples for stable r
        corr_by_cat[cat] = subset[features + ['reach']].corr().loc['reach', features]
    else:
        corr_by_cat[cat] = np.nan

# ── Now make a working copy for filtering & modeling ─────────────────────────
df = df_full.copy()

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

# ── Retrain model on the filtered set ────────────────────────────────────────
X = df[features].fillna(0)
y = df['reach'].fillna(0)
model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# ── Formatter ─────────────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n): return "-"
    if n >= 1e6:   return f"{n/1e6:.2f}M"
    if n >= 1e3:   return f"{n/1e3:.1f}K"
    return str(int(n))

# ── Summary Insights ──────────────────────────────────────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent Totals)")
ve = df[df['performance'].isin(['Viral','Excellent'])]
ta = ve['reach'].sum()
tp = ve['predicted_reach'].sum()
dev = (abs(ta - tp) / tp * 100) if tp else 0
c1, c2, c3 = st.columns(3)
c1.metric("Total Actual Reach",    fmt(ta))
c2.metric("Total Predicted Reach", fmt(tp))
c3.metric("Deviation %",           f"{dev:.2f}%")

# ── Viral & Excellent Table ───────────────────────────────────────────────────
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    out = ve[['post_date','caption'] + features + ['reach','predicted_reach','performance']].copy()
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
    st.write("No Viral/Excellent reels in this date range.")

# ── Performance Distribution ──────────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
dist = df['performance'].value_counts().reindex(cats, fill_value=0)
st.bar_chart(dist)

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

# ── Engagement Correlation by Category (FULL‐DATA) ─────────────────────────────
st.subheader("🔗 Engagement Correlation by Category")
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
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general",{}).get("OPENAI_API_KEY")
if api_key and 'caption' in df.columns:
    client = openai.OpenAI(api_key=api_key)
    sample = df_full['caption'].dropna().astype(str).sample(min(5,len(df_full))).tolist()
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
            model="gpt-4", messages=[{"role":"user","content":prompt}]
        )
        st.markdown(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"NLP error: {e}")
else:
    st.info("Add OPENAI_API_KEY to secrets to enable Content Intelligence.")

# ── Strategic Takeaways ───────────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
1. **Double-down on saveable “how-to” tips** – high saves drive long-term reach.  
2. **Drive share prompts** (“tag a friend”) for algorithmic uplift.  
3. **Leverage trending audio** for extra boost.  
4. **Post Wed/Thu evenings (6–9 PM IST)** for peak engagement.  
5. **Collaborate** – co-tags double viral odds.  
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
    st.info(f"🔎 Actual Reach: {fmt(ar)}")
    st.info(f"🎯 Category: **{perf}**")

    st.markdown("**📈 Engagement Correlations by Category (full data):**")
    st.table(corr_by_cat)

    promotion = {'Poor':'Average','Average':'Good','Good':'Excellent','Excellent':'Viral'}
    if perf in promotion:
        target = promotion[perf]
        rcat   = corr_by_cat[target]
        if rcat.notna().any():
            best_m, best_v = rcat.idxmax(), rcat.max()
            st.markdown(
                f"🔜 To move **{perf}** → **{target}**, focus on "
                f"**{best_m.capitalize()}** (r={best_v:.2f})."
            )
        else:
            st.write("🔍 Not enough data to recommend next steps.")
    else:
        st.markdown("✅ You’re already in **Viral** territory—awesome work!")
