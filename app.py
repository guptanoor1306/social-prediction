import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import altair as alt
import openai

# ── 1) PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Instagram Reels Performance Dashboard", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# ── 2) LOAD & CLEAN FULL DATA ─────────────────────────────────────────────────
df_full = pd.read_csv("posts_zero1byzerodha.csv")
df_full.columns = df_full.columns.str.strip().str.lower()
if 'type' in df_full.columns:
    df_full = df_full[df_full['type'].str.lower() == 'reel']

# Clean numeric columns - no regex replacement needed as data is already numeric
for col in ['reach','shares','saved','comments','likes']:
    if col in df_full.columns:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0)

# ── 3) TRAIN MODEL ON FULL DATA ──────────────────────────────────────────────
features = [c for c in ['shares','saved','comments','likes'] if c in df_full.columns]

# Full-data regression for categorization
X_full = df_full[features].fillna(0)
y_full = df_full['reach'].fillna(0)
full_model = LinearRegression().fit(X_full, y_full)
df_full['pred_full'] = full_model.predict(X_full)

def categorize_ratio(actual, pred):
    if pred <= 0:
        return 'Poor'
    r = actual / pred
    if r > 2.0:
        return 'Viral'
    elif r > 1.5:
        return 'Excellent'
    elif r > 1.0:
        return 'Good'
    elif r > 0.5:
        return 'Average'
    else:
        return 'Poor'

df_full['performance_full'] = df_full.apply(
    lambda r: categorize_ratio(r['reach'], r['pred_full']), axis=1
)

# ── 4) FIXED: CORRELATIONS BY CATEGORY ON FULL DATA ──────────────────────────
cats = ['Viral','Excellent','Good','Average','Poor']
corr_by_cat = pd.DataFrame(index=features, columns=cats)

# Remove outliers for better correlation calculation
df_corr = df_full.copy()
for f in features + ['reach']:
    if f in df_corr.columns:
        lo, hi = df_corr[f].quantile([0.05, 0.95])  # Less aggressive outlier removal
        df_corr = df_corr[df_corr[f].between(lo, hi)]

# Calculate correlations for each category
for cat in cats:
    sub = df_corr[df_corr['performance_full'] == cat]
    if len(sub) >= 5:  # require at least 5 posts for meaningful correlation
        corr_matrix = sub[features + ['reach']].corr()
        if 'reach' in corr_matrix.index:
            corr_by_cat[cat] = corr_matrix.loc['reach', features]
    else:
        corr_by_cat[cat] = np.nan

# Display category counts for debugging
st.sidebar.write("Category Distribution (Full Data):")
cat_counts = df_full['performance_full'].value_counts()
for cat in cats:
    count = cat_counts.get(cat, 0)
    st.sidebar.write(f"{cat}: {count}")

# ── 5) COPY FOR FILTERING & APPLY DATE FILTER ────────────────────────────────
df = df_full.copy()
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

# ── 6) RETRAIN ON FILTERED DATA ───────────────────────────────────────────────
X = df[features].fillna(0)
y = df['reach'].fillna(0)
model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# ── 7) RATIO-BASED CATEGORIZATION ON FILTERED DATA ────────────────────────────
df['performance'] = df.apply(
    lambda r: categorize_ratio(r['reach'], r['predicted_reach']), axis=1
)

# ── 8) FORMAT HELPER ──────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n) or n == 0:
        return "-"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(int(n))

# ── 9) FIXED: SUMMARY INSIGHTS: ALL POSTS ────────────────────────────────────
st.subheader("📈 Summary Insights (All Categories)")
act_all = df['reach'].sum()
pred_all = df['predicted_reach'].sum()

# Calculate proper error metrics
mae = np.mean(np.abs(df['reach'] - df['predicted_reach']))
mape = np.mean(np.abs((df['reach'] - df['predicted_reach']) / df['reach'].replace(0, np.nan))) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Actual Reach", fmt(act_all))
c2.metric("Total Predicted Reach", fmt(pred_all))
c3.metric("Mean Absolute Error", fmt(mae))
c4.metric("Mean % Error", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")

# ── 10) FIXED: SUMMARY INSIGHTS: VIRAL & EXCELLENT ──────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent)")
ve = df[df['performance'].isin(['Viral','Excellent'])]
if not ve.empty:
    act_ve = ve['reach'].sum()
    pred_ve = ve['predicted_reach'].sum()
    mae_ve = np.mean(np.abs(ve['reach'] - ve['predicted_reach']))
    mape_ve = np.mean(np.abs((ve['reach'] - ve['predicted_reach']) / ve['reach'].replace(0, np.nan))) * 100
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Actual Reach", fmt(act_ve))
    c6.metric("Total Predicted Reach", fmt(pred_ve))
    c7.metric("Mean Absolute Error", fmt(mae_ve))
    c8.metric("Mean % Error", f"{mape_ve:.2f}%" if not np.isnan(mape_ve) else "N/A")
else:
    st.write("No Viral/Excellent reels in this date range.")

# ── 11) VIRAL & EXCELLENT REELS TABLE ─────────────────────────────────────────
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    table_cols = ['post_date'] if 'post_date' in ve.columns else []
    table_cols += ['caption'] + features + ['reach','predicted_reach','performance']
    
    table = ve[table_cols].copy()
    table['reach'] = table['reach'].apply(fmt)
    table['predicted_reach'] = table['predicted_reach'].apply(fmt)
    
    rename_dict = {
        'caption':'Caption',
        'shares':'Shares','saved':'Saves','comments':'Comments','likes':'Likes',
        'reach':'Reach','predicted_reach':'Predicted Reach','performance':'Performance'
    }
    if 'post_date' in table.columns:
        rename_dict['post_date'] = 'Date'
    
    table = table.rename(columns=rename_dict)
    st.dataframe(
        table.style.set_properties(subset=['Caption'], **{'white-space':'pre-wrap'}),
        use_container_width=True
    )
else:
    st.write("No Viral/Excellent reels in this date range.")

# ── 12) PERFORMANCE DISTRIBUTION ───────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
dist = df['performance'].value_counts().reindex(cats, fill_value=0)
st.bar_chart(dist)

# ── 13) REACH TREND OVER TIME ─────────────────────────────────────────────────
st.subheader("📈 Reach Trend Over Time")
if 'post_date_dt' in df.columns and not df.empty:
    ts = df.set_index('post_date_dt')[['reach','predicted_reach']].rename(
        columns={'reach':'Actual Reach','predicted_reach':'Predicted Reach'}
    )
    st.line_chart(ts)
else:
    st.write("No date-indexed data to plot trend.")

# ── 14) FIXED: ENGAGEMENT CORRELATIONS (FULL DATA) ───────────────────────────
st.subheader("🔗 Engagement Correlations by Category (Full Dataset)")

# Display correlation table for reference
st.write("**Correlation Matrix by Performance Category:**")
corr_display = corr_by_cat.round(3)
st.dataframe(corr_display)

# Create visualization only for categories with data
corr_viz = corr_by_cat.dropna(axis=1, how='all')  # Remove columns with all NaN
if not corr_viz.empty:
    corr_src = (
        corr_viz
         .reset_index()
         .melt('index', var_name='Category', value_name='Correlation')
         .rename(columns={'index':'Engagement'})
         .dropna(subset=['Correlation'])  # Remove NaN values
    )
    
    if not corr_src.empty:
        chart = (
            alt.Chart(corr_src)
               .mark_bar()
               .encode(
                   x=alt.X('Engagement:N', title='Engagement Metric'),
                   xOffset='Category:N',
                   y=alt.Y('Correlation:Q', scale=alt.Scale(domain=[-1, 1])),
                   color=alt.Color('Category:N', scale=alt.Scale(scheme='category10')),
                   tooltip=['Engagement', 'Category', 'Correlation']
               )
               .properties(width='container', height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Not enough data to display correlations chart.")
else:
    st.warning("No correlation data available for visualization.")

# ── 15) CONTENT INTELLIGENCE ──────────────────────────────────────────────────
st.subheader("🧠 Content Intelligence")
api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("general",{}).get("OPENAI_API_KEY")
if api_key and 'caption' in df_full.columns:
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
            model="gpt-4",
            messages=[{"role":"user","content":prompt}]
        )
        st.markdown(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"NLP error: {e}")
else:
    st.info("Add OPENAI_API_KEY to Streamlit secrets to enable Content Intelligence.")

# ── 16) STRATEGIC TAKEAWAYS ───────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
1. **Double-down on saveable "how-to" tips** – high saves drive long-term reach.  
2. **Drive share prompts** ("tag a friend") for algorithmic uplift.  
3. **Leverage trending audio** for extra boost.  
4. **Post Wed/Thu evenings (6–9 PM IST)** for peak engagement.  
5. **Collaborate** – co-tags double your viral odds.  
6. **Optimize captions** with targeted hashtags + emojis.
""")

# ── 17) DOWNLOAD & PREDICT & DIAGNOSE ─────────────────────────────────────────
st.subheader("⬇️ Download Full Data")
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode('utf-8'),
    "reels_with_predictions.csv",
    "text/csv"
)

st.subheader("🎯 Predict & Diagnose for a New Reel")
with st.form("diagnose_form"):
    s  = st.number_input("Shares",      0, value=100)
    sv = st.number_input("Saves",       0, value=50) 
    c  = st.number_input("Comments",    0, value=25)
    l  = st.number_input("Likes",       0, value=1000)
    ar = st.number_input("Actual Reach (0 if predicting)", 0, value=0)
    go = st.form_submit_button("Diagnose")

if go:
    # Use the model trained on filtered data for consistency
    inp = np.array([[s, sv, c, l]])  # Proper 2D array format
    pred = model.predict(inp)[0]
    
    # If actual reach is provided, categorize performance
    if ar > 0:
        perf = categorize_ratio(ar, pred)
        st.success(f"📢 Predicted Reach: {fmt(pred)}")
        st.info(f"🔎 Actual Reach: {fmt(ar)}")
        st.info(f"🎯 Performance Category: **{perf}**")
        
        # Show performance ratio
        ratio = ar / pred if pred > 0 else 0
        st.info(f"📊 Performance Ratio: {ratio:.2f}x predicted")
    else:
        st.success(f"📢 Predicted Reach: {fmt(pred)}")
        st.info("🔎 Provide actual reach to get performance analysis")

    # Show model coefficients for insights
    st.markdown("**🔍 Model Insights:**")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Impact': ['High' if abs(c) > np.std(model.coef_) else 'Medium' if abs(c) > np.std(model.coef_)/2 else 'Low' 
                  for c in model.coef_]
    }).sort_values('Coefficient', key=abs, ascending=False)
    st.dataframe(coef_df)

    # Recommendations based on correlations
    st.markdown("**📈 Correlations by Performance Category:**")
    st.dataframe(corr_by_cat.round(3))
    
    if ar > 0:  # Only show recommendations if we have actual performance
        promo = {'Poor':'Average','Average':'Good','Good':'Excellent','Excellent':'Viral'}
        if perf in promo:
            tgt = promo[perf]
            if tgt in corr_by_cat.columns:
                rcat = corr_by_cat[tgt].dropna()
                if not rcat.empty:
                    best_metric = rcat.idxmax()
                    best_corr = rcat.max()
                    st.markdown(
                        f"🔜 To move from **{perf}** → **{tgt}**, focus on "
                        f"**{best_metric.capitalize()}** (correlation: {best_corr:.3f})"
                    )
                else:
                    st.write("🔍 Not enough data to recommend next steps.")
        else:
            st.markdown("✅ You're already in **Viral** territory—awesome work!")
