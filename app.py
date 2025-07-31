import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt
import openai

# ── 1) PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Instagram Reels Performance Dashboard", layout="wide")
st.title("📊 Instagram Reels Performance Dashboard")

# ── 2) LOAD & CLEAN DATA ─────────────────────────────────────────────────────
df_full = pd.read_csv("posts_zero1byzerodha.csv")
df_full.columns = df_full.columns.str.strip().str.lower()

# Filter to Reels only
if 'type' in df_full.columns:
    df_full = df_full[df_full['type'].str.lower() == 'reel']

# Clean numeric columns
for col in ['reach','shares','saved','comments','likes']:
    if col in df_full.columns:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0)

# ── 3) TRAIN MODEL ON FULL DATA FOR CATEGORIZATION ──────────────────────────
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
    if r > 2.0:    return 'Viral'
    if r > 1.5:    return 'Excellent'
    if r > 1.0:    return 'Good'
    if r > 0.5:    return 'Average'
    return 'Poor'

df_full['performance_full'] = df_full.apply(
    lambda r: categorize_ratio(r['reach'], r['pred_full']), axis=1
)

# ── 4) CALCULATE CORRELATIONS BY CATEGORY ────────────────────────────────────
cats = ['Viral','Excellent','Good','Average','Poor']
corr_by_cat = pd.DataFrame(index=features, columns=cats)

# Calculate correlations for each category
for cat in cats:
    sub = df_full[df_full['performance_full'] == cat]
    if len(sub) >= 3:
        try:
            corr_matrix = sub[features + ['reach']].corr()
            if 'reach' in corr_matrix.index and not corr_matrix.loc['reach', features].isna().all():
                corr_by_cat[cat] = corr_matrix.loc['reach', features]
        except:
            corr_by_cat[cat] = np.nan
    else:
        corr_by_cat[cat] = np.nan

# ── 5) DATE FILTER ───────────────────────────────────────────────────────────
df = df_full.copy()
date_col = next((c for c in df.columns if 'date' in c), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df['post_date'] = df[date_col].dt.date
    df['post_date_dt'] = pd.to_datetime(df['post_date'])
    
    st.sidebar.subheader("📅 Filter by Post Date")
    dates = st.sidebar.date_input(
        "Select date range",
        value=[df['post_date'].min(), df['post_date'].max()],
        key="date_range"
    )
    
    # Handle different date input formats
    if isinstance(dates, (list, tuple)):
        if len(dates) == 2:
            start_date, end_date = dates[0], dates[1]
        elif len(dates) == 1:
            start_date = end_date = dates[0]
        else:
            start_date, end_date = df['post_date'].min(), df['post_date'].max()
    else:
        start_date = end_date = dates
    
    # Filter data by date range
    df = df[df['post_date'].between(start_date, end_date)]
else:
    st.sidebar.info("No date column found for filtering.")

# ── 6) RETRAIN ON FILTERED DATA ──────────────────────────────────────────────
X = df[features].fillna(0)
y = df['reach'].fillna(0)
model = LinearRegression().fit(X, y)
df['predicted_reach'] = model.predict(X)

# Compute accuracy metrics
r2 = r2_score(y, df['predicted_reach'])
rmse = np.sqrt(mean_squared_error(y, df['predicted_reach']))

# ── 7) CATEGORIZE PERFORMANCE ON FILTERED DATA ──────────────────────────────
df['performance'] = df.apply(
    lambda r: categorize_ratio(r['reach'], r['predicted_reach']), axis=1
)

# ── 8) FORMAT HELPER ─────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n) or n == 0:
        return "-"
    if n >= 1e6:   return f"{n/1e6:.2f}M"
    if n >= 1e3:   return f"{n/1e3:.1f}K"
    return str(int(n))

# ── 9) SUMMARY INSIGHTS: ALL POSTS ───────────────────────────────────────────
st.subheader("📈 Summary Insights (All Categories)")

total_posts = len(df)
total_actual = df['reach'].sum()
total_predicted = df['predicted_reach'].sum()
mae = np.mean(np.abs(df['reach'] - df['predicted_reach']))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Posts", str(total_posts))
c2.metric("Total Actual Reach", fmt(total_actual))
c3.metric("Mean Abs Error", fmt(mae))
c4.metric("Model R² Score", f"{r2:.3f}")

# Explain the "Poor" category
poor_count = len(df[df['performance'] == 'Poor'])
poor_pct = (poor_count / total_posts) * 100 if total_posts > 0 else 0

if poor_pct > 30:
    st.warning(f"⚠️ **{poor_count} posts ({poor_pct:.1f}%) are in 'Poor' category** - these posts underperformed compared to their predicted reach based on engagement metrics. This suggests external factors (timing, trends, hashtags) significantly impact performance.")

# ── 10) SUMMARY INSIGHTS: VIRAL & EXCELLENT ─────────────────────────────────
st.subheader("📈 Summary Insights (Viral & Excellent)")
ve = df[df['performance'].isin(['Viral','Excellent'])]

if not ve.empty:
    ve_count = len(ve)
    ve_pct = (ve_count / len(df)) * 100
    ve_total_actual = ve['reach'].sum()
    ve_total_predicted = ve['predicted_reach'].sum()
    avg_virality = (ve['reach'] / np.where(ve['predicted_reach'] > 0, ve['predicted_reach'], 1)).mean()
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("V&E Posts", f"{ve_count} ({ve_pct:.1f}%)")
    c6.metric("Total Actual Reach", fmt(ve_total_actual))
    c7.metric("Total Predicted Reach", fmt(ve_total_predicted))
    c8.metric("Avg Virality Ratio", f"{avg_virality:.2f}x")
    
    with st.expander("🧠 Understanding High Performance"):
        st.markdown("- **Viral & Excellent** posts significantly outperform predictions")
        st.markdown("- These posts likely benefited from trending topics, optimal timing, or algorithmic boosts")
        st.markdown("- Focus on replicating patterns from these high-performers")
else:
    st.write("No Viral/Excellent reels in this date range.")

# ── 11) VIRAL & EXCELLENT REELS TABLE ────────────────────────────────────────
st.subheader("🔥 Viral & Excellent Reels")
if not ve.empty:
    table_cols = (['post_date'] if 'post_date' in ve.columns else []) + ['caption'] + features + ['reach','predicted_reach','performance']
    table = ve[[c for c in table_cols if c in ve.columns]].copy()
    
    # Format numbers
    table['reach'] = table['reach'].apply(fmt)
    table['predicted_reach'] = table['predicted_reach'].apply(fmt)
    
    # Rename columns
    rename_dict = {
        'caption':'Caption', 'shares':'Shares','saved':'Saves',
        'comments':'Comments','likes':'Likes', 'reach':'Reach',
        'predicted_reach':'Predicted Reach','performance':'Performance'
    }
    if 'post_date' in table.columns:
        rename_dict['post_date'] = 'Date'
    
    table = table.rename(columns=rename_dict)
    st.dataframe(
        table.style.set_properties(subset=['Caption'], **{'white-space':'pre-wrap'}),
        use_container_width=True
    )
else:
    st.write("No Viral or Excellent reels in this range.")

# ── 12) PERFORMANCE DISTRIBUTION ─────────────────────────────────────────────
st.subheader("📊 Performance Distribution")
perf_counts = df['performance'].value_counts().reindex(cats, fill_value=0)
st.bar_chart(perf_counts)

# ── 13) REACH TREND OVER TIME ────────────────────────────────────────────────
st.subheader("📈 Reach Trend Over Time")
if 'post_date_dt' in df.columns and not df.empty:
    weekly = (
        df.set_index('post_date_dt')
          .resample('W')[['reach','predicted_reach']]
          .mean()
          .rename(columns={'reach':'Actual Reach','predicted_reach':'Predicted Reach'})
    )
    st.line_chart(weekly)
else:
    st.write("No date data available for trend analysis.")

# ── 14) ENGAGEMENT CORRELATIONS BY CATEGORY ──────────────────────────────────
st.subheader("🔗 Engagement Correlations by Category (Full Dataset)")

# Display correlation matrix
st.write("**Correlation Matrix by Performance Category:**")
corr_display = corr_by_cat.round(3)
st.dataframe(corr_display)

# Create the original format chart - engagement metrics on X-axis, categories as different colors
corr_viz = corr_by_cat.dropna(axis=1, how='all')
if not corr_viz.empty:
    # Convert to the format the original chart expects
    corr_src = (
        corr_viz
         .reset_index()
         .melt('index', var_name='Category', value_name='Correlation')
         .rename(columns={'index':'Engagement'})
         .dropna(subset=['Correlation'])
    )
    
    if not corr_src.empty:
        chart = (
            alt.Chart(corr_src)
               .mark_bar()
               .encode(
                   x=alt.X('Engagement:N', title='Engagement Metric'),
                   xOffset='Category:N',
                   y=alt.Y('Correlation:Q', scale=alt.Scale(domain=[-1, 1])),
                   color='Category:N',
                   tooltip=['Engagement', 'Category', alt.Tooltip('Correlation:Q', format='.3f')]
               )
               .properties(width='container', height=300)
        )
        st.altair_chart(chart, use_container_width=True)
            
    st.info("**Note:** Negative correlations may indicate controversial content that generates discussion but doesn't always boost reach algorithmically.")

# ── 15) TOP VIRAL CONTENT ────────────────────────────────────────────────────
st.subheader("📈 Top Content by Virality Score")
df['virality_score'] = np.where(df['predicted_reach'] > 0, 
                               df['reach'] / df['predicted_reach'], 0)
df['virality_score'] = df['virality_score'].replace([np.inf, -np.inf], 0)

top5 = df.nlargest(5, 'virality_score')
if 'caption' in top5.columns and not top5.empty:
    t5 = top5[['caption','reach','predicted_reach','virality_score']].copy()
    t5['reach'] = t5['reach'].apply(fmt)
    t5['predicted_reach'] = t5['predicted_reach'].apply(fmt)
    t5['virality_score'] = t5['virality_score'].apply(lambda x: f"{x:.2f}x")
    t5 = t5.rename(columns={
        'caption':'Caption','reach':'Reach',
        'predicted_reach':'Predicted Reach','virality_score':'Virality Score'
    })
    st.dataframe(t5, use_container_width=True)

# ── 16) CONTENT INTELLIGENCE ─────────────────────────────────────────────────
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

# ── 17) STRATEGIC TAKEAWAYS ──────────────────────────────────────────────────
st.subheader("🚀 Strategic Takeaways")
st.markdown("""
1. **Double‑down on saveable "how‑to" tips** – high saves = strong long‑term reach.  
2. **Drive share prompts** ("tag a friend") for algorithmic uplift.  
3. **Leverage trending audio** formats & sounds for extra boost.  
4. **Post mid‑week evenings** (Wed/Thu 6–9 PM IST) for peak visibility.  
5. **Pursue collabs** – tagged reels double viral odds.  
6. **Optimize captions** with 2–3 targeted hashtags + emojis for tone & reach.
""")

# ── 18) DOWNLOAD & PREDICT ───────────────────────────────────────────────────
st.subheader("⬇️ Download Full Data")
st.download_button(
    "Download CSV", 
    df.to_csv(index=False).encode('utf-8'),
    "reels_with_predictions.csv", 
    mime="text/csv"
)

st.subheader("🎯 Predict Reach for a New Reel")
st.markdown("⚠️ Enter engagement values to predict reach:")

with st.form("predict_form"):
    input_values = {}
    # Set more realistic default values to avoid negative predictions
    defaults = {'shares': 50, 'saved': 30, 'comments': 20, 'likes': 500}
    
    for feature in features:
        input_values[feature] = st.number_input(
            feature.capitalize(), 
            min_value=0, 
            value=defaults.get(feature, 10), 
            help=f"Enter number of {feature}"
        )
    
    actual_reach = st.number_input(
        "Actual Reach (optional - for performance analysis)", 
        min_value=0, 
        value=0,
        help="Leave as 0 if you're predicting future performance"
    )
    
    predict_btn = st.form_submit_button("🎯 Predict & Analyze")

if predict_btn:
    # Make prediction using the trained model
    input_df = pd.DataFrame([input_values])
    predicted_reach = model.predict(input_df)[0]
    
    # Fix negative predictions by ensuring minimum realistic reach
    predicted_reach = max(predicted_reach, 1000)  # Minimum 1K reach
    
    st.success(f"📢 **Predicted Reach: {fmt(predicted_reach)}**")
    
    if actual_reach > 0:
        performance_cat = categorize_ratio(actual_reach, predicted_reach)
        virality_ratio = actual_reach / predicted_reach if predicted_reach > 0 else 0
        
        st.info(f"🔎 **Actual Reach: {fmt(actual_reach)}**")
        st.info(f"🎯 **Performance Category: {performance_cat}**")
        st.info(f"📊 **Virality Ratio: {virality_ratio:.2f}x predicted**")
        
        # Simple improvement tip
        if performance_cat in ['Poor', 'Average', 'Good']:
            promo_map = {'Poor':'Average', 'Average':'Good', 'Good':'Excellent'}
            target_cat = promo_map.get(performance_cat)
            if target_cat:
                st.markdown(f"💡 **Tip:** Focus on increasing engagement to reach **{target_cat}** performance level")
