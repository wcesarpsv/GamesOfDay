# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import re

st.set_page_config(page_title="Auto Recommendation Parameter Backtest", layout="wide")
st.title("🎯 Auto Recommendation Parameter Backtest")

# ──────────────────────────────────────────────────────────────────────────────
# 🔒 Internal league filter (NOT shown in UI)
EXCLUDED_LEAGUE_KEYWORDS = ["Cup", "Copa", "Copas", "UEFA", "Friendly", "Super Cup"]
_EXC_PATTERN = re.compile("|".join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS)),
                          flags=re.IGNORECASE) if EXCLUDED_LEAGUE_KEYWORDS else None

# ──────────────────────────────────────────────────────────────────────────────
# Load CSVs (same as your existing backtest)
# ──────────────────────────────────────────────────────────────────────────────
GAMES_FOLDER = "GamesDay"

if not os.path.isdir(GAMES_FOLDER):
    st.error(f"❌ Folder '{GAMES_FOLDER}' not found.")
    st.stop()

all_dfs = []
for file in sorted(os.listdir(GAMES_FOLDER)):
    if file.endswith(".csv"):
        df_path = os.path.join(GAMES_FOLDER, file)
        try:
            df = pd.read_csv(df_path)
        except Exception:
            continue

        required = {"Goals_H_FT","Goals_A_FT","Diff_Power","OU_Total","Odd_H","Odd_D","Odd_A","Date","M_H","M_A"}
        if not required.issubset(df.columns):
            continue

        # 🔧 Garante colunas extras
        for col in ["Diff_HT_P", "M_HT_H", "M_HT_A"]:
            if col not in df.columns:
                df[col] = float("nan")

        if _EXC_PATTERN and "League" in df.columns:
            df = df[~df["League"].astype(str).str.contains(_EXC_PATTERN, na=False)]

        df = df.dropna(subset=["Goals_H_FT","Goals_A_FT"])
        if df.empty:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data with goal columns found.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")

if _EXC_PATTERN and "League" in df_all.columns:
    df_all = df_all[~df_all["League"].astype(str).str.contains(_EXC_PATTERN, na=False)]

df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# 🎛️ PARAMETER CONTROL PANEL
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Auto Recommendation Parameters")

# Reset button
if st.sidebar.button("🔄 Reset All Parameters"):
    for key in list(st.session_state.keys()):
        if key.startswith("param_"):
            del st.session_state[key]
    st.rerun()

# Main parameters
st.sidebar.subheader("📊 Core Parameters")
M_DIFF_MARGIN = st.sidebar.slider("M_DIFF_MARGIN", 0.10, 0.50, 0.30, 0.05, key="param_m_diff_margin")
POWER_MARGIN = st.sidebar.slider("POWER_MARGIN", 5, 20, 10, 1, key="param_power_margin")
DOMINANT_THRESHOLD = st.sidebar.slider("DOMINANT_THRESHOLD", 0.80, 0.95, 0.90, 0.01, key="param_dominant_threshold")

# Condition-specific parameters
st.sidebar.subheader("🎯 Condition Parameters")
DIFF_MID_LO = st.sidebar.slider("DIFF_MID_LO", 0.10, 0.40, 0.20, 0.05, key="param_diff_mid_lo")
DIFF_MID_HI = st.sidebar.slider("DIFF_MID_HI", 0.50, 1.00, 0.80, 0.05, key="param_diff_mid_hi")
DIFF_MID_HI_HIGHVAR = st.sidebar.slider("DIFF_MID_HI_HIGHVAR", 0.50, 1.00, 0.75, 0.05, key="param_diff_mid_hi_highvar")
POWER_GATE = st.sidebar.slider("POWER_GATE", 1, 10, 1, 1, key="param_power_gate")
POWER_GATE_HIGHVAR = st.sidebar.slider("POWER_GATE_HIGHVAR", 1, 15, 5, 1, key="param_power_gate_highvar")

# Draw-specific parameters
st.sidebar.subheader("⚪ Draw Parameters")
ODD_D_MIN = st.sidebar.slider("ODD_D_MIN", 2.0, 4.0, 2.5, 0.1, key="param_odd_d_min")
ODD_D_MAX = st.sidebar.slider("ODD_D_MAX", 4.0, 7.0, 6.0, 0.1, key="param_odd_d_max")
DIFF_POWER_DRAW_MIN = st.sidebar.slider("DIFF_POWER_DRAW_MIN", -15, 0, -10, 1, key="param_diff_power_draw_min")
DIFF_POWER_DRAW_MAX = st.sidebar.slider("DIFF_POWER_DRAW_MAX", 0, 15, 10, 1, key="param_diff_power_draw_max")

# ──────────────────────────────────────────────────────────────────────────────
# 🎯 AUTO RECOMMENDATION ENGINE WITH CUSTOM PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
def classify_leagues_variation(history_df):
    agg = (
        history_df.groupby('League')
        .agg(
            M_H_Min=('M_H','min'), M_H_Max=('M_H','max'),
            M_A_Min=('M_A','min'), M_A_Max=('M_A','max'),
            Hist_Games=('M_H','count')
        ).reset_index()
    )
    agg['Variation_Total'] = (agg['M_H_Max'] - agg['M_H_Min']) + (agg['M_A_Max'] - agg['M_A_Min'])
    def label(v):
        if v > 6.0: return "High Variation"
        if v >= 3.0: return "Medium Variation"
        return "Low Variation"
    agg['League_Classification'] = agg['Variation_Total'].apply(label)
    return agg[['League','League_Classification','Variation_Total','Hist_Games']]

def compute_league_bands(history_df):
    hist = history_df.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']
    diff_q = (
        hist.groupby('League')['M_Diff']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'P20_Diff', 0.8:'P80_Diff'})
            .reset_index()
    )
    home_q = (
        hist.groupby('League')['M_H']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Home_P20', 0.8:'Home_P80'})
            .reset_index()
    )
    away_q = (
        hist.groupby('League')['M_A']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Away_P20', 0.8:'Away_P80'})
            .reset_index()
    )
    out = diff_q.merge(home_q, on='League', how='inner').merge(away_q, on='League', how='inner')
    return out

def dominant_side(row, threshold=DOMINANT_THRESHOLD):
    m_h, m_a = row['M_H'], row['M_A']
    if (m_h >= threshold) and (m_a <= -threshold):
        return "Both extremes (Home↑ & Away↓)"
    if (m_a >= threshold) and (m_h <= -threshold):
        return "Both extremes (Away↑ & Home↓)"
    if m_h >= threshold:
        return "Home strong"
    if m_h <= -threshold:
        return "Home weak"
    if m_a >= threshold:
        return "Away strong"
    if m_a <= -threshold:
        return "Away weak"
    return "Mixed / Neutral"

def auto_recommendation_custom(row):
    # Calculate bands and dominant for this row
    band_home = 'Top 20%' if row['M_H'] >= row.get('Home_P80', 0.8) else ('Bottom 20%' if row['M_H'] <= row.get('Home_P20', 0.2) else 'Balanced')
    band_away = 'Top 20%' if row['M_A'] >= row.get('Away_P80', 0.8) else ('Bottom 20%' if row['M_A'] <= row.get('Away_P20', 0.2) else 'Balanced')
    dominant = dominant_side(row)
    
    diff_m = row.get('M_Diff', 0)
    diff_pow = row.get('Diff_Power', 0)
    league_cls = row.get('League_Classification', 'Medium Variation')
    m_a = row.get('M_A', 0)
    m_h = row.get('M_H', 0)
    odd_d = row.get('Odd_D', 0)

    # 1) Strong edges -> Direct Back
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return '🟢 Back Home'
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return '🟠 Back Away'

    if dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong'] and band_away != 'Top 20%':
        if diff_m is not None and diff_m >= 0.90:
            return '🟢 Back Home'
    if dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong'] and band_home == 'Balanced':
        if diff_m is not None and diff_m <= -0.90:
            return '🟪 X2 (Away/Draw)'

    # 2) Both Balanced (with thresholds)
    if (band_home == 'Balanced') and (band_away == 'Balanced') and (diff_m is not None) and (diff_pow is not None):
        if league_cls == 'High Variation':
            if (diff_m >= 0.45 and diff_m < DIFF_MID_HI_HIGHVAR and diff_pow >= POWER_GATE_HIGHVAR):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -0.45 and diff_m > -DIFF_MID_HI_HIGHVAR and diff_pow <= -POWER_GATE_HIGHVAR):
                return '🟪 X2 (Away/Draw)'
        else:
            if (diff_m >= DIFF_MID_LO and diff_m < DIFF_MID_HI and diff_pow >= POWER_GATE):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -DIFF_MID_LO and diff_m > -DIFF_MID_HI and diff_pow <= -POWER_GATE):
                return '🟪 X2 (Away/Draw)'

    # 3) Balanced vs Bottom20%
    if (band_home == 'Balanced') and (band_away == 'Bottom 20%'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Balanced') and (band_home == 'Bottom 20%'):
        return '🟪 X2 (Away/Draw)'

    # 4) Top20% vs Balanced
    if (band_home == 'Top 20%') and (band_away == 'Balanced'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Top 20%') and (band_home == 'Balanced'):
        return '🟪 X2 (Away/Draw)'

    # 5) Filtro Draw (com parâmetros customizados)
    if (odd_d is not None and ODD_D_MIN <= odd_d <= ODD_D_MAX) and (diff_pow is not None and DIFF_POWER_DRAW_MIN <= diff_pow <= DIFF_POWER_DRAW_MAX):
        if (m_h is not None and 0 <= m_h <= 1) or (m_a is not None and 0 <= m_a <= 0.5):
            return '⚪ Back Draw'

    # 6) Fallback
    return '❌ Avoid'

# ──────────────────────────────────────────────────────────────────────────────
# 📊 APPLY AUTO RECOMMENDATION AND CALCULATE PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
st.header("📊 Backtest Results with Custom Parameters")

# Prepare data with bands and classifications
st.info("🔄 Calculating league bands and classifications...")
league_class = classify_leagues_variation(df_all)
league_bands = compute_league_bands(df_all)

df_backtest = df_all.merge(league_class, on='League', how='left')
df_backtest = df_backtest.merge(league_bands, on='League', how='left')

# Calculate M_Diff
df_backtest['M_Diff'] = df_backtest['M_H'] - df_backtest['M_A']

# Apply Auto Recommendation
st.info("🎯 Applying Auto Recommendation with custom parameters...")
df_backtest['Auto_Recommendation'] = df_backtest.apply(auto_recommendation_custom, axis=1)

# Calculate profit for each recommendation type
def calculate_profit_auto(row):
    rec = row['Auto_Recommendation']
    h_goals = row['Goals_H_FT']
    a_goals = row['Goals_A_FT']
    
    if rec == '🟢 Back Home':
        return (row['Odd_H'] - 1) if h_goals > a_goals else -1
    elif rec == '🟠 Back Away':
        return (row['Odd_A'] - 1) if a_goals > h_goals else -1
    elif rec == '⚪ Back Draw':
        return (row['Odd_D'] - 1) if h_goals == a_goals else -1
    elif rec == '🟦 1X (Home/Draw)':
        return (row.get('Odd_1X', 1.3) - 1) if h_goals >= a_goals else -1
    elif rec == '🟪 X2 (Away/Draw)':
        return (row.get('Odd_X2', 1.3) - 1) if a_goals >= h_goals else -1
    else:
        return 0  # Avoid bets

# Calculate Odd_1X and Odd_X2 if not present
if 'Odd_1X' not in df_backtest.columns:
    probs = pd.DataFrame()
    probs['p_H'] = 1 / df_backtest['Odd_H']
    probs['p_D'] = 1 / df_backtest['Odd_D']
    probs['p_A'] = 1 / df_backtest['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)
    df_backtest['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
    df_backtest['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])

df_backtest['Profit'] = df_backtest.apply(calculate_profit_auto, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# 📈 PERFORMANCE DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🎯 Performance by Recommendation Type")

# Summary by recommendation type
rec_performance = df_backtest[df_backtest['Auto_Recommendation'] != '❌ Avoid'].groupby('Auto_Recommendation').agg({
    'Profit': ['count', 'sum', 'mean'],
    'Odd_H': 'mean',
    'Odd_D': 'mean', 
    'Odd_A': 'mean',
    'Odd_1X': 'mean',
    'Odd_X2': 'mean'
}).round(3)

rec_performance.columns = ['Bets', 'Total_Profit', 'Avg_Profit', 'Avg_Odd_H', 'Avg_Odd_D', 'Avg_Odd_A', 'Avg_Odd_1X', 'Avg_Odd_X2']
rec_performance['Win_Rate'] = (df_backtest[df_backtest['Auto_Recommendation'] != '❌ Avoid']
                              .groupby('Auto_Recommendation')
                              .apply(lambda x: (x['Profit'] > 0).sum() / len(x) * 100)).round(1)
rec_performance['ROI'] = (rec_performance['Total_Profit'] / rec_performance['Bets'] * 100).round(1)

# Display performance table
st.dataframe(rec_performance.style.format({
    'Total_Profit': '{:.2f}',
    'Avg_Profit': '{:.3f}',
    'Win_Rate': '{:.1f}%',
    'ROI': '{:.1f}%',
    'Avg_Odd_H': '{:.2f}',
    'Avg_Odd_D': '{:.2f}',
    'Avg_Odd_A': '{:.2f}',
    'Avg_Odd_1X': '{:.2f}',
    'Avg_Odd_X2': '{:.2f}'
}), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 📊 CUMULATIVE PROFIT CHARTS (BY BET NUMBER)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Cumulative Profit by Bet Number")

# Calculate cumulative profit by bet sequence for each recommendation type
df_backtest_sorted = df_backtest.sort_values('Date')
profit_data = []

for rec_type in rec_performance.index:
    df_rec = df_backtest_sorted[df_backtest_sorted['Auto_Recommendation'] == rec_type].copy()
    if len(df_rec) > 0:
        df_rec = df_rec.reset_index(drop=True)
        df_rec['Bet_Number'] = range(1, len(df_rec) + 1)
        df_rec['Cumulative_Profit'] = df_rec['Profit'].cumsum()
        df_rec['Recommendation_Type'] = rec_type
        profit_data.append(df_rec[['Bet_Number', 'Cumulative_Profit', 'Recommendation_Type', 'Date']])

if profit_data:
    profit_df = pd.concat(profit_data)
    
    fig = px.line(profit_df, x='Bet_Number', y='Cumulative_Profit', color='Recommendation_Type',
                  title='Cumulative Profit by Bet Number (Smooth Progression)',
                  labels={'Cumulative_Profit': 'Cumulative Profit (Units)', 'Bet_Number': 'Bet Sequence Number'},
                  hover_data=['Date'])
    
    # Add a zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 📊 ADDITIONAL: PROFIT DISTRIBUTION BY BET
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Profit Distribution by Bet Type")

# Create a box plot showing profit distribution for each recommendation type
fig_box = px.box(df_backtest[df_backtest['Auto_Recommendation'] != '❌ Avoid'], 
                 x='Auto_Recommendation', y='Profit',
                 title='Profit Distribution per Bet Type',
                 labels={'Profit': 'Profit per Bet (Units)', 'Auto_Recommendation': 'Recommendation Type'})
fig_box.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
st.plotly_chart(fig_box, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 🔍 BEST COMBINATIONS ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🏆 Best Performing Combinations")

# Analyze conditions that lead to successful bets
successful_bets = df_backtest[df_backtest['Profit'] > 0]

if len(successful_bets) > 0:
    # Top conditions for each recommendation type
    for rec_type in ['🟢 Back Home', '🟠 Back Away', '🟦 1X (Home/Draw)', '🟪 X2 (Away/Draw)', '⚪ Back Draw']:
        rec_bets = successful_bets[successful_bets['Auto_Recommendation'] == rec_type]
        if len(rec_bets) > 0:
            st.write(f"**{rec_type} - Successful Conditions:**")
            
            # Show average stats for successful bets
            avg_stats = rec_bets[['M_H', 'M_A', 'M_Diff', 'Diff_Power', 'Odd_H', 'Odd_D', 'Odd_A']].mean().round(3)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg M_H", f"{avg_stats['M_H']:.3f}")
            col2.metric("Avg M_A", f"{avg_stats['M_A']:.3f}")
            col3.metric("Avg M_Diff", f"{avg_stats['M_Diff']:.3f}")
            col4.metric("Avg Diff_Power", f"{avg_stats['Diff_Power']:.1f}")
            
            # Show most common bands combination
            if 'Home_Band' in rec_bets.columns and 'Away_Band' in rec_bets.columns:
                band_combo = rec_bets.groupby(['Home_Band', 'Away_Band']).size().reset_index(name='Count')
                band_combo = band_combo.sort_values('Count', ascending=False).head(3)
                st.write("Most common band combinations:")
                st.dataframe(band_combo, use_container_width=True)
            
            st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# 📋 DETAILED RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Detailed Backtest Results")

# Show filtered results
cols_to_show = ['Date', 'League', 'Home', 'Away', 'Auto_Recommendation', 'Profit', 
                'M_H', 'M_A', 'M_Diff', 'Diff_Power', 'Odd_H', 'Odd_D', 'Odd_A']

available_cols = [col for col in cols_to_show if col in df_backtest.columns]

st.dataframe(
    df_backtest[available_cols].sort_values(['Date', 'League']),
    use_container_width=True,
    height=600
)

# ──────────────────────────────────────────────────────────────────────────────
# 💾 SAVE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("💾 Save Configuration")

config_name = st.sidebar.text_input("Configuration Name", "My Config")
if st.sidebar.button("💾 Save Current Parameters"):
    # Save current parameters to session state
    st.session_state['saved_configs'] = st.session_state.get('saved_configs', {})
    st.session_state['saved_configs'][config_name] = {
        'M_DIFF_MARGIN': M_DIFF_MARGIN,
        'POWER_MARGIN': POWER_MARGIN,
        'DOMINANT_THRESHOLD': DOMINANT_THRESHOLD,
        'DIFF_MID_LO': DIFF_MID_LO,
        'DIFF_MID_HI': DIFF_MID_HI,
        'DIFF_MID_HI_HIGHVAR': DIFF_MID_HI_HIGHVAR,
        'POWER_GATE': POWER_GATE,
        'POWER_GATE_HIGHVAR': POWER_GATE_HIGHVAR,
        'ODD_D_MIN': ODD_D_MIN,
        'ODD_D_MAX': ODD_D_MAX,
        'DIFF_POWER_DRAW_MIN': DIFF_POWER_DRAW_MIN,
        'DIFF_POWER_DRAW_MAX': DIFF_POWER_DRAW_MAX
    }
    st.sidebar.success(f"✅ Configuration '{config_name}' saved!")

# Load saved configurations
saved_configs = st.session_state.get('saved_configs', {})
if saved_configs:
    st.sidebar.subheader("📂 Load Configuration")
    selected_config = st.sidebar.selectbox("Choose configuration", list(saved_configs.keys()))
    if st.sidebar.button("🔄 Load Selected"):
        config = saved_configs[selected_config]
        for key, value in config.items():
            st.session_state[f"param_{key.lower()}"] = value
        st.sidebar.success("✅ Configuration loaded! Refresh to apply.")
