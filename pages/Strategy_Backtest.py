# ──────────────────────────────────────────────────────────────────────────────
# 🎯 Filters (Dynamic Cascading)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Filter Matches")

# 🔘 Botão Reset
if st.sidebar.button("🔄 Reset filters"):
    for key in list(st.session_state.keys()):
        if any(prefix in key for prefix in ["mh", "ma", "diff_power", "odd_h", "odd_d", "odd_a",
                                            "date", "bet_on", "diff_htp", "mht_h", "mht_a"]):
            del st.session_state[key]
    st.rerun()

# Escolha de aposta
bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Draw", "Away"], key="bet_on")
st.sidebar.divider()

# Dataset filtrado progressivamente
df_filtered = df_all.copy()

# 📅 Período (Date)
date_start, date_end = date_range_filter_hybrid("🗓️ Period (Date)", df_filtered["Date"], key_prefix="date")
df_filtered = df_filtered[(df_filtered["Date"] >= date_start) & (df_filtered["Date"] <= date_end)]

# 📊 M_H
if not df_filtered.empty:
    mh_min, mh_max = float(df_filtered["M_H"].min()), float(df_filtered["M_H"].max())
    mh_sel = range_filter_hybrid("📊 M_H", mh_min, mh_max, step=0.01, key_prefix="mh")
    df_filtered = df_filtered[(df_filtered["M_H"] >= mh_sel[0]) & (df_filtered["M_H"] <= mh_sel[1])]

# 📊 M_A
if not df_filtered.empty:
    ma_min, ma_max = float(df_filtered["M_A"].min()), float(df_filtered["M_A"].max())
    ma_sel = range_filter_hybrid("📊 M_A", ma_min, ma_max, step=0.01, key_prefix="ma")
    df_filtered = df_filtered[(df_filtered["M_A"] >= ma_sel[0]) & (df_filtered["M_A"] <= ma_sel[1])]

# 📊 Diff_Power
if not df_filtered.empty:
    dp_min, dp_max = float(df_filtered["Diff_Power"].min()), float(df_filtered["Diff_Power"].max())
    diff_power_sel = range_filter_hybrid("📊 Diff_Power", dp_min, dp_max, step=0.01, key_prefix="diff_power")
    df_filtered = df_filtered[(df_filtered["Diff_Power"] >= diff_power_sel[0]) & (df_filtered["Diff_Power"] <= diff_power_sel[1])]

# 💰 Odd_H
if not df_filtered.empty:
    oh_min, oh_max = float(df_filtered["Odd_H"].min()), float(df_filtered["Odd_H"].max())
    odd_h_sel = range_filter_hybrid("💰 Odd_H (Home win)", oh_min, oh_max, step=0.01, key_prefix="odd_h")
    df_filtered = df_filtered[(df_filtered["Odd_H"] >= odd_h_sel[0]) & (df_filtered["Odd_H"] <= odd_h_sel[1])]

# 💰 Odd_D
if not df_filtered.empty:
    od_min, od_max = float(df_filtered["Odd_D"].min()), float(df_filtered["Odd_D"].max())
    odd_d_sel = range_filter_hybrid("💰 Odd_D (Draw)", od_min, od_max, step=0.01, key_prefix="odd_d")
    df_filtered = df_filtered[(df_filtered["Odd_D"] >= odd_d_sel[0]) & (df_filtered["Odd_D"] <= odd_d_sel[1])]

# 💰 Odd_A
if not df_filtered.empty:
    oa_min, oa_max = float(df_filtered["Odd_A"].min()), float(df_filtered["Odd_A"].max())
    odd_a_sel = range_filter_hybrid("💰 Odd_A (Away win)", oa_min, oa_max, step=0.01, key_prefix="odd_a")
    df_filtered = df_filtered[(df_filtered["Odd_A"] >= odd_a_sel[0]) & (df_filtered["Odd_A"] <= odd_a_sel[1])]

# ➕ Filtros extras
extra_filters = st.sidebar.multiselect(
    "➕ Filtros extras (opcionais)",
    options=["Diff_HT_P", "M_HT_H", "M_HT_A"]
)

if "Diff_HT_P" in extra_filters and not df_filtered.empty:
    htp_min, htp_max = float(df_filtered["Diff_HT_P"].min()), float(df_filtered["Diff_HT_P"].max())
    diff_ht_p_sel = range_filter_hybrid("📉 Diff_HT_P", htp_min, htp_max, step=0.01, key_prefix="diff_htp")
    df_filtered = df_filtered[(df_filtered["Diff_HT_P"] >= diff_ht_p_sel[0]) & (df_filtered["Diff_HT_P"] <= diff_ht_p_sel[1])]

if "M_HT_H" in extra_filters and not df_filtered.empty:
    mht_h_min, mht_h_max = float(df_filtered["M_HT_H"].min()), float(df_filtered["M_HT_H"].max())
    mht_h_sel = range_filter_hybrid("📊 M_HT_H", mht_h_min, mht_h_max, step=0.01, key_prefix="mht_h")
    df_filtered = df_filtered[(df_filtered["M_HT_H"] >= mht_h_sel[0]) & (df_filtered["M_HT_H"] <= mht_h_sel[1])]

if "M_HT_A" in extra_filters and not df_filtered.empty:
    mht_a_min, mht_a_max = float(df_filtered["M_HT_A"].min()), float(df_filtered["M_HT_A"].max())
    mht_a_sel = range_filter_hybrid("📊 M_HT_A", mht_a_min, mht_a_max, step=0.01, key_prefix="mht_a")
    df_filtered = df_filtered[(df_filtered["M_HT_A"] >= mht_a_sel[0]) & (df_filtered["M_HT_A"] <= mht_a_sel[1])]