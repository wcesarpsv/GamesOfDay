import streamlit as st
import pandas as pd
import numpy as np

st.title("🧮 Teste de Cálculo de Handicap Asiático")

# --- Funções ---

def calc_handicap_result(margin, asian_line_decimal, invert=False):
    if pd.isna(asian_line_decimal) or pd.isna(margin):
        return np.nan
    if invert:
        margin = -margin
    if margin > asian_line_decimal:
        return 1.0
    elif margin == asian_line_decimal:
        return 0.5
    else:
        return 0.0

def determine_handicap_result(row):
    gh, ga, asian_line_decimal = row['Goals_H'], row['Goals_A'], row['Asian_Line_Decimal']
    margin = gh - ga
    res = calc_handicap_result(margin, asian_line_decimal, invert=False)
    if res == 1.0:
        return "HOME_COVERED"
    elif res == 0.5:
        return "HALF_HOME_COVERED"
    elif res == 0.0:
        return "HOME_NOT_COVERED"
    else:
        return None

def calculate_handicap_profit(rec, handicap_result, odd_home, odd_away, asian_line_decimal):
    if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid' or pd.isna(asian_line_decimal):
        return 0

    rec = str(rec).upper()
    is_home_bet = any(k in rec for k in ['HOME', 'FAVORITO HOME', 'VALUE NO HOME'])
    is_away_bet = any(k in rec for k in ['AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])

    if not (is_home_bet or is_away_bet):
        return 0

    odd = odd_home if is_home_bet else odd_away
    result = str(handicap_result).upper()

    if result == "PUSH":
        return 0
    elif result == "HALF_HOME_COVERED":
        if is_home_bet:
            return odd / 2
        elif is_away_bet:
            return -0.5
    elif result == "HOME_COVERED":
        if is_home_bet:
            return odd
        elif is_away_bet:
            return -1
    elif result == "HOME_NOT_COVERED":
        if is_home_bet:
            return -1
        elif is_away_bet:
            return odd
    else:
        return 0

    return 0

# --- Casos de teste ---
df_test = pd.DataFrame([
    ["+0.25", 1.0, 1.0, "AWAY", 1.95, 1.95, 0.25],
    ["+0.25", 2.0, 1.0, "AWAY", 1.95, 1.95, 0.25],
    ["+0.25", 1.0, 2.0, "AWAY", 1.95, 1.95, 0.25],
    ["+1.00", 2.0, 2.0, "AWAY", 1.90, 1.90, 1.00],
    ["-0.75", 2.0, 1.0, "HOME", 1.90, 1.90, -0.75],
    ["-1.00", 3.0, 1.0, "HOME", 1.90, 1.90, -1.00],
], columns=["Asian_Line", "Goals_H", "Goals_A", "Recomendacao", "Odd_H_Asi", "Odd_A_Asi", "Asian_Line_Decimal"])

# --- Cálculos ---
df_test["Handicap_Result"] = df_test.apply(determine_handicap_result, axis=1)
df_test["Profit"] = df_test.apply(
    lambda r: calculate_handicap_profit(r["Recomendacao"], r["Handicap_Result"], r["Odd_H_Asi"], r["Odd_A_Asi"], r["Asian_Line_Decimal"]),
    axis=1
)

st.dataframe(df_test)
