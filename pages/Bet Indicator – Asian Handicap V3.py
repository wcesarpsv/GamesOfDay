##################### BLOCO 1 – IMPORTS & CONFIG #####################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import io
import zipfile
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta

st.set_page_config(page_title="Bet Indicator – Asian Handicap V3", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (V3 – Enriched Features)")

PAGE_PREFIX = "AsianHandicap"
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup","copas","uefa","afc","sudamericana","copa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

... (todo o código completo que você colou antes, já ajustado com Strict, blocos organizados e download de modelos)
