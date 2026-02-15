import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ----------------------------
# Configuration
# ----------------------------

RESULTS_FILE = "cache_results.csv"
PLOTS_DIR = Path("plots")

POLICY_ORDER = [
    "l2-disabled",
    "mip-none",
    "mip-plus1",
    "lip-none",
    "lip-plus1",
]

# ----------------------------
# Reset plots directory
# ----------------------------

if PLOTS_DIR.exists():
    shutil.rmtree(PLOTS_DIR)
PLOTS_DIR.mkdir()

# ----------------------------
# Load and normalize data
# ----------------------------

df = pd.read_csv(RESULTS_FILE)

df["C2"] = df["C2"].fillna(0).astype(int)
df["S2"] = df["S2"].fillna(0).astype(int)

df["Rep"] = df["Rep"].fillna("none").str.lower().str.strip()
df["Pref"] = df["Pref"].fillna("none").str.lower().str.strip()

def make_policy(row):
    if row["L2_en"] == 0 or row["C2"] == 0:
        return "l2-disabled"
    return f"{row['Rep']}-{row['Pref']}"

df["L2_policy"] = df.apply(make_policy, axis=1)

df["L2_policy"] = pd.Categorical(
    df["L2_policy"],
    categories=POLICY_ORDER,
    ordered=True,
)

# ----------------------------
# 🔑 FILTER: keep only min-AAT block size
# ----------------------------

GROUP_COLS = [
    "trace",
    "C1",
    "C2",
    "S1",
    "S2",
    "L2_policy",
]

df = (
    df.loc[df.groupby(GROUP_COLS)["L1_AAT"].idxmin()]
    .reset_index(drop=True)
)

# Plot here
