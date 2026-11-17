"""Shared data code: load the cleaned California data, define feature groups, split by location, and preprocess."""
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Works no matter which folder you run from
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "clean_ca_2016_2020.parquet"
# Saved region -> train/val/test assignment from split(df, seed=42), made in notebooks/data_splitting.ipynb
SPLIT_PATH = Path(__file__).resolve().parent.parent / "docs" / "region_split_seed42.json"

# Feature Groups
WEATHER = [
    # Day of ignition
    "pr", "tmmn", "tmmx", "rmin", "rmax", "sph", "vs", "srad", "etr", "vpd", "fm100", "fm1000", "bi", "erc",
    # Long-term normals for that place and day of year
    "pr_Normal", "tmmn_Normal", "tmmx_Normal", "rmin_Normal", "rmax_Normal",
    "fm100_Normal", "fm1000_Normal", "bi_Normal", "vpd_Normal", "erc_Normal",
    "Annual_precipitation", "Annual_tempreture",
    # How unusual the day is (converted to percentile numbers in the notebook)
    "tmmn_Percentile", "tmmx_Percentile", "sph_Percentile", "vs_Percentile",
    "fm100_Percentile", "bi_Percentile", "vpd_Percentile", "erc_Percentile",
    # Season
    "DISCOVERY_DOY", "DISCOVERY_MONTH",
]

VEGETATION_BASE = [
    "NDVI-1day",
    "NDVI_min_avg", "NDVI_min_last", "NDVI_max_avg", "NDVI_max_last",
    "NDVI_mean_avg", "NDVI_mean_last", "MOD_NDVI_12m_avg", "MOD_NDVI_12m_last",
]

GEOGRAPHY = [
    "LATITUDE", "LONGITUDE",
    "Elevation", "Slope", "TPI",
    "Elevation_1km", "Slope_1km", "TPI_1km",
    "Aspect_sin", "Aspect_cos", "Aspect_1km_sin", "Aspect_1km_cos",
]

HUMAN = ["Population", "GAP_Prity", "EPL_MOBILE"]

TARGET = "TARGET"
GROUP = "REGION"   # 0.5 x 0.5 degree grid block, created in the cleaning notebook


def load():
    """Load the cleaned data saved by notebooks/data_cleaning.ipynb."""
    return pd.read_parquet(DATA_PATH)


def feature_sets(df):
    """The three feature combinations to compare. Adds the one-hot land cover/EVT columns automatically."""
    one_hot = [c for c in df.columns if c.startswith(("LC_", "EVT_"))]
    vegetation = VEGETATION_BASE + one_hot

    def keep(cols):
        # Skip columns that are missing, and warn about any that aren't numeric (the models need numbers)
        present = [c for c in cols if c in df.columns]
        bad = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            print("Skipping non-numeric columns:", bad)
        return [c for c in present if c not in bad]

    return {
        "model1_weather": keep(WEATHER),
        "model2_weather_veg": keep(WEATHER + vegetation),
        "model3_all": keep(WEATHER + vegetation + GEOGRAPHY),
        "model4_human": keep(WEATHER + vegetation + GEOGRAPHY + HUMAN),
    }


def split(df, seed=42):
    """Quick location split for experiments: whole grid blocks go to train, val, or test (about 70/15/15)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trval_idx, test_idx = next(gss.split(df, groups=df[GROUP]))
    trval, test = df.iloc[trval_idx], df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    tr_idx, val_idx = next(gss2.split(trval, groups=trval[GROUP]))
    return trval.iloc[tr_idx], trval.iloc[val_idx], test


def cv_folds(df, n_splits=5):
    """Location cross-validation for final results: each fold holds out a different set of grid blocks.
    Yields (train, test) DataFrames. Report metrics as mean +/- std across folds."""
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in gkf.split(df, groups=df[GROUP]):
        yield df.iloc[tr_idx], df.iloc[te_idx]


def prepare(train, val, test, features):
    """Fill missing values and scale, fit on train only. Returns (X, y) for each split."""
    imputer = SimpleImputer(strategy="median").fit(train[features])
    scaler = StandardScaler().fit(imputer.transform(train[features]))

    def transform(d):
        return scaler.transform(imputer.transform(d[features]))

    return (
        (transform(train), train[TARGET]),
        (transform(val), val[TARGET]),
        (transform(test), test[TARGET]),
    )

# to add separate the mutable and immutable features for the model explainability
def mutability(df):
    lc_evt = [c for c in df.columns if c.startswith(("LC_", "EVT_"))]
    mutable = ["NDVI-1day"] + lc_evt  + HUMAN        # fuel / vegetation
    immutable = WEATHER + GEOGRAPHY    # weather, season, terrain, location
    return {"mutable": [c for c in mutable if c in df.columns],
            "immutable": [c for c in immutable if c in df.columns]}

def temporal_split(df):
    """Split by year: train 2016-2018, validation 2019, test 2020."""
    train = df[df['FIRE_YEAR'] < 2019]
    val = df[df['FIRE_YEAR'] == 2019]
    test = df[df['FIRE_YEAR'] == 2020]
    return train, val, test

def log_target(df):
    """Log-transform the target variable."""
    return np.log1p(df["FIRE_SIZE"])

def to_acres(pred_log):
    """Convert log-transformed predictions back to acres."""
    return np.expm1(pred_log)

def random_split(df, seed=42):
    """Randomly split the data into train, validation, and test sets to compare with location and temporal splits."""
    train, rest = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[TARGET])
    # rest is 30% of the data, so splitting it in half gives 15% val and 15% test
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest[TARGET])
    return train, val, test

def combined_split(df, split_map=None):
    """Train on 2016-2018 fires in training regions, validate on 2019 fires in validation regions,
    test on 2020 fires in test regions. Uses the saved regional split unless a split_map is given."""
    if split_map is None:
        with open(SPLIT_PATH) as f:
            split_map = json.load(f)
    s = df["REGION"].map(split_map)
    train = df[(df["FIRE_YEAR"] <= 2018) & (s == "train")]
    val = df[(df["FIRE_YEAR"] == 2019) & (s == "val")]
    test = df[(df["FIRE_YEAR"] == 2020) & (s == "test")]
    return train, val, test