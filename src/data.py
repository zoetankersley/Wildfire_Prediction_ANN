"""Shared data code: load the cleaned data, define feature groups, split by location, and preprocess."""
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# Works no matter which folder you run from
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "clean_2016_2020.parquet"

# Feature Groups
WEATHER = [
    # Day of ignition
    "pr", "tmmn", "tmmx", "rmin", "rmax", "vs", "vpd", "fm100", "fm1000", "bi", "erc",
    # Long-term normals for that place and day of year
    "pr_Normal", "tmmn_Normal", "tmmx_Normal", "rmin_Normal", "rmax_Normal",
    "fm100_Normal", "fm1000_Normal", "bi_Normal", "vpd_Normal", "erc_Normal",
    "Annual_precipitation", "Annual_tempreture",
    # How unusual the day is (converted to 0-5 in the notebook)
    "tmmn_Percentile", "tmmx_Percentile", "sph_Percentile", "vs_Percentile",
    "fm100_Percentile", "bi_Percentile", "vpd_Percentile", "erc_Percentile",
    # Season
    "DISCOVERY_DOY", "DISCOVERY_MONTH",
]

VEGETATION_BASE = [
    "NDVI-1day", "NDVI_min", "NDVI_max", "NDVI_mean", "MOD_NDVI_12m",
    "CheatGrass", "ExoticAnnualGrass",
]

GEOGRAPHY = [
    "LATITUDE", "LONGITUDE",
    "Elevation", "Slope", "TPI",
    "Elevation_1km", "Slope_1km", "TPI_1km",
    "Aspect_sin", "Aspect_cos", "Aspect_1km_sin", "Aspect_1km_cos",
]

TARGET = "TARGET"
GROUP = "REGION"


def load():
    """Load the cleaned data saved by notebooks/01_eda.ipynb."""
    return pd.read_parquet(DATA_PATH)


def feature_sets(df):
    """The three feature combinations to compare. Adds the one-hot land cover/EVT columns automatically."""
    one_hot = [c for c in df.columns if c.startswith(("LC_", "EVT_"))]
    vegetation = VEGETATION_BASE + one_hot

    def keep(cols):
        return [c for c in cols if c in df.columns]   # skip any column that's missing

    return {
        "model1_weather": keep(WEATHER),
        "model2_weather_veg": keep(WEATHER + vegetation),
        "model3_all": keep(WEATHER + vegetation + GEOGRAPHY),
    }


def split(df, seed=42):
    """Location split: each 1x1 degree grid block goes entirely to train, val, or test (70/15/15)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trval_idx, test_idx = next(gss.split(df, groups=df[GROUP]))
    trval, test = df.iloc[trval_idx], df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    tr_idx, val_idx = next(gss2.split(trval, groups=trval[GROUP]))
    return trval.iloc[tr_idx], trval.iloc[val_idx], test


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