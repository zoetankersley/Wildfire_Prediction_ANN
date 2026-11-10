"""Readable names for features, used only for plots and tables (the data keeps its original column names)."""

NAMES = {
    # Weather on the day of ignition (gridMET)
    "pr": "Precipitation (mm)",
    "tmmn": "Min temperature (K)",
    "tmmx": "Max temperature (K)",
    "rmin": "Min relative humidity (%)",
    "rmax": "Max relative humidity (%)",
    "sph": "Specific humidity",
    "vs": "Wind speed (m/s)",
    "srad": "Solar radiation (W/m²)",
    "etr": "Reference evapotranspiration (mm)",
    "vpd": "Vapor pressure deficit (kPa)",
    "fm100": "100-hr dead fuel moisture (%)",
    "fm1000": "1000-hr dead fuel moisture (%)",
    "bi": "Burning index",
    "erc": "Energy release component",
    "Annual_precipitation": "Annual precipitation",
    "Annual_tempreture": "Annual temperature",
    "DISCOVERY_DOY": "Day of year",
    "DISCOVERY_MONTH": "Month",
    # Vegetation
    "NDVI-1day": "Greenness (NDVI), day before",
    "NDVI_min": "Greenness (NDVI), 12-mo min",
    "NDVI_max": "Greenness (NDVI), 12-mo max",
    "NDVI_mean": "Greenness (NDVI), 12-mo mean",
    "MOD_NDVI_12m": "Greenness (MODIS NDVI), prior year",
    "CheatGrass": "Cheatgrass cover (%)",
    "ExoticAnnualGrass": "Exotic annual grass cover (%)",
    # Geography
    "LATITUDE": "Latitude",
    "LONGITUDE": "Longitude",
    "Elevation": "Elevation (m)",
    "Slope": "Slope (°)",
    "TPI": "Topographic position (ridge vs. valley)",
    "Aspect_sin": "Slope faces east (+) / west (−)",
    "Aspect_cos": "Slope faces north (+) / south (−)",
}

# NLCD land cover codes
LAND_COVER = {
    11: "Open water", 21: "Developed, open space", 22: "Developed, low intensity",
    23: "Developed, medium intensity", 24: "Developed, high intensity", 31: "Barren",
    41: "Deciduous forest", 42: "Evergreen forest", 43: "Mixed forest", 52: "Shrub/scrub",
    71: "Grassland", 81: "Pasture/hay", 82: "Cultivated crops", 90: "Woody wetlands",
    95: "Emergent herbaceous wetlands",
}


def pretty(col):
    """Turn a column name into a readable label, e.g. 'erc_Normal' -> 'Energy release component (30-yr normal)'."""
    if col.startswith("LC_"):
        code = col[3:]
        try:
            return "Land cover: " + LAND_COVER.get(int(float(code)), code)
        except ValueError:
            return "Land cover: " + code          # "other" / "missing"
    if col.startswith("EVT_"):
        return "Vegetation type " + col[4:]       # look up codes in LANDFIRE if needed
    for suffix, note in [("_Normal", " (30-yr normal)"), ("_Percentile", " (percentile vs. history)"), ("_1km", " (1 km avg)")]:
        if col.endswith(suffix):
            return pretty(col[: -len(suffix)]) + note
    return NAMES.get(col, col)