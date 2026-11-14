# Data Dictionary: `clean_ca_2016_2020.parquet`

California wildfires discovered 2016–2020. **43,561 rows × 96 columns**, one row per fire.

- **Source:** FPA FOD-Attributes (Pourmohamad et al., 2024), filtered to California by [`notebooks/data_cleaning.ipynb`](../notebooks/data_cleaning.ipynb).
- **Load with:** `from src.data import load; df = load()`
- **Groups** follow the feature lists in [`src/data.py`](../src/data.py) (`WEATHER`, `VEGETATION_BASE` + one-hot columns, `GEOGRAPHY`). Columns that are not model inputs are listed at the end.
- **Ranges** are the min–max values in this file. **Missing** is the share of rows that are empty (filled with the training median in `prepare()`).
- Weather variables come from gridMET (daily, ~4 km grid). Temperatures are in **Kelvin** (subtract 273.15 for °C).

---

## 1. Weather (36 columns)

### 1a. Conditions on the day of discovery

| Column | Description | Units | Range | Missing |
|---|---|---|---|---|
| `pr` | Precipitation | mm/day | 0 – 132.4 | 0% |
| `tmmn` | Minimum air temperature | K | 258.1 – 307.9 | 0% |
| `tmmx` | Maximum air temperature | K | 268.6 – 322.7 | 0% |
| `rmin` | Minimum relative humidity | % | 1 – 100 | 0% |
| `rmax` | Maximum relative humidity | % | 8 – 100 | 0% |
| `sph` | Specific humidity | kg/kg | 0.0003 – 0.0205 | 0% |
| `vs` | Wind speed at 10 m | m/s | 0.4 – 13.9 | 0% |
| `srad` | Downward shortwave solar radiation | W/m² | 22.2 – 387.7 | 0% |
| `etr` | Reference evapotranspiration (alfalfa) | mm/day | 0.2 – 24.7 | 0% |
| `vpd` | Vapor pressure deficit (how "thirsty" the air is) | kPa | 0 – 7.48 | 0% |
| `fm100` | 100-hour dead fuel moisture (sticks ~1–3 in. diameter) | % | 2 – 28.5 | 0% |
| `fm1000` | 1000-hour dead fuel moisture (logs ~3–8 in. diameter) | % | 2.7 – 34.6 | 0% |
| `bi` | Burning Index (NFDRS): expected flame length / difficulty of control | index | 0 – 143 | 0% |
| `erc` | Energy Release Component (NFDRS): potential heat released at the flaming front | index | 0 – 114 | 0% |

### 1b. Long-term normals (typical value for that location and day of year)

| Column | Description | Units | Range | Missing |
|---|---|---|---|---|
| `pr_Normal` | Normal precipitation | mm/day | 0 – 13.01 | 0% |
| `tmmn_Normal` | Normal minimum temperature | K | 263.6 – 301.9 | 0% |
| `tmmx_Normal` | Normal maximum temperature | K | 276.0 – 317.4 | 0% |
| `rmin_Normal` | Normal minimum relative humidity | % | 7.6 – 70 | 1.9% |
| `rmax_Normal` | Normal maximum relative humidity | % | 24 – 99.9 | 1.9% |
| `fm100_Normal` | Normal 100-hour fuel moisture | % | 3.1 – 23.4 | 0% |
| `fm1000_Normal` | Normal 1000-hour fuel moisture | % | 3.4 – 29.4 | 1.9% |
| `bi_Normal` | Normal Burning Index | index | 4.8 – 92.9 | 0% |
| `vpd_Normal` | Normal vapor pressure deficit | kPa | 0.17 – 5.32 | 0% |
| `erc_Normal` | Normal Energy Release Component | index | 3.1 – 106.8 | 0% |
| `Annual_precipitation` | Average annual precipitation | mm/year | 35 – 5,377 | 0% |
| `Annual_tempreture` | Average annual temperature *(column name is misspelled in the source)* | K | 274 – 299 | 0% |

### 1c. Percentiles (how unusual the day was compared with that location's history)

The source stores these as text ranges (`"<10th"`, `"10th-30th"`, `">90th"`). The cleaning notebook converts each range to its midpoint so the order is preserved, giving the values **5, 20, 40, 60, 80, 95**.

| Column | Description | Values | Missing |
|---|---|---|---|
| `tmmn_Percentile` | Percentile of the day's minimum temperature | **95 only** ⚠️ | 0% |
| `tmmx_Percentile` | Percentile of the day's maximum temperature | **95 only** ⚠️ | 0% |
| `sph_Percentile` | Percentile of specific humidity | 5, 20, 40, 60, 80, 95 | 0% |
| `vs_Percentile` | Percentile of wind speed | 5, 20, 40, 60, 80, 95 | 0.2% |
| `fm100_Percentile` | Percentile of 100-hour fuel moisture | 5, 20, 40, 60, 80, 95 | 0% |
| `bi_Percentile` | Percentile of Burning Index | 5, 20, 40, 60, 80, 95 | 0.1% |
| `vpd_Percentile` | Percentile of vapor pressure deficit (finer bins than the others) | 1, 3.5, 7.5, 15, 25, 40, 60, 75, 85, 92.5, 96.5, 99 | 0.1% |
| `erc_Percentile` | Percentile of Energy Release Component | 5, 20, 40, 60, 80, 95 | 0.1% |

### 1d. Season (grouped with weather in `src/data.py`)

| Column | Description | Units | Range | Missing |
|---|---|---|---|---|
| `DISCOVERY_DOY` | Day of year the fire was discovered | day | 1 – 366 | 0% |
| `DISCOVERY_MONTH` | Month the fire was discovered (derived from `DISCOVERY_DATE`) | month | 1 – 12 | 0% |

---

## 2. Vegetation (38 columns)

### 2a. Greenness

| Column | Description | Units | Range | Missing |
|---|---|---|---|---|
| `NDVI-1day` | Normalized Difference Vegetation Index just before discovery. Higher means greener, denser vegetation; near 0 or negative means bare ground, water, or pavement. | unitless (−1 to 1) | −0.32 – 1.03 ⚠️ | 2.4% |

### 2b. Land cover: `LC_<code>` (one-hot, 16 columns)

Each column is 1 if the ignition point falls in that **NLCD land cover class** and 0 otherwise. Exactly one is 1 per row.

| Column | NLCD class | % of fires |
|---|---|---|
| `LC_0.0` | Unclassified / no data | <0.1% |
| `LC_11.0` | Open water | 0.3% |
| `LC_21.0` | Developed, open space | 18.6% |
| `LC_22.0` | Developed, low intensity | 17.9% |
| `LC_23.0` | Developed, medium intensity | 25.5% |
| `LC_24.0` | Developed, high intensity | 8.4% |
| `LC_31.0` | Barren land | 0.4% |
| `LC_41.0` | Deciduous forest | 0.2% |
| `LC_42.0` | Evergreen forest | 9.4% |
| `LC_43.0` | Mixed forest | 0.8% |
| `LC_52.0` | Shrub / scrub | 10.0% |
| `LC_71.0` | Grassland / herbaceous | 5.5% |
| `LC_81.0` | Pasture / hay | 0.4% |
| `LC_82.0` | Cultivated crops | 2.2% |
| `LC_90.0` | Woody wetlands | 0.2% |
| `LC_95.0` | Emergent herbaceous wetlands | 0.3% |

### 2c. Existing vegetation type: `EVT_<code>` (one-hot, 22 columns)

Each column is 1 if the ignition point falls in that **LANDFIRE Existing Vegetation Type**. The cleaning notebook kept the 20 most common codes and lumped the rest into `EVT_other`.

LANDFIRE code names depend on the product version, so look up the exact names in the LANDFIRE EVT attribute table for the version used by FPA FOD-Attributes. Codes 7296–7299 are LANDFIRE's "Developed" classes.

| Column | % of fires | Column | % of fires |
|---|---|---|---|
| `EVT_7027.0` | 1.5% | `EVT_7298.0` | 0.6% |
| `EVT_7028.0` | 2.1% | `EVT_7299.0` | **68.7%** |
| `EVT_7030.0` | 0.9% | `EVT_7901.0` | 0.5% |
| `EVT_7032.0` | 0.8% | `EVT_7904.0` | 0.6% |
| `EVT_7043.0` | 0.8% | `EVT_7913.0` | 0.7% |
| `EVT_7092.0` | 0.8% | `EVT_7914.0` | 1.1% |
| `EVT_7098.0` | 0.8% | `EVT_7984.0` | 0.6% |
| `EVT_7105.0` | 1.1% | `EVT_9301.0` | 2.4% |
| `EVT_7110.0` | 1.1% | `EVT_9337.0` | 0.6% |
| `EVT_7296.0` | 1.2% | `EVT_other` | 12.0% |
| `EVT_7297.0` | 1.1% | | |

---

## 3. Geography (12 columns)

Columns without a suffix are at the ignition point (30 m cell). `_1km` columns are averaged over the surrounding ~1 km.

| Column | Description | Units | Range | Missing |
|---|---|---|---|---|
| `LATITUDE` | Latitude of the ignition point | degrees N | 32.55 – 42.42 | 0% |
| `LONGITUDE` | Longitude of the ignition point | degrees E | −124.4 – −114.1 | 0% |
| `Elevation` | Elevation (negative = below sea level, e.g. Death Valley, Salton Sea) | m | −83 – 3,650 | 0% |
| `Slope` | Steepness of the ground | degrees | 0 – 58 | 0% |
| `TPI` | Topographic Position Index: positive = ridge or hilltop, negative = valley, ~0 = flat or mid-slope | index | −12.38 – 40.5 | 0% |
| `Elevation_1km` | Elevation, 1 km average | m | −65.6 – 2,981 | 0% |
| `Slope_1km` | Slope, 1 km average | degrees | −0.81 – 31.47 ⚠️ | 0% |
| `TPI_1km` | Topographic Position Index, 1 km scale (different scale than `TPI`) | index | −0.48 – 0.45 | 0% |
| `Aspect_sin` | East–west facing direction: +1 = east, −1 = west, 0 = flat | −1 to 1 | −1 – 1 | 0% |
| `Aspect_cos` | North–south facing direction: +1 = north, −1 = south, 0 = flat | −1 to 1 | −1 – 1 | 0% |
| `Aspect_1km_sin` | East–west facing direction, 1 km scale | −1 to 1 | −0.98 – 1 | 0% |
| `Aspect_1km_cos` | North–south facing direction, 1 km scale | −1 to 1 | −1 – 0.99 | 0% |

**Why sine and cosine?** Aspect is a compass bearing, so 359° and 1° both face north even though the numbers are far apart. Splitting it into sin/cos lets the model treat nearby directions as similar. Where slope is under 2°, both values are set to 0 ("no direction") so flat farmland and towns aren't mislabeled as north-facing.

---

## 4. Other columns (not model inputs)

### Target and split

| Column | Description | Values |
|---|---|---|
| `TARGET` | **What the model predicts.** Fire size class with the rare classes merged. | `A` 27,279 · `B` 13,678 · `C` 1,693 · `D-E` 591 · `F-G` 320 |
| `REGION` | 0.5° × 0.5° grid block (~50 km), formatted `"<lat block>_<lon block>"`. Used to keep whole areas together in train/test splits so the model is tested on places it hasn't seen. | 197 blocks |

### Fire record (analysis only: these are known only after the fire, so using them as inputs would leak the answer)

| Column | Description | Values |
|---|---|---|
| `FIRE_SIZE` | Final burned area | 0.01 – 589,368 acres |
| `FIRE_SIZE_CLASS` | Official NWCG size class: A ≤ 0.25 ac · B 0.26–9.9 · C 10–99.9 · D 100–299 · E 300–999 · F 1,000–4,999 · G ≥ 5,000 | A 27,279 · B 13,678 · C 1,693 · D 394 · E 197 · F 177 · G 143 |
| `FIRE_YEAR` | Year of discovery | 2016 – 2020 |
| `DISCOVERY_DATE` | Date of discovery | 2016-01-01 – 2020-12-31 |
| `STATE` | State | `CA` only |

### Human / land-use context (kept but not in any feature group yet)

| Column | Description | Range | Missing |
|---|---|---|---|
| `Population` | Population density near the ignition point (check the source documentation for units) | 0 – 396.5 | 0.3% |
| `GAP_Prity` | Protected-area (PAD-US GAP) related code. Its meaning still needs to be confirmed in the source documentation. | 0, 6, 7, 8, 9 | 0% |
| `EPL_MOBILE` | CDC Social Vulnerability Index: percentile rank of the share of housing that is mobile homes | 0 – 1 | 0.1% |

---

## ⚠️ Data issues to discuss with the team

1. **`tmmn_Percentile` and `tmmx_Percentile` are the same value (95) in every row**, so they give the model no information. The text-to-number conversion in the cleaning notebook (step 7) may be collapsing their labels. Check the raw values.
2. **`src/data.py` lists vegetation columns that aren't in the file:** `NDVI_min`, `NDVI_max`, `NDVI_mean`, `MOD_NDVI_12m`. `feature_sets()` silently skips them, so the vegetation group is really just `NDVI-1day` plus the LC/EVT one-hot columns. They were probably dropped in step 5 for being more than 50% missing.
3. **`Slope_1km` is negative in 9,582 rows (22%)**, which isn't possible for a slope in degrees. It may be a resampling artifact or a different unit.
4. **`NDVI-1day` has values above 1** (max 1.03), outside NDVI's valid −1 to 1 range.
5. **`EVT_7299.0` covers 68.7% of fires.** It's worth confirming its LANDFIRE name, since one category this dominant will strongly shape the vegetation model.
6. **There is no `LC_other` column, but there is an `EVT_other`.** California only has 16 land cover codes, so all of them fit in the top 20.
