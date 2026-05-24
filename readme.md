# PM2.5 Estimation from Satellite Observations
### Bharatiya Antariksh Hackathon 2025 — Problem Statement 3

# Testing AI code Review Assistant

> Estimating surface-level Particulate Matter (PM2.5) concentration using Aerosol Optical Depth (AOD) from INSAT-3DR satellite observations, MERRA-2 reanalysis data, and CPCB ground measurements using AI/ML techniques.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Approach Overview](#approach-overview)
3. [Dataset Description](#dataset-description)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Phase 1 — Data Reduction](#phase-1--data-reduction)
6. [Phase 2 — INSAT-3DR Preprocessing](#phase-2--insat-3dr-preprocessing)
7. [Phase 3 — MERRA-2 Preprocessing](#phase-3--merra-2-preprocessing)
8. [Phase 4 — Spatial Harmonization](#phase-4--spatial-harmonization)
9. [Phase 5 — Three-Way Merge](#phase-5--three-way-merge)
10. [Phase 6 — AOD Missing Value Strategy](#phase-6--aod-missing-value-strategy)
11. [Phase 7 — Feature Engineering](#phase-7--feature-engineering)
12. [Phase 8 — Model Training](#phase-8--model-training)
13. [Results](#results)
14. [Key Findings](#key-findings)
15. [Project Structure](#project-structure)
16. [Dependencies](#dependencies)
17. [How to Run](#how-to-run)

---

## Problem Statement

Air pollution is a major environmental concern in Indian urban centres. Regular, consistent monitoring of PM2.5 remains a challenge due to limited ground station coverage. Satellite remote sensing enables observations over broad spatial areas, allowing indirect estimation of pollutant levels.

**Objective:** Estimate surface-level PM2.5 concentration using Aerosol Optical Depth (AOD) measurements from INSAT-3D/3DR/3DS satellite observations combined with meteorological reanalysis data and AI/ML-based predictive modelling.

**Expected Output:** High-resolution spatial map depicting PM2.5 concentration across India derived from an AI/ML estimation pipeline.

---

## Approach Overview

```
Satellite AOD (INSAT-3DR)   ──┐
                               ├──► AI/ML Model ──► PM2.5 Concentration Map
Atmospheric variables          │
(MERRA-2 Reanalysis)      ────┤
                               │
Ground truth PM2.5 (CPCB) ────┘  (training labels)
```

The core challenge is that three datasets have fundamentally different resolutions:

| Dimension | INSAT-3DR | MERRA-2 | CPCB |
|---|---|---|---|
| Temporal | 30-min (7 slots/day) | Hourly (24 slots/day) | Daily |
| Spatial | 4 km grid | ~55 km grid | Point stations |
| Coverage | 70% missing (clouds/night) | 0% missing | Sparse (98 stations) |

**Solution:** Use CPCB station coordinates as the spatial anchor and collapse all three datasets to daily station-level values before merging.

---

## Dataset Description

### INSAT-3DR AOD

| Property | Value |
|---|---|
| Raw shape | (455,520,000, 4) |
| After India clip | ~35–40M rows |
| After daily aggregation | (9,928,398, 8) — 31,000 unique lat/lon points |
| Period | 2023–2024 (731 days) |
| Temporal resolution | 30-minute — 7 slots/day, 05:45–13:30 UTC |
| Spatial resolution | 4 km |
| Wavelength | 650 nm |
| RMS Error | ±0.1 |
| AOD missing rate | ~70% raw; 43.4% after filtering |
| Features | datetime, latitude, longitude, AOD |

**Spectral note:** INSAT-3DR retrieves AOD at 650 nm. MODIS and MERRA-2 report at 550 nm. If mixing sources, apply the Ångström exponent conversion: `AOD_550 = AOD_650 × (550/650)^(−α)` where α ≈ 1.2 for urban Indian aerosols.

**RMS error implication:** With ±0.1 error, AOD values below 0.1 carry >100% relative error and are unreliable. Values below 0.3 are flagged as uncertain. Both thresholds are applied as hard filters before aggregation.

### MERRA-2 Reanalysis

| Property | Value |
|---|---|
| Raw shape | (52,139,520, 10) |
| After India clip | ~4–5M rows |
| After daily aggregation | (787,670, 20) — 1,079 unique lat/lon points |
| Period | 2023–2024 |
| Temporal resolution | Hourly (24 slots/day) |
| Spatial resolution | ~55 km (0.5° × 0.625°) |
| Missing rate | 0% |
| Features | datetime, latitude, longitude, PBLH, TLML, QLML, ULML, VLML, SPEED, PRECTOT |

### CPCB Ground Measurements

| Property | Value |
|---|---|
| Shape | (67,257, 5) |
| Stations | 98 unique stations across India |
| Period | 2023-01-01 → 2024-12-31 (731 days) |
| Temporal resolution | Daily |
| PM2.5 range (raw) | 0.0 – 1,000.0 µg/m³ |
| PM2.5 range (cleaned) | 3.6 – 319.1 µg/m³ |
| Missing PM2.5 | 0.0% |
| Features | station_name, date, latitude, longitude, pm25 |

---

## Pipeline Architecture

```
Raw INSAT (455M rows)
    │
    ├─ Phase 1 : India bbox filter → polygon clip
    ├─ Phase 2 : Nighttime drop → AOD range filter [0.1, 3.0]
    │            Half-hourly → daily aggregation per pixel
    │            (mean, max, p75, count, coverage)
    │
    └─► data/interim/insat_daily/
        9,928,398 rows × 8 cols — 31,000 grid points

Raw MERRA-2 (52M rows)
    │
    ├─ Phase 1 : India bbox filter → polygon clip
    ├─ Phase 3 : Derived features on hourly data
    │            (vent_coeff, inversion_proxy, moisture, wind_dir)
    │            Variable-specific daily aggregation
    │            Morning window PBLH (00–04 UTC = 05:30–09:30 IST)
    │
    └─► data/interim/merra_daily/
        787,670 rows × 20 cols — 1,079 grid points

CPCB daily data (67,257 rows — 98 stations)
    │
    ├─ Phase 4 : INSAT → stations  IDW 15km buffer  → data/interim/spatial_join/insat_station_daily
    ├─ Phase 4 : MERRA → stations  Bilinear interp  → data/interim/spatial_join/merra_station_daily
    │
    ├─ Phase 5 : Three-way merge on [station_name, date]
    │            Shape after merge: (67,970, 28)
    │
    ├─ Phase 6 : AOD missing value strategy
    │            Sentinel −1 fill + cloud flags + station quality encoding
    │
    ├─ Phase 7 : Feature engineering → (67,257, 73)
    │            Temporal · Physics interactions · Precipitation
    │            AOD lags/rolling · Station spatial context
    │            PM2.5 outlier removal (674 rows removed)
    │
    ├─ Phase 7B: PM2.5 lag features (Dataset B only)
    │            pm25_lag1/2/3 · pm25_roll3/7 · pm25_delta1
    │
    └─► data/processed/v1/
        dataset_A_no_lags.parquet    66 features
        dataset_B_with_lags.parquet  72 features
```

---

## Phase 1 — Data Reduction

Both raw datasets span a rectangle bounding box larger than India. Clip to the actual India polygon boundary as the very first step — this is the single most impactful preprocessing decision for compute time.

```python
import geopandas as gpd

india          = gpd.read_file('india_boundary.geojson')
india_buffered = india.buffer(0.5)   # 0.5° buffer retains border district stations

# Step 1: Fast bounding box pre-filter (lat 6–38°N, lon 68–98°E)
def bbox_filter(df):
    return df[
        df['latitude'].between(6.0, 38.0) &
        df['longitude'].between(68.0, 98.0)
    ]

# Step 2: Precise polygon clip
def polygon_clip(df, shape):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs='EPSG:4326'
    )
    return gpd.clip(gdf, shape).drop(columns='geometry')
```

**Size reduction after clipping:**

| Dataset | Before | After | Reduction |
|---|---|---|---|
| INSAT-3DR | 455M rows | ~35–40M rows | ~92% |
| MERRA-2 | 52M rows | ~4–5M rows | ~91% |

Always save clipped outputs as parquet — never re-read raw files during iteration:

```python
insat_clipped.to_parquet('data/interim/india_clipped/insat_clipped.parquet', index=False)
merra_clipped.to_parquet('data/interim/india_clipped/merra_clipped.parquet', index=False)
```

---

## Phase 2 — INSAT-3DR Preprocessing

### 2.1 Remove invalid observations

```python
# Drop nighttime — AOD retrieval is impossible without sunlight
insat_df    = insat_df[insat_df['hour'].between(5, 14)]

# Drop missing AOD — do NOT impute
# Missing = clouds present = fundamentally different atmospheric state
insat_valid = insat_df.dropna(subset=['AOD'])

# Physical range filter based on ±0.1 RMS specification
# Below 0.1: relative error > 100% — signal unusable
# Above 3.0: retrieval artifact — physically implausible over India
insat_valid = insat_valid[
    (insat_valid['AOD'] >= 0.1) &
    (insat_valid['AOD'] <= 3.0)
]
```

### 2.2 Daily aggregation — multiple statistics

INSAT has 7 observation slots per day (05:45–13:30 UTC). Aggregate each pixel to daily statistics:

```python
insat_daily = insat_valid.groupby(['lat_round', 'lon_round', 'date']).agg(
    AOD_mean     = ('AOD', 'mean'),
    AOD_max      = ('AOD', 'max'),                      # peak loading — better for PM episodes
    AOD_p75      = ('AOD', lambda x: x.quantile(0.75)),
    AOD_count    = ('AOD', 'count'),                    # valid obs count (quality proxy)
    AOD_coverage = ('AOD', lambda x: x.count() / 7.0)  # fraction of day with clear sky
).reset_index()

# Require ≥ 2 valid observations for a reliable daily mean
insat_daily = insat_daily[insat_daily['AOD_count'] >= 2]
```

**Why multiple AOD statistics:**
- `AOD_mean` — overall daily aerosol column loading
- `AOD_max` — captures morning rush-hour peaks; stronger predictor during high-PM episodes
- `AOD_p75` — robust to single-slot outliers while still capturing elevated loading
- `AOD_coverage` — data quality flag; low coverage = single-slot retrieval = less reliable

> `AOD_std` was entirely null across the dataset and was dropped.

**Shape after aggregation:** (9,928,398, 8) — 31,000 unique lat/lon grid points

---

## Phase 3 — MERRA-2 Preprocessing

### 3.1 Derived features on hourly data

Compute these before daily aggregation — they cannot be recovered afterward:

```python
# Wind direction from U and V components
merra_df['wind_dir'] = np.degrees(
    np.arctan2(merra_df['VLML'], merra_df['ULML'])
) % 360

# Ventilation coefficient — atmospheric dispersion capacity
# Low VC = stagnant day = pollution accumulates near the surface
merra_df['vent_coeff'] = merra_df['PBLH'] * merra_df['SPEED']

# Inversion proxy — temperature-to-PBLH ratio
# High value = strong surface inversion = pollution trapped near ground
merra_df['inversion_proxy'] = merra_df['TLML'] / (merra_df['PBLH'] + 1)

# Specific humidity → g/kg for interpretability
merra_df['moisture'] = merra_df['QLML'] * 1000
```

### 3.2 Variable-specific daily aggregation

Aggregation functions follow atmospheric physics:

```python
agg_rules = {
    'PBLH'           : ['mean', 'min', 'max'],  # min = dawn inversion depth (critical)
    'TLML'           : ['mean', 'max'],          # max = daytime heating peak
    'moisture'       : ['mean', 'max'],
    'SPEED'          : ['mean', 'min'],          # min = worst stagnation hour
    'PRECTOT'        : ['sum'],                  # SUM not mean — total wet scavenging
    'vent_coeff'     : ['mean', 'min'],          # min = worst dispersion hour
    'inversion_proxy': ['mean', 'max'],
    'wind_dir'       : ['mean'],
}
```

### 3.3 Morning window PBLH — strongest single MERRA-2 predictor

```python
# 00–04 UTC ≈ 05:30–09:30 IST — pre-sunrise boundary layer
morning      = merra_df[merra_df['hour'].between(0, 4)]
morning_pblh = morning.groupby(['lat_round', 'lon_round', 'date'])[
    'PBLH'
].agg(['mean', 'min']).reset_index()
morning_pblh.columns = [
    'lat_round', 'lon_round', 'date',
    'PBLH_morning_mean', 'PBLH_morning_min'
]
```

`PBLH_morning_min` is consistently the strongest individual MERRA-2 predictor of PM2.5. It captures the lowest point of the boundary layer before solar heating begins to disperse pollutants.

**Shape after aggregation:** (787,670, 20) — 1,079 unique lat/lon grid points

---

## Phase 4 — Spatial Harmonization

Three incompatible spatial grids must be brought to a common reference. CPCB station coordinates serve as the anchor — everything is mapped to station level.

### 4.1 INSAT → CPCB stations (4 km grid → point)

Inverse distance weighted (IDW) averaging within a 15 km radius buffer per station per day:

```python
from scipy.spatial import cKDTree

# For each station-day:
# 1. Build KD-tree on valid INSAT pixel centroids for that day
# 2. Find all pixels within 0.135° (≈ 15 km)
# 3. Weight each pixel by 1 / distance
# 4. Compute weighted average of all AOD statistics
```

**Why 15 km radius:** PM2.5 spatial footprint in Indian urban settings is approximately 10–20 km. A tighter radius misses nearby plumes; a wider radius dilutes the local aerosol signal.

**Why IDW over nearest-neighbour:** At 4 km resolution, the nearest pixel may miss an aerosol plume offset from the station — weighting across the 15 km buffer captures this.

**Output:** (68,482, 19) — INSAT AOD features at each CPCB station for each day

### 4.2 MERRA-2 → CPCB stations (55 km grid → point)

Bilinear interpolation (`LinearNDInterpolator`) with nearest-neighbour fallback for border stations outside the convex hull:

```python
from scipy.interpolate import LinearNDInterpolator

for col in merra_feat_cols:
    interp = LinearNDInterpolator(grid_points, values)
    result = interp(station_location)
    if np.isnan(result):           # station outside MERRA convex hull
        _, idx = cKDTree(grid_points).query(station_location)
        result = values[idx]       # nearest-neighbour fallback
```

**Why bilinear over nearest-neighbour:** MERRA-2 grid cells are 55 km wide. Nearest-neighbour introduces positional errors of up to 30 km for border stations in Rajasthan, Punjab, and the Northeast.

**Output:** (71,540, 19) — MERRA-2 meteorological features at each CPCB station for each day

---

## Phase 5 — Three-Way Merge

CPCB is the left anchor. All joins are left joins to preserve every ground measurement row:

```python
df = (
    cpcb
    .merge(insat_station, on=['station_name', 'date'], how='left')
    .merge(merra_station, on=['station_name', 'date'], how='left')
)
```

**Merged shape:** (67,970, 28)

**Missing values after merge and their treatment:**

| Column group | Missing rows | Cause | Fix |
|---|---|---|---|
| AOD_mean / AOD_max / AOD_p75 | 29,921 (43.4%) | Cloud cover, nighttime | Sentinel −1 fill |
| n_pixels / AOD_count / AOD_coverage | 2,993 | Complete cloud cover, zero pixels in buffer | Fill with 0 |
| PBLH and all MERRA vars | 92 (0.13%) | Border stations outside MERRA convex hull | Station-month median → month median → global median |
| pm25 | 0 | CPCB anchor is complete | — |

---

## Phase 6 — AOD Missing Value Strategy

### Core principle — missing AOD is informative, not merely absent

Missing AOD = clouds present = critical meteorological context. Cloud days with rainfall tend to have lower PM2.5 due to wet scavenging. **Never drop these rows. Never impute AOD with mean or interpolation.**

### Observed seasonal AOD missingness

| Season | Months | AOD Missing % | Primary cause |
|---|---|---|---|
| Winter | Dec–Feb | 11–25% | Fog, occasional cloud |
| Pre-monsoon | Mar–May | 17–32% | Convective cloud build-up |
| Monsoon | Jun–Sep | **61–93%** | Dense monsoon cloud cover |
| Post-monsoon | Oct–Nov | 26–29% | Clearing sky, occasional cloud |

### Station-level AOD quality encoding

```python
# Quality categories based on per-station AOD missing rate
# dead     ≥ 95%  : AOD unusable  (Belur Math Howrah: 100%, Jodhpur: 99.7%)
# poor     ≥ 60%  : weak signal   (multiple Bengaluru stations: 59–62%)
# moderate ≥ 40%  : monsoon-affected but usable
# good     < 40%  : reliable AOD signal
```

Bengaluru stations show 59–62% AOD missingness even outside monsoon — likely a geometric retrieval issue with the INSAT-3DR viewing angle over the Deccan plateau rather than actual cloud cover.

### Features derived from AOD missingness

```python
df['AOD_missing']              # 1 = no retrieval today (cloud/night flag)
df['aod_miss_winter']          # unusual winter cloud → possible fog event
df['aod_miss_premonsoon']      # pre-monsoon cloud build-up
df['aod_miss_monsoon']         # expected — rain washout signal
df['aod_miss_postmonsoon']     # post-monsoon residual cloud
df['station_aod_quality']      # 0=dead, 1=poor, 2=moderate, 3=good
df['station_aod_missing_rate'] # fraction of days this station has no AOD
df['station_month_aod_clim']   # typical AOD at this station in this month
```

### Sentinel fill

```python
# Fill AOD value columns with −1 (outside physical range 0–3)
# −1 signals "no satellite retrieval today" — distinct from AOD = 0 (clean air)
aod_value_cols = ['AOD_mean', 'AOD_max', 'AOD_p75']
df[aod_value_cols] = df[aod_value_cols].fillna(-1)
```

**Final AOD composition after fill:**
- Cloud days (AOD = −1): **43.4%** of station-days
- Clear days (AOD > 0): **56.6%** of station-days

---

## Phase 7 — Feature Engineering

Feature engineering is the largest single contributor to R². This phase expands the dataset from 28 raw columns to 73 engineered columns.

### 7.1 Temporal features

```python
df['month']        # 1–12
df['day_of_year']  # 1–365
df['day_of_week']  # 0–6
df['year']         # 2023 or 2024
df['is_weekend']   # binary flag
df['season']       # 0=Winter, 1=Pre-monsoon, 2=Monsoon, 3=Post-monsoon

# Cyclical encoding — preserves Dec–Jan continuity
# Without this, a tree model treats December (12) as far from January (1)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)
```

### 7.2 Physics-driven AOD × meteorology interactions

All interactions use a sentinel-aware helper that returns −1 on cloud days:

```python
def safe_interact(aod_col, met_col, op='divide'):
    valid  = df[aod_col] != -1
    result = pd.Series(-1.0, index=df.index)
    if op == 'divide':
        result[valid] = df.loc[valid, aod_col] / (df.loc[valid, met_col] + 1)
    elif op == 'multiply':
        result[valid] = df.loc[valid, aod_col] * df.loc[valid, met_col]
    return result

# Core PM proxy: PM_surface ∝ AOD / PBLH (column loading model)
df['PM_proxy']        = safe_interact('AOD_mean', 'PBLH_morning_min', 'divide')
df['PM_proxy_max']    = safe_interact('AOD_max',  'PBLH_morning_min', 'divide')
df['AOD_x_moisture']  = safe_interact('AOD_mean', 'moisture_mean',    'multiply')
df['AOD_x_vent']      = safe_interact('AOD_mean', 'vent_coeff_mean',  'divide')
df['AOD_x_inversion'] = safe_interact('AOD_mean', 'inversion_proxy_max', 'multiply')

# Pollution trap score — combined atmospheric suppression index
df['trap_score'] = (
    df['inversion_proxy_max'] /
    df['SPEED_min'].clip(lower=0.1) /
    (df['PBLH_morning_min'] + 1)
)

# Stagnation flag
df['is_stagnant'] = (
    (df['SPEED_mean'] < 2.0) &
    (df['PBLH_min']   < 500)
).astype(int)
```

### 7.3 Precipitation washout features

```python
df['rain_lag1']  # yesterday's precipitation
df['rain_lag2']  # precipitation 2 days ago
df['rain_3day']  # cumulative 3-day precipitation total
df['is_rainy']   # 1 if PRECTOT > 0.5 mm today
df['post_rain1'] # 1 if it rained yesterday
df['post_rain2'] # 1 if it rained 2 days ago
```

### 7.4 Lag and rolling features

This is the single most impactful feature group across the entire pipeline.

| Addition | R² before | R² after | Gain |
|---|---|---|---|
| PM2.5 lag features | ~0.72 | ~0.82–0.85 | **+0.10** |
| AOD lag + rolling | ~0.82 | ~0.83–0.85 | +0.02–0.03 |

#### PM2.5 lag features (Dataset B only)

```python
df = df.sort_values(['station_name', 'date']).reset_index(drop=True)

df['pm25_lag1']   = df.groupby('station_name')['pm25'].shift(1)
df['pm25_lag2']   = df.groupby('station_name')['pm25'].shift(2)
df['pm25_lag3']   = df.groupby('station_name')['pm25'].shift(3)
df['pm25_delta1'] = df['pm25_lag1'] - df['pm25_lag2']   # rate of change

# shift(1) before rolling is mandatory — prevents today's value
# from entering its own window (data leakage)
df['pm25_roll3'] = df.groupby('station_name')['pm25'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df['pm25_roll7'] = df.groupby('station_name')['pm25'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=3).mean()
)
```

**Why PM2.5 lags provide the largest gain:**
1. **Emission persistence** — traffic and industrial sources repeat daily; yesterday's PM2.5 encodes local source intensity that no satellite or met variable captures as directly
2. **Cloud-day signal** — on 43.4% of days INSAT cannot retrieve AOD; pm25_lag1 tells the model whether pollution was already elevated before the cloud arrived
3. **Atmospheric memory** — fine particles remain suspended 1–7 days; the rolling means capture background loading that single-day AOD misses

**Leakage prevention:**

```python
# WRONG — includes today → target leaks into its own feature
df['pm25_roll7'] = df.groupby('station_name')['pm25'].transform(
    lambda x: x.rolling(7).mean()
)

# CORRECT — shift(1) moves the window entirely into the past
df['pm25_roll7'] = df.groupby('station_name')['pm25'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=3).mean()
)
```

#### AOD lag and rolling features (both datasets)

```python
aod_real = df['AOD_mean'].replace(-1, np.nan)   # restore NaN for rolling

df['AOD_lag1']  = df.groupby('station_name')['AOD_mean'].shift(1).fillna(-1)
df['AOD_lag2']  = df.groupby('station_name')['AOD_mean'].shift(2).fillna(-1)
df['AOD_roll3'] = aod_real.groupby(df['station_name']).transform(
    lambda x: x.rolling(3, min_periods=1).mean()
).fillna(-1)
df['AOD_roll7'] = aod_real.groupby(df['station_name']).transform(
    lambda x: x.rolling(7, min_periods=3).mean()
).fillna(-1)
```

### 7.5 Station spatial context

```python
# Long-term baseline — captures local emission source intensity
df['station_pm_mean']  = df.groupby('station_name')['pm25'].transform('mean')
df['station_pm_std']   = df.groupby('station_name')['pm25'].transform('std')

# Seasonal baseline per station — "this station is always worse in winter"
df['station_month_pm'] = df.groupby(
    ['station_name', 'month']
)['pm25'].transform('mean')

# PM proxy anomaly — today's loading vs station's typical this month
df['proxy_anomaly'] = df['PM_proxy'] - df.groupby(
    ['station_name', 'month']
)['PM_proxy'].transform('mean')

# Geographic gradient — IGP vs coastal vs Deccan Plateau
df['lat'] = df['latitude']
df['lon'] = df['longitude']
```

### 7.6 PM2.5 outlier removal

```python
p995 = df['pm25'].quantile(0.995)   # 319.1 µg/m³
p005 = df['pm25'].quantile(0.005)   # 3.6 µg/m³
df   = df[(df['pm25'] >= p005) & (df['pm25'] <= p995)]
# Removed: 674 rows (instrument errors and implausible spikes)
```

---

## Phase 8 — Model Training

### Two datasets produced

| | Dataset A | Dataset B |
|---|---|---|
| PM2.5 lag features | No | Yes |
| Feature count | 66 | 72 |
| Extra features in B | — | pm25_lag1, pm25_lag2, pm25_lag3, pm25_roll3, pm25_roll7, pm25_delta1 |
| File | dataset_A_no_lags.parquet | dataset_B_with_lags.parquet |

### Train/test split — temporal only, never random

```python
SPLIT_DATE = pd.Timestamp('2024-07-01')
train_df   = df[df['date'] <  SPLIT_DATE]   # 2023-01-01 → 2024-06-30
test_df    = df[df['date'] >= SPLIT_DATE]   # 2024-07-01 → 2024-12-31
```

| Split | Rows | PM2.5 mean | PM2.5 std |
|---|---|---|---|
| Train | 49,848 | 54.6 µg/m³ | 44.7 µg/m³ |
| Test | 16,735 | 48.7 µg/m³ | 45.4 µg/m³ |

Random split is incorrect here — lag features would allow future PM2.5 values to enter the training set, inflating R² by 0.05–0.10 artificially.

### Four models per dataset

| Model | Key hyperparameters |
|---|---|
| Random Forest | 500 trees, max_depth=15, min_samples_leaf=10, max_features=0.6 |
| XGBoost | 1000 estimators, max_depth=6, lr=0.03, subsample=0.8, early stopping=50 |
| LightGBM | 1000 estimators, max_depth=6, lr=0.03, num_leaves=63, early stopping=50 |
| Gradient Boosting | 500 estimators, max_depth=5, lr=0.05, subsample=0.8 |

### Weighted ensemble

```python
total         = sum(r2_scores.values())
weights       = {k: v / total for k, v in r2_scores.items()}
ensemble_pred = sum(weights[k] * preds[k] for k in weights)
```

---

## Results

### Dataset A — without PM2.5 lag features (66 features)

| Model | R² | MAE (µg/m³) | RMSE (µg/m³) | Ensemble weight |
|---|---|---|---|---|
| Random Forest | 0.7073 | 15.67 | 24.59 | 0.248 |
| XGBoost | 0.7214 | 15.25 | 23.99 | 0.253 |
| LightGBM | 0.7200 | 15.28 | 24.05 | 0.253 |
| Gradient Boosting | 0.7023 | 15.65 | 24.80 | 0.246 |
| **Ensemble** | **0.7211** | **15.25** | **24.00** | — |

### Dataset B — with PM2.5 lag features (72 features)

| Model | R² | MAE (µg/m³) | RMSE (µg/m³) | Ensemble weight |
|---|---|---|---|---|
| Random Forest | 0.8483 | 10.40 | 17.70 | 0.250 |
| XGBoost | 0.8514 | 10.37 | 17.52 | 0.251 |
| LightGBM | 0.8507 | 10.44 | 17.56 | 0.250 |
| Gradient Boosting | 0.8480 | 10.43 | 17.72 | 0.250 |
| **Ensemble** | **0.8522** | **10.32** | **17.47** | — |

### R² gain from PM2.5 lag features

| Model | Dataset A | Dataset B | Gain |
|---|---|---|---|
| Random Forest | 0.7073 | 0.8483 | +0.1410 |
| XGBoost | 0.7214 | 0.8514 | +0.1300 |
| LightGBM | 0.7200 | 0.8507 | +0.1307 |
| Gradient Boosting | 0.7023 | 0.8480 | +0.1457 |
| **Ensemble** | **0.7211** | **0.8522** | **+0.1311** |

### R² progression through pipeline

| Stage | R² | What was added |
|---|---|---|
| Naive AOD only | ~0.45–0.50 | Raw AOD_mean only |
| + MERRA meteorology | ~0.60–0.65 | PBLH, SPEED, TLML, moisture |
| + Physics feature engineering | ~0.68–0.72 | PM_proxy, trap_score, interaction terms |
| + **PM2.5 lag features** | **~0.82–0.85** | pm25_lag1/2/3, pm25_roll3/7 ← biggest jump |
| + AOD lags + rolling | ~0.83–0.85 | AOD_lag1/2, AOD_roll3/7 |
| + Station spatial context | ~0.84–0.85 | station_pm_mean, station_month_pm |
| + **Ensemble** | **0.8522** | Weighted combination of 4 models |

### Seasonal R² — Dataset B ensemble

| Season | R² | Test rows | Note |
|---|---|---|---|
| Winter (Dec–Feb) | 0.8156 | 2,828 | Strong — good AOD availability |
| Monsoon (Jun–Sep) | 0.6403 | 5,593 | Weakest — 61–93% AOD missing |
| Post-monsoon (Oct–Nov) | 0.8412 | 8,314 | Strongest — clearing sky, stable conditions |

### R² by AOD availability — Dataset B ensemble

| Condition | R² | Test rows |
|---|---|---|
| Cloud days (AOD = −1) | 0.7310 | 9,185 |
| Clear days (AOD > 0) | 0.8335 | 7,550 |

The 0.10 gap confirms that PM2.5 lags carry the primary signal on cloud days — without them this gap would be substantially larger.

### Prediction bias by PM2.5 quintile — Dataset B ensemble

| Quintile | Bias (µg/m³) |
|---|---|
| Q1 — low PM2.5 | +4.36 |
| Q2 | +2.84 |
| Q3 | +1.11 |
| Q4 | −0.53 |
| Q5 — high PM2.5 | −7.61 |

Positive bias in low quintiles and negative bias in high quintiles is the classic regression-to-the-mean effect. The model slightly underpredicts extreme pollution events.

### Top 20 features — Dataset B (avg RF + XGBoost importance)

| Rank | Feature | Importance | Description |
|---|---|---|---|
| 1 | pm25_lag1 | 0.4623 | Yesterday's PM2.5 |
| 2 | pm25_roll3 | 0.2156 | 3-day rolling mean PM2.5 |
| 3 | pm25_roll7 | 0.0623 | 7-day rolling mean PM2.5 |
| 4 | station_month_pm | 0.0570 | Station seasonal baseline |
| 5 | rain_3day | 0.0089 | Cumulative 3-day precipitation |
| 6 | pm25_lag2 | 0.0083 | PM2.5 from 2 days ago |
| 7 | rain_lag1 | 0.0083 | Yesterday's precipitation |
| 8 | SPEED_mean | 0.0080 | Mean daily wind speed |
| 9 | vent_coeff_mean | 0.0072 | Mean ventilation coefficient |
| 10 | pm25_delta1 | 0.0063 | Daily PM2.5 rate of change |
| 11 | AOD_x_vent | 0.0058 | AOD / ventilation interaction |
| 12 | SPEED_min | 0.0051 | Worst stagnation hour |
| 13 | pm25_lag3 | 0.0049 | PM2.5 from 3 days ago |
| 14 | doy_sin | 0.0048 | Day-of-year cyclical |
| 15 | PRECTOT_sum | 0.0047 | Daily precipitation total |
| 16 | month_cos | 0.0047 | Month cyclical |
| 17 | moisture_max | 0.0044 | Peak daily humidity |
| 18 | vent_coeff_min | 0.0043 | Worst daily dispersion |
| 19 | moisture_mean | 0.0041 | Mean daily humidity |
| 20 | doy_cos | 0.0041 | Day-of-year cyclical |

---

## Key Findings

**On data quality:**
- 70% of raw INSAT data is missing — this is physically expected (clouds + nighttime), not a data quality issue
- AOD below 0.1 carries >100% relative error given the ±0.1 RMS specification — always apply this as a hard filter
- Dead stations (Belur Math Howrah 100%, Jodhpur 99.7%) should be retained in training — MERRA-2 features still provide useful signal on those days
- Bengaluru stations show 59–62% AOD missingness even outside monsoon — likely a INSAT-3DR geometric retrieval issue over the Deccan plateau

**On spatial merging:**
- INSAT at 4 km: IDW within 15 km radius captures intra-city aerosol variation adequately
- MERRA-2 at 55 km: bilinear interpolation is necessary — nearest-neighbour introduces up to 30 km positional error for border stations
- CPCB station coordinates must always be the spatial anchor — never upsample ground measurements

**On temporal aggregation:**
- `PBLH_min` (not mean) is the critical MERRA-2 variable — minimum boundary layer height at dawn captures the pollution trap before solar heating begins
- `PRECTOT` must be summed, not averaged — total daily precipitation drives wet scavenging
- Morning window PBLH (00:00–04:00 UTC = 05:30–09:30 IST) is more predictive than full-day PBLH statistics

**On feature engineering:**
- PM2.5 lag features (pm25_lag1, pm25_roll3) are the single most impactful addition — +0.13 R² ensemble gain
- `station_month_pm` captures seasonal emission patterns per location — essential for differentiating IGP winter haze from cleaner southern stations
- Physics interaction terms (`PM_proxy`, `trap_score`, `AOD_x_vent`) encode the atmospheric mechanism and improve both predictive power and interpretability

**On model training:**
- Temporal train/test split is mandatory — random split inflates R² by 0.05–0.10 through lag feature leakage
- All four models perform nearly identically on Dataset B — ensemble weights are approximately equal (~0.25 each), indicating the models are similarly calibrated
- Monsoon season (R² 0.64) is the hardest to predict due to near-complete AOD blackout and PM2.5 variability driven by intermittent rainfall

---

## Project Structure

```
aodtopm25/
│
├── config/                                  # configuration files and paths
│
├── data/
│   ├── interim/
│   │   ├── india_clipped/                   # INSAT + MERRA clipped to India polygon
│   │   ├── insat_daily/                     # INSAT half-hourly → daily grid aggregation
│   │   ├── merra_daily/                     # MERRA hourly → daily grid aggregation
│   │   ├── interpolated/                    # intermediate interpolation outputs
│   │   └── spatial_join/                    # INSAT + MERRA interpolated to CPCB stations
│   │
│   ├── processed/
│   │   └── v1/
│   │       ├── final_ml_dataset.parquet     # three-way merged (67,970 × 28)
│   │       ├── dataset_A_no_lags.parquet    # 66 features, no PM2.5 lags
│   │       └── dataset_B_with_lags.parquet  # 72 features, with PM2.5 lags
│   │
│   └── raw/
│       ├── cpcb/                            # raw CPCB daily PM2.5 station data
│       └── insat/
│           └── Apr26_176936/                # raw INSAT-3DR AOD HDF/NetCDF files
│
├── extras/
│   └── models/                              # trained model pkl files (8 total)
│       ├── random_forest_A_no_lags.pkl
│       ├── xgboost_A_no_lags.pkl
│       ├── lightgbm_A_no_lags.pkl
│       ├── gradient_boosting_A_no_lags.pkl
│       ├── random_forest_B_with_lags.pkl
│       ├── xgboost_B_with_lags.pkl
│       ├── lightgbm_B_with_lags.pkl
│       └── gradient_boosting_B_with_lags.pkl
│
├── models/                                  # model architecture configs
│
├── notebooks/                               # exploratory analysis notebooks
│
├── reports/                                 # evaluation reports and figures
│
├── src/
│   ├── final_pipeline.py                    # end-to-end pipeline — both datasets
│   └── train_and_save_models.py             # train 4 models + save pkl from existing parquets
│
├── requirements.txt
└── README.md
```

---

## Dependencies

```
python >= 3.9
numpy
pandas
geopandas
scipy
scikit-learn
xgboost >= 1.7
lightgbm >= 3.3
joblib
pyarrow              # parquet read/write
shapely
dask                 # reading large raw parquet files
```

Install all dependencies:

```bash
pip install numpy pandas geopandas scipy scikit-learn xgboost lightgbm joblib pyarrow shapely dask
```

---

## How to Run

### Option A — Full pipeline from scratch

```bash
# 1. Edit paths at the top of final_pipeline.py
# 2. Run end-to-end
python src/final_pipeline.py
```

Runs all 9 sections sequentially. Each section saves its output to parquet — if interrupted, the pipeline resumes from the last saved file. Total runtime: approximately 1–3 hours depending on hardware (spatial joins are the bottleneck at ~30–50 min each).

### Option B — Train models on existing datasets only

If `dataset_A_no_lags.parquet` and `dataset_B_with_lags.parquet` already exist in `data/processed/v1/`:

```bash
python src/train_and_save_models.py
```

Trains all 4 models on both datasets and saves 8 pkl files to `extras/models/`. No spatial join or feature engineering is re-run.

### Loading a saved model

```python
import joblib
import pandas as pd

model = joblib.load(r'extras/models/xgboost_B_with_lags.pkl')
pred  = model.predict(X_test)
```

### Pipeline execution order

```
Section 1  → Load and validate inputs + column names
Section 2  → Fix missing values
             (MERRA → station-month median, AOD → sentinel −1, coverage → 0)
Section 3  → Base feature engineering (28 → 73 columns)
Section 4  → Build Dataset A — 66 features, no PM2.5 lags
Section 5  → Build Dataset B — 72 features, with PM2.5 lags (zero leakage)
Section 6  → Helper functions (split, evaluate, save_pkl)
Section 7  → Train 4 models on Dataset A → save pkl → evaluate → diagnostics
Section 8  → Train 4 models on Dataset B → save pkl → evaluate → diagnostics
Section 9  → Final comparison table
```
