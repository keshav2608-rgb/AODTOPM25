import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os

# =========================
# FILE PATHS
# =========================
CPCB_FILE = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\raw\cpcb\merged.csv"
INSAT_FILE = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\insat_daily\insatdaily_grid.parquet"
OUTPUT_FILE = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\spatial_join\insat_station_daily.parquet"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# =========================
# LOAD CPCB DATA
# =========================
cpcb_df = pd.read_csv(CPCB_FILE)
cpcb_df['date'] = pd.to_datetime(cpcb_df['date']).dt.date
cpcb_df['latitude'] = pd.to_numeric(cpcb_df['latitude'], errors='coerce')
cpcb_df['longitude'] = pd.to_numeric(cpcb_df['longitude'], errors='coerce')
cpcb_df = cpcb_df.dropna(subset=['latitude', 'longitude'])
stations = cpcb_df[['station_name', 'latitude', 'longitude']].drop_duplicates()

print(f"Total CPCB stations : {len(stations)}")
print(f"Total CPCB rows     : {len(cpcb_df)}")
print(f"Date range          : {cpcb_df['date'].min()} to {cpcb_df['date'].max()}")
print(f"PM25 missing %      : {cpcb_df['pm25'].isna().mean()*100:.1f}%")

# This tells expected merged size
print(f"\nExpected merged rows: ~{len(cpcb_df):,}")

# =========================
# LOAD INSAT DATA
# =========================
insat_daily = pd.read_parquet(INSAT_FILE)
insat_daily['date'] = pd.to_datetime(insat_daily['date']).dt.date

# =========================
# SPATIAL MATCHING
# =========================
results_insat = []

grouped = insat_daily.groupby('date')
total_dates = len(grouped)

for i, (date, insat_day) in enumerate(grouped):
    if i % 100 == 0:
        print(f"Processing {i}/{total_dates} dates...")

    insat_coords = insat_day[['lat_round', 'lon_round']].values
    tree = cKDTree(insat_coords)

    for _, station in stations.iterrows():
        station_pt = [[station['latitude'], station['longitude']]]

        # 0.135 degrees ≈ 15km
        indices = tree.query_ball_point(station_pt, r=0.135)[0]

        if len(indices) == 0:
            results_insat.append({
                'station_name': station['station_name'],
                'date'          : date,
                'AOD_mean'      : np.nan,
                'AOD_max'       : np.nan,
                'AOD_p75'       : np.nan,
                'AOD_count'     : 0,
                'AOD_coverage': 0.0,
                'n_pixels'    : 0
            })
            continue

        nearby = insat_day.iloc[indices].copy()

        # Inverse distance weighting
        dists = np.sqrt(
            (nearby['lat_round'].values - station['latitude'])**2 +
            (nearby['lon_round'].values - station['longitude'])**2
        )
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()

        row = {
            'station_name': station['station_name'],
            'date'          : date,
            'n_pixels'    : len(indices)
        }

        for col in ['AOD_mean', 'AOD_max', 'AOD_p75']:
            valid = nearby[col].notna()
            if valid.any():
                w = weights[valid.values]
                w /= w.sum()
                row[col] = np.average(nearby.loc[valid.index, col], weights=w)
            else:
                row[col] = np.nan

        row['AOD_count']    = nearby['AOD_count'].max()
        row['AOD_coverage'] = nearby['AOD_coverage'].max()

        results_insat.append(row)

# =========================
# SAVE OUTPUT
# =========================
insat_station = pd.DataFrame(results_insat)
insat_station.to_parquet(OUTPUT_FILE, index=False)

print(f"\nINSAT station shape: {insat_station.shape}")
print(f"Saved to: {OUTPUT_FILE}")