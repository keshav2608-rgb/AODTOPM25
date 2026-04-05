import numpy as np
import pandas as pd
import warnings
import time
import os
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

# ==============================
# FILE PATHS
# ==============================
merra_file = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\merra_daily\merra_daily_grid.parquet"
cpcb_file  = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\raw\cpcb\merged.csv"
output_file = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\spatial_join\merra_station_daily.parquet"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
merra = pd.read_parquet(merra_file)
cpcb  = pd.read_csv(cpcb_file)

merra['date'] = pd.to_datetime(merra['date']).dt.date
cpcb['date']  = pd.to_datetime(cpcb['date']).dt.date

stations = (
    cpcb[['station_name', 'latitude', 'longitude']]
    .drop_duplicates(subset=['station_name'])
    .reset_index(drop=True)
)

print("Stations:", len(stations))
print("MERRA shape:", merra.shape)

LAT_COL = 'lat_round'
LON_COL = 'lon_round'

feat_cols = [c for c in merra.columns if c not in [LAT_COL, LON_COL, 'date']]
print("Features:", feat_cols)

# ==============================
# INTERPOLATION
# ==============================
results = []
grouped = merra.groupby('date')
n_dates = len(grouped)
t0 = time.time()

for i, (date, day_df) in enumerate(grouped):

    if i % 50 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (n_dates - i) if i > 0 else 0
        print(f"{i}/{n_dates} | {elapsed/60:.1f} min | ETA {eta/60:.1f} min")

    pts = day_df[[LAT_COL, LON_COL]].values
    tree = cKDTree(pts)

    # Build interpolator once per day per feature
    interpolators = {}
    for col in feat_cols:
        vals = day_df[col].values.astype(float)
        if np.isnan(vals).all():
            interpolators[col] = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interpolators[col] = LinearNDInterpolator(pts, vals)

    # Loop stations
    for _, stn in stations.iterrows():
        row = {'station_name': stn['station_name'], 'date': date}
        target = np.array([stn['latitude'], stn['longitude']])

        for col in feat_cols:
            interp = interpolators[col]

            if interp is None:
                row[col] = np.nan
                continue

            val = interp(target)

            # Outside convex hull → nearest neighbour
            if val is None or len(val) == 0:
               val = np.nan
            else:
                    val = val[0]

            if np.isnan(val):
               _, idx = tree.query(target)
               val = day_df[col].values[idx]

            row[col] = float(val)

        results.append(row)

# ==============================
# SAVE
# ==============================
merra_station = pd.DataFrame(results)
merra_station.to_parquet(output_file, index=False)

print("\nSaved:", output_file)
print("Shape:", merra_station.shape)
print("\nMissing %:\n", merra_station.isna().mean() * 100)