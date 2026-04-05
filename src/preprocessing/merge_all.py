import pandas as pd
import os

# ==============================
# LOAD FILES
# ==============================
cpcb = pd.read_csv(
    r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\raw\cpcb\merged.csv"
)

insat_station = pd.read_parquet(
    r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\spatial_join\insat_station_daily.parquet"
)

merra_station = pd.read_parquet(
    r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\spatial_join\merra_station_daily.parquet"
)

# ==============================
# DATE FORMAT
# ==============================
for d in [cpcb, insat_station, merra_station]:
    d['date'] = pd.to_datetime(d['date']).dt.date



# ==============================
# MERGE USING station_name
# ==============================
df = (
    cpcb
    .merge(insat_station, on=['station_name', 'date'], how='left')
    .merge(merra_station, on=['station_name', 'date'], how='left')
)

print("Merged shape:", df.shape)
print("PM25 missing %:", df['pm25'].isna().mean() * 100)
print("AOD missing %:", df['AOD_mean'].isna().mean() * 100)
print("PBLH missing %:", df['PBLH_mean'].isna().mean() * 100)

# ==============================
# SAVE FINAL DATASET
# ==============================
output_file = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\processed\final_ml_dataset.parquet"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df.to_parquet(output_file, index=False)

print("Saved final dataset:", output_file)