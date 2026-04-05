import pandas as pd
import os

# Load MERRA data
file_path = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\india_clipped\merra_features.parquet"
merra_df = pd.read_parquet(file_path)

# Create output folder (if not exists)
output_dir = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\merra_daily"
os.makedirs(output_dir, exist_ok=True)

merra_df['date'] = merra_df['datetime'].dt.date
merra_df['lat_round'] = merra_df['latitude'].round(2)
merra_df['lon_round'] = merra_df['longitude'].round(2)

agg_rules = {
    'PBLH'          : ['mean', 'min', 'max'],
    'TLML'          : ['mean', 'max'],
    'moisture'      : ['mean', 'max'],
    'SPEED'         : ['mean', 'min'],
    'PRECTOT'       : ['sum'],
    'vent_coeff'    : ['mean', 'min'],
    'inversion_proxy': ['mean', 'max'],
    'wind_dir'      : ['mean'],
}

merra_daily = merra_df.groupby(
    ['lat_round', 'lon_round', 'date']
).agg(agg_rules).reset_index()

merra_daily.columns = [
    '_'.join(col).strip('_') 
    for col in merra_daily.columns.values
]

morning = merra_df[merra_df['hour'].between(0, 4)]

morning_pblh = morning.groupby(
    ['lat_round', 'lon_round', 'date']
)['PBLH'].agg(['mean', 'min']).reset_index()

morning_pblh.columns = [
    'lat_round', 'lon_round', 'date',
    'PBLH_morning_mean', 'PBLH_morning_min'
]

merra_daily = merra_daily.merge(
    morning_pblh, on=['lat_round', 'lon_round', 'date'], how='left'
)

# Save file
output_path = os.path.join(output_dir, "merra_daily_grid.parquet")
merra_daily.to_parquet(output_path, index=False)

print("Saved at:", output_path)