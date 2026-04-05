import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os

# ==============================
# FILE PATHS
# ==============================
input_file = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\india_clipped\insat_india_clean.parquet"
output_file = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\insat_daily\insat_daily_grid.parquet"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

pf = pq.ParquetFile(input_file)

chunk_results = []

# ==============================
# STEP 1 — CHUNK AGGREGATION
# ==============================
for i in range(pf.num_row_groups):
    print(f"Processing row group {i+1}/{pf.num_row_groups}")
    
    df = pf.read_row_group(i).to_pandas()
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    
    df["lat_round"] = df["latitude"].round(3)
    df["lon_round"] = df["longitude"].round(3)

    # Aggregate per chunk
    agg = df.groupby(
        ['lat_round', 'lon_round', 'date']
    ).agg(
        AOD_mean=('AOD', 'mean'),
        AOD_max=('AOD', 'max'),
        AOD_std=('AOD', 'std'),
        AOD_p75=('AOD', lambda x: x.quantile(0.75)),
        AOD_count=('AOD', 'count')
    ).reset_index()

    chunk_results.append(agg)

# ==============================
# STEP 2 — FINAL AGGREGATION
# ==============================
print("Merging chunk results...")

all_chunks = pd.concat(chunk_results, ignore_index=True)

insat_daily = all_chunks.groupby(
    ['lat_round', 'lon_round', 'date']
).agg(
    AOD_mean=('AOD_mean', 'mean'),
    AOD_max=('AOD_max', 'max'),
    AOD_std=('AOD_std', 'mean'),
    AOD_p75=('AOD_p75', 'mean'),
    AOD_count=('AOD_count', 'sum')
).reset_index()

# ==============================
# STEP 3 — COVERAGE FILTER
# ==============================
insat_daily['AOD_coverage'] = insat_daily['AOD_count'] / 7.0

insat_daily = insat_daily[insat_daily['AOD_count'] >= 2]

# ==============================
# SAVE FINAL FILE
# ==============================
insat_daily.to_parquet(output_file, index=False)

print("Saved:", output_file)
print("Final shape:", insat_daily.shape)