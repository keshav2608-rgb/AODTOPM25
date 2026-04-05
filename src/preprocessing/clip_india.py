import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# File paths
insat_file = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\insat_aod_merged.parquet"
merra_file = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\merra_merged.parquet"

# Output files
insat_out = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\india_clipped/insat_india.parquet"
merra_out = r"C:\Users\kesha\OneDrive - dtu.ac.in\Desktop\aodtopm25\data\interim\india_clipped/merra_india.parquet"

# Load India boundary
india = gpd.read_file("india_boundary.geo.json").to_crs("EPSG:4326")
india_buffered = india.buffer(0.5)

def process_parquet(input_file, output_file):
    pf = pq.ParquetFile(input_file)
    writer = None

    for i in range(pf.num_row_groups):
        print(f"Processing row group {i+1}/{pf.num_row_groups}")

        # Read chunk
        df = pf.read_row_group(i).to_pandas()

        # Step 1 — Bounding box filter (FAST)
        df = df[
            (df["latitude"].between(6.0, 38.0)) &
            (df["longitude"].between(68.0, 98.0))
        ]

        if len(df) == 0:
            continue

        # Step 2 — Polygon clip (ACCURATE)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )

        gdf = gpd.clip(gdf, india_buffered)
        df = gdf.drop(columns="geometry")

        if len(df) == 0:
            continue

        # Step 3 — Write chunk to parquet
        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)

        writer.write_table(table)

    if writer:
        writer.close()

    print("Saved:", output_file)


# Run for both datasets
process_parquet(insat_file, insat_out)
process_parquet(merra_file, merra_out)