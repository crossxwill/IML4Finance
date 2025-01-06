# %%

import polars as pl

# %%
# convert accepted_2007_to_2018Q4.csv to parquet

df_lc_approved = pl.read_csv("accepted_2007_to_2018Q4.csv", infer_schema_length=1_000_000)

df_lc_approved.write_parquet("accepted_2007_to_2018Q4.parquet")

# %%

# convert rejected_2007_to_2018Q4.csv to parquet

df_lc_rejected = pl.read_csv("rejected_2007_to_2018Q4.csv", infer_schema_length=1_000_000)

df_lc_rejected.write_parquet("rejected_2007_to_2018Q4.parquet")

# %%