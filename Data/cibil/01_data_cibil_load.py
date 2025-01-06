# %%

import polars as pl

# %%

# read External_Cibil_Dataset.xlsx and then convert to parquet

external_df = pl.read_excel('External_Cibil_Dataset.xlsx')

external_df.write_parquet('External_Cibil_Dataset.parquet')

# %%

# read Internal_Bank_Dataset.xlsx and then convert to parquet

internal_df = pl.read_excel('Internal_Bank_Dataset.xlsx')

internal_df.write_parquet('Internal_Bank_Dataset.parquet')

# %%

# read Unseen_Dataset.xlsx and then convert to parquet

unseen_df = pl.read_excel('Unseen_Dataset.xlsx')

unseen_df.write_parquet('Unseen_Dataset.parquet')