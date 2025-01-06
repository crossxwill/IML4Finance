# %%
# The script randomly samples high LTV loans from 
# the Fannie Mae dataset and exports the sampled loans 
# to a parquet file

import duckdb
import polars as pl
import zipfile
from typing import Dict
import tempfile, os

# Sample unique loan IDs

sampled_loans = duckdb.sql("""
    WITH CTE_UNIQUE_LOANS AS (
        SELECT DISTINCT loan_id 
        FROM read_parquet('parquet-fanniemae-SFLP/*.parquet')
        WHERE orig_ltv > 80
        ORDER BY loan_id
    ),
    CTE_SAMPLED AS (
        SELECT loan_id
        FROM CTE_UNIQUE_LOANS
        USING SAMPLE bernoulli(1 PERCENT)
        REPEATABLE(42)
    )
    SELECT A.*
    FROM read_parquet('parquet-fanniemae-SFLP/*.parquet') AS A
    WHERE A.loan_id IN (SELECT loan_id FROM CTE_SAMPLED)
    ORDER BY A.loan_id, A.monthly_rpt_period
""").pl()

# Display the first few rows

display(sampled_loans)
# %%

# export sampled_loans to parquet

sampled_loans.write_parquet('tbl_fanniemae_SFLP_sample.parquet')

# %%
