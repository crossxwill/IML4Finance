# %%
# The script reads the full and sampled Fannie Mae datasets
# and calculates the average current interest rate,
# average current unpaid balance, and delinquency rate
# for each report period. The script then concatenates the
# full and sampled datasets, converts the report period to
# a date, and plots the delinquency rate over time.

import duckdb
import polars as pl
import zipfile
from typing import Dict
import tempfile, os
import plotnine as p9

# %%
agg_full = duckdb.sql("""
    SELECT 'FULL' AS SRC,
        monthly_rpt_period,
        avg(curr_int_rate) as avg_curr_int_rate,
        avg(curr_act_upb) as avg_curr_upb,
        SUM(CASE 
            WHEN curr_loan_delinq_status <> '00'
            THEN 1 
            ELSE 0 END) / COUNT(loan_id) AS DQ_RATE
    FROM read_parquet('parquet-fanniemae-SFLP/*.parquet')
    WHERE orig_ltv > 80
    GROUP BY monthly_rpt_period
    ORDER BY monthly_rpt_period
""").pl()

# %%

agg_sample = duckdb.sql("""
    SELECT 'SAMPLE' AS SRC,
        monthly_rpt_period,
        avg(curr_int_rate) as avg_curr_int_rate,
        avg(curr_act_upb) as avg_curr_upb,
        SUM(CASE 
            WHEN curr_loan_delinq_status <> '00'
            THEN 1 
            ELSE 0 END) / COUNT(loan_id) AS DQ_RATE
    FROM read_parquet('tbl_fanniemae_SFLP_sample.parquet')
    GROUP BY monthly_rpt_period
    ORDER BY monthly_rpt_period
""").pl()                        

# %%

display(agg_full.describe())

display(agg_sample.describe())
# %%
agg_concat = pl.concat([agg_full, agg_sample])

# %%
agg_concat = (agg_concat
    .with_columns(
        pl.col('monthly_rpt_period')
        .str.strptime(pl.Date, format='%m%Y')
        .dt.month_end()
        .alias('dt_monthly_rpt_period')
    )
    .with_columns(
        DQ_RATE = pl.col('DQ_RATE') * 100 
    )
    .sort(['SRC', 'dt_monthly_rpt_period'])
    )

# %%
(p9.ggplot(agg_concat, 
    p9.aes(x='dt_monthly_rpt_period', y='DQ_RATE', color='SRC')) +
    p9.geom_line() +
    p9.labs(x='Report Period', y='Delinquency Rate (%)', color='Data Source') +
    p9.theme_minimal()
)
# %%
