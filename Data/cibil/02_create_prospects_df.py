# %%

import polars as pl
import numpy as np

np.random.seed(42)

df_external = pl.read_parquet("External_Cibil_Dataset.parquet")
df_internal = pl.read_parquet("Internal_Bank_Dataset.parquet") 

df_prospects_orig = df_external.join(df_internal, on="PROSPECTID")

remove_cols = ['Approved_Flag']

df_prospects_orig = df_prospects_orig.drop(remove_cols)

# %%

df_prospects_CC = df_prospects_orig.filter(pl.col('CC_utilization') > 0)

df_prospects_CC_expanded = pl.concat([df_prospects_CC] * 20, how="vertical")

df_prospects_baseline = pl.concat([df_prospects_orig] * 10, how="vertical")

df_prospects = df_prospects_CC_expanded.with_columns(
    CC_utilization = pl.col('CC_utilization') + pl.Series(np.random.normal(0, 0.01, df_prospects_CC_expanded.height))
)

df_prospects = pl.concat([df_prospects_CC_expanded, df_prospects_baseline], how="vertical")

df_prospects = df_prospects.with_columns(
    CC_utilization = pl.col('CC_utilization').clip(None, 1.0)
)

np.random.seed(42)

df_prospects = df_prospects.sample(fraction=1.0, shuffle=True, seed=42)

df_prospects = df_prospects.with_columns(
    PROSPECTID = pl.arange(1, df_prospects.height + 1).cast(pl.Utf8)
)

# %%
np_prospects = df_prospects.get_column('PROSPECTID').to_numpy()

np_cc_util = df_prospects.get_column('CC_utilization').to_numpy() 
np_pl_util = df_prospects.get_column('PL_utilization').to_numpy()

np_last_inq_cc_pl_flag = np.where(df_prospects.get_column('last_prod_enq2').is_in(['CC', 'PL']), 1, 0)

np_PL_inq_L6m = df_prospects.get_column('PL_enq_L6m').to_numpy()
np_CC_inq_L6m = df_prospects.get_column('CC_enq_L6m').to_numpy()

# %%
df_campaign_history = pl.DataFrame({
    'PROSPECTID': np_prospects,
    'campaign_id': np.random.choice(['1', '2', '3'], size=df_prospects.height, p=[0.66667, 0.16667, 0.16666]),
    'log_odds': -3.5 + 0.85 * np.maximum(0, np_cc_util)
                     + 0.20 * np.maximum(0, np_cc_util - 0.5)
                     + 0.25 * np.maximum(0, np_cc_util - 0.6)
                     + 0.30 * np.maximum(0, np_cc_util - 0.7)
                     + 0.35 * np.maximum(0, np_cc_util - 0.8)
                     + 0.40 * np.maximum(0, np_cc_util - 0.9)

                     + 0.50 * np.maximum(0, np_pl_util)
                     + 0.15 * np.maximum(0, np_pl_util - 0.5)
                     + 0.20 * np.maximum(0, np_pl_util - 0.6)
                     + 0.25 * np.maximum(0, np_pl_util - 0.7)
                     + 0.30 * np.maximum(0, np_pl_util - 0.8)
                     + 0.35 * np.maximum(0, np_pl_util - 0.9)

                     + 0.50 * np.maximum(0, np_cc_util) * np.maximum(0, np_pl_util)

                     + 0.10 * np_last_inq_cc_pl_flag 
                     + 0.10 * np.maximum(0, np_CC_inq_L6m)
                     + 0.10 * np.maximum(0, np_PL_inq_L6m)                     
                     + 0.10 * np.maximum(0, np_PL_inq_L6m) * np.maximum(0, np_CC_inq_L6m)
})

df_campaign_history = df_campaign_history.with_columns(
    prob = 1 / (1 + (-pl.col('log_odds')).exp())
)

df_campaign_history = df_campaign_history.with_columns(
    response_flag = pl.Series(np.random.binomial(n=1, p=df_campaign_history.get_column('prob').to_numpy()))
)

# %%
keep_cols = ['PROSPECTID', 'campaign_id', 'response_flag']

df_campaign_history = df_campaign_history.select(keep_cols)
# %%

df_campaign_eval = (df_campaign_history
                    .filter(pl.col('campaign_id') == '3')
                    .sort(pl.col(['PROSPECTID', 'campaign_id']))
                    )
# %%

df_campaign_history = df_campaign_history.with_columns(
    direct_mail_flag = pl.when(pl.col('campaign_id').is_in(['1', '2']))
    .then(pl.lit('Y'))
    .otherwise(None),
        
    response_flag = pl.when(pl.col('campaign_id') == '3')
    .then(None)
    .otherwise(pl.col('response_flag'))
)
# %%

df_campaign_eval.write_parquet("campaign_eval.parquet")
df_campaign_history.write_parquet("campaign_history.parquet") 
df_prospects.write_parquet("prospects.parquet")
# %%
