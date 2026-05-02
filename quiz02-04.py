from PyALE._src.lib import quantile_ied

import pandas as pd

import numpy as np

grid_size = 100

df_ale = train_data.sample(40_000, random_state=2025)

quantiles = np.linspace(0, 1, grid_size + 1, endpoint=True)

cut_points = [df_ale.CC_utilization.min()] + quantile_ied(df_ale.CC_utilization, quantiles).to_list()

df_ale['binned_CC_utilization'] = pd.cut(df_ale.CC_utilization, cut_points, include_lowest=True, duplicates='drop')

df_ale.binned_CC_utilization.value_counts().sort_index()