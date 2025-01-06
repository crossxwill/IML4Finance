# %% 

import pandas as pd
import os
import glob

# %%

# Get list of all pkl files in simulated-data folder
pkl_files = glob.glob('simulated-data/*.pkl')

# Process each file
for pkl_file in pkl_files:
    # Read the pkl file
    df = pd.read_pickle(pkl_file)
    
    # Create output filename by replacing .pkl with .parquet
    parquet_file = pkl_file.replace('.pkl', '.parquet')
    
    # Save as parquet
    df.to_parquet(parquet_file)