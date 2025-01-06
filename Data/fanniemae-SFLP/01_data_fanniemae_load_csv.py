# %%
# The script extracts CSV files from a zip file
# and loads them into parquet files using Polars.

import duckdb
import polars as pl
import zipfile
from typing import Dict
import tempfile, os

# %%
def load_csv_from_zip(zip_path: str,
                    column_names: Dict[str, str],
                    output_folder_path: str,
                    output_file_prefix: str) -> pl.LazyFrame:
    """
    Load CSV files from a zip file into Parquet files using Polars.
    
    Args:
        zip_path: Path to the zip file containing CSVs
        column_names: Dictionary where keys are column names and values 
            are DuckDB data types
        output_folder_path: Path to store the Parquet files
        output_file_prefix: Name prefix for the Parquet files
    """
    
    # Convert DuckDB types to Polars types
    type_mapping = {
        'VARCHAR': pl.Utf8,
        'INTEGER': pl.Int32,
        'DOUBLE': pl.Float64
    }
    
    schema = {col: type_mapping[dtype] for col, dtype in column_names.items()}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Process zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.csv'):
                # Extract CSV file from zip to disk
                zip_ref.extract(file_name)
                    
                df = pl.scan_csv(
                    file_name,
                    has_header=False,
                    new_columns=list(column_names.keys()),
                    separator='|',
                    schema_overrides=schema,
                    low_memory=True
                )
                
                # Write to Parquet with compression
                output_path = os.path.join(output_folder_path,
                                        f"{output_file_prefix}_{file_name[:-4]}.parquet")
                
                df.sink_parquet(
                    output_path
                )
                
                os.remove(file_name)
                    
    return pl.scan_parquet(output_folder_path)


# %%

column_names = {
    'ref_pool_id': 'VARCHAR',
    'loan_id': 'VARCHAR',
    'monthly_rpt_period': 'VARCHAR',
    'channel': 'VARCHAR',
    'seller_name': 'VARCHAR',
    'servicer_name': 'VARCHAR',
    'master_servicer': 'VARCHAR',
    'orig_int_rate': 'DOUBLE',
    'curr_int_rate': 'DOUBLE',
    'orig_upb': 'DOUBLE',
    'upb_issuance': 'DOUBLE',
    'curr_act_upb': 'DOUBLE',
    'orig_loan_term': 'INTEGER',
    'orig_date': 'VARCHAR',
    'first_pmt_date': 'VARCHAR',
    'loan_age': 'INTEGER',
    'rem_mths_legal_mat': 'INTEGER',
    'rem_mths_mat': 'INTEGER',
    'mat_date': 'VARCHAR',
    'orig_ltv': 'DOUBLE',
    'orig_cltv': 'DOUBLE',
    'num_borrowers': 'INTEGER',
    'dti': 'DOUBLE',
    'borr_credit_score_orig': 'INTEGER',
    'co_borr_credit_score_orig': 'INTEGER',
    'first_time_home_buyer': 'VARCHAR',
    'loan_purpose': 'VARCHAR',
    'prop_type': 'VARCHAR',
    'num_units': 'INTEGER',
    'occupancy_status': 'VARCHAR',
    'prop_state': 'VARCHAR',
    'msa': 'VARCHAR',
    'zip_code_short': 'VARCHAR',
    'mortgage_ins_pct': 'DOUBLE',
    'amort_type': 'VARCHAR',
    'prepay_penalty_ind': 'VARCHAR',
    'int_only_loan_ind': 'VARCHAR',
    'int_only_first_pmt_date': 'VARCHAR',
    'mths_to_amort': 'INTEGER',
    'curr_loan_delinq_status': 'VARCHAR',
    'loan_pmt_history': 'VARCHAR',
    'mod_flag': 'VARCHAR',
    'mortgage_ins_cancel_ind': 'VARCHAR',
    'zero_balance_cd': 'VARCHAR',
    'zero_balance_eff_date': 'VARCHAR',
    'upb_removal': 'DOUBLE',
    'repurchase_date': 'VARCHAR',
    'sched_prin_curr': 'DOUBLE',
    'total_prin_curr': 'DOUBLE',
    'unsched_prin_curr': 'DOUBLE',
    'last_paid_inst_date': 'VARCHAR',
    'foreclosure_date': 'VARCHAR',
    'disposition_date': 'VARCHAR',
    'foreclosure_costs': 'DOUBLE',
    'prop_pres_repair_costs': 'DOUBLE',
    'asset_recovery_costs': 'DOUBLE',
    'misc_holding_exp_credits': 'DOUBLE',
    'assoc_taxes_holding_prop': 'DOUBLE',
    'net_sales_proceeds': 'DOUBLE',
    'credit_enhance_proceeds': 'DOUBLE',
    'repurchase_make_whole_proceeds': 'DOUBLE',
    'other_foreclosure_proceeds': 'DOUBLE',
    'mod_non_int_bearing_upb': 'DOUBLE',
    'prin_forgiveness_amt': 'DOUBLE',
    'orig_list_start_date': 'VARCHAR',
    'orig_list_price': 'DOUBLE',
    'curr_list_start_date': 'VARCHAR',
    'curr_list_price': 'DOUBLE',
    'borr_credit_score_issuance': 'INTEGER',
    'co_borr_credit_score_issuance': 'INTEGER',
    'borr_credit_score_curr': 'INTEGER',
    'co_borr_credit_score_curr': 'INTEGER',
    'mortgage_ins_type': 'VARCHAR',
    'serv_activity_ind': 'VARCHAR',
    'curr_mod_loss_amt': 'DOUBLE',
    'cumulative_mod_loss_amt': 'DOUBLE',
    'curr_credit_event_net_gain_loss': 'DOUBLE',
    'cumulative_credit_event_net_gain_loss': 'DOUBLE',
    'special_eligibility_prog': 'VARCHAR',
    'foreclosure_prin_writeoff_amt': 'DOUBLE',
    'relocation_mortgage_ind': 'VARCHAR',
    'zero_balance_cd_chg_date': 'VARCHAR',
    'loan_holdback_ind': 'VARCHAR',
    'loan_holdback_eff_date': 'VARCHAR',
    'delinquent_accrued_int': 'DOUBLE',
    'prop_val_method': 'VARCHAR',
    'high_balance_loan_ind': 'VARCHAR',
    'arm_init_fixed_rate_period_ind': 'VARCHAR',
    'arm_product_type': 'VARCHAR',
    'init_fixed_rate_period': 'INTEGER',
    'int_rate_adj_freq': 'INTEGER',
    'next_int_rate_adj_date': 'VARCHAR',
    'next_pmt_chg_date': 'VARCHAR',
    'rate_index': 'VARCHAR',
    'arm_cap_structure': 'VARCHAR',
    'init_int_rate_cap_up_pct': 'DOUBLE',
    'periodic_int_rate_cap_up_pct': 'DOUBLE',
    'lifetime_int_rate_cap_up_pct': 'DOUBLE',
    'mortgage_margin': 'DOUBLE',
    'arm_balloon_ind': 'VARCHAR',
    'arm_plan_num': 'INTEGER',
    'borr_assist_plan': 'VARCHAR',
    'high_ltv_refi_opt_ind': 'VARCHAR',
    'deal_name': 'VARCHAR',
    'repurchase_make_whole_proceeds_flag': 'VARCHAR',
    'alt_delinq_resolution': 'VARCHAR',
    'alt_delinq_resolution_count': 'INTEGER',
    'total_deferral_amt': 'DOUBLE',
    'pmt_deferral_mod_event_ind': 'VARCHAR',
    'int_bearing_upb': 'DOUBLE'
}

# %%

## WARNING: 
#     load_csv_from_zip() takes 7 hours 
#     to run on a 14-core machine with 32GB RAM

big_df = load_csv_from_zip(
    zip_path = 'Performance_All.zip', 
    column_names = column_names,
    output_folder_path = 'parquet-fanniemae-SFLP',
    output_file_prefix = 'tbl_fanniemae_SFLP'
)

# %%
