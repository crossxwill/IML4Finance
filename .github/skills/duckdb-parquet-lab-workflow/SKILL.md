---
name: duckdb-parquet-lab-workflow
description: Use DuckDB to query Parquet files, inspect metadata, join tables, and convert results to pandas for analysis; commonly precedes ydata-eda-profiling for EDA on extracted tables.
---

# DuckDB Parquet Lab Workflow

## Purpose
Standardize the pattern of loading Parquet files into DuckDB, inspecting schema, running SQL joins, and converting results to pandas DataFrames.

## Usage
- "load Parquet with DuckDB and join tables"
- "describe DuckDB table schema"
- "convert DuckDB query to pandas"

## Instructions
1. Read Parquet data with `duckdb.query` or `duckdb.sql` using SQL strings.
2. Inspect schema using `DESCRIBE SELECT * FROM <table>` and display with `.show()`.
3. Use explicit joins with clear `LEFT` or `RIGHT` semantics to preserve row counts.
4. Convert results to pandas with `.to_df()` for downstream modeling.
5. Use `./templates/duckdb_snippets.md` for the standard SQL patterns.
