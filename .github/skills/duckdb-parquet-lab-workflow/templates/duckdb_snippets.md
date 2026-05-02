# DuckDB Parquet Patterns

## Load Parquet
```sql
SELECT *
FROM '../Data/<dataset>/<file>.parquet'
```

## Describe schema
```sql
DESCRIBE SELECT * FROM <table_name>
```

## Join pattern
```sql
SELECT a.*, b.*
FROM <table_a> AS a
LEFT JOIN <table_b> AS b
ON a.key = b.key
ORDER BY a.key
```

## Convert to pandas
Use `.to_df()` on the DuckDB relation
