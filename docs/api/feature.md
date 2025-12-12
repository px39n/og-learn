# Feature Engineering API

Utilities for spatiotemporal feature engineering.

---

## simple_feature_engineering

```python
from og_learn.feature import simple_feature_engineering

df_processed, feature_cols = simple_feature_engineering(
    df,
    feature_cols,
    time_col='time',
    k=3,
    standardize=True,
    add_temporal=True
)
```

Apply complete feature engineering pipeline.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input data |
| `feature_cols` | list | - | Original feature column names |
| `time_col` | str | `'time'` | Name of time column |
| `k` | int | `3` | Number of spatial harmonics |
| `standardize` | bool | `True` | Apply StandardScaler |
| `add_temporal` | bool | `True` | Add temporal features |

**Returns:** 
- `df_processed` (DataFrame): Processed data
- `feature_cols` (list): Updated feature column names

---

## compute_spatial_harmonics

```python
from og_learn.feature import compute_spatial_harmonics

df = compute_spatial_harmonics(df, k=3, lon_col='longitude', lat_col='latitude')
```

Add spatial harmonic features.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input data |
| `k` | int | `3` | Number of harmonics |
| `lon_col` | str | `'longitude'` | Longitude column name |
| `lat_col` | str | `'latitude'` | Latitude column name |

**Returns:** DataFrame with added harmonic columns

**Created columns:**
- `lon_sin_1`, `lon_cos_1`, ..., `lon_sin_k`, `lon_cos_k`
- `lat_sin_1`, `lat_cos_1`, ..., `lat_sin_k`, `lat_cos_k`

---

## compute_temporal_features

```python
from og_learn.feature import compute_temporal_features

df = compute_temporal_features(df, time_col='time')
```

Add temporal features from datetime column.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input data |
| `time_col` | str | `'time'` | Time column name (datetime) |

**Returns:** DataFrame with added temporal columns

**Created columns:**
- `time_month`: Month (1-12), normalized to [0, 1]
- `time_hour`: Hour (0-23), normalized to [0, 1]
- `time_day_of_month`: Day (1-31), normalized to [0, 1]

---

## Example

```python
import pandas as pd
from og_learn.feature import (
    compute_spatial_harmonics,
    compute_temporal_features,
    simple_feature_engineering
)

# Load data
df = pd.read_parquet('data.parquet')

# Option 1: Step-by-step
df = compute_temporal_features(df, time_col='time')
df = compute_spatial_harmonics(df, k=3)

# Option 2: Complete pipeline
df, feature_cols = simple_feature_engineering(
    df,
    feature_cols=['longitude', 'latitude', 'temp', 'pressure'],
    k=3,
    standardize=True
)
```

