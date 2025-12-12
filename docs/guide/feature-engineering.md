# Feature Engineering

OG-Learn provides utilities for spatiotemporal feature engineering.

## Overview

Effective feature engineering is crucial for spatiotemporal models. OG-Learn includes functions for:

- Loading and preprocessing data
- Computing spatial harmonics
- Creating temporal features
- Feature standardization

---

## Loading Data

```python
from og_learn import load_data

# Load and preprocess
df = load_data(
    data_path='data/observations.parquet',
    feature_cols=['longitude', 'latitude', 'temp', 'humidity'],
    target_col='ozone'
)
```

The `load_data` function:

- Converts `time` column to datetime
- Converts float64 to float32 for memory efficiency
- Filters to valid feature columns

---

## Spatial Harmonics

Transform longitude/latitude into harmonic features:

```python
from og_learn.feature import compute_spatial_harmonics

# Add spatial harmonics
df = compute_spatial_harmonics(
    df,
    k=3,  # Number of harmonics
    lon_col='longitude',
    lat_col='latitude'
)
```

This creates features:

- `lon_sin_1`, `lon_cos_1`, `lat_sin_1`, `lat_cos_1`
- `lon_sin_2`, `lon_cos_2`, `lat_sin_2`, `lat_cos_2`
- `lon_sin_3`, `lon_cos_3`, `lat_sin_3`, `lat_cos_3`

The transformation is:

$$\sin(k \cdot \frac{\text{lon}}{180} \cdot \pi), \quad \cos(k \cdot \frac{\text{lon}}{180} \cdot \pi)$$

---

## Temporal Features

Extract time-based features:

```python
from og_learn.feature import compute_temporal_features

# Add temporal features
df = compute_temporal_features(df, time_col='time')
```

Created features:

| Feature | Description |
|---------|-------------|
| `time_month` | Month (1-12), normalized |
| `time_hour` | Hour (0-23), normalized |
| `time_day_of_month` | Day (1-31), normalized |

---

## Complete Feature Engineering Pipeline

```python
from og_learn.feature import simple_feature_engineering

# Apply full pipeline
df_processed, feature_cols = simple_feature_engineering(
    df,
    feature_cols=['longitude', 'latitude', 'temp', 'humidity', 'pressure'],
    time_col='time',
    k=3,                  # Spatial harmonics
    standardize=True,     # Apply StandardScaler
    add_temporal=True     # Add temporal features
)

print(f"Original features: 5")
print(f"Processed features: {len(feature_cols)}")
```

Output:

```
Original features: 5
Processed features: 20
```

---

## Sanity Check

Validate your data before training:

```python
from og_learn import sanity_check

sanity_check(df, feature_cols, target_col='ozone')
```

Output:

```
============================================================
              Data Sanity Check
============================================================
Total samples: 50,000
Features: 10
Missing values: 0

Feature Engineering Options:
  • Spatial harmonics: k=3 (adds 12 features)
  • Temporal features: month, hour, day (adds 3 features)
  • Standardization: recommended

Data looks good! ✓
============================================================
```

---

## Best Practices

!!! tip "Spatial Harmonics"
    - Use `k=3` for most cases
    - Higher `k` captures finer spatial patterns but may overfit

!!! tip "Standardization"
    - Always standardize features for neural network LV models
    - Apply standardization **before** spatial harmonics

!!! warning "Data Leakage"
    - Fit StandardScaler on training data only
    - Use `split_test_train` before feature engineering

