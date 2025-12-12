# Quick Start

This guide will walk you through the basics of using OG-Learn.

## Basic Usage

### 1. Import and Create Model

```python
from og_learn import OGModel

# Create OG model with presets
model = OGModel(
    hv='lightgbm',      # High-variance model
    lv='mlp',           # Low-variance model
    oscillation=0.05,   # Feature noise regularization
    sampling_alpha=0.1  # Density-aware sampling weight
)
```

### 2. Prepare Your Data

OG-Learn works with standard NumPy arrays or Pandas DataFrames:

```python
import numpy as np
from og_learn import calculate_density

# Your data
X_train = ...  # Features (n_samples, n_features)
y_train = ...  # Target (n_samples,)

# Calculate spatial density (required for OG)
# Assumes columns 0, 1 are longitude, latitude
density_train = calculate_density(X_train[:, 0], X_train[:, 1])
```

### 3. Train the Model

```python
model.fit(
    X_train, y_train,
    density=density_train,
    epochs=100,
    X_valid=X_valid,  # Optional: for early stopping
    y_valid=y_valid
)
```

### 4. Make Predictions

```python
predictions = model.predict(X_test)
```

### 5. Evaluate

```python
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
```

---

## Complete Example

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from og_learn import OGModel, calculate_density

# Generate sample data
np.random.seed(42)
n_samples = 1000
lon = np.random.uniform(-180, 180, n_samples)
lat = np.random.uniform(-90, 90, n_samples)
X = np.column_stack([lon, lat, np.random.randn(n_samples, 5)])
y = np.sin(lon/30) + np.cos(lat/20) + 0.1 * np.random.randn(n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Calculate density
density_train = calculate_density(X_train[:, 0], X_train[:, 1])

# Create and train OG model
model = OGModel(
    hv='lightgbm',
    lv='mlp',
    oscillation=0.05,
    sampling_alpha=0.1,
    seed=42
)

model.fit(X_train, y_train, density=density_train, epochs=50)

# Evaluate
predictions = model.predict(X_test)
print(f"Test R²: {r2_score(y_test, predictions):.4f}")
```

---

## Next Steps

- Learn about the [OG Framework](../guide/og-framework.md) in detail
- Explore [Preset Models](../guide/presets.md)
- Create [Custom Models](../guide/custom-models.md)
- Compare models with [Model Comparison](../guide/og-framework.md#model-comparison)

