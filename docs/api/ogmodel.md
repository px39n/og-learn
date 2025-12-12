# OGModel

The main class for the Overfit-to-Generalization framework.

## Class Definition

::: og_learn.framework.OGModel
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - fit
        - predict

---

## Constructor

```python
OGModel(
    hv='lightgbm',
    lv='mlp',
    oscillation=0.05,
    sampling_alpha=0.1,
    epochs=100,
    early_stopping=True,
    patience=10,
    seed=42,
    verbose=True,
    tensorboard_dir=None,
    tensorboard_name=None,
    eval_every_epochs=10
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hv` | str or model | `'lightgbm'` | High-variance model. Can be a preset name or a model instance with `fit()`/`predict()` methods |
| `lv` | str or model | `'mlp'` | Low-variance model. Can be a preset name or a model instance |
| `oscillation` | float | `0.05` | Noise injection strength for pseudo-label generation |
| `sampling_alpha` | float | `0.1` | Exponent for density-aware sampling weights |
| `epochs` | int | `100` | Number of training epochs for LV model |
| `early_stopping` | bool | `True` | Whether to use early stopping |
| `patience` | int | `10` | Early stopping patience (epochs without improvement) |
| `seed` | int | `42` | Random seed for reproducibility |
| `verbose` | bool | `True` | Whether to print training progress |
| `tensorboard_dir` | str | `None` | Directory for TensorBoard logs |
| `tensorboard_name` | str | `None` | Name for this run in TensorBoard |
| `eval_every_epochs` | int | `10` | Frequency of evaluation/logging |

---

## Methods

### fit

```python
model.fit(X, y, density=None, X_valid=None, y_valid=None, epochs=None)
```

Train the OG model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Training features, shape (n_samples, n_features) |
| `y` | array-like | Training target, shape (n_samples,) |
| `density` | array-like | Spatial density for each sample, shape (n_samples,) |
| `X_valid` | array-like | Validation features (optional) |
| `y_valid` | array-like | Validation target (optional) |
| `epochs` | int | Override epochs from constructor |

**Returns:** `self`

**Example:**

```python
model = OGModel(hv='lightgbm', lv='mlp')
model.fit(
    X_train, y_train,
    density=density_train,
    X_valid=X_valid,
    y_valid=y_valid,
    epochs=100
)
```

---

### predict

```python
predictions = model.predict(X)
```

Make predictions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Features, shape (n_samples, n_features) |

**Returns:** `numpy.ndarray` - Predictions, shape (n_samples,)

**Example:**

```python
predictions = model.predict(X_test)
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_hv_model` | object | Fitted HV model instance |
| `_lv_model` | object | Fitted LV model instance |
| `hv_name` | str | Name of HV model |
| `lv_name` | str | Name of LV model |

---

## Complete Example

```python
from og_learn import OGModel, calculate_density
from sklearn.metrics import r2_score
import numpy as np

# Prepare data
density = calculate_density(X_train[:, 0], X_train[:, 1])

# Create and train model
model = OGModel(
    hv='lightgbm',
    lv='resnet',
    oscillation=0.05,
    sampling_alpha=0.1,
    epochs=100,
    early_stopping=True,
    patience=15,
    seed=42,
    tensorboard_dir='runs/og_resnet'
)

model.fit(
    X_train, y_train,
    density=density,
    X_valid=X_valid,
    y_valid=y_valid
)

# Evaluate
predictions = model.predict(X_test)
print(f"Test R²: {r2_score(y_test, predictions):.4f}")
```

