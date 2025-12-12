# Presets API

Functions for accessing preset model configurations.

---

## list_presets

```python
from og_learn import list_presets

list_presets()
```

Print all available HV and LV presets.

---

## get_hv_model

```python
from og_learn.presets import get_hv_model

model = get_hv_model(name)
```

Get a High-Variance model instance.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Preset name: `'lightgbm'`, `'biglightgbm'`, `'xgboost'`, `'catboost'`, `'random_forest'`, `'decision_tree'`, `'linear_regression'` |

**Returns:** Model instance with `fit()`/`predict()` methods

**Example:**

```python
from og_learn.presets import get_hv_model

lgb = get_hv_model('lightgbm')
lgb.fit(X_train, y_train)
predictions = lgb.predict(X_test)
```

---

## get_lv_model

```python
from og_learn.presets import get_lv_model

model = get_lv_model(name, num_features, epochs=100)
```

Get a Low-Variance model instance.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Preset name: `'mlp'`, `'bigmlp'`, `'resnet'`, `'transformer'` |
| `num_features` | int | Number of input features |
| `epochs` | int | Training epochs (default: 100) |

**Returns:** Model instance with `fit()`/`predict()` methods

**Example:**

```python
from og_learn.presets import get_lv_model

mlp = get_lv_model('mlp', num_features=10, epochs=50)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
```

---

## Preset Configurations

### HV_PRESETS

Dictionary of HV model configurations:

```python
HV_PRESETS = {
    'lightgbm': {
        'n_estimators': 500,
        'num_leaves': 1200,
        'max_depth': 9,
        'learning_rate': 0.05,
        # ...
    },
    'biglightgbm': { ... },
    'xgboost': { ... },
    'catboost': { ... },
    'random_forest': { ... },
    'decision_tree': { ... },
    'linear_regression': { ... },
}
```

### LV_PRESETS

Dictionary of LV model configurations:

```python
LV_PRESETS = {
    'mlp': {
        'hidden_layers': [256, 128, 64],
        'dropout': 0.3,
        'batch_size': 256,
        'learning_rate': 0.001,
    },
    'bigmlp': { ... },
    'resnet': { ... },
    'transformer': { ... },
}
```

