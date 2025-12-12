# Preset Models

OG-Learn provides carefully tuned preset configurations for both HV and LV models.

## High-Variance (HV) Presets

These models are used in Stage 1 for pseudo-label generation.

### LightGBM Presets

=== "lightgbm (default)"

    Standard LightGBM configuration for most use cases.
    
    ```python
    OGModel(hv='lightgbm', lv='mlp')
    ```
    
    | Parameter | Value |
    |-----------|-------|
    | n_estimators | 500 |
    | num_leaves | 1200 |
    | max_depth | 9 |
    | learning_rate | 0.05 |

=== "biglightgbm"

    Larger LightGBM for complex datasets.
    
    ```python
    OGModel(hv='biglightgbm', lv='mlp')
    ```
    
    | Parameter | Value |
    |-----------|-------|
    | n_estimators | 500 |
    | num_leaves | 1200 |
    | max_depth | 9 |
    | min_child_samples | 1 |

### XGBoost

```python
OGModel(hv='xgboost', lv='mlp')
```

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 11 |
| learning_rate | 0.05 |

### CatBoost

```python
OGModel(hv='catboost', lv='mlp')
```

| Parameter | Value |
|-----------|-------|
| iterations | 500 |
| depth | 9 |
| learning_rate | 0.05 |

### Other HV Models

```python
# Random Forest
OGModel(hv='random_forest', lv='mlp')

# Decision Tree
OGModel(hv='decision_tree', lv='mlp')

# Linear Regression (baseline)
OGModel(hv='linear_regression', lv='mlp')
```

---

## Low-Variance (LV) Presets

These neural network models are used in Stage 2 for learning from pseudo-labels.

### MLP (Multi-Layer Perceptron)

=== "mlp (default)"

    Standard MLP architecture.
    
    ```python
    OGModel(hv='lightgbm', lv='mlp')
    ```
    
    | Parameter | Value |
    |-----------|-------|
    | hidden_layers | [256, 128, 64] |
    | dropout | 0.3 |
    | batch_size | 256 |
    | learning_rate | 0.001 |

=== "bigmlp"

    Larger MLP for complex patterns.
    
    ```python
    OGModel(hv='lightgbm', lv='bigmlp')
    ```
    
    | Parameter | Value |
    |-----------|-------|
    | hidden_layers | [512, 256, 128] |
    | dropout | 0.3 |
    | batch_size | 256 |
    | learning_rate | 0.001 |

### ResNet

Deep residual network with skip connections.

```python
OGModel(hv='lightgbm', lv='resnet')
```

| Parameter | Value |
|-----------|-------|
| d_main | 256 |
| d_hidden | 512 |
| n_blocks | 2 |
| dropout | 0.2 |

### Transformer

Attention-based architecture for capturing complex dependencies.

```python
OGModel(hv='lightgbm', lv='transformer')
```

| Parameter | Value |
|-----------|-------|
| n_layers | 3 |
| d_token | 192 |
| n_heads | 8 |
| d_ffn | 256 |

---

## Listing Available Presets

```python
from og_learn import list_presets

list_presets()
```

Output:

```
============================================================
              Available Presets
============================================================

HV (High-Variance) Presets:
  • lightgbm
  • biglightgbm
  • xgboost
  • catboost
  • random_forest
  • decision_tree
  • linear_regression

LV (Low-Variance) Presets:
  • mlp
  • bigmlp
  • resnet
  • transformer

============================================================
```

---

## Getting Preset Models Directly

You can also get preset models without using OGModel:

```python
from og_learn.presets import get_hv_model, get_lv_model

# Get HV model
hv_model = get_hv_model('lightgbm')

# Get LV model (requires num_features)
lv_model = get_lv_model('mlp', num_features=10, epochs=100)

# Use directly
hv_model.fit(X_train, y_train)
predictions = hv_model.predict(X_test)
```

