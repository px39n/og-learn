# Tutorial

This tutorial walks through a complete OG-Learn workflow.

## Setup

```python
import numpy as np
import pandas as pd
import warnings
import shutil
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Import og_learn
import og_learn
from og_learn import (
    OGModel, 
    compare_models,
    calculate_density,
    split_test_train,
    load_data,
    sanity_check,
    save_split_indices,
    load_split_indices,
    launch_tensorboard,
    list_presets
)
from og_learn.feature import simple_feature_engineering
from og_learn.presets import get_hv_model, get_lv_model
```

---

## 1. Data Preparation

### Load Data

```python
# Configuration
DATA_PATH = '../data/your_data.parquet'
SAVE_DIR = '../diagnostics'
SAMPLE = 10000  # Use subset for quick testing

# Define features
FEATURE_COLS = [
    'longitude', 'latitude',
    'temperature', 'humidity', 'pressure'
]
TARGET_COL = 'ozone'

# Load data
df = load_data(DATA_PATH, FEATURE_COLS, TARGET_COL)

# Sample for testing
if SAMPLE:
    df = df.sample(n=min(SAMPLE, len(df)), random_state=42)
    
print(f"Data shape: {df.shape}")
```

### Validate Data

```python
sanity_check(df, FEATURE_COLS, TARGET_COL)
```

### Feature Engineering

```python
df, FEATURE_COLS = simple_feature_engineering(
    df, FEATURE_COLS,
    time_col='time',
    k=3,
    standardize=True,
    add_temporal=True
)

print(f"Features after engineering: {len(FEATURE_COLS)}")
```

### Calculate Density

```python
density = calculate_density(df['longitude'], df['latitude'])
df['density'] = density
```

### Split Data

```python
# Site-wise split to prevent data leakage
train_idx, test_idx = split_test_train(df, method='site', test_ratio=0.2)

# Save for reproducibility
save_split_indices(train_idx, test_idx, SAVE_DIR)

# Prepare arrays
X_train = df.loc[train_idx, FEATURE_COLS].values
y_train = df.loc[train_idx, TARGET_COL].values
X_test = df.loc[test_idx, FEATURE_COLS].values
y_test = df.loc[test_idx, TARGET_COL].values
density_train = df.loc[train_idx, 'density'].values

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
```

---

## 2. Model Training

### Single OG Model

```python
model = OGModel(
    hv='lightgbm',
    lv='mlp',
    oscillation=0.05,
    sampling_alpha=0.1,
    epochs=100,
    seed=42
)

model.fit(X_train, y_train, density=density_train)

predictions = model.predict(X_test)
print(f"Test R²: {r2_score(y_test, predictions):.4f}")
```

### Compare Multiple Models

```python
# Clear old TensorBoard logs
TB_LOG_DIR = os.path.join(SAVE_DIR, 'tensorboard')
shutil.rmtree(TB_LOG_DIR, ignore_errors=True)

# Define models
models = {
    'MLP': get_lv_model('mlp', num_features=X_train.shape[1]),
    'BigMLP': get_lv_model('bigmlp', num_features=X_train.shape[1]),
    'OG_LightGBM_MLP': OGModel(hv='lightgbm', lv='mlp'),
    'OG_XGBoost_MLP': OGModel(hv='xgboost', lv='mlp'),
    'OG_CatBoost_ResNet': OGModel(hv='catboost', lv='resnet'),
}

# Run comparison
results = compare_models(
    models,
    X_train, y_train,
    X_test, y_test,
    density=density_train,
    tensorboard_dir=TB_LOG_DIR,
    save_dir=os.path.join(SAVE_DIR, 'models'),
    eval_every_epochs=5
)

print(results)
```

---

## 3. TensorBoard Visualization

```python
# Launch TensorBoard
tb_process = launch_tensorboard(TB_LOG_DIR, open_browser=True)

# View at http://localhost:6006

# When done:
# tb_process.terminate()
```

---

## 4. Results Analysis

```python
import matplotlib.pyplot as plt

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
results.plot(kind='bar', x='Model', y=['Train_R2', 'Test_R2'], ax=ax)
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison')
ax.legend(['Train', 'Test'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## Next Steps

- Try different HV/LV combinations
- Tune `oscillation` and `sampling_alpha`
- Use `early_stopping` with validation data
- Explore [Custom Models](../guide/custom-models.md)

