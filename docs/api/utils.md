# Utilities API

Helper functions for data loading, validation, and more.

---

## load_data

```python
from og_learn import load_data

df = load_data(data_path, feature_cols, target_col)
```

Load and preprocess data from file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_path` | str | Path to data file (CSV, Parquet, etc.) |
| `feature_cols` | list | Expected feature column names |
| `target_col` | str | Target column name |

**Returns:** `pandas.DataFrame`

**Operations performed:**
- Converts `time` column to datetime
- Converts float64 to float32
- Filters to valid feature columns

---

## sanity_check

```python
from og_learn import sanity_check

sanity_check(df, feature_cols, target_col)
```

Validate data and print summary.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | DataFrame | Data to validate |
| `feature_cols` | list | Feature column names |
| `target_col` | str | Target column name |

---

## calculate_density

```python
from og_learn import calculate_density

density = calculate_density(longitude, latitude, method='kde')
```

Calculate spatial density for each point.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `longitude` | array | - | Longitude values |
| `latitude` | array | - | Latitude values |
| `method` | str | `'kde'` | Density estimation method |

**Returns:** `numpy.ndarray` - Density values

---

## split_test_train

```python
from og_learn import split_test_train

train_idx, test_idx = split_test_train(df, method='site', test_ratio=0.2)
```

Split data into train/test sets.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Data with site information |
| `method` | str | `'site'` | Split method: `'site'`, `'random'`, `'temporal'` |
| `test_ratio` | float | `0.2` | Fraction for test set |

**Returns:** 
- `train_idx`: Training indices
- `test_idx`: Test indices

---

## save_split_indices / load_split_indices

```python
from og_learn import save_split_indices, load_split_indices

# Save indices
save_split_indices(train_idx, test_idx, save_dir='data/splits')

# Load indices
train_idx, test_idx = load_split_indices(save_dir='data/splits')
```

Persist train/test splits for reproducibility.

---

## launch_tensorboard

```python
from og_learn import launch_tensorboard

tb_process = launch_tensorboard(log_dir, port=6006, open_browser=True)
```

Start TensorBoard server.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | str | - | Directory containing TensorBoard logs |
| `port` | int | `6006` | Port for TensorBoard server |
| `open_browser` | bool | `True` | Open browser automatically |

**Returns:** `subprocess.Popen` - TensorBoard process handle

**Stop TensorBoard:**

```python
tb_process.terminate()
```

---

## compare_models

```python
from og_learn import compare_models

results = compare_models(
    models,
    X_train, y_train,
    X_test, y_test,
    density=None,
    tensorboard_dir=None,
    save_dir=None,
    eval_every_epochs=10
)
```

Train and compare multiple models.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | dict | Dictionary of {name: model} |
| `X_train` | array | Training features |
| `y_train` | array | Training target |
| `X_test` | array | Test features |
| `y_test` | array | Test target |
| `density` | array | Density for OG models (optional) |
| `tensorboard_dir` | str | TensorBoard log directory (optional) |
| `save_dir` | str | Directory to save trained models (optional) |
| `eval_every_epochs` | int | Logging frequency |

**Returns:** `pandas.DataFrame` with columns `['Model', 'Train_R2', 'Test_R2']`

