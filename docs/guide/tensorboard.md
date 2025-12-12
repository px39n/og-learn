# TensorBoard Integration

OG-Learn provides built-in TensorBoard support for visualizing training progress.

## Basic Setup

### Enable TensorBoard Logging

```python
from og_learn import OGModel

model = OGModel(
    hv='lightgbm',
    lv='mlp',
    tensorboard_dir='runs/experiment1',  # Log directory
    tensorboard_name='og_mlp',           # Run name
    eval_every_epochs=5                   # Log frequency
)

model.fit(
    X_train, y_train,
    density=density_train,
    X_valid=X_valid,  # Required for validation metrics
    y_valid=y_valid,
    epochs=100
)
```

### Launch TensorBoard

```python
from og_learn import launch_tensorboard

# Start TensorBoard server
tb_process = launch_tensorboard('runs/experiment1', open_browser=True)
```

Or from command line:

```bash
tensorboard --logdir=runs/experiment1
```

Then open http://localhost:6006 in your browser.

---

## Comparing Multiple Models

```python
from og_learn import OGModel, compare_models
from og_learn.presets import get_lv_model

# Clear old logs
import shutil
shutil.rmtree('runs/comparison', ignore_errors=True)

# Define models
models = {
    'MLP': get_lv_model('mlp', num_features=X_train.shape[1]),
    'OG_LightGBM_MLP': OGModel(hv='lightgbm', lv='mlp'),
    'OG_XGBoost_MLP': OGModel(hv='xgboost', lv='mlp'),
    'OG_CatBoost_ResNet': OGModel(hv='catboost', lv='resnet'),
}

# Run comparison with TensorBoard logging
results = compare_models(
    models,
    X_train, y_train,
    X_test, y_test,
    density=density_train,
    tensorboard_dir='runs/comparison',  # All models log here
    eval_every_epochs=5
)

# Launch TensorBoard to compare
launch_tensorboard('runs/comparison', open_browser=True)
```

---

## What Gets Logged

| Metric | Description |
|--------|-------------|
| `R2_train` | Training R² score |
| `R2_val` | Validation R² score |
| `Loss_train` | Training loss (MSE) |

---

## Viewing in TensorBoard

### Scalars Tab

View training curves:

- **R² scores** over epochs
- **Loss** over epochs
- Compare multiple runs side-by-side

### Text Tab

View run configuration and parameters.

---

## Tips

!!! tip "Clear Old Logs"
    Before running new experiments, clear old logs to avoid confusion:
    ```python
    import shutil
    shutil.rmtree('runs/', ignore_errors=True)
    ```

!!! tip "Logging Frequency"
    Set `eval_every_epochs=1` for detailed curves, or higher values (5-10) for faster training.

!!! tip "Validation Data"
    Always provide `X_valid` and `y_valid` to see validation metrics in TensorBoard.

!!! warning "Background Process"
    `launch_tensorboard()` runs in background. To stop:
    ```python
    tb_process.terminate()
    ```

