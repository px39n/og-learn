# Installation

## Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- NumPy, Pandas, Scikit-learn

## Install via pip

```bash
pip install og-learn
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/your-username/og-learn.git
cd og-learn
pip install -e .
```

## Dependencies

OG-Learn will automatically install the following dependencies:

| Package | Purpose |
|---------|---------|
| `torch` | Neural network models (MLP, ResNet, Transformer) |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `scikit-learn` | Preprocessing and metrics |
| `lightgbm` | LightGBM HV model preset |
| `xgboost` | XGBoost HV model preset |
| `catboost` | CatBoost HV model preset |
| `tqdm` | Progress bars |
| `tensorboard` | Training visualization |

## Optional Dependencies

For full functionality, you may also want to install:

```bash
# For TensorBoard visualization
pip install tensorboard

# For Jupyter notebook support
pip install jupyter ipywidgets
```

## Verify Installation

```python
import og_learn
print(og_learn.__version__)

# Check available presets
from og_learn import list_presets
list_presets()
```

Expected output:

```
Available HV Presets:
  - lightgbm
  - biglightgbm
  - xgboost
  - catboost
  - random_forest
  - decision_tree
  - linear_regression

Available LV Presets:
  - mlp
  - bigmlp
  - resnet
  - transformer
```

