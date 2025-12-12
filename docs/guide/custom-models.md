# Custom Models

OG-Learn allows you to use any model that follows the scikit-learn interface.

## Custom HV Model

Any model with `fit(X, y)` and `predict(X)` methods can be used as an HV model:

```python
from og_learn import OGModel
from sklearn.ensemble import GradientBoostingRegressor

# Create custom HV model
custom_hv = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1
)

# Use in OGModel
model = OGModel(
    hv=custom_hv,  # Pass model instance
    lv='mlp',
    oscillation=0.05,
    sampling_alpha=0.1
)

model.fit(X_train, y_train, density=density_train)
```

## Custom LV Model

For custom LV models, you need to implement a class with `fit()` and `predict()` methods:

```python
import torch
import torch.nn as nn
from og_learn import OGModel

class CustomMLP:
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.model = None
        
    def _build_model(self):
        layers = []
        prev_dim = self.input_dim
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def fit(self, X, y, epochs=100, **kwargs):
        self.model = self._build_model()
        # Training logic here...
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            return self.model(X_tensor).numpy().flatten()

# Use custom LV model
custom_lv = CustomMLP(input_dim=X_train.shape[1])
model = OGModel(
    hv='lightgbm',
    lv=custom_lv,  # Pass model instance
    oscillation=0.05
)
```

---

## Using OG Core Functions Directly

For maximum flexibility, use the core OG functions:

```python
from og_learn.og_core import (
    initialize_OG_componment,
    generate_OG_componment
)

# Stage 1: Initialize and fit HV model
hv_model = initialize_OG_componment(X_train, y_train, model_type='lightgbm')

# Generate pseudo-labels with density-aware sampling
X_pseudo, y_pseudo = generate_OG_componment(
    X_train, y_train,
    density=density_train,
    hv_model=hv_model,
    oscillation=0.05,
    sampling_alpha=0.1
)

# Stage 2: Train your own LV model on pseudo-labels
from sklearn.neural_network import MLPRegressor

lv_model = MLPRegressor(hidden_layer_sizes=(256, 128, 64))
lv_model.fit(X_pseudo, y_pseudo)

# Predict
predictions = lv_model.predict(X_test)
```

---

## Hybrid Approaches

Combine preset and custom models:

```python
from og_learn import OGModel
from og_learn.presets import get_hv_model
from catboost import CatBoostRegressor

# Custom CatBoost with specific parameters
custom_catboost = CatBoostRegressor(
    iterations=1000,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=5,
    verbose=False
)

# Use with preset LV model
model = OGModel(
    hv=custom_catboost,
    lv='resnet',
    oscillation=0.03,
    sampling_alpha=0.15
)

model.fit(X_train, y_train, density=density_train, epochs=150)
```

---

## Tips for Custom Models

!!! tip "HV Model Requirements"
    - Must have `fit(X, y)` method
    - Must have `predict(X)` method
    - Should be capable of learning local patterns (tree-based models work well)

!!! tip "LV Model Requirements"
    - Must have `fit(X, y, ...)` method
    - Must have `predict(X)` method
    - For full OG integration, implement `OG_componment` support in `fit()`

!!! warning "Avoid"
    - Linear models as HV (won't capture local patterns)
    - Very complex models as LV (defeats the purpose of generalization)

