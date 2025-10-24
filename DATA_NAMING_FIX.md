# Data Variable Naming Fix

## Problem

When creating multiple state space models in a loop (e.g., one per subject in panel data), all models attempted to register their data with PyMC using the same variable name `"data"`. This caused naming conflicts and prevented multiple models from coexisting in the same PyMC model.

## Solution

Modified the data registration process to use prefixed data variable names when a model has a `name` attribute:

1. **Updated `add_data_to_active_model` function** (`pymc_extras/statespace/utils/data_tools.py`)
   - Added `name` parameter with default value `"data"`
   - Updated docstring to explain the parameter and its purpose
   - Variable name is now customizable to avoid conflicts

2. **Updated `register_data_with_pymc` function** (`pymc_extras/statespace/utils/data_tools.py`)
   - Added `data_name` parameter with default value `"data"`
   - Passes `data_name` to both `add_data_to_active_model` and `pytensor.shared`
   - Updated docstring with comprehensive parameter documentation

3. **Updated all calls to `register_data_with_pymc`** (`pymc_extras/statespace/core/statespace.py`)
   - Modified three calls in `build_statespace_graph` and `_sample_unconditional`
   - Each call now passes `data_name=self._prefix_name("data")`
   - When `self.name` is None, data variable is named `"data"` (backward compatible)
   - When `self.name` is set (e.g., `"model1"`), data variable is named `"model1_data"`

## Example Usage

### Before (caused conflicts):

```python
models = [BayesianSARIMA(order=(1,0,1)) for _ in range(3)]

with pm.Model() as m:
    for i, model in enumerate(models):
        # ... define parameters ...
        model.build_statespace_graph(data[i])
        # ERROR: All models try to create "data" variable
```

### After (works correctly):

```python
models = [BayesianSARIMA(order=(1,0,1), name=f"subject_{i}") for i in range(3)]

with pm.Model() as m:
    for i, model in enumerate(models):
        # ... define parameters with f"subject_{i}_" prefix ...
        model.build_statespace_graph(data[i])
        # Creates "subject_0_data", "subject_1_data", "subject_2_data"
```

## Backward Compatibility

The changes are fully backward compatible:

- Default parameter values preserve original behavior
- Models without a `name` still create a variable called `"data"`
- No changes required to existing code that doesn't use the `name` feature

## Testing

Created `test_data_naming.py` which verifies:

1. Single model without name creates `"data"` variable
2. Single model with name creates prefixed data variable (e.g., `"model1_data"`)
3. Multiple models with different names create separate data variables
4. Loop creation of models (panel data scenario) works without conflicts

## Files Modified

- `pymc_extras/statespace/utils/data_tools.py`: Added `name`/`data_name` parameters
- `pymc_extras/statespace/core/statespace.py`: Updated function calls to use prefixed names
- `FEATURE_SUMMARY.md`: Documented the change
- `USAGE_GUIDE.md`: Added section explaining data variable naming
- `test_data_naming.py`: New test file demonstrating the feature
