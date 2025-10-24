# State Space Model Naming Feature

## Summary

Added a `name` parameter to the `PyMCStateSpace` class constructor to allow multiple state space models to coexist in a single PyMC model without variable name conflicts.

## Problem

Previously, when trying to use more than one state space model in a single PyMC model, variable names would collide because all state space models used the same parameter names (e.g., "x0", "P0", "rho", "sigma", "obs", etc.).

## Solution

Added an optional `name` parameter to `PyMCStateSpace.__init__()`. When provided, all variables registered in the PyMC model are prefixed with this name, preventing conflicts.

## Changes Made

### 1. **Constructor Update** (`__init__`)
   - Added `name: str | None = None` parameter
   - Stored as `self.name` attribute

### 2. **Helper Method** (`_prefix_name`)
   - Created new method to prefix variable names when `self.name` is set
   - Returns unprefixed name if `self.name` is None (backward compatible)

### 3. **Variable Registration** (`make_and_register_variable`, `make_and_register_data`, `register_variable`)
   - Updated documentation to note that prefixing happens during lookup
   - Internal storage uses unprefixed names as keys
   - Added new `register_variable()` method for registering pre-existing variables (not symbolic placeholders)
     - Used when variables are passed to the model constructor
     - Validates that variable name is in `param_names`
     - Stores in `_name_to_variable` with unprefixed key

### 4. **Variable Insertion** (`_insert_random_variables`, `_insert_data_variables`)
   - Modified to use prefixed names when looking up variables in PyMC model
   - Enhanced error messages to include model name when provided

### 5. **Matrix Registration** (`_register_matrices_with_pymc_model`)
   - Updated to use prefixed names when creating Deterministic variables
   - Converted `_register_kalman_filter_outputs_with_pymc_model` from static to instance method to access `self.name`

### 6. **Observation Variables**
   - Updated `build_statespace_graph` to prefix "obs" variable
   - Updated all sampling methods to prefix their output variables:
     - `_sample_conditional`: prefixes "filtered_prior", "smoothed_posterior", etc.
     - `_sample_unconditional`: prefixes "prior_latent", "posterior_observed", etc.
     - `forecast`: prefixes "forecast_latent", "forecast_observed", etc.
     - `impulse_response_function`: prefixes "irf", "x0_new", "initial_shock", etc.

### 7. **Data Variables** (`register_data_with_pymc`, `add_data_to_active_model`)
   - Added `data_name` parameter to `register_data_with_pymc` function
   - Added `name` parameter to `add_data_to_active_model` function  
   - Updated all calls to `register_data_with_pymc` in `build_statespace_graph` and sampling methods to use prefixed data names
   - Data variable is now named `{model_name}_data` instead of just `"data"` when model has a name
   - Prevents naming conflicts when multiple state space models are used in the same PyMC model

## Usage Example

```python
import pymc as pm
from pymc_extras.statespace.models import BayesianSARIMA

# Create two SARIMAX models with different names
model1 = BayesianSARIMA(order=(1, 0, 1), name="series1")
model2 = BayesianSARIMA(order=(1, 0, 1), name="series2")

with pm.Model() as combined_model:
    # Model 1 parameters (prefixed with "series1_")
    series1_x0 = pm.Normal("series1_x0", shape=model1.k_states)
    series1_ar_params = pm.Normal("series1_ar_params", shape=1)
    series1_ma_params = pm.Normal("series1_ma_params", shape=1)
    series1_sigma = pm.Exponential("series1_sigma", 1)
    
    # Model 2 parameters (prefixed with "series2_")
    series2_x0 = pm.Normal("series2_x0", shape=model2.k_states)
    series2_ar_params = pm.Normal("series2_ar_params", shape=1)
    series2_ma_params = pm.Normal("series2_ma_params", shape=1)
    series2_sigma = pm.Exponential("series2_sigma", 1)
    
    # Build both models without name conflicts
    model1.build_statespace_graph(data1)
    model2.build_statespace_graph(data2)
    
    # Sample from the combined model
    idata = pm.sample()
```

## Backward Compatibility

The change is fully backward compatible:
- `name` parameter defaults to `None`
- When `name=None`, no prefixing occurs (original behavior)
- Existing code continues to work without modification

## Testing

A test script has been created at `test_multiple_statespace.py` that verifies:
1. Single models without names work as before
2. Multiple models with different names can coexist
3. All variables are properly prefixed when names are provided

## Files Modified

- `pymc_extras/statespace/core/statespace.py`: Core implementation of the naming feature
- `test_multiple_statespace.py`: Test script demonstrating the feature
