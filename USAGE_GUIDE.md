# Using Multiple State Space Models in a Single PyMC Model

## Overview

As of this update, you can now use multiple state space models in a single PyMC model by giving each model a unique name. This prevents variable name conflicts that would previously occur.

## Quick Start

### Basic Usage

```python
import pymc as pm
from pymc_extras.statespace.models import BayesianSARIMA

# Create two models with unique names
model_a = BayesianSARIMA(order=(1, 0, 1), name="region_a")
model_b = BayesianSARIMA(order=(1, 0, 1), name="region_b")

with pm.Model() as combined_model:
    # Define priors for model A (prefix all parameter names with "region_a_")
    region_a_x0 = pm.Normal("region_a_x0", shape=model_a.k_states)
    region_a_P0_diag = pm.Exponential("region_a_P0_diag", 1, shape=model_a.k_states)
    region_a_P0 = pm.Deterministic("region_a_P0", pt.diag(region_a_P0_diag))
    region_a_ar_params = pm.Normal("region_a_ar_params", shape=1)
    region_a_ma_params = pm.Normal("region_a_ma_params", shape=1)
    region_a_sigma_state = pm.Exponential("region_a_sigma_state", 1)
    
    # Define priors for model B (prefix all parameter names with "region_b_")
    region_b_x0 = pm.Normal("region_b_x0", shape=model_b.k_states)
    region_b_P0_diag = pm.Exponential("region_b_P0_diag", 1, shape=model_b.k_states)
    region_b_P0 = pm.Deterministic("region_b_P0", pt.diag(region_b_P0_diag))
    region_b_ar_params = pm.Normal("region_b_ar_params", shape=1)
    region_b_ma_params = pm.Normal("region_b_ma_params", shape=1)
    region_b_sigma_state = pm.Exponential("region_b_sigma_state", 1)
    
    # Build both state space graphs
    model_a.build_statespace_graph(data_a)
    model_b.build_statespace_graph(data_b)
    
    # Sample from the combined model
    idata = pm.sample()
```

## Important Details

### Parameter Naming Convention

When you create a state space model with a name, **all parameters must be prefixed** with that name:

- If your model has `name="model1"` and expects a parameter called `"x0"`, you must create it in PyMC as `"model1_x0"`
- The prefix format is always: `{name}_{parameter_name}`

### Registering Variables in Custom Models

When creating custom state space models, you have two ways to register parameters:

#### 1. Using `make_and_register_variable()` for symbolic placeholders

This method creates symbolic PyTensor variables that will be replaced with PyMC random variables later:

```python
class MyModel(PyMCStateSpace):
    def __init__(self, name=None):
        super().__init__(k_states=2, k_posdef=1, name=name)
        # Create symbolic placeholders - use UNPREFIXED names
        self.A = self.make_and_register_variable("A", shape=(2, 2))
        self.B = self.make_and_register_variable("B", shape=(2, 1))
```

#### 2. Using `register_variable()` for existing variables

This method registers pre-existing PyTensor variables (e.g., those passed to the constructor):

```python
class MyModel(PyMCStateSpace):
    def __init__(self, A, B, var_eta, var_epsilon, name=None):
        super().__init__(k_states=2, k_posdef=1, name=name)
        # Store the variables
        self.A = A
        self.B = B
        self.var_eta = var_eta
        self.var_epsilon = var_epsilon
        
        # Register them - use UNPREFIXED names
        self.register_variable("A", self.A)
        self.register_variable("B", self.B)
        self.register_variable("var_eta", self.var_eta)
        self.register_variable("var_epsilon", self.var_epsilon)
    
    @property
    def param_names(self):
        return ["A", "B", "var_eta", "var_epsilon"]
```

**Important**: When calling these methods, always use **unprefixed** parameter names (e.g., `"A"`, not `"model1_A"`). The library handles the prefixing internally when looking up variables in the PyMC model.

### Finding Required Parameters

To see what parameters a model requires, check the model's `param_names` property:

```python
model = BayesianSARIMA(order=(1, 0, 1), name="my_model")
print(model.param_names)
# Output: ['x0', 'P0', 'ar_params', 'ma_params', 'sigma_state']

# In your PyMC model, you need to create:
# - my_model_x0
# - my_model_P0
# - my_model_ar_params
# - my_model_ma_params
# - my_model_sigma_state
```

### Backward Compatibility

If you don't provide a `name` parameter, the model works exactly as before:

```python
# This still works - no name prefix needed
model = BayesianSARIMA(order=(1, 0, 1))

with pm.Model() as mod:
    x0 = pm.Normal("x0", shape=model.k_states)  # No prefix
    # ... other parameters without prefixes
    model.build_statespace_graph(data)
```

### Observation Variables

The observation variable (typically called `"obs"`) is also prefixed:

```python
model = BayesianSARIMA(order=(1, 0, 1), name="series1")

with pm.Model() as mod:
    # ... define parameters ...
    model.build_statespace_graph(data)
    
    # The observation variable will be named "series1_obs"
    # Access it via: mod["series1_obs"]
```

### Output Variables in Sampling Methods

All output variables from sampling methods are also prefixed:

```python
model = BayesianSARIMA(order=(1, 0, 1), name="series1")

# ... fit the model ...

# Forecasting
forecast_idata = model.forecast(idata, start=100, periods=10)
# Output variables: "series1_forecast_latent", "series1_forecast_observed"

# Impulse response
irf_idata = model.impulse_response_function(idata)
# Output variable: "series1_irf"

# Conditional posterior
posterior_idata = model.sample_conditional_posterior(idata)
# Output variables: "series1_filtered_posterior", "series1_smoothed_posterior", etc.
```

## Common Use Cases

### Multiple Time Series from Different Regions

```python
regions = ["north", "south", "east", "west"]
models = {region: BayesianSARIMA(order=(1,0,1), name=region) for region in regions}

with pm.Model() as hierarchical_model:
    # Hierarchical priors
    global_ar_mean = pm.Normal("global_ar_mean", 0, 1)
    global_ar_sd = pm.Exponential("global_ar_sd", 1)
    
    for region in regions:
        # Region-specific parameters
        ar = pm.Normal(f"{region}_ar_params", global_ar_mean, global_ar_sd, shape=1)
        # ... define other parameters ...
        
        models[region].build_statespace_graph(data[region])
    
    idata = pm.sample()
```

### Panel Data Analysis

```python
individuals = ["person_1", "person_2", "person_3"]
models = {ind: StructuralTimeSeries(name=ind) for ind in individuals}

with pm.Model() as panel_model:
    for ind in individuals:
        # Individual-specific parameters
        # ... define parameters with f"{ind}_" prefix ...
        models[ind].build_statespace_graph(individual_data[ind])
    
    idata = pm.sample()
```

## Error Messages

If you forget to prefix a parameter, you'll get a helpful error message:

```python
model = BayesianSARIMA(order=(1,0,1), name="series1")

with pm.Model() as mod:
    x0 = pm.Normal("x0", shape=model.k_states)  # Forgot to prefix!
    # ...
    model.build_statespace_graph(data)
    
# Error: The following required model parameters were not found in the PyMC model 
#        (model name: 'series1'): series1_x0
```

## Tips and Best Practices

1. **Use descriptive names**: Choose names that clearly identify what each model represents
   - Good: `name="northeast_region"`, `name="product_sales"`
   - Avoid: `name="m1"`, `name="model"`

2. **Be consistent**: Use the same naming convention throughout your code

3. **Document your models**: Keep track of which model corresponds to which data series

4. **Check param_names**: Always verify the required parameters using `model.param_names`

5. **Test with simple data first**: Verify your setup works with small datasets before scaling up
