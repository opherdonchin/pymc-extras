"""
Test script to verify that multiple state space models can coexist in a single PyMC model.
"""

import numpy as np
import pymc as pm
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.models.utilities import make_default_coords

floatX = "float64"


class SimpleStateSpace(PyMCStateSpace):
    @property
    def param_names(self):
        return ["rho", "sigma"]

    @property
    def state_names(self):
        return ["state_1", "state_2"]

    @property
    def observed_states(self):
        return ["state_1"]

    @property
    def shock_names(self):
        return ["shock_1"]

    @property
    def coords(self):
        return make_default_coords(self)

    def make_symbolic_graph(self):
        rho = self.make_and_register_variable("rho", ())
        sigma = self.make_and_register_variable("sigma", ())

        # Set up simple statespace matrices
        self.ssm["transition", 0, 0] = rho
        self.ssm["transition", 1, 0] = 0.5
        self.ssm["state_cov", 0, 0] = sigma
        self.ssm["initial_state", :] = 0.0
        self.ssm["initial_state_cov", :, :] = np.eye(2, dtype=floatX)


# Generate some test data
np.random.seed(42)
data1 = np.random.randn(50, 1).astype(floatX)
data2 = np.random.randn(50, 1).astype(floatX)

# Test 1: Without names (should work as before)
print("Test 1: Single model without name")
ss_mod1 = SimpleStateSpace(k_states=2, k_endog=1, k_posdef=1, verbose=False)

# Set up basic statespace structure
Z = np.array([[1.0, 0.0]], dtype=floatX)
R = np.array([[1.0], [0.0]], dtype=floatX)
H = np.array([[0.1]], dtype=floatX)
Q = np.array([[0.8]], dtype=floatX)

ss_mod1.ssm["design", :, :] = Z
ss_mod1.ssm["selection", :, :] = R
ss_mod1.ssm["obs_cov", :, :] = H

with pm.Model() as model1:
    rho = pm.Normal("rho", 0, 1)
    sigma = pm.Exponential("sigma", 1)

    ss_mod1.build_statespace_graph(data1, register_data=True)

    # Check that variables were created
    assert "rho" in model1.named_vars
    assert "sigma" in model1.named_vars
    assert "obs" in model1.named_vars
    print("✓ Single model without name works correctly")

# Test 2: Two models with different names
print("\nTest 2: Two models with different names")
ss_mod_a = SimpleStateSpace(
    k_states=2, k_endog=1, k_posdef=1, verbose=False, name="model_a"
)
ss_mod_b = SimpleStateSpace(
    k_states=2, k_endog=1, k_posdef=1, verbose=False, name="model_b"
)

# Set up the same basic structure for both models
for ss_mod in [ss_mod_a, ss_mod_b]:
    ss_mod.ssm["design", :, :] = Z
    ss_mod.ssm["selection", :, :] = R
    ss_mod.ssm["obs_cov", :, :] = H

with pm.Model() as model2:
    # Model A parameters
    rho_a = pm.Normal("model_a_rho", 0, 1)
    sigma_a = pm.Exponential("model_a_sigma", 1)

    # Model B parameters
    rho_b = pm.Normal("model_b_rho", 0, 1)
    sigma_b = pm.Exponential("model_b_sigma", 1)

    # Build both models
    ss_mod_a.build_statespace_graph(data1, register_data=True)
    ss_mod_b.build_statespace_graph(data2, register_data=True)

    # Check that all variables were created with proper prefixes
    assert "model_a_rho" in model2.named_vars
    assert "model_a_sigma" in model2.named_vars
    assert "model_a_obs" in model2.named_vars

    assert "model_b_rho" in model2.named_vars
    assert "model_b_sigma" in model2.named_vars
    assert "model_b_obs" in model2.named_vars

    # Make sure unprefixed names don't exist (would conflict)
    assert (
        "rho" not in model2.named_vars
        or model2.named_vars["rho"] == rho_a
        or model2.named_vars["rho"] == rho_b
    )
    assert (
        "sigma" not in model2.named_vars
        or model2.named_vars["sigma"] == sigma_a
        or model2.named_vars["sigma"] == sigma_b
    )

    print("✓ Two models with different names work correctly")
    print(
        f"  Model variables: {sorted([k for k in model2.named_vars.keys() if 'model_' in k])}"
    )

print(
    "\n✅ All tests passed! Multiple state space models can now coexist in a single PyMC model."
)
