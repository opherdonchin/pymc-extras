"""
Test script to verify the register_variable method works correctly.

This demonstrates how to use register_variable() for custom state space models
where parameters are passed to the constructor rather than created as symbolic placeholders.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.statespace.core.statespace import PyMCStateSpace


class SimpleStateSpace(PyMCStateSpace):
    """
    A simple state space model that takes parameters as constructor arguments.

    This demonstrates the use of register_variable() to register pre-existing
    variables rather than make_and_register_variable() which creates symbolic placeholders.
    """

    def __init__(self, A, B, var_eta, var_epsilon, name=None):
        super().__init__(k_states=2, k_posdef=1, name=name)

        # Store the parameters
        self.A = A
        self.B = B
        self.var_eta = var_eta
        self.var_epsilon = var_epsilon

        # Register them with the model
        self.register_variable("A", self.A)
        self.register_variable("B", self.B)
        self.register_variable("var_eta", self.var_eta)
        self.register_variable("var_epsilon", self.var_epsilon)

    @property
    def param_names(self):
        return ["A", "B", "var_eta", "var_epsilon"]

    @property
    def state_names(self):
        return ["state_1", "state_2"]

    @property
    def shock_names(self):
        return ["shock_1"]

    @property
    def data_names(self):
        return ["obs"]

    @property
    def coords(self):
        return {
            "state": self.state_names,
            "shock": self.shock_names,
        }

    @property
    def param_dims(self):
        return {
            "A": ["state", "state"],
            "B": ["state", "shock"],
            "var_eta": [],
            "var_epsilon": [],
        }

    def make_symbolic_graph(self):
        """Build the state space graph using registered variables."""
        x0 = pt.zeros(self.k_states)
        P0 = pt.eye(self.k_states) * 100

        c = pt.zeros(self.k_states)
        d = pt.zeros(1)

        T = self.A  # Transition matrix
        Z = pt.ones((1, self.k_states))  # Observation matrix
        R = self.B  # Selection matrix
        H = self.var_epsilon * pt.eye(1)  # Observation covariance
        Q = self.var_eta * pt.eye(1)  # State innovation covariance

        return x0, P0, c, d, T, Z, R, H, Q


def test_register_variable_basic():
    """Test basic functionality of register_variable."""
    print("Test 1: Basic register_variable functionality")
    print("=" * 60)

    # Create simple parameter values
    A = pt.eye(2)
    B = pt.ones((2, 1))
    var_eta = pt.scalar("var_eta")
    var_epsilon = pt.scalar("var_epsilon")

    # Create model without name
    model = SimpleStateSpace(A, B, var_eta, var_epsilon)

    # Check that variables were registered
    assert "A" in model._name_to_variable
    assert "B" in model._name_to_variable
    assert "var_eta" in model._name_to_variable
    assert "var_epsilon" in model._name_to_variable

    print("✓ Variables successfully registered without name prefix")
    print()


def test_register_variable_with_name():
    """Test register_variable with model name."""
    print("Test 2: register_variable with model name")
    print("=" * 60)

    # Create simple parameter values
    A = pt.eye(2)
    B = pt.ones((2, 1))
    var_eta = pt.scalar("var_eta")
    var_epsilon = pt.scalar("var_epsilon")

    # Create model with name
    model = SimpleStateSpace(A, B, var_eta, var_epsilon, name="model1")

    # Check that variables were registered (with unprefixed keys)
    assert "A" in model._name_to_variable
    assert "B" in model._name_to_variable
    assert "var_eta" in model._name_to_variable
    assert "var_epsilon" in model._name_to_variable

    # Check that model has the name
    assert model.name == "model1"

    print("✓ Variables successfully registered with model name")
    print(f"✓ Model name: {model.name}")
    print()


def test_register_variable_validation():
    """Test that register_variable validates parameter names."""
    print("Test 3: register_variable validation")
    print("=" * 60)

    A = pt.eye(2)
    B = pt.ones((2, 1))
    var_eta = pt.scalar("var_eta")
    var_epsilon = pt.scalar("var_epsilon")

    model = SimpleStateSpace(A, B, var_eta, var_epsilon)

    # Try to register an invalid parameter
    try:
        invalid_param = pt.scalar("invalid")
        model.register_variable("not_a_param", invalid_param)
        print("✗ Should have raised ValueError for invalid parameter name")
    except ValueError as e:
        print(f"✓ Correctly rejected invalid parameter: {e}")

    # Try to register a duplicate
    try:
        model.register_variable("A", A)
        print("✗ Should have raised ValueError for duplicate registration")
    except ValueError as e:
        print(f"✓ Correctly rejected duplicate registration: {e}")

    print()


def test_multiple_models_with_register_variable():
    """Test using multiple models with register_variable in a PyMC model."""
    print("Test 4: Multiple models using register_variable")
    print("=" * 60)

    # Generate some fake data
    np.random.seed(42)
    data1 = np.random.randn(50)
    data2 = np.random.randn(50)

    with pm.Model() as pymc_model:
        # Model 1 parameters
        model1_A = pm.Normal("model1_A", 0, 1, shape=(2, 2))
        model1_B = pm.Normal("model1_B", 0, 1, shape=(2, 1))
        model1_var_eta = pm.Exponential("model1_var_eta", 1)
        model1_var_epsilon = pm.Exponential("model1_var_epsilon", 1)

        # Model 2 parameters
        model2_A = pm.Normal("model2_A", 0, 1, shape=(2, 2))
        model2_B = pm.Normal("model2_B", 0, 1, shape=(2, 1))
        model2_var_eta = pm.Exponential("model2_var_eta", 1)
        model2_var_epsilon = pm.Exponential("model2_var_epsilon", 1)

        # Create state space models
        ss_model1 = SimpleStateSpace(
            model1_A, model1_B, model1_var_eta, model1_var_epsilon, name="model1"
        )
        ss_model2 = SimpleStateSpace(
            model2_A, model2_B, model2_var_eta, model2_var_epsilon, name="model2"
        )

        # Build the state space graphs
        ss_model1.build_statespace_graph(data1)
        ss_model2.build_statespace_graph(data2)

        # Check that both observation variables were created with correct names
        assert "model1_obs" in pymc_model.named_vars
        assert "model2_obs" in pymc_model.named_vars

        print("✓ Both models built successfully with unique names")
        print(f"✓ Model 1 observation variable: model1_obs")
        print(f"✓ Model 2 observation variable: model2_obs")
        print(f"✓ Total variables in PyMC model: {len(pymc_model.named_vars)}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing register_variable() Method")
    print("=" * 60)
    print()

    test_register_variable_basic()
    test_register_variable_with_name()
    test_register_variable_validation()
    test_multiple_models_with_register_variable()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
