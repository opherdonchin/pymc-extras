"""
Test script to verify that data variables are properly prefixed when using multiple state space models.

This demonstrates that the data registration process correctly handles model names to avoid conflicts.
"""

import numpy as np
import pymc as pm
from pymc_extras.statespace.models import BayesianSARIMA


def test_single_model_without_name():
    """Test that a single model without a name creates 'data' variable."""
    print("Test 1: Single model without name")
    print("=" * 60)

    np.random.seed(42)
    data = np.random.randn(50)

    model = BayesianSARIMA(order=(1, 0, 1))

    with pm.Model() as pymc_model:
        # Define parameters
        x0 = pm.Normal("x0", 0, 1, shape=model.k_states)
        P0_diag = pm.Exponential("P0_diag", 1, shape=model.k_states)
        P0 = pm.Deterministic("P0", pm.math.stack([P0_diag]))
        ar_params = pm.Normal("ar_params", 0, 1, shape=1)
        ma_params = pm.Normal("ma_params", 0, 1, shape=1)
        sigma_state = pm.Exponential("sigma_state", 1)

        # Build the model
        model.build_statespace_graph(data)

        # Check that "data" variable exists (no prefix)
        assert "data" in pymc_model.named_vars, "Expected 'data' variable not found"
        print("✓ Data variable correctly named 'data' (no prefix)")

    print()


def test_single_model_with_name():
    """Test that a single model with a name creates prefixed data variable."""
    print("Test 2: Single model with name")
    print("=" * 60)

    np.random.seed(42)
    data = np.random.randn(50)

    model = BayesianSARIMA(order=(1, 0, 1), name="model1")

    with pm.Model() as pymc_model:
        # Define parameters with prefix
        x0 = pm.Normal("model1_x0", 0, 1, shape=model.k_states)
        P0_diag = pm.Exponential("model1_P0_diag", 1, shape=model.k_states)
        P0 = pm.Deterministic("model1_P0", pm.math.stack([P0_diag]))
        ar_params = pm.Normal("model1_ar_params", 0, 1, shape=1)
        ma_params = pm.Normal("model1_ma_params", 0, 1, shape=1)
        sigma_state = pm.Exponential("model1_sigma_state", 1)

        # Build the model
        model.build_statespace_graph(data)

        # Check that "model1_data" variable exists
        assert (
            "model1_data" in pymc_model.named_vars
        ), "Expected 'model1_data' variable not found"
        print("✓ Data variable correctly named 'model1_data' (with prefix)")

    print()


def test_multiple_models_with_names():
    """Test that multiple models with different names create separate data variables."""
    print("Test 3: Multiple models with different names")
    print("=" * 60)

    np.random.seed(42)
    data1 = np.random.randn(50)
    data2 = np.random.randn(50)

    model1 = BayesianSARIMA(order=(1, 0, 1), name="north")
    model2 = BayesianSARIMA(order=(1, 0, 1), name="south")

    with pm.Model() as pymc_model:
        # Model 1 parameters
        north_x0 = pm.Normal("north_x0", 0, 1, shape=model1.k_states)
        north_P0_diag = pm.Exponential("north_P0_diag", 1, shape=model1.k_states)
        north_P0 = pm.Deterministic("north_P0", pm.math.stack([north_P0_diag]))
        north_ar_params = pm.Normal("north_ar_params", 0, 1, shape=1)
        north_ma_params = pm.Normal("north_ma_params", 0, 1, shape=1)
        north_sigma_state = pm.Exponential("north_sigma_state", 1)

        # Model 2 parameters
        south_x0 = pm.Normal("south_x0", 0, 1, shape=model2.k_states)
        south_P0_diag = pm.Exponential("south_P0_diag", 1, shape=model2.k_states)
        south_P0 = pm.Deterministic("south_P0", pm.math.stack([south_P0_diag]))
        south_ar_params = pm.Normal("south_ar_params", 0, 1, shape=1)
        south_ma_params = pm.Normal("south_ma_params", 0, 1, shape=1)
        south_sigma_state = pm.Exponential("south_sigma_state", 1)

        # Build both models
        model1.build_statespace_graph(data1)
        model2.build_statespace_graph(data2)

        # Check that both data variables exist with correct names
        assert (
            "north_data" in pymc_model.named_vars
        ), "Expected 'north_data' variable not found"
        assert (
            "south_data" in pymc_model.named_vars
        ), "Expected 'south_data' variable not found"

        # Verify no unprefixed "data" variable exists
        assert (
            "data" not in pymc_model.named_vars
        ), "Unprefixed 'data' variable should not exist"

        print("✓ Data variables correctly named 'north_data' and 'south_data'")
        print(f"✓ Total named variables in model: {len(pymc_model.named_vars)}")

    print()


def test_loop_multiple_subjects():
    """Test creating multiple models in a loop (simulating panel data)."""
    print("Test 4: Multiple models created in a loop")
    print("=" * 60)

    np.random.seed(42)
    subjects = ["subject_1", "subject_2", "subject_3"]
    subject_data = {s: np.random.randn(50) for s in subjects}

    models = {s: BayesianSARIMA(order=(1, 0, 1), name=s) for s in subjects}

    with pm.Model() as pymc_model:
        for subject in subjects:
            model = models[subject]

            # Create subject-specific parameters
            x0 = pm.Normal(f"{subject}_x0", 0, 1, shape=model.k_states)
            P0_diag = pm.Exponential(f"{subject}_P0_diag", 1, shape=model.k_states)
            P0 = pm.Deterministic(f"{subject}_P0", pm.math.stack([P0_diag]))
            ar_params = pm.Normal(f"{subject}_ar_params", 0, 1, shape=1)
            ma_params = pm.Normal(f"{subject}_ma_params", 0, 1, shape=1)
            sigma_state = pm.Exponential(f"{subject}_sigma_state", 1)

            # Build the model with subject-specific data
            model.build_statespace_graph(subject_data[subject])

        # Check that all data variables exist with correct names
        for subject in subjects:
            expected_data_name = f"{subject}_data"
            assert (
                expected_data_name in pymc_model.named_vars
            ), f"Expected '{expected_data_name}' variable not found"
            print(f"✓ Data variable '{expected_data_name}' created successfully")

        # Verify no conflicts
        print(f"✓ All {len(subjects)} models created without naming conflicts")
        print(f"✓ Total named variables in model: {len(pymc_model.named_vars)}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Data Variable Naming with Model Prefixes")
    print("=" * 60)
    print()

    test_single_model_without_name()
    test_single_model_with_name()
    test_multiple_models_with_names()
    test_loop_multiple_subjects()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
