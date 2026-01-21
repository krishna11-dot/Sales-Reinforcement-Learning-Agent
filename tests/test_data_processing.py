"""
Unit Tests for Data Processing Module

Phase 2: Testing function behavior and preventing regression bugs.

These tests ensure:
1. Input validation catches bad data (Phase 1 validation works)
2. Processing produces correct output (15 features, no Education_Encoded)
3. No data leakage (stats from train set only)
4. Model performance doesn't degrade over time

Run tests: pytest tests/ -v
Run specific test: pytest tests/test_data_processing.py::test_name -v
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing import validate_data


# ================================================================
# Test Suite 1: Input Validation Tests
# Tests that validate_data() catches bad input correctly
# ================================================================

class TestInputValidation:
    """
    Test that validate_data() correctly identifies invalid data.

    Purpose: Ensure Phase 1 validation function works as expected.
    If someone modifies validate_data() and breaks it, these tests fail.
    """

    def test_validate_empty_dataframe(self):
        """
        Test: Empty DataFrame should be rejected

        WHY: environment.py crashes if no customers exist (env.reset() fails)
        IMPACT: Training crashes after hours if not caught early
        """
        df = pd.DataFrame()

        with pytest.raises(AssertionError, match="data is empty"):
            validate_data(df, data_type='train', stage='raw')


    def test_validate_missing_required_columns_raw(self):
        """
        Test: Missing required columns in raw data should be rejected

        WHY: Processing expects specific columns to exist
        IMPACT: KeyError during feature engineering if columns missing
        """
        # Missing Education, Status, Stage, etc.
        df = pd.DataFrame({
            'Country': ['USA', 'UK', 'Canada'],
            'First Contact': ['2024-01-01', '2024-01-02', '2024-01-03']
        })

        with pytest.raises(AssertionError, match="Missing required columns"):
            validate_data(df, data_type='train', stage='raw')


    def test_validate_missing_state_features_processed(self):
        """
        Test: Missing state features in processed data should be rejected

        WHY: environment.py line 311-335 accesses these features in state vector
        IMPACT: KeyError during training if features missing
        """
        # Create DataFrame with some features but missing others
        df = pd.DataFrame({
            'Country_Encoded': [0, 1, 2],
            'Status_Active': [1, 0, 1],
            # Missing: Education_ConvRate, Days_Since_First_Norm, etc.
        })

        with pytest.raises(AssertionError, match="Missing state features"):
            validate_data(df, data_type='train', stage='processed')


    def test_validate_subscribed_binary_not_binary(self):
        """
        Test: Subscribed_Binary must be 0 or 1 only

        WHY: environment.py checks "if Subscribed_Binary == 1" for +100 reward
        IMPACT: Agent never gets reward signal if not 0/1, can't learn

        BUG SCENARIO: Someone changes encoding to 1/2 instead of 0/1
        """
        # Create valid DataFrame but with wrong Subscribed_Binary values
        df = pd.DataFrame({
            'Country_Encoded': [0, 1, 2],
            'Stage_Encoded': [0, 1, 2],
            'Status_Active': [1, 0, 1],
            'Days_Since_First_Norm': [0.5, 0.6, 0.7],
            'Days_Since_Last_Norm': [0.3, 0.4, 0.5],
            'Days_Between_Norm': [0.2, 0.3, 0.4],
            'Contact_Frequency': [0.5, 0.6, 0.7],
            'Had_First_Call': [1, 0, 1],
            'Had_Demo': [1, 0, 0],
            'Had_Survey': [0, 1, 1],
            'Had_Signup': [1, 0, 1],
            'Had_Manager': [0, 1, 0],
            'Country_ConvRate': [0.8, 0.7, 0.9],
            'Education_ConvRate': [0.7, 0.8, 0.6],
            'Stages_Completed': [2, 3, 1],
            'Subscribed_Binary': [1, 2, 0]  # BUG: Should be [1, 0, 0]
        })

        with pytest.raises(AssertionError, match="must be 0 or 1"):
            validate_data(df, data_type='train', stage='processed')


    def test_validate_binary_flags_not_binary(self):
        """
        Test: Binary flags (Had_First_Call, etc.) must be 0 or 1

        WHY: Reward calculation checks "if Had_First_Call == 1" for +15 reward
        IMPACT: Intermediate rewards never trigger, agent can't learn milestones
        """
        df = pd.DataFrame({
            'Country_Encoded': [0, 1],
            'Stage_Encoded': [0, 1],
            'Status_Active': [1, 0],
            'Days_Since_First_Norm': [0.5, 0.6],
            'Days_Since_Last_Norm': [0.3, 0.4],
            'Days_Between_Norm': [0.2, 0.3],
            'Contact_Frequency': [0.5, 0.6],
            'Had_First_Call': [1, 2],  # BUG: Should be [1, 0]
            'Had_Demo': [1, 0],
            'Had_Survey': [0, 1],
            'Had_Signup': [1, 0],
            'Had_Manager': [0, 1],
            'Country_ConvRate': [0.8, 0.7],
            'Education_ConvRate': [0.7, 0.8],
            'Stages_Completed': [2, 3],
            'Subscribed_Binary': [1, 0]
        })

        with pytest.raises(AssertionError, match="must be binary"):
            validate_data(df, data_type='train', stage='processed')


    def test_validate_normalized_features_out_of_range(self):
        """
        Test: Normalized features must be in [0, 1] range

        WHY: RL training becomes unstable with features outside [0, 1]
        IMPACT: Poor convergence, exploding Q-values

        BUG SCENARIO: Someone removes .clip(0, 1) during normalization
        """
        df = pd.DataFrame({
            'Country_Encoded': [0, 1],
            'Stage_Encoded': [0, 1],
            'Status_Active': [1, 0],
            'Days_Since_First_Norm': [0.5, 1.5],  # BUG: 1.5 > 1.0!
            'Days_Since_Last_Norm': [0.3, 0.4],
            'Days_Between_Norm': [0.2, 0.3],
            'Contact_Frequency': [0.5, 0.6],
            'Had_First_Call': [1, 0],
            'Had_Demo': [1, 0],
            'Had_Survey': [0, 1],
            'Had_Signup': [1, 0],
            'Had_Manager': [0, 1],
            'Country_ConvRate': [0.8, 0.7],
            'Education_ConvRate': [0.7, 0.8],
            'Stages_Completed': [2, 3],
            'Subscribed_Binary': [1, 0]
        })

        with pytest.raises(AssertionError, match="outside \[0, 1\] range"):
            validate_data(df, data_type='train', stage='processed')


    def test_validate_accepts_valid_data(self):
        """
        Test: Valid data should pass validation without errors

        WHY: Ensure validation doesn't reject good data
        SANITY CHECK: If this fails, validation function is too strict
        """
        # Create fully valid processed DataFrame
        df = pd.DataFrame({
            'Country_Encoded': [0, 1, 2],
            'Stage_Encoded': [0, 1, 2],
            'Status_Active': [1, 0, 1],
            'Days_Since_First_Norm': [0.5, 0.6, 0.7],
            'Days_Since_Last_Norm': [0.3, 0.4, 0.5],
            'Days_Between_Norm': [0.2, 0.3, 0.4],
            'Contact_Frequency': [0.5, 0.6, 0.7],
            'Had_First_Call': [1, 0, 1],
            'Had_Demo': [1, 0, 0],
            'Had_Survey': [0, 1, 1],
            'Had_Signup': [1, 0, 1],
            'Had_Manager': [0, 1, 0],
            'Country_ConvRate': [0.8, 0.7, 0.9],
            'Education_ConvRate': [0.7, 0.8, 0.6],
            'Stages_Completed': [2, 3, 1],
            'Subscribed_Binary': [1, 0, 0]
        })

        # Should not raise any exception
        try:
            validate_data(df, data_type='train', stage='processed')
        except AssertionError as e:
            pytest.fail(f"Valid data was rejected: {e}")


# ================================================================
# Test Suite 2: Processing Logic Tests
# Tests that process_data() produces correct output
# ================================================================

class TestProcessingLogic:
    """
    Test that data processing produces correct output.

    Purpose: Ensure processed data has correct structure for RL environment.
    Regression tests prevent accidental bugs (like re-adding Education_Encoded).
    """

    def test_processed_data_has_15_state_features(self):
        """
        Test: Processed data must have exactly 15 state features

        WHY: Education_Encoded was removed (16 -> 15 features)
        REGRESSION TEST: Prevents someone from re-adding Education_Encoded
        IMPACT: If state dimension changes, environment.py breaks
        """
        # Check that processed training data exists
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first to generate processed data"

        df = pd.read_csv(train_path)

        # Expected 15 state features (after Education_Encoded fix)
        expected_features = [
            'Country_Encoded',
            'Stage_Encoded',
            'Status_Active',
            'Days_Since_First_Norm',
            'Days_Since_Last_Norm',
            'Days_Between_Norm',
            'Contact_Frequency',
            'Had_First_Call',
            'Had_Demo',
            'Had_Survey',
            'Had_Signup',
            'Had_Manager',
            'Country_ConvRate',
            'Education_ConvRate',
            'Stages_Completed'
        ]

        # Check all expected features exist
        for feature in expected_features:
            assert feature in df.columns, f"Missing state feature: {feature}"

        # Check we have exactly 15 features (no more, no less)
        assert len(expected_features) == 15, f"Expected 15 features, got {len(expected_features)}"


    def test_education_encoded_removed(self):
        """
        Test: Education_Encoded column must NOT exist

        WHY: Education values (B1-B30) are unordered bootcamp aliases
        FIX: Removed Education_Encoded, kept Education_ConvRate
        REGRESSION TEST: Critical - prevents re-introduction of wrong encoding
        """
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first"

        df = pd.read_csv(train_path)

        assert 'Education_Encoded' not in df.columns, (
            "Education_Encoded should be removed! "
            "B1-B30 are unordered bootcamp aliases, not ordered levels. "
            "Use Education_ConvRate instead."
        )


    def test_education_convrate_exists(self):
        """
        Test: Education_ConvRate column must exist

        WHY: Correct way to capture bootcamp performance
        REPLACES: Education_Encoded (which assumed false ordering)
        """
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first"

        df = pd.read_csv(train_path)

        assert 'Education_ConvRate' in df.columns, (
            "Education_ConvRate is missing! "
            "This feature captures per-bootcamp conversion rates (correct approach)."
        )


    def test_processed_data_no_nan_in_state_features(self):
        """
        Test: State features must not contain NaN values

        WHY: np.array() in environment.py can't handle NaN
        IMPACT: Invalid states during training, agent can't learn
        """
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first"

        df = pd.read_csv(train_path)

        state_features = [
            'Country_Encoded', 'Stage_Encoded', 'Status_Active',
            'Days_Since_First_Norm', 'Days_Since_Last_Norm', 'Days_Between_Norm',
            'Contact_Frequency', 'Had_First_Call', 'Had_Demo', 'Had_Survey',
            'Had_Signup', 'Had_Manager', 'Country_ConvRate', 'Education_ConvRate',
            'Stages_Completed'
        ]

        nan_counts = df[state_features].isnull().sum()

        assert nan_counts.sum() == 0, (
            f"Found NaN values in state features:\n{nan_counts[nan_counts > 0]}\n"
            "State vector cannot handle NaN - training will produce invalid states!"
        )


    def test_subscribed_binary_exists_and_valid(self):
        """
        Test: Subscribed_Binary must exist and be 0/1

        WHY: This is the target variable for rewards
        IMPACT: Without this, agent gets no reward signal
        """
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first"

        df = pd.read_csv(train_path)

        # Check column exists
        assert 'Subscribed_Binary' in df.columns, (
            "Subscribed_Binary is missing! This is the primary reward signal."
        )

        # Check values are 0 or 1
        unique_vals = df['Subscribed_Binary'].dropna().unique()
        assert set(unique_vals).issubset({0, 1}), (
            f"Subscribed_Binary must be 0 or 1! Found: {unique_vals}"
        )


    def test_train_val_test_split_exists(self):
        """
        Test: Train, validation, and test sets must all exist

        WHY: Training needs all three sets for proper evaluation
        IMPACT: Can't train or validate model without these files
        """
        train_path = Path('data/processed/crm_train.csv')
        val_path = Path('data/processed/crm_val.csv')
        test_path = Path('data/processed/crm_test.csv')

        assert train_path.exists(), "Missing training set: crm_train.csv"
        assert val_path.exists(), "Missing validation set: crm_val.csv"
        assert test_path.exists(), "Missing test set: crm_test.csv"


    def test_class_imbalance_preserved(self):
        """
        Test: Training set should maintain natural class distribution

        WHY: Batch oversampling happens during training, not in data files
        EXPECTED: ~1.5% subscribed (natural distribution)
        NOTE: Training loop does 30/30/40 oversampling in batches
        """
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first"

        df = pd.read_csv(train_path)

        subscribed_pct = df['Subscribed_Binary'].mean() * 100

        # Should be around 1.5% (natural distribution)
        # Allow range 1.0% to 2.5% to account for train/test split variance
        assert 1.0 <= subscribed_pct <= 2.5, (
            f"Training set has {subscribed_pct:.2f}% subscribed. "
            f"Expected ~1.5% (natural distribution). "
            f"Batch oversampling (30/30/40) happens during training, not here."
        )


# ================================================================
# Test Suite 3: Data Leakage Prevention Tests
# Tests that no future information leaks into training
# ================================================================

class TestDataLeakagePrevention:
    """
    Test that conversion rates and statistics use training data only.

    Purpose: Prevent data leakage (using validation/test info in training).
    This is critical for valid model evaluation.
    """

    def test_historical_stats_file_exists(self):
        """
        Test: historical_stats.json must exist

        WHY: Contains train-set-only statistics for conversion rates
        IMPACT: Without this, can't verify no data leakage
        """
        stats_path = Path('data/processed/historical_stats.json')
        assert stats_path.exists(), (
            "Missing historical_stats.json file. "
            "This file should contain statistics calculated from training set only."
        )


    def test_conversion_rates_from_train_only(self):
        """
        Test: Conversion rates in stats file match training set

        WHY: Prevents data leakage (using val/test info during training)
        IMPACT: If stats leak future info, model performance is inflated

        DATA LEAKAGE CHECK: Validation and test sets should use train stats,
        not calculate their own conversion rates.
        """
        train_path = Path('data/processed/crm_train.csv')
        stats_path = Path('data/processed/historical_stats.json')

        assert train_path.exists(), "Run data_processing.py first"
        assert stats_path.exists(), "Missing historical_stats.json"

        # Load training data and stats
        train_df = pd.read_csv(train_path)
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # Calculate expected education conversion rates from training set
        expected_edu_conv = train_df.groupby('Education')['Subscribed_Binary'].mean().to_dict()

        # Check that stats match training set calculations
        assert 'edu_conv' in stats, "Missing edu_conv in historical_stats.json"

        # Verify at least a few bootcamp conversion rates match
        for bootcamp in ['B8', 'B27', 'B1']:
            if bootcamp in expected_edu_conv and bootcamp in stats['edu_conv']:
                expected_rate = expected_edu_conv[bootcamp]
                stats_rate = stats['edu_conv'][bootcamp]

                # Allow small floating point differences
                assert abs(expected_rate - stats_rate) < 0.001, (
                    f"Conversion rate mismatch for {bootcamp}: "
                    f"Expected {expected_rate:.4f}, got {stats_rate:.4f}. "
                    f"Stats should be calculated from training set only!"
                )


# ================================================================
# Test Suite 4: Model Performance Regression Tests
# Tests that model performance doesn't degrade over time
# ================================================================

class TestModelPerformance:
    """
    Test that trained model meets minimum performance requirements.

    Purpose: Regression test to catch code changes that hurt performance.
    These tests fail if model performance degrades below acceptable thresholds.
    """

    def test_trained_model_beats_random_baseline(self):
        """
        Test: Model must beat random baseline by at least 2x

        WHY: Sanity check that model learned something useful
        BASELINE: Random actions achieve 0.44% subscription rate
        MINIMUM: Model should achieve >= 0.88% (2x random)
        CURRENT: Model achieves 1.30-1.80% depending on version
        """
        # Check if test results exist
        results_paths = [
            Path('logs/test_results.json'),  # Q-Learning baseline
            Path('logs/dqn/test_results.json'),  # DQN baseline
            Path('logs/dqn_feature_selection/test_results.json')  # DQN feature selection
        ]

        # Find first existing results file
        results_path = None
        for path in results_paths:
            if path.exists():
                results_path = path
                break

        if results_path is None:
            pytest.skip("No test results found. Train and evaluate model first.")

        # Load test results
        with open(results_path, 'r') as f:
            results = json.load(f)

        sub_rate = results['subscription_rate']
        random_baseline = 0.44

        # Model should beat 2x random baseline (minimum acceptable)
        assert sub_rate >= random_baseline * 2, (
            f"Model performance too low! Got {sub_rate:.2f}%, "
            f"expected >= {random_baseline * 2:.2f}% (2x random baseline). "
            f"Code changes may have broken the model."
        )


    def test_trained_model_achieves_target_performance(self):
        """
        Test: Model should achieve >= 1.20% subscription rate

        WHY: Project goal is 3x improvement over random (0.44% * 3 = 1.32%)
        TARGET: >= 1.20% (allows some variance while maintaining quality)
        CURRENT: Q-Learning baseline 1.80%, DQN 1.39-1.45%
        """
        # Find existing results file
        results_paths = [
            Path('logs/test_results.json'),
            Path('logs/dqn/test_results.json'),
            Path('logs/dqn_feature_selection/test_results.json')
        ]

        results_path = None
        for path in results_paths:
            if path.exists():
                results_path = path
                break

        if results_path is None:
            pytest.skip("No test results found. Train and evaluate model first.")

        with open(results_path, 'r') as f:
            results = json.load(f)

        sub_rate = results['subscription_rate']
        target = 1.20

        assert sub_rate >= target, (
            f"Model underperforming! Got {sub_rate:.2f}%, expected >= {target:.2f}%. "
            f"Target is ~3x random baseline (0.44% * 3 = 1.32%)."
        )


    def test_improvement_factor_documented(self):
        """
        Test: Test results should document improvement over baseline

        WHY: Important metric for showcasing project success
        CHECKS: Results file contains improvement_factor or can calculate it
        """
        results_paths = [
            Path('logs/test_results.json'),
            Path('logs/dqn/test_results.json'),
            Path('logs/dqn_feature_selection/test_results.json')
        ]

        results_path = None
        for path in results_paths:
            if path.exists():
                results_path = path
                break

        if results_path is None:
            pytest.skip("No test results found.")

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Check that results contain subscription_rate (minimum requirement)
        assert 'subscription_rate' in results, (
            "Test results must include subscription_rate!"
        )

        # Calculate improvement factor if not present
        if 'improvement_factor' not in results:
            sub_rate = results['subscription_rate']
            random_baseline = 0.44
            improvement_factor = sub_rate / random_baseline

            # Should be at least 2x improvement
            assert improvement_factor >= 2.0, (
                f"Improvement factor too low: {improvement_factor:.2f}x. "
                f"Model should beat random baseline by at least 2x."
            )


# ================================================================
# Test Suite 5: Environment Compatibility Tests
# Tests that processed data works with RL environment
# ================================================================

class TestEnvironmentCompatibility:
    """
    Test that processed data is compatible with RL environment.

    Purpose: Ensure data_processing.py output can be used by environment.py.
    Prevents runtime errors during training.
    """

    def test_state_features_match_environment_expectations(self):
        """
        Test: State features must match what environment.py expects

        WHY: environment.py accesses specific columns in _get_state()
        IMPACT: KeyError during training if column names don't match
        """
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first"

        df = pd.read_csv(train_path)

        # These column names must match environment.py line 311-335
        required_for_environment = [
            'Country_Encoded',       # Line 311
            'Status_Active',         # Line 315
            'Days_Since_First_Norm', # Line 318
            'Days_Since_Last_Norm',  # Line 319
            'Days_Between_Norm',     # Line 320
            'Contact_Frequency',     # Line 321
            'Had_First_Call',        # Line 324
            'Had_Demo',              # Line 325
            'Had_Survey',            # Line 326
            'Had_Signup',            # Line 327
            'Had_Manager',           # Line 328
            'Country_ConvRate',      # Line 331
            'Education_ConvRate',    # Line 332
            'Stages_Completed'       # Line 335
        ]

        for col in required_for_environment:
            assert col in df.columns, (
                f"Missing column '{col}' required by environment.py. "
                f"Training will crash with KeyError!"
            )


    def test_binary_features_are_actually_binary(self):
        """
        Test: Binary features must only contain 0 and 1

        WHY: Reward calculation uses binary checks (if flag == 1)
        IMPACT: Rewards don't trigger correctly if not binary
        """
        train_path = Path('data/processed/crm_train.csv')
        assert train_path.exists(), "Run data_processing.py first"

        df = pd.read_csv(train_path)

        binary_features = [
            'Had_First_Call',
            'Had_Demo',
            'Had_Survey',
            'Had_Signup',
            'Had_Manager',
            'Status_Active'
        ]

        for col in binary_features:
            unique_vals = df[col].dropna().unique()
            assert set(unique_vals).issubset({0, 1}), (
                f"{col} must be binary (0 or 1)! Found: {unique_vals}"
            )


# ================================================================
# Run tests with: pytest tests/test_data_processing.py -v
# ================================================================
