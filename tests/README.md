# Test Suite - Phase 2: Unit Testing

## Overview

This test suite validates function behavior and prevents regression bugs in the Sales RL Agent project.

**Phase 1 (Existing):** Input validation in `src/data_processing.py` - catches bad data at runtime
**Phase 2 (This Suite):** Unit tests with pytest - catches code bugs before deployment

---

## Purpose

These tests ensure:

1. **Phase 1 validation works correctly** - Tests that `validate_data()` catches bad input
2. **Processing produces correct output** - Tests that data has 15 features, no Education_Encoded
3. **No data leakage** - Tests that statistics come from training set only
4. **Model performance doesn't degrade** - Regression tests catch performance drops
5. **Environment compatibility** - Tests that processed data works with RL environment

---

## Installation

```bash
# Install pytest
pip install pytest pytest-cov

# Verify installation
pytest --version
```

---

## Running Tests

### Run all tests
```bash
# From project root directory
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run specific test file
```bash
pytest tests/test_data_processing.py -v
```

### Run specific test class
```bash
pytest tests/test_data_processing.py::TestInputValidation -v
```

### Run specific test
```bash
pytest tests/test_data_processing.py::TestInputValidation::test_validate_empty_dataframe -v
```

### Run tests matching pattern
```bash
# Run all tests with "validation" in name
pytest tests/ -v -k "validation"

# Run all tests with "education" in name
pytest tests/ -v -k "education"
```

### Run tests and stop on first failure
```bash
pytest tests/ -v -x
```

---

## Test Structure

```
tests/
├── __init__.py                  # Package marker
├── conftest.py                  # Pytest configuration and fixtures
├── test_data_processing.py      # Main test suite (5 test classes, 20+ tests)
└── README.md                    # This file
```

---

## Test Suites

### 1. TestInputValidation
**Purpose:** Test that `validate_data()` catches invalid input

**Tests:**
- `test_validate_empty_dataframe` - Rejects empty DataFrames
- `test_validate_missing_required_columns_raw` - Catches missing columns in raw data
- `test_validate_missing_state_features_processed` - Catches missing features in processed data
- `test_validate_subscribed_binary_not_binary` - Rejects non-binary target variable
- `test_validate_binary_flags_not_binary` - Rejects non-binary flags (Had_First_Call, etc.)
- `test_validate_normalized_features_out_of_range` - Catches values outside [0, 1]
- `test_validate_accepts_valid_data` - Ensures valid data passes

**Why this matters:** If someone modifies `validate_data()` and breaks it, these tests fail immediately.

---

### 2. TestProcessingLogic
**Purpose:** Test that data processing produces correct output

**Tests:**
- `test_processed_data_has_15_state_features` - Verifies 15 features (after Education fix)
- `test_education_encoded_removed` - **CRITICAL:** Prevents re-adding Education_Encoded
- `test_education_convrate_exists` - Ensures correct encoding is present
- `test_processed_data_no_nan_in_state_features` - No NaN in state vector
- `test_subscribed_binary_exists_and_valid` - Target variable exists and is binary
- `test_train_val_test_split_exists` - All three data files exist
- `test_class_imbalance_preserved` - Natural distribution preserved (~1.5% subscribed)

**Why this matters:** Regression tests prevent accidental bugs (like re-adding Education_Encoded).

---

### 3. TestDataLeakagePrevention
**Purpose:** Ensure no future information leaks into training

**Tests:**
- `test_historical_stats_file_exists` - Statistics file exists
- `test_conversion_rates_from_train_only` - Conversion rates calculated from training set only

**Why this matters:** Data leakage inflates model performance - these tests ensure valid evaluation.

---

### 4. TestModelPerformance
**Purpose:** Regression tests to catch performance degradation

**Tests:**
- `test_trained_model_beats_random_baseline` - Model must beat 2x random (0.88%)
- `test_trained_model_achieves_target_performance` - Model must achieve >= 1.20%
- `test_improvement_factor_documented` - Results include improvement factor

**Why this matters:** If code changes hurt performance, these tests fail before deployment.

---

### 5. TestEnvironmentCompatibility
**Purpose:** Ensure processed data works with RL environment

**Tests:**
- `test_state_features_match_environment_expectations` - Column names match environment.py
- `test_binary_features_are_actually_binary` - Binary flags are 0/1 only

**Why this matters:** Prevents runtime errors during training (KeyError, wrong rewards).

---

## Prerequisites for Tests

### Some tests require processed data:
```bash
# Generate processed data first
python src/data_processing.py
```

### Some tests require trained model:
```bash
# Train and evaluate model first
python src/train.py
python src/evaluate.py
```

**Note:** Tests that require files will be **skipped** (not failed) if files don't exist.

---

## Understanding Test Output

### Example: All tests pass
```bash
$ pytest tests/ -v

tests/test_data_processing.py::TestInputValidation::test_validate_empty_dataframe PASSED
tests/test_data_processing.py::TestInputValidation::test_validate_missing_required_columns_raw PASSED
tests/test_data_processing.py::TestProcessingLogic::test_education_encoded_removed PASSED
tests/test_data_processing.py::TestProcessingLogic::test_education_convrate_exists PASSED
tests/test_data_processing.py::TestModelPerformance::test_trained_model_beats_random_baseline PASSED

==================== 20 passed in 2.3s ====================
```

### Example: Test fails (caught a bug!)
```bash
$ pytest tests/ -v

tests/test_data_processing.py::TestProcessingLogic::test_education_encoded_removed FAILED

================================ FAILURES =================================
_________ TestProcessingLogic.test_education_encoded_removed __________

    def test_education_encoded_removed(self):
>       assert 'Education_Encoded' not in df.columns, (
            "Education_Encoded should be removed! "
            "B1-B30 are unordered bootcamp aliases, not ordered levels. "
            "Use Education_ConvRate instead."
        )
E       AssertionError: Education_Encoded should be removed!
E       B1-B30 are unordered bootcamp aliases, not ordered levels.
E       Use Education_ConvRate instead.

tests/test_data_processing.py:312: AssertionError
==================== 1 failed, 19 passed in 2.5s ====================
```

**This is a good thing!** The test caught that someone re-added Education_Encoded. Fix the bug and rerun tests.

---

## Continuous Integration (CI/CD)

### GitHub Actions (Optional - Phase 3)

These tests can run automatically on every commit using GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --cov=src
```

---

## Best Practices

### When to Run Tests

1. **Before committing code** - Catch bugs before they reach GitHub
2. **After modifying code** - Ensure changes don't break existing functionality
3. **Before deploying to production** - Final check that everything works

### When to Write New Tests

1. **When you find a bug** - Write a test that catches it, then fix the bug
2. **When adding new features** - Test that new code works correctly
3. **When modifying critical functions** - Prevent regression bugs

### Test Naming Convention

- `test_<what>_<expected_behavior>` - Clear, descriptive names
- Example: `test_education_encoded_removed` - Tests that Education_Encoded is removed
- Example: `test_validate_empty_dataframe` - Tests validation of empty DataFrames

---

## Troubleshooting

### Tests fail with "Run data_processing.py first"
**Solution:** Generate processed data before running tests:
```bash
python src/data_processing.py
pytest tests/ -v
```

### Tests fail with "No test results found"
**Solution:** Train and evaluate model before running performance tests:
```bash
python src/train.py
python src/evaluate.py
pytest tests/ -v
```

### Import errors (ModuleNotFoundError)
**Solution:** Ensure you're running pytest from project root:
```bash
cd c:\Users\krish\Downloads\Sales_Optimization_Agent
pytest tests/ -v
```

### Tests skip with "SKIPPED"
**Reason:** Some tests require files that don't exist yet
**Solution:** This is normal - tests skip gracefully if prerequisites missing

---

## Coverage Report

### Generate coverage report
```bash
pytest tests/ --cov=src --cov-report=html

# Open htmlcov/index.html in browser to see coverage
```

### Example coverage output
```
---------- coverage: platform win32, python 3.10.x -----------
Name                      Stmts   Miss  Cover
---------------------------------------------
src/data_processing.py      156      8    95%
src/agent.py                 89      3    97%
src/environment.py          245     12    95%
---------------------------------------------
TOTAL                       490     23    95%
```

---

## Interview Talking Points

When asked: **"Do you write tests?"**

**Answer:**
> "Yes, I implemented a comprehensive pytest test suite for my RL project with 20+ unit tests across 5 test classes:
>
> 1. **Input Validation Tests** - Verify that data quality checks work correctly
> 2. **Processing Logic Tests** - Regression tests prevent bugs (like re-adding Education_Encoded)
> 3. **Data Leakage Tests** - Ensure statistics come from training set only
> 4. **Performance Tests** - Catch performance degradation before deployment
> 5. **Compatibility Tests** - Verify processed data works with RL environment
>
> This caught several bugs during development, like when I accidentally removed the wrong column. Tests provide safety when modifying code after weeks away from it."

---

## Related Documentation

- **Phase 1:** `src/data_processing.py` - Input validation at runtime
- **Phase 2:** `tests/test_data_processing.py` - Unit tests (this suite)
- **Phase 3:** `.github/workflows/test.yml` - CI/CD automation (optional)

---

## Summary

**Phase 1 (validate_data):** Catches bad data during execution
**Phase 2 (pytest):** Catches code bugs before deployment
**Together:** Comprehensive quality assurance for the project

**Run tests regularly** to catch bugs early and maintain code quality!
