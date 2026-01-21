"""
Pytest Configuration and Shared Fixtures

This file configures pytest behavior and provides reusable test fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_valid_processed_data():
    """
    Fixture: Returns a small valid processed DataFrame for testing

    Use this for tests that need valid data without loading full dataset.
    """
    return pd.DataFrame({
        'Country_Encoded': [0, 1, 2, 3, 4],
        'Stage_Encoded': [0, 1, 2, 1, 0],
        'Status_Active': [1, 0, 1, 1, 0],
        'Days_Since_First_Norm': [0.5, 0.6, 0.7, 0.4, 0.8],
        'Days_Since_Last_Norm': [0.3, 0.4, 0.5, 0.2, 0.6],
        'Days_Between_Norm': [0.2, 0.3, 0.4, 0.1, 0.5],
        'Contact_Frequency': [0.5, 0.6, 0.7, 0.4, 0.8],
        'Had_First_Call': [1, 0, 1, 1, 0],
        'Had_Demo': [1, 0, 0, 1, 0],
        'Had_Survey': [0, 1, 1, 0, 0],
        'Had_Signup': [1, 0, 1, 1, 0],
        'Had_Manager': [0, 1, 0, 0, 1],
        'Country_ConvRate': [0.8, 0.7, 0.9, 0.6, 0.75],
        'Education_ConvRate': [0.7, 0.8, 0.6, 0.9, 0.65],
        'Stages_Completed': [2, 3, 1, 4, 2],
        'Subscribed_Binary': [1, 0, 0, 1, 0]
    })


@pytest.fixture
def sample_invalid_data_empty():
    """
    Fixture: Returns empty DataFrame for testing validation rejection
    """
    return pd.DataFrame()


@pytest.fixture
def sample_invalid_data_missing_columns():
    """
    Fixture: Returns DataFrame with missing required columns
    """
    return pd.DataFrame({
        'Country': ['USA', 'UK', 'Canada'],
        'Status': ['Active', 'Inactive', 'Active']
        # Missing: Education, Stage, etc.
    })


@pytest.fixture
def sample_invalid_data_non_binary_target():
    """
    Fixture: Returns DataFrame with invalid Subscribed_Binary values
    """
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
        'Subscribed_Binary': [1, 2, 0]  # Invalid: should be [1, 0, 0]
    })
    return df


@pytest.fixture
def project_root():
    """
    Fixture: Returns path to project root directory

    Useful for constructing paths to data files, checkpoints, etc.
    """
    return Path(__file__).parent.parent


@pytest.fixture
def processed_data_path(project_root):
    """
    Fixture: Returns path to processed data directory
    """
    return project_root / 'data' / 'processed'


@pytest.fixture
def train_data_path(processed_data_path):
    """
    Fixture: Returns path to training data file
    """
    return processed_data_path / 'crm_train.csv'


@pytest.fixture
def test_results_paths(project_root):
    """
    Fixture: Returns list of possible test results file paths

    Checks Q-Learning and DQN results locations.
    """
    return [
        project_root / 'logs' / 'test_results.json',
        project_root / 'logs' / 'dqn' / 'test_results.json',
        project_root / 'logs' / 'dqn_feature_selection' / 'test_results.json'
    ]


# Configure pytest output
def pytest_configure(config):
    """
    Configure pytest with custom markers and settings
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require full dataset)"
    )
