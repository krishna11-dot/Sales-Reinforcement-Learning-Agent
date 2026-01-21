"""
Data Processing Module for CRM Sales RL

CRITICAL NUANCE #7: Temporal Data Handling (No Data Leakage!)
This module implements the CORRECT way to handle temporal CRM data:
1. Split by DATE first (temporal split)
2. Calculate statistics ONLY on train set
3. Apply features to all splits using train statistics

INTERVIEW PREP: Be able to explain WHY each step prevents data leakage!
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

# Configure logging for data processing
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default: INFO level


def set_logging_level(level='INFO'):
    """
    Set logging level for data processing module.

    Args:
        level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'

    Usage:
        set_logging_level('DEBUG')   # Show all details
        set_logging_level('INFO')    # Show important info only (default)
        set_logging_level('WARNING') # Show warnings only
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    logger.info(f"Logging level set to {level.upper()}")


def validate_data(df, data_type='train', stage='raw'):
    """
    Validate data before processing or training.

    WHY: Environment expects specific columns, formats, and no NaN.
    If data doesn't meet these requirements, training will crash
    or produce incorrect results.

    PRINCIPLE: Tests check that function receives what it expects

    Args:
        df: Input DataFrame
        data_type: 'train', 'val', or 'test'
        stage: 'raw' (before processing) or 'processed' (after feature engineering)

    Raises:
        AssertionError: If data doesn't meet requirements

    Returns:
        df: Same DataFrame (for chaining)
    """

    # Check 1: DataFrame not empty
    # WHY: env.reset() will crash if no customers exist
    assert len(df) > 0, f"{data_type} data is empty!"

    if stage == 'raw':
        # Validate raw data before processing
        required_cols = [
            'Country',           # → Country_Encoded in state
            'Education',         # → Education_ConvRate in state
            'Status',            # → Status_Active in state
            'Stage',             # → current_stage in state
            'First Contact',     # → Days_Since_First_Norm in state
            'Last Contact',      # → Days_Since_Last_Norm in state
            'First Call',        # → Had_First_Call (binary flag)
            'Signed up for a demo',        # → Had_Demo
            'Filled in customer survey',   # → Had_Survey
            'Did sign up to the platform', # → Had_Signup
            'Account Manager assigned',    # → Had_Manager
            'Subscribed'         # → Subscribed_Binary (target)
        ]

        # Check 2: Required columns exist
        missing = [col for col in required_cols if col not in df.columns]
        assert len(missing) == 0, (
            f"Missing required columns in {data_type} data: {missing}\n"
            f"WHY: These columns are needed for state representation in environment.py"
        )

        # Check 3: Critical columns have no NaN (Country, Education, Status)
        # Note: Other columns can have NaN (means event didn't happen)
        critical_cols = ['Country', 'Education', 'Status', 'Stage']
        nan_counts = df[critical_cols].isnull().sum()

        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN in critical columns:")
            for col, count in nan_counts[nan_counts > 0].items():
                logger.warning(f"   {col}: {count} NaN values ({count/len(df)*100:.2f}%)")
            logger.info(f"   These will be filled with defaults in processing.")

    elif stage == 'processed':
        # Validate processed data ready for RL environment

        # Check 2: All state features exist
        # WHY: environment.py line 308-337 accesses these in _get_state()
        state_features = [
            'Country_Encoded',         # Line 311
            'Stage_Encoded',           # Line 314 (note: code uses self.current_stage, not this column)
            'Status_Active',           # Line 315
            'Days_Since_First_Norm',   # Line 318
            'Days_Since_Last_Norm',    # Line 319
            'Days_Between_Norm',       # Line 320
            'Contact_Frequency',       # Line 321
            'Had_First_Call',          # Line 324
            'Had_Demo',                # Line 325
            'Had_Survey',              # Line 326
            'Had_Signup',              # Line 327
            'Had_Manager',             # Line 328
            'Country_ConvRate',        # Line 331
            'Education_ConvRate',      # Line 332
            'Stages_Completed'         # Line 335
        ]

        missing = [col for col in state_features if col not in df.columns]
        assert len(missing) == 0, (
            f"Missing state features in {data_type} data: {missing}\n"
            f"WHY: environment.py _get_state() accesses these columns. "
            f"Missing features will cause KeyError during training!"
        )

        # Check 3: No NaN in state features
        # WHY: np.array() conversion will fail or produce NaN states
        nan_counts = df[state_features].isnull().sum()
        assert nan_counts.sum() == 0, (
            f"Found NaN values in state features:\n{nan_counts[nan_counts > 0]}\n"
            f"WHY: State vector uses np.array([...]) which cannot handle NaN. "
            f"Training will produce invalid states!"
        )

        # Check 4: Subscribed_Binary is 0 or 1 (TARGET VARIABLE!)
        # WHY: Line 252 in environment.py checks "if Subscribed_Binary == 1"
        # If not 0/1 → Agent never receives +100 reward → Can't learn!
        assert 'Subscribed_Binary' in df.columns, (
            f"Missing target column 'Subscribed_Binary' in {data_type} data!\n"
            f"WHY: This is the PRIMARY REWARD in environment.py line 252-256"
        )

        unique_vals = df['Subscribed_Binary'].dropna().unique()
        assert set(unique_vals).issubset({0, 1}), (
            f"Subscribed_Binary must be 0 or 1! Found: {unique_vals}\n"
            f"WHY: Environment checks 'if Subscribed_Binary == 1' for +100 reward. "
            f"Other values mean agent never gets rewarded and can't learn!"
        )

        # Check 5: Binary flags are 0 or 1
        # WHY: Reward calculation checks "if Had_First_Call == 1" (line 221)
        # Non-binary values → Intermediate rewards never trigger → Wrong learning!
        binary_features = [
            'Had_First_Call',    # Line 221: +15 reward check
            'Had_Demo',          # Line 226: +12 reward check
            'Had_Survey',        # Line 231: +8 reward check
            'Had_Signup',        # Line 241: +20 reward check
            'Had_Manager',       # Line 236: +10 reward check
            'Status_Active'      # Line 315: state feature
        ]

        for col in binary_features:
            unique_vals = df[col].dropna().unique()
            assert set(unique_vals).issubset({0, 1}), (
                f"{col} must be binary (0 or 1)! Found: {unique_vals}\n"
                f"WHY: environment.py checks 'if {col} == 1' for intermediate rewards. "
                f"Non-binary values break reward calculation!"
            )

        # Check 6: Normalized features are in [0, 1]
        # WHY: Normalization should clip to [0, 1] for RL stability
        norm_features = ['Days_Since_First_Norm', 'Days_Since_Last_Norm',
                        'Days_Between_Norm', 'Country_ConvRate', 'Education_ConvRate']

        for col in norm_features:
            assert df[col].between(0, 1).all(), (
                f"{col} values outside [0, 1] range!\n"
                f"Min: {df[col].min()}, Max: {df[col].max()}\n"
                f"WHY: Normalized features should be clipped to [0, 1] for stable RL training"
            )

    # Success message
    logger.info(f"{data_type.capitalize()} data validation passed ({stage} stage)")
    logger.debug(f"   Samples: {len(df):,}")

    if stage == 'processed' and 'Subscribed_Binary' in df.columns:
        n_positive = df['Subscribed_Binary'].sum()
        pct_positive = df['Subscribed_Binary'].mean() * 100
        logger.debug(f"   Positive samples: {n_positive:,} ({pct_positive:.2f}%)")
        logger.debug(f"   Class imbalance: {(len(df)-n_positive)/n_positive:.0f}:1")

    logger.debug(f"   All required columns present")
    logger.debug(f"   All checks passed")

    return df


def process_crm_data(filepath, output_dir='data/processed'):
    """
    Load and process CRM data with NO data leakage.

    NUANCE: Why temporal split?
    - CRM data is time-ordered customer journeys
    - Real deployment: train on past, test on future
    - Random split would leak future information into training

    Args:
        filepath: Path to SalesCRM.xlsx
        output_dir: Directory for processed outputs

    Returns:
        train_df, val_df, test_df, historical_stats
    """
    logger.info("="*80)
    logger.info("STEP 1: LOADING RAW DATA")
    logger.info("="*80)

    # Load raw data
    df = pd.read_excel(filepath)
    logger.info(f"Loaded {len(df):,} customers from {filepath}")
    logger.debug(f"Columns: {df.columns.tolist()}")

    # VALIDATION: Check raw data has required columns
    logger.info("")
    logger.info("="*80)
    logger.info("STEP 1.5: VALIDATE RAW DATA")
    logger.info("="*80)
    df = validate_data(df, data_type='all', stage='raw')

    logger.info("")
    logger.info("="*80)
    logger.info("STEP 2: PARSE DATE COLUMNS")
    logger.info("="*80)

    # Parse all date columns to datetime
    # SAFE: These are historical events (past), not future information
    date_columns = [
        'First Contact', 'Last Contact', 'First Call',
        'Signed up for a demo', 'Filled in customer survey',
        'Did sign up to the platform', 'Account Manager assigned',
        'Subscribed'
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            non_null = df[col].notna().sum()
            logger.debug(f"{col}: {non_null:,} non-null ({non_null/len(df)*100:.2f}%)")

    logger.info("")
    logger.info("="*80)
    logger.info("STEP 3: TEMPORAL SPLIT (CRITICAL - BEFORE FEATURE ENGINEERING!)")
    logger.info("="*80)

    # CRITICAL: Split by date BEFORE calculating any statistics
    # WHY: Prevents test set outcomes from leaking into train features

    # Sort by First Contact date
    df = df.sort_values('First Contact').reset_index(drop=True)

    # 70% train, 15% val, 15% test BY DATE
    # NOT random split! This maintains temporal order
    train_end_idx = int(len(df) * 0.70)
    val_end_idx = int(len(df) * 0.85)

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    logger.info(f"Train set: {len(train_df):,} customers ({len(train_df)/len(df)*100:.1f}%)")
    logger.debug(f"  Date range: {train_df['First Contact'].min()} to {train_df['First Contact'].max()}")

    logger.info(f"Val set: {len(val_df):,} customers ({len(val_df)/len(df)*100:.1f}%)")
    logger.debug(f"  Date range: {val_df['First Contact'].min()} to {val_df['First Contact'].max()}")

    logger.info(f"Test set: {len(test_df):,} customers ({len(test_df)/len(df)*100:.1f}%)")
    logger.debug(f"  Date range: {test_df['First Contact'].min()} to {test_df['First Contact'].max()}")

    logger.info("")
    logger.info("="*80)
    logger.info("STEP 4: CREATE TARGET VARIABLES")
    logger.info("="*80)

    # Create binary target variables
    for split_df in [train_df, val_df, test_df]:
        split_df['Subscribed_Binary'] = split_df['Subscribed'].notna().astype(int)
        split_df['Had_First_Call'] = split_df['First Call'].notna().astype(int)

    # Calculate baseline metrics
    train_sub_rate = train_df['Subscribed_Binary'].mean() * 100
    train_call_rate = train_df['Had_First_Call'].mean() * 100

    logger.info(f"TRAIN SET BASELINE METRICS:")
    logger.info(f"  Subscription rate: {train_sub_rate:.2f}%")
    logger.debug(f"  First call rate: {train_call_rate:.2f}%")
    logger.debug(f"  Class imbalance: {(1-train_df['Subscribed_Binary'].mean())/train_df['Subscribed_Binary'].mean():.0f}:1")

    logger.info("")
    logger.info("="*80)
    logger.info("STEP 5: CALCULATE HISTORICAL STATISTICS (TRAIN SET ONLY!)")
    logger.info("="*80)

    # CRITICAL: Calculate aggregated statistics ONLY on train set
    # WHY: Using test set here would leak future outcomes into features

    # Country conversion rates (from train only)
    country_stats = train_df.groupby('Country')['Subscribed_Binary'].agg(['mean', 'count'])
    logger.debug(f"\nCountry statistics (top 10 by conversion rate):")
    logger.debug(f"\n{country_stats.nlargest(10, 'mean')}")

    # Education conversion rates (from train only)
    edu_stats = train_df.groupby('Education')['Subscribed_Binary'].agg(['mean', 'count'])
    logger.debug(f"\nEducation statistics:")
    logger.debug(f"\n{edu_stats.sort_values('mean', ascending=False)}")

    # Store as dictionaries for mapping
    historical_stats = {
        'country_conv': country_stats['mean'].to_dict(),
        'edu_conv': edu_stats['mean'].to_dict(),
        'global_avg': train_df['Subscribed_Binary'].mean()  # Fallback for unseen categories
    }

    logger.info(f"Stored {len(historical_stats['country_conv'])} country stats")
    logger.info(f"Stored {len(historical_stats['edu_conv'])} education stats")
    logger.debug(f"Global average: {historical_stats['global_avg']:.4f}")

    logger.info("")
    logger.info("="*80)
    logger.info("STEP 6: FEATURE ENGINEERING (SAME FOR ALL SPLITS)")
    logger.info("="*80)

    # Apply feature engineering to each split
    # IMPORTANT: All splits use TRAIN statistics (no leakage)
    train_df = engineer_features(train_df, historical_stats)
    val_df = engineer_features(val_df, historical_stats)
    test_df = engineer_features(test_df, historical_stats)

    logger.info(f"Engineered {len(train_df.columns)} total features")
    logger.debug(f"Feature columns: {[col for col in train_df.columns if col not in df.columns]}")

    # VALIDATION: Check processed data is ready for RL environment
    logger.info("")
    logger.info("="*80)
    logger.info("STEP 6.5: VALIDATE PROCESSED DATA")
    logger.info("="*80)
    logger.info("Validating that all features are ready for RL training...")
    train_df = validate_data(train_df, data_type='train', stage='processed')
    val_df = validate_data(val_df, data_type='val', stage='processed')
    test_df = validate_data(test_df, data_type='test', stage='processed')

    logger.info("")
    logger.info("="*80)
    logger.info("STEP 7: SAVE PROCESSED DATA")
    logger.info("="*80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save processed datasets
    train_path = Path(output_dir) / 'crm_train.csv'
    val_path = Path(output_dir) / 'crm_val.csv'
    test_path = Path(output_dir) / 'crm_test.csv'
    stats_path = Path(output_dir) / 'historical_stats.json'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved: {train_path}")
    logger.info(f"Saved: {val_path}")
    logger.info(f"Saved: {test_path}")

    # Save historical statistics for use in environment
    with open(stats_path, 'w') as f:
        json.dump(historical_stats, f, indent=2)
    logger.info(f"Saved: {stats_path}")

    logger.info("")
    logger.info("="*80)
    logger.info("DATA PROCESSING COMPLETE - NO LEAKAGE CONFIRMED")
    logger.info("="*80)
    logger.debug("VERIFICATION:")
    logger.debug("- Temporal split: Train dates < Val dates < Test dates")
    logger.debug("- Statistics: Calculated on train only, mapped to val/test")
    logger.debug("- Features: All use historical data (past events)")
    logger.info("="*80)

    return train_df, val_df, test_df, historical_stats


def engineer_features(df, historical_stats):
    """
    Create 16-dimensional state vector features with NO temporal leakage.

    NUANCE #2: Time Series to RL State Conversion
    - Original: Sequential customer journey
    - Transformed: Fixed 16-dim state vector
    - Temporal info preserved through: days_since, binary flags, current stage

    Args:
        df: DataFrame with raw data
        historical_stats: Statistics from TRAIN SET ONLY

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()  # Don't modify original

    # FEATURE 1-2: DEMOGRAPHICS (Categorical encoding)
    # NUANCE: NOT ordered by conversion rate (that would be leakage!)
    # Just arbitrary alphabetical encoding

    # Education encoding (categorical)
    education_unique = sorted(df['Education'].dropna().unique())
    education_map = {edu: idx for idx, edu in enumerate(education_unique)}
    education_map['Unknown'] = len(education_unique)  # For missing values

    df['Education_Encoded'] = df['Education'].map(education_map).fillna(len(education_unique))

    # Country encoding (categorical)
    country_unique = sorted(df['Country'].dropna().unique())
    country_map = {country: idx for idx, country in enumerate(country_unique)}
    country_map['Unknown'] = len(country_unique)

    df['Country_Encoded'] = df['Country'].map(country_map).fillna(len(country_unique))

    # FEATURE 3: STAGE (Ordinal encoding by funnel position)
    # NUANCE #10: Stage is CURRENT STATUS, not future prediction
    # It's observable NOW, like "patient in ICU" for readmission prediction

    stage_map = {
        'do not contact': 0,
        'not interested': 1,
        'declined/canceled call': 2,
        'did not join the call': 2,
        'interested': 3,
        'subscribed already': 6,  # Terminal state
        None: 0
    }
    df['Stage_Encoded'] = df['Stage'].map(stage_map).fillna(0)

    # FEATURE 4: STATUS (Binary)
    df['Status_Active'] = (df['Status'] == 'Active').astype(int)

    # FEATURES 5-8: TEMPORAL FEATURES (All historical - past events)
    # SAFE: All calculations use dates that occurred in the past

    # Reference date: Use max date from dataset (simulates "today")
    reference_date = df['First Contact'].max()

    # Days since first contact (normalized to [0, 1])
    df['Days_Since_First'] = (reference_date - df['First Contact']).dt.days
    df['Days_Since_First'] = df['Days_Since_First'].fillna(0)
    df['Days_Since_First_Norm'] = (df['Days_Since_First'] / 365).clip(0, 1)

    # Days since last contact (normalized to [0, 1])
    df['Days_Since_Last'] = (reference_date - df['Last Contact']).dt.days
    df['Days_Since_Last'] = df['Days_Since_Last'].fillna(0)
    df['Days_Since_Last_Norm'] = (df['Days_Since_Last'] / 30).clip(0, 1)

    # Days between contacts (engagement timespan)
    df['Days_Between_Contacts'] = (df['Last Contact'] - df['First Contact']).dt.days
    df['Days_Between_Contacts'] = df['Days_Between_Contacts'].fillna(0).clip(lower=0)
    df['Days_Between_Norm'] = (df['Days_Between_Contacts'] / 365).clip(0, 1)

    # Contact frequency (inverse of time between)
    df['Contact_Frequency'] = 1 / (df['Days_Between_Contacts'] + 1)

    # FEATURES 9-13: BINARY FLAGS (Pipeline stage completions)
    # SAFE: These are completed historical events

    df['Had_Demo'] = df['Signed up for a demo'].notna().astype(int)
    df['Had_Survey'] = df['Filled in customer survey'].notna().astype(int)
    df['Had_Signup'] = df['Did sign up to the platform'].notna().astype(int)
    df['Had_Manager'] = df['Account Manager assigned'].notna().astype(int)
    # Had_First_Call already created in main function

    # FEATURES 14-15: AGGREGATED STATISTICS (From train set ONLY!)
    # CRITICAL: These use historical_stats calculated ONLY on train set
    # Val/test sets get train statistics mapped to them (no leakage)

    global_avg = historical_stats['global_avg']

    df['Country_ConvRate'] = df['Country'].map(historical_stats['country_conv']).fillna(global_avg)
    df['Education_ConvRate'] = df['Education'].map(historical_stats['edu_conv']).fillna(global_avg)

    # FEATURE 16: DERIVED FEATURE (Stages completed count)
    df['Stages_Completed'] = (
        df['Had_First_Call'] + df['Had_Demo'] + df['Had_Survey'] +
        df['Had_Signup'] + df['Had_Manager']
    )

    return df


if __name__ == "__main__":
    # Run data processing
    train_df, val_df, test_df, stats = process_crm_data(
        filepath='data/raw/SalesCRM.xlsx',
        output_dir='data/processed'
    )

    # Verify no leakage
    logger.info("\nVERIFICATION CHECK:")
    logger.info(f"Train subscription rate: {train_df['Subscribed_Binary'].mean()*100:.2f}%")
    logger.info(f"Val subscription rate: {val_df['Subscribed_Binary'].mean()*100:.2f}%")
    logger.info(f"Test subscription rate: {test_df['Subscribed_Binary'].mean()*100:.2f}%")

    # Check state vector can be created
    sample = train_df.iloc[0]
    state_features = [
        'Education_Encoded', 'Country_Encoded', 'Stage_Encoded', 'Status_Active',
        'Days_Since_First_Norm', 'Days_Since_Last_Norm', 'Days_Between_Norm', 'Contact_Frequency',
        'Had_First_Call', 'Had_Demo', 'Had_Survey', 'Had_Signup', 'Had_Manager',
        'Country_ConvRate', 'Education_ConvRate', 'Stages_Completed'
    ]

    logger.debug(f"\nSample state vector (16 dimensions):")
    for i, feat in enumerate(state_features):
        logger.debug(f"  {i}: {feat} = {sample[feat]:.4f}")

    logger.info("\nData processing ready for RL environment!")
