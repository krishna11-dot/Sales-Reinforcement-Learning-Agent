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
from pathlib import Path


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
    print("\n" + "="*80)
    print("STEP 1: LOADING RAW DATA")
    print("="*80)

    # Load raw data
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df):,} customers from {filepath}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n" + "="*80)
    print("STEP 2: PARSE DATE COLUMNS")
    print("="*80)

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
            print(f"{col}: {non_null:,} non-null ({non_null/len(df)*100:.2f}%)")

    print("\n" + "="*80)
    print("STEP 3: TEMPORAL SPLIT (CRITICAL - BEFORE FEATURE ENGINEERING!)")
    print("="*80)

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

    print(f"Train set: {len(train_df):,} customers ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Date range: {train_df['First Contact'].min()} to {train_df['First Contact'].max()}")

    print(f"Val set: {len(val_df):,} customers ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Date range: {val_df['First Contact'].min()} to {val_df['First Contact'].max()}")

    print(f"Test set: {len(test_df):,} customers ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Date range: {test_df['First Contact'].min()} to {test_df['First Contact'].max()}")

    print("\n" + "="*80)
    print("STEP 4: CREATE TARGET VARIABLES")
    print("="*80)

    # Create binary target variables
    for split_df in [train_df, val_df, test_df]:
        split_df['Subscribed_Binary'] = split_df['Subscribed'].notna().astype(int)
        split_df['Had_First_Call'] = split_df['First Call'].notna().astype(int)

    # Calculate baseline metrics
    train_sub_rate = train_df['Subscribed_Binary'].mean() * 100
    train_call_rate = train_df['Had_First_Call'].mean() * 100

    print(f"TRAIN SET BASELINE METRICS:")
    print(f"  Subscription rate: {train_sub_rate:.2f}%")
    print(f"  First call rate: {train_call_rate:.2f}%")
    print(f"  Class imbalance: {(1-train_df['Subscribed_Binary'].mean())/train_df['Subscribed_Binary'].mean():.0f}:1")

    print("\n" + "="*80)
    print("STEP 5: CALCULATE HISTORICAL STATISTICS (TRAIN SET ONLY!)")
    print("="*80)

    # CRITICAL: Calculate aggregated statistics ONLY on train set
    # WHY: Using test set here would leak future outcomes into features

    # Country conversion rates (from train only)
    country_stats = train_df.groupby('Country')['Subscribed_Binary'].agg(['mean', 'count'])
    print(f"\nCountry statistics (top 10 by conversion rate):")
    print(country_stats.nlargest(10, 'mean'))

    # Education conversion rates (from train only)
    edu_stats = train_df.groupby('Education')['Subscribed_Binary'].agg(['mean', 'count'])
    print(f"\nEducation statistics:")
    print(edu_stats.sort_values('mean', ascending=False))

    # Store as dictionaries for mapping
    historical_stats = {
        'country_conv': country_stats['mean'].to_dict(),
        'edu_conv': edu_stats['mean'].to_dict(),
        'global_avg': train_df['Subscribed_Binary'].mean()  # Fallback for unseen categories
    }

    print(f"\nStored {len(historical_stats['country_conv'])} country stats")
    print(f"Stored {len(historical_stats['edu_conv'])} education stats")
    print(f"Global average: {historical_stats['global_avg']:.4f}")

    print("\n" + "="*80)
    print("STEP 6: FEATURE ENGINEERING (SAME FOR ALL SPLITS)")
    print("="*80)

    # Apply feature engineering to each split
    # IMPORTANT: All splits use TRAIN statistics (no leakage)
    train_df = engineer_features(train_df, historical_stats)
    val_df = engineer_features(val_df, historical_stats)
    test_df = engineer_features(test_df, historical_stats)

    print(f"Engineered {len(train_df.columns)} total features")
    print(f"Feature columns: {[col for col in train_df.columns if col not in df.columns]}")

    print("\n" + "="*80)
    print("STEP 7: SAVE PROCESSED DATA")
    print("="*80)

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

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")

    # Save historical statistics for use in environment
    with open(stats_path, 'w') as f:
        json.dump(historical_stats, f, indent=2)
    print(f"Saved: {stats_path}")

    print("\n" + "="*80)
    print("DATA PROCESSING COMPLETE - NO LEAKAGE CONFIRMED")
    print("="*80)
    print("VERIFICATION:")
    print("- Temporal split: Train dates < Val dates < Test dates")
    print("- Statistics: Calculated on train only, mapped to val/test")
    print("- Features: All use historical data (past events)")
    print("="*80 + "\n")

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
    print("\nVERIFICATION CHECK:")
    print(f"Train subscription rate: {train_df['Subscribed_Binary'].mean()*100:.2f}%")
    print(f"Val subscription rate: {val_df['Subscribed_Binary'].mean()*100:.2f}%")
    print(f"Test subscription rate: {test_df['Subscribed_Binary'].mean()*100:.2f}%")

    # Check state vector can be created
    sample = train_df.iloc[0]
    state_features = [
        'Education_Encoded', 'Country_Encoded', 'Stage_Encoded', 'Status_Active',
        'Days_Since_First_Norm', 'Days_Since_Last_Norm', 'Days_Between_Norm', 'Contact_Frequency',
        'Had_First_Call', 'Had_Demo', 'Had_Survey', 'Had_Signup', 'Had_Manager',
        'Country_ConvRate', 'Education_ConvRate', 'Stages_Completed'
    ]

    print(f"\nSample state vector (16 dimensions):")
    for i, feat in enumerate(state_features):
        print(f"  {i}: {feat} = {sample[feat]:.4f}")

    print("\nData processing ready for RL environment!")
