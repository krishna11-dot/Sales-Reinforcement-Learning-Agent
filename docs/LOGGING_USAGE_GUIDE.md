# Logging Usage Guide

## 🎯 Quick Start

### **Default Usage (INFO level)**
```python
from data_processing import validate_data
import pandas as pd

df = pd.read_csv('data/processed/crm_train.csv')
validate_data(df, 'train', 'processed')
```

**Output:**
```
2026-01-14 22:39:46 - data_processing - INFO - Train data validation passed (processed stage)
```

---

## 🎚️ Logging Levels

### **1. INFO Level (Default - Recommended)**

Shows important validation results only.

```python
from data_processing import validate_data, set_logging_level

set_logging_level('INFO')  # Default
validate_data(df, 'train', 'processed')
```

**Output:**
```
2026-01-14 22:39:46 - INFO - Train data validation passed (processed stage)
```

**Use when:**
- ✅ Running normal training
- ✅ Production deployment
- ✅ You only care if validation passed/failed

---

### **2. DEBUG Level (Detailed)**

Shows all validation details.

```python
set_logging_level('DEBUG')
validate_data(df, 'train', 'processed')
```

**Output:**
```
2026-01-14 22:40:16 - INFO - Train data validation passed (processed stage)
2026-01-14 22:40:16 - DEBUG -    Samples: 7,722
2026-01-14 22:40:16 - DEBUG -    Positive samples: 19 (0.25%)
2026-01-14 22:40:16 - DEBUG -    Class imbalance: 405:1
2026-01-14 22:40:16 - DEBUG -    All required columns present
2026-01-14 22:40:16 - DEBUG -    All checks passed
```

**Use when:**
- 🔍 Debugging data issues
- 🔍 Investigating class imbalance
- 🔍 First time running on new data

---

### **3. WARNING Level (Minimal)**

Shows only warnings and errors.

```python
set_logging_level('WARNING')
validate_data(df, 'train', 'processed')
```

**Output:**
```
(No output if everything is OK)

If there are warnings:
2026-01-14 22:41:00 - WARNING - Found NaN in critical columns:
2026-01-14 22:41:00 - WARNING -    Country: 5 NaN values (0.06%)
```

**Use when:**
- ⚠️ Running automated tests (pytest)
- ⚠️ Only want to see problems
- ⚠️ Batch processing many files

---

## 📊 Comparison with print()

### **Before (with print()):**
```python
def validate_data(df):
    print("[OK] Validation passed!")
    print(f"   - {len(df)} samples")
    print(f"   - {df.shape[1]} columns")
```

**Output (always shows everything):**
```
[OK] Validation passed!
   - 7,722 samples
   - 32 columns
```

---

### **After (with logging):**
```python
def validate_data(df):
    logger.info("Validation passed!")
    logger.debug(f"   Samples: {len(df)}")
    logger.debug(f"   Columns: {df.shape[1]}")
```

**Output (controlled by level):**

**INFO level:**
```
2026-01-14 22:39:46 - INFO - Validation passed!
```

**DEBUG level:**
```
2026-01-14 22:39:46 - INFO - Validation passed!
2026-01-14 22:39:46 - DEBUG -    Samples: 7,722
2026-01-14 22:39:46 - DEBUG -    Columns: 32
```

---

## 🧪 Using with Pytest

### **Test File:**
```python
# tests/test_data_validation.py
import pandas as pd
from data_processing import validate_data, set_logging_level

def test_validate_data():
    # Set to WARNING to keep test output clean
    set_logging_level('WARNING')

    df = pd.DataFrame({
        'Country_Encoded': [1, 2, 3],
        'Subscribed_Binary': [0, 1, 0],
        # ... all required columns
    })

    result = validate_data(df, 'test', 'processed')
    assert result is not None
```

**Running tests:**
```bash
# Clean output (only shows test results)
pytest tests/ -v

# Show INFO logs if needed
pytest tests/ -v --log-cli-level=INFO

# Show all DEBUG logs
pytest tests/ -v --log-cli-level=DEBUG
```

---

## 🔧 Advanced: Save Logs to File

### **Option 1: Save to file + show on screen**
```python
import logging

# Configure to save to file AND print to screen
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_validation.log'),
        logging.StreamHandler()  # Also print to screen
    ]
)

from data_processing import validate_data
validate_data(df, 'train', 'processed')
```

**Result:**
- Screen: Shows INFO messages
- File `logs/data_validation.log`: Contains all DEBUG messages

---

### **Option 2: Separate files for different levels**
```python
import logging

# INFO and above to one file
info_handler = logging.FileHandler('logs/info.log')
info_handler.setLevel(logging.INFO)

# WARNING and above to another file
warning_handler = logging.FileHandler('logs/warnings.log')
warning_handler.setLevel(logging.WARNING)

# Configure logger
logger = logging.getLogger('data_processing')
logger.addHandler(info_handler)
logger.addHandler(warning_handler)
logger.setLevel(logging.DEBUG)
```

---

## 📋 Common Usage Patterns

### **Pattern 1: Training Script**
```python
# src/train_dqn.py
from data_processing import validate_data, set_logging_level

# Normal training - show important info only
set_logging_level('INFO')

train_df = pd.read_csv('data/processed/crm_train.csv')
validate_data(train_df, 'train', 'processed')

# Start training...
```

---

### **Pattern 2: Data Processing Script**
```python
# src/process_data.py
from data_processing import process_crm_data, set_logging_level

# Show all details during data processing
set_logging_level('DEBUG')

train_df, val_df, test_df, stats = process_crm_data(
    filepath='data/raw/SalesCRM.xlsx'
)
```

---

### **Pattern 3: Pytest Tests**
```python
# tests/test_validation.py
import pytest
from data_processing import validate_data, set_logging_level

@pytest.fixture(autouse=True)
def setup_logging():
    """Set WARNING level for all tests to keep output clean."""
    set_logging_level('WARNING')

def test_valid_data():
    # Test code here
    # Only warnings/errors will show
    pass
```

---

## 🎯 Summary

| Level | Shows | Use Case |
|-------|-------|----------|
| **DEBUG** | Everything | First run, debugging, investigation |
| **INFO** | Important info | Normal training, default |
| **WARNING** | Problems only | Pytest, batch processing |
| **ERROR** | Errors only | Production monitoring |

**Default:** INFO level (shows validation passed/failed)

**Change level:** `set_logging_level('DEBUG')` before calling validation

**Pytest:** Use WARNING level to keep test output clean
