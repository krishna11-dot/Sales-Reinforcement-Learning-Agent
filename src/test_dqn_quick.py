"""
Quick DQN Test - Verify DQN works with environment.py

This script tests that:
1. DQN can load your environment
2. DQN can take actions
3. DQN can learn (even if just for a few steps)

Run this before full training to catch any issues early!
"""

import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from stable_baselines3 import DQN
from environment import CRMSalesFunnelEnv

logger.info("="*80)
logger.info("QUICK DQN SANITY CHECK")
logger.info("="*80)
logger.info("Testing DQN with your environment.py...")

# Test 1: Load environment
logger.info("Test 1: Loading environment...")
try:
    env = CRMSalesFunnelEnv(
        data_path='data/processed/crm_train.csv',
        stats_path='data/processed/historical_stats.json',
        mode='train'
    )
    logger.info("[OK] Environment loaded successfully!")
    logger.debug(f"   State space: {env.observation_space}")
    logger.debug(f"   Action space: {env.action_space}")
except Exception as e:
    logger.error(f"[FAIL] Failed to load environment: {e}")
    sys.exit(1)

# Test 2: Create DQN agent
logger.info("Test 2: Creating DQN agent...")
try:
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        verbose=1
    )
    logger.info("[OK] DQN agent created successfully!")
    logger.debug(f"   Neural network parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}")
except Exception as e:
    logger.error(f"[FAIL] Failed to create DQN agent: {e}")
    sys.exit(1)

# Test 3: Train for a few steps
logger.info("Test 3: Training for 1000 steps (quick test)...")
try:
    model.learn(total_timesteps=1000, progress_bar=True)
    logger.info("[OK] Training completed successfully!")
except Exception as e:
    logger.error(f"[FAIL] Training failed: {e}")
    sys.exit(1)

# Test 4: Make predictions
logger.info("Test 4: Making predictions...")
try:
    state, _ = env.reset()
    action, _ = model.predict(state, deterministic=True)
    action_int = int(action)  # Convert numpy array to int
    logger.info(f"[OK] Prediction successful!")
    logger.debug(f"   State shape: {state.shape}")
    logger.debug(f"   Action: {action_int} ({env.action_names[action_int]})")
except Exception as e:
    logger.error(f"[FAIL] Prediction failed: {e}")
    sys.exit(1)

# Test 5: Save and load
logger.info("Test 5: Saving and loading model...")
try:
    import os
    os.makedirs('checkpoints/dqn/test', exist_ok=True)
    model.save('checkpoints/dqn/test/quick_test')
    loaded_model = DQN.load('checkpoints/dqn/test/quick_test', env=env)
    logger.info("[OK] Save/load successful!")
except Exception as e:
    logger.error(f"[FAIL] Save/load failed: {e}")
    sys.exit(1)

logger.info("="*80)
logger.info("ALL TESTS PASSED!")
logger.info("="*80)
logger.info("Your environment works perfectly with DQN!")
logger.info("Ready for full training:")
logger.info("  python src/train_dqn.py")
logger.info("="*80)
