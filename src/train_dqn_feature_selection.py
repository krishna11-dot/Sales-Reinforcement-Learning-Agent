"""
DQN Training with Feature Selection - The REAL Test!

This script tests DQN on the feature selection environment where Q-Learning FAILED.

COMPARISON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q-Learning (Feature Selection):
  - State space: 522,619 states
  - Performance: 0.80% (FAILED - state space explosion)
  - Problem: Too sparse, can't learn

DQN (Feature Selection):
  - State space: Continuous (handles infinite states)
  - Expected: 1.2-1.5% (should work!)
  - Advantage: Generalizes via neural network
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is THE KEY TEST to prove DQN > Q-Learning for large state spaces!
"""

import os
import sys
import logging
import numpy as np
import json
from pathlib import Path

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

logger.info("Loading libraries (this may take 10-30 seconds)...")

logger.info("Loading Stable-Baselines3 and PyTorch...")
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

logger.info("Libraries loaded successfully!")

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from environment_feature_selection import CRMFeatureSelectionEnv


class MetricsCallback(BaseCallback):
    """Track business metrics during training."""

    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.subscriptions = []
        self.first_calls = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1

            info = self.locals.get('infos', [{}])[0]
            episode_reward = np.sum(self.locals.get('rewards', [0]))
            self.episode_rewards.append(episode_reward)

            got_subscription = info.get('subscribed', 0) == 1
            self.subscriptions.append(1 if got_subscription else 0)

            got_first_call = episode_reward > 10
            self.first_calls.append(1 if got_first_call else 0)

            if self.episode_count % self.log_interval == 0:
                recent_rewards = self.episode_rewards[-self.log_interval:]
                recent_subs = self.subscriptions[-self.log_interval:]
                recent_calls = self.first_calls[-self.log_interval:]

                avg_reward = np.mean(recent_rewards)
                sub_rate = np.mean(recent_subs) * 100
                call_rate = np.mean(recent_calls) * 100

                logger.info(f"{'='*80}")
                logger.info(f"Episode {self.episode_count:,}")
                logger.info(f"{'='*80}")
                logger.info(f"BUSINESS METRICS:")
                logger.info(f"  Subscription Rate: {sub_rate:.2f}% (baseline: 0.44%)")
                logger.debug(f"  Q-Learning Feature Selection: 0.80% (FAILED)")
                logger.info(f"  Improvement: {sub_rate/0.44:.1f}x subscriptions")
                logger.debug(f"  vs Q-Learning FS: {sub_rate/0.80:.1f}x better")
                logger.info(f"{'='*80}")

        return True

    def _on_training_end(self) -> None:
        metrics = {
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'subscriptions': [int(s) for s in self.subscriptions],
            'first_calls': [int(c) for c in self.first_calls],
            'final_subscription_rate': float(np.mean(self.subscriptions[-1000:]) * 100) if len(self.subscriptions) >= 1000 else 0.0,
        }

        os.makedirs('logs/dqn_feature_selection', exist_ok=True)
        with open('logs/dqn_feature_selection/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"{'='*80}")
        logger.info("TRAINING COMPLETE - DQN FEATURE SELECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Total Episodes: {self.episode_count:,}")
        logger.info(f"Final Subscription Rate: {metrics['final_subscription_rate']:.2f}%")
        logger.debug(f"Q-Learning Feature Selection: 0.80% (FAILED)")
        logger.info(f"DQN Improvement: {metrics['final_subscription_rate']/0.80:.1f}x better")
        logger.info(f"{'='*80}")


def train_dqn_feature_selection(
    n_timesteps=100000,
    log_interval=1000,
    save_interval=10000,
    output_dir='checkpoints/dqn_feature_selection'
):
    """
    Train DQN on feature selection environment.

    THE KEY TEST: Can DQN handle 522k states where Q-Learning failed?
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs/dqn_feature_selection', exist_ok=True)

    logger.info("="*80)
    logger.info("DQN FEATURE SELECTION TRAINING")
    logger.info("="*80)
    logger.info("THE KEY TEST: Large State Space")
    logger.debug("")
    logger.debug("Q-Learning Feature Selection:")
    logger.debug("  State space: 522,619 states")
    logger.debug("  Performance: 0.80% (FAILED)")
    logger.debug("  Problem: State space explosion")
    logger.debug("")
    logger.debug("DQN Feature Selection:")
    logger.debug("  State space: Continuous (handles infinite)")
    logger.debug("  Expected: 1.2-1.5% (should work!)")
    logger.debug("  Advantage: Neural network generalizes")
    logger.info("="*80)

    # Load feature selection environment
    env = CRMFeatureSelectionEnv(
        data_path='data/processed/crm_train.csv',
        stats_path='data/processed/historical_stats.json',
        mode='train'
    )

    logger.info("Feature Selection Environment loaded!")
    logger.debug(f"State space: {env.observation_space}")
    logger.debug(f"Action space: {env.action_space}")
    logger.debug(f"Features: Agent selects features dynamically")
    logger.debug(f"State includes: 15 customer features + 15 binary feature toggles = 30-dim")

    # Create DQN agent
    logger.info("Creating DQN agent for feature selection...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        gamma=0.95,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=None  # Disabled due to environment compatibility
    )

    logger.info("DQN agent created!")
    logger.debug(f"Total parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}")

    # Setup callbacks
    metrics_callback = MetricsCallback(log_interval=log_interval)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval,
        save_path=output_dir,
        name_prefix='dqn_fs_agent'
    )

    # Train
    logger.info("="*80)
    logger.info("STARTING TRAINING - FEATURE SELECTION")
    logger.info("="*80)
    logger.info(f"Total timesteps: {n_timesteps:,}")
    logger.debug(f"This is where DQN should SHINE!")
    logger.info("="*80)

    model.learn(
        total_timesteps=n_timesteps,
        callback=[metrics_callback, checkpoint_callback],
        log_interval=log_interval,
        progress_bar=True
    )

    # Save final model
    final_path = os.path.join(output_dir, 'dqn_fs_agent_final')
    model.save(final_path)
    logger.info(f"{'='*80}")
    logger.info(f"Final model saved to: {final_path}")
    logger.info(f"{'='*80}")

    return model


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("DQN FEATURE SELECTION - THE REAL TEST")
    logger.info("="*80)
    logger.debug("Q-Learning FAILED here (0.80% on 522k states)")
    logger.debug("DQN should SUCCEED (generalizes via neural network)")
    logger.info("="*80)

    # Train DQN on feature selection
    model = train_dqn_feature_selection(
        n_timesteps=100000,
        log_interval=1000,
        save_interval=10000
    )

    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("  1. Run: python src/evaluate_dqn_feature_selection.py")
    logger.debug("  2. Compare with Q-Learning Feature Selection (0.80%)")
    logger.debug("  3. DQN should be MUCH better (1.2-1.5% expected)")
    logger.info("="*80)
