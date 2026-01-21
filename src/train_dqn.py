"""
DQN Baseline Training - Standard Environment

Compare DQN vs Q-Learning on baseline environment:
- Q-Learning Baseline: 1.30% (works well)
- DQN Baseline: Should be similar or better!

This tests DQN on the simpler 16-dimensional state space.
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

logger.info("Loading Stable-Baselines3...")
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

logger.info("Libraries loaded successfully!")

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

logger.info("Loading environment...")
from environment import CRMSalesFunnelEnv
logger.info("Environment loaded! Ready to train.")


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
                logger.debug(f"  Q-Learning Baseline: 1.30%")
                logger.info(f"  Improvement: {sub_rate/0.44:.1f}x subscriptions")
                logger.info(f"{'='*80}")

        return True

    def _on_training_end(self) -> None:
        metrics = {
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'subscriptions': [int(s) for s in self.subscriptions],
            'first_calls': [int(c) for c in self.first_calls],
            'final_subscription_rate': float(np.mean(self.subscriptions[-1000:]) * 100) if len(self.subscriptions) >= 1000 else 0.0,
        }

        os.makedirs('logs/dqn', exist_ok=True)
        with open('logs/dqn/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"{'='*80}")
        logger.info("TRAINING COMPLETE - DQN BASELINE")
        logger.info(f"{'='*80}")
        logger.info(f"Total Episodes: {self.episode_count:,}")
        logger.info(f"Final Subscription Rate: {metrics['final_subscription_rate']:.2f}%")
        logger.debug(f"Q-Learning Baseline: 1.30%")
        logger.info(f"{'='*80}")


def train_dqn_baseline(
    n_timesteps=100000,
    log_interval=1000,
    save_interval=10000,
    output_dir='checkpoints/dqn'
):
    """
    Train DQN on baseline environment.

    Compare with Q-Learning baseline (1.30%) on same 16-dim state space.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs/dqn', exist_ok=True)

    logger.info("="*80)
    logger.info("DQN BASELINE TRAINING")
    logger.info("="*80)
    logger.info("Baseline Environment Test")
    logger.debug("")
    logger.debug("Q-Learning Baseline:")
    logger.debug("  State space: 16-dimensional continuous")
    logger.debug("  Performance: 1.30% (3.0x improvement)")
    logger.debug("")
    logger.debug("DQN Baseline:")
    logger.debug("  State space: Same 16-dimensional")
    logger.debug("  Expected: 1.1-1.4% (should be comparable)")
    logger.debug("  Advantage: Neural network vs lookup table")
    logger.info("="*80)

    # Load baseline environment
    env = CRMSalesFunnelEnv(
        data_path='data/processed/crm_train.csv',
        stats_path='data/processed/historical_stats.json',
        mode='train'
    )

    logger.info("Baseline Environment loaded!")
    logger.debug(f"State space: {env.observation_space}")
    logger.debug(f"Action space: {env.action_space}")

    # Create DQN agent
    logger.info("Creating DQN agent for baseline environment...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        gamma=0.95,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
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
        name_prefix='dqn_agent'
    )

    # Train
    logger.info("="*80)
    logger.info("STARTING TRAINING - BASELINE")
    logger.info("="*80)
    logger.info(f"Total timesteps: {n_timesteps:,}")
    logger.info("="*80)

    model.learn(
        total_timesteps=n_timesteps,
        callback=[metrics_callback, checkpoint_callback],
        log_interval=log_interval,
        progress_bar=True
    )

    # Save final model
    final_path = os.path.join(output_dir, 'best_model')
    model.save(final_path)
    logger.info(f"{'='*80}")
    logger.info(f"Final model saved to: {final_path}")
    logger.info(f"{'='*80}")

    return model


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("DQN BASELINE - STANDARD ENVIRONMENT TEST")
    logger.info("="*80)
    logger.debug("Compare DQN vs Q-Learning on 16-dim state space")
    logger.debug("Q-Learning: 1.30% (tabular lookup)")
    logger.debug("DQN: Neural network approximation")
    logger.info("="*80)

    # Train DQN on baseline
    model = train_dqn_baseline(
        n_timesteps=100000,
        log_interval=1000,
        save_interval=10000
    )

    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("  1. Run: python src/evaluate_dqn.py")
    logger.debug("  2. Compare with Q-Learning Baseline (1.30%)")
    logger.info("="*80)
