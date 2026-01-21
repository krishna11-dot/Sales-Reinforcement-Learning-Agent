"""
DQN Feature Selection Evaluation - Test on Natural Distribution

Compare DQN vs Q-Learning on feature selection:
- Q-Learning: 0.80% (FAILED at 522k states)
- DQN: Should work better!
"""

import sys
import logging
import numpy as np
import json
import os
from pathlib import Path
from stable_baselines3 import DQN
from tqdm import tqdm

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

sys.path.append(str(Path(__file__).parent))

from environment_feature_selection import CRMFeatureSelectionEnv


def evaluate_dqn_feature_selection(
    model_path='checkpoints/dqn_feature_selection/dqn_fs_agent_final',
    n_episodes=None,
    verbose=True
):
    """Evaluate DQN on feature selection test set."""

    logger.info("="*80)
    logger.info("DQN FEATURE SELECTION EVALUATION")
    logger.info("="*80)

    # Load test environment
    env = CRMFeatureSelectionEnv(
        data_path='data/processed/crm_test.csv',
        stats_path='data/processed/historical_stats.json',
        mode='test'
    )

    if n_episodes is None:
        n_episodes = len(env.df)

    logger.info(f"Test set size: {len(env.df):,} customers")
    logger.info(f"Evaluating on: {n_episodes:,} episodes")
    logger.debug(f"Q-Learning Feature Selection: 0.80% (FAILED)")

    # Load trained model
    logger.info(f"Loading DQN model from: {model_path}")
    try:
        model = DQN.load(model_path, env=env)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Make sure you've trained the model first:")
        logger.info("  python src/train_dqn_feature_selection.py")
        return None

    # Evaluation
    episode_rewards = []
    subscriptions = []
    first_calls = []

    logger.info("="*80)
    logger.info("RUNNING EVALUATION")
    logger.info("="*80)

    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        state, info = env.reset()
        done, truncated = False, False
        episode_reward = 0
        episode_steps = 0
        max_steps_per_episode = 50  # Prevent infinite loops from feature toggling

        got_subscription = False
        got_first_call = False

        while not (done or truncated) and episode_steps < max_steps_per_episode:
            action, _ = model.predict(state, deterministic=True)
            action = int(action)  # Convert numpy array to int

            next_state, reward, done, truncated, step_info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if step_info.get('subscribed', 0) == 1 and done:
                got_subscription = True
            if step_info.get('stage_reward', 0) == 15:
                got_first_call = True

            state = next_state

        episode_rewards.append(episode_reward)
        subscriptions.append(1 if got_subscription else 0)
        first_calls.append(1 if got_first_call else 0)

    # Calculate metrics
    metrics = {
        'subscription_rate': np.mean(subscriptions) * 100,
        'first_call_rate': np.mean(first_calls) * 100,
        'avg_reward': np.mean(episode_rewards),
        'baseline_sub_rate': 0.44,
        'q_learning_fs_rate': 0.80,
        'improvement_factor': (np.mean(subscriptions) * 100) / 0.44,
        'vs_q_learning_fs': (np.mean(subscriptions) * 100) / 0.80,
        'n_episodes': n_episodes,
        'total_subscriptions': sum(subscriptions),
    }

    # Print results
    if verbose:
        logger.info("="*80)
        logger.info("DQN FEATURE SELECTION RESULTS")
        logger.info("="*80)
        logger.info(f"BUSINESS METRICS:")
        logger.info(f"  Subscription Rate: {metrics['subscription_rate']:.2f}%")
        logger.debug(f"  Random Baseline: 0.44%")
        logger.debug(f"  Q-Learning Feature Selection: 0.80% (FAILED)")
        logger.info(f"  Improvement over baseline: {metrics['improvement_factor']:.2f}x")
        logger.debug(f"  vs Q-Learning FS: {metrics['vs_q_learning_fs']:.2f}x better")
        logger.info(f"TECHNICAL METRICS:")
        logger.info(f"  Average Reward: {metrics['avg_reward']:.2f}")
        logger.info(f"  Total Subscriptions: {metrics['total_subscriptions']}")

        logger.info(f"{'='*80}")
        logger.info("COMPARISON - STATE SPACE HANDLING")
        logger.info(f"{'='*80}")
        logger.debug("Q-Learning Feature Selection:")
        logger.debug("  State space: 522,619 discrete states")
        logger.debug("  Performance: 0.80% (state space explosion)")
        logger.debug("  Problem: Too sparse to learn")
        logger.debug("")
        logger.info("DQN Feature Selection:")
        logger.info(f"  State space: Continuous (handles infinite)")
        logger.info(f"  Performance: {metrics['subscription_rate']:.2f}%")

        if metrics['subscription_rate'] > 1.0:
            logger.info(f"  SUCCESS: {metrics['vs_q_learning_fs']:.2f}x better than Q-Learning!")
        elif metrics['subscription_rate'] > 0.80:
            logger.info("  WORKS: Better than Q-Learning!")
        else:
            logger.warning("  ISSUE: Underperformed (may need more training)")

        logger.info("="*80)

    # Save results (convert numpy types to Python native types for JSON)
    output_path = 'logs/dqn_feature_selection/test_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types to Python native types
    metrics_json = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in metrics.items()
    }

    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    logger.info("="*80)

    return metrics


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("DQN FEATURE SELECTION EVALUATION")
    logger.info("="*80)
    logger.debug("Testing DQN on large state space (522k states)")
    logger.debug("Q-Learning FAILED here (0.80%)")
    logger.info("="*80)

    model_path = 'checkpoints/dqn_feature_selection/dqn_fs_agent_final'
    if not os.path.exists(model_path + '.zip'):
        logger.error("Trained model not found!")
        logger.error(f"Expected: {model_path}.zip")
        logger.info("Please train the model first:")
        logger.info("  python src/train_dqn_feature_selection.py")
    else:
        metrics = evaluate_dqn_feature_selection(
            model_path=model_path,
            n_episodes=None,
            verbose=True
        )

        if metrics:
            logger.info("="*80)
            logger.info("EVALUATION COMPLETE!")
            logger.info("="*80)
            logger.info("KEY RESULT:")
            if metrics['subscription_rate'] > 1.0:
                logger.info(f"  DQN: {metrics['subscription_rate']:.2f}% - SUCCESS!")
                logger.debug(f"  Q-Learning: 0.80% - FAILED")
                logger.info(f"  DQN is {metrics['vs_q_learning_fs']:.2f}x BETTER!")
                logger.info("  PROOF: DQN handles large state spaces!")
            else:
                logger.info(f"  DQN: {metrics['subscription_rate']:.2f}%")
                logger.debug(f"  Q-Learning: 0.80%")
                logger.info("  Try training longer for better results")
            logger.info("="*80)
