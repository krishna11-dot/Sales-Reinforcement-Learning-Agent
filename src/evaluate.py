"""
Evaluation Script for Trained RL Agent

Evaluates agent on test set and generates business insights.
"""

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import sys

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

from environment import CRMSalesFunnelEnv
from agent import QLearningAgent


def evaluate_agent(agent_path='checkpoints/agent_final.pkl',
                   data_path='data/processed/crm_test.csv',
                   n_episodes=1000):
    """
    Evaluate trained agent on test set.
    """
    logger.info("="*80)
    logger.info("EVALUATION ON TEST SET")
    logger.info("="*80)

    # Load trained agent
    agent = QLearningAgent()
    agent.load(agent_path)

    # Create test environment (no oversampling)
    env = CRMSalesFunnelEnv(
        data_path=data_path,
        stats_path='data/processed/historical_stats.json',
        mode='test'
    )

    # Evaluation metrics
    results = {
        'subscriptions': [],
        'first_calls': [],
        'rewards': [],
        'steps': []
    }

    # Run episodes
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        state, info = env.reset()
        done, truncated = False, False

        episode_reward = 0
        episode_steps = 0
        got_subscription = False
        got_first_call = False

        while not (done or truncated):
            # Use greedy policy (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, step_info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if step_info.get('subscribed', 0) == 1 and done:
                got_subscription = True
            if step_info.get('stage_reward', 0) == 15:
                got_first_call = True

            state = next_state

        results['subscriptions'].append(1 if got_subscription else 0)
        results['first_calls'].append(1 if got_first_call else 0)
        results['rewards'].append(episode_reward)
        results['steps'].append(episode_steps)

    # Calculate metrics
    sub_rate = np.mean(results['subscriptions']) * 100
    call_rate = np.mean(results['first_calls']) * 100
    avg_reward = np.mean(results['rewards'])

    logger.info("="*80)
    logger.info("TEST SET RESULTS")
    logger.info("="*80)
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"BUSINESS METRICS:")
    logger.info(f"  Subscription Rate: {sub_rate:.2f}% (baseline: 0.44%)")
    logger.debug(f"  First Call Rate: {call_rate:.2f}% (baseline: 4.0%)")
    logger.info(f"  Improvement: {sub_rate/0.44:.1f}x subscriptions")
    logger.info(f"TECHNICAL METRICS:")
    logger.info(f"  Avg Reward: {avg_reward:.2f}")
    logger.debug(f"  Avg Steps: {np.mean(results['steps']):.2f}")
    logger.info("="*80)

    # Save results
    with open('logs/test_results.json', 'w') as f:
        json.dump({
            'subscription_rate': sub_rate,
            'first_call_rate': call_rate,
            'avg_reward': avg_reward,
            'baseline_sub_rate': 0.44,
            'baseline_call_rate': 4.0,
            'improvement_factor': sub_rate / 0.44
        }, f, indent=2)

    return results


if __name__ == "__main__":
    results = evaluate_agent()
