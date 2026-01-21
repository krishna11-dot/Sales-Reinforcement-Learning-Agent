"""
Training Loop for CRM Sales RL Agent

NUANCE #8: Technical vs Business Metrics (CRITICAL!)
This script tracks BOTH metric types separately:
- Technical: For debugging/optimization (Q-convergence, epsilon, etc.)
- Business: For stakeholder value (conversion rate, ROI, etc.)

INTERVIEW: "Your Q-values converged. Is the model good?"
ANSWER: "NO! Technical convergence != Business success.
  Must check BOTH: Technical (model learned) AND Business (learned RIGHT thing)"
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from pathlib import Path

# Add parent directory to path for imports
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


def train_agent(
    n_episodes=100000,
    log_interval=1000,
    save_interval=10000,
    output_dir='checkpoints'
):
    """
    Train Q-Learning agent on CRM sales data.

    NUANCE #8: Separated Metrics
    - Track technical AND business metrics separately
    - Both must succeed for project success

    Args:
        n_episodes: Total training episodes
        log_interval: Episodes between progress logs
        save_interval: Episodes between checkpoints
        output_dir: Directory for saved models

    Returns:
        agent, technical_metrics, business_metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Initialize environment and agent
    env = CRMSalesFunnelEnv(
        data_path='data/processed/crm_train.csv',
        stats_path='data/processed/historical_stats.json',
        mode='train'
    )

    agent = QLearningAgent(
        n_actions=6,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    # NUANCE #8: SEPARATED METRICS
    # Technical: For YOU (debugging, optimization)
    technical_metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'epsilon_values': [],
        'subscriptions': [],
        'first_calls': [],
        'q_table_sizes': []
    }

    # Business: For STAKEHOLDERS (value, ROI)
    business_metrics = {
        'subscription_rate': [],
        'first_call_rate': [],
        'avg_cost': []
    }

    logger.info("="*80)
    logger.info("TRAINING START")
    logger.info("="*80)
    logger.info(f"Episodes: {n_episodes:,}")
    logger.debug(f"Log interval: {log_interval:,}")
    logger.debug(f"Save interval: {save_interval:,}")
    logger.debug(f"Batch sampling: 30% subscribed, 30% first call, 40% random")
    logger.info("="*80)

    # Training loop with progress bar
    for episode in tqdm(range(n_episodes), desc="Training"):
        state, info = env.reset()
        done, truncated = False, False

        episode_reward = 0
        episode_steps = 0
        got_subscription = False
        got_first_call = False

        # Episode loop
        while not (done or truncated):
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, step_info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1

            # Track achievements
            if step_info.get('subscribed', 0) == 1 and done:
                got_subscription = True
            if step_info.get('stage_reward', 0) == 15:
                got_first_call = True

            state = next_state

        # Decay exploration
        agent.decay_epsilon()

        # Track metrics
        technical_metrics['episode_rewards'].append(episode_reward)
        technical_metrics['episode_steps'].append(episode_steps)
        technical_metrics['epsilon_values'].append(agent.epsilon)
        technical_metrics['subscriptions'].append(1 if got_subscription else 0)
        technical_metrics['first_calls'].append(1 if got_first_call else 0)
        technical_metrics['q_table_sizes'].append(len(agent.q_table))

        # Log progress
        if (episode + 1) % log_interval == 0:
            recent_rewards = technical_metrics['episode_rewards'][-log_interval:]
            recent_subs = technical_metrics['subscriptions'][-log_interval:]
            recent_calls = technical_metrics['first_calls'][-log_interval:]

            avg_reward = np.mean(recent_rewards)
            sub_rate = np.mean(recent_subs) * 100
            call_rate = np.mean(recent_calls) * 100

            logger.info(f"{'='*80}")
            logger.info(f"Episode {episode + 1:,} / {n_episodes:,}")
            logger.info(f"{'='*80}")
            logger.info(f"TECHNICAL METRICS (for debugging):")
            logger.info(f"  Avg Reward: {avg_reward:.2f}")
            logger.debug(f"  Epsilon: {agent.epsilon:.4f}")
            logger.debug(f"  Q-table size: {len(agent.q_table):,} states")
            logger.info(f"BUSINESS METRICS (for stakeholders):")
            logger.info(f"  Subscription Rate: {sub_rate:.2f}% (baseline: 0.44%)")
            logger.debug(f"  First Call Rate: {call_rate:.2f}% (baseline: 4.0%)")
            logger.info(f"  Improvement: {sub_rate/0.44:.1f}x subscriptions")
            logger.info(f"{'='*80}")

            # Store business metrics
            business_metrics['subscription_rate'].append(sub_rate)
            business_metrics['first_call_rate'].append(call_rate)
            business_metrics['avg_cost'].append(-np.mean([r for r in recent_rewards if r < 0]))

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f'agent_episode_{episode+1}.pkl')
            agent.save(checkpoint_path)

            # Save metrics
            metrics_path = os.path.join('logs', f'metrics_episode_{episode+1}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'technical': {k: [float(v) for v in vals] for k, vals in technical_metrics.items()},
                    'business': business_metrics
                }, f, indent=2)

    # Save final model
    agent.save(os.path.join(output_dir, 'agent_final.pkl'))

    # Save final metrics
    with open(os.path.join('logs', 'training_metrics_final.json'), 'w') as f:
        json.dump({
            'technical': {k: [float(v) for v in vals] for k, vals in technical_metrics.items()},
            'business': business_metrics
        }, f, indent=2)

    # Plot training curves
    plot_training_curves(technical_metrics, 'visualizations')

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Final Q-table size: {len(agent.q_table):,} states")
    logger.debug(f"Final epsilon: {agent.epsilon:.4f}")

    final_sub_rate = np.mean(technical_metrics['subscriptions'][-1000:]) * 100
    final_call_rate = np.mean(technical_metrics['first_calls'][-1000:]) * 100

    logger.info(f"Final Performance (last 1000 episodes):")
    logger.info(f"  Subscription rate: {final_sub_rate:.2f}% (baseline: 0.44%)")
    logger.debug(f"  First call rate: {final_call_rate:.2f}% (baseline: 4.0%)")
    logger.info(f"  Improvement: {final_sub_rate/0.44:.1f}x subscriptions")
    logger.info("="*80)

    return agent, technical_metrics, business_metrics


def plot_training_curves(metrics, output_dir):
    """
    Plot training metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def smooth(values, window=1000):
        return pd.Series(values).rolling(window=window, min_periods=1).mean()

    # Rewards
    axes[0, 0].plot(smooth(metrics['episode_rewards']), color='blue', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards (smoothed)', fontsize=12)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Subscription rate
    sub_rate = smooth([v * 100 for v in metrics['subscriptions']])
    axes[0, 1].plot(sub_rate, color='green', alpha=0.7)
    axes[0, 1].axhline(y=0.44, color='r', linestyle='--', label='Baseline', linewidth=2)
    axes[0, 1].axhline(y=1.0, color='g', linestyle='--', label='Target', linewidth=2)
    axes[0, 1].set_title('Subscription Rate (%)', fontsize=12)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Rate (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Epsilon decay
    axes[1, 0].plot(metrics['epsilon_values'], color='purple', alpha=0.7)
    axes[1, 0].set_title('Exploration Rate (Epsilon)', fontsize=12)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True, alpha=0.3)

    # Q-table growth
    axes[1, 1].plot(metrics['q_table_sizes'], color='orange', alpha=0.7)
    axes[1, 1].set_title('Q-Table Size (States Visited)', fontsize=12)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Number of States')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    logger.info(f"Training curves saved to: {output_dir}/training_curves.png")
    plt.close()


if __name__ == "__main__":
    # Run training
    agent, tech_metrics, bus_metrics = train_agent(
        n_episodes=100000,
        log_interval=1000,
        save_interval=10000
    )

    logger.info("Training complete! Check:")
    logger.info("  - checkpoints/ for saved models")
    logger.info("  - logs/ for metrics")
    logger.info("  - visualizations/ for plots")
