"""
Training Loop for Feature Selection RL Agent

Trains Q-Learning agent with feature selection capability.

Key differences from baseline training:
1. Uses CRMFeatureSelectionEnv (32-dim state, 22 actions)
2. Uses QLearningAgentFeatureSelection (22 actions)
3. Tracks feature selection metrics (toggles, active features)
"""

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

from environment_feature_selection import CRMFeatureSelectionEnv
from agent_feature_selection import QLearningAgentFeatureSelection


def train_agent(
    n_episodes=100000,
    log_interval=1000,
    save_interval=10000,
    output_dir='checkpoints'
):
    """
    Train Q-Learning agent with feature selection on CRM sales data.

    Args:
        n_episodes: Total training episodes
        log_interval: Episodes between progress logs
        save_interval: Episodes between checkpoints
        output_dir: Directory for saved models

    Returns:
        agent, technical_metrics, business_metrics, feature_metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Initialize environment and agent
    env = CRMFeatureSelectionEnv(
        data_path='data/processed/crm_train.csv',
        stats_path='data/processed/historical_stats.json',
        mode='train'
    )

    agent = QLearningAgentFeatureSelection(
        n_actions=22,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    # Technical metrics
    technical_metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'epsilon_values': [],
        'subscriptions': [],
        'first_calls': [],
        'q_table_sizes': []
    }

    # Business metrics
    business_metrics = {
        'subscription_rate': [],
        'first_call_rate': [],
        'avg_cost': []
    }

    # Feature selection metrics (NEW)
    feature_metrics = {
        'avg_toggles_per_episode': [],
        'avg_features_selected': [],
        'feature_selection_counts': []
    }

    print("\n" + "="*80)
    print("TRAINING START - FEATURE SELECTION AGENT")
    print("="*80)
    print(f"Episodes: {n_episodes:,}")
    print(f"Log interval: {log_interval:,}")
    print(f"Save interval: {save_interval:,}")
    print(f"Environment: 32-dim state (16 mask + 16 features)")
    print(f"Actions: 22 (16 toggles + 6 CRM)")
    print(f"Batch sampling: 30% subscribed, 30% first call, 40% random")
    print("="*80 + "\n")

    # Training loop with progress bar
    for episode in tqdm(range(n_episodes), desc="Training"):
        state, info = env.reset()
        done, truncated = False, False

        episode_reward = 0
        episode_steps = 0
        got_subscription = False
        got_first_call = False
        feature_toggles = 0
        final_features_selected = 0

        # Episode loop
        while not (done or truncated):
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, step_info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1

            # Track feature toggles
            if step_info.get('action_type') == 'feature_toggle':
                feature_toggles += 1

            # Track achievements
            if step_info.get('subscribed', 0) == 1 and done:
                got_subscription = True
            if step_info.get('stage_reward', 0) == 15:
                got_first_call = True

            # Track final feature selection
            if done:
                final_features_selected = step_info.get('n_active_features', 0)

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

            print(f"\n{'='*80}")
            print(f"Episode {episode + 1:,} / {n_episodes:,}")
            print(f"{'='*80}")
            print(f"TECHNICAL METRICS (for debugging):")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-table size: {len(agent.q_table):,} states")
            print(f"\nBUSINESS METRICS (for stakeholders):")
            print(f"  Subscription Rate: {sub_rate:.2f}% (baseline: 0.44%)")
            print(f"  First Call Rate: {call_rate:.2f}% (baseline: 4.0%)")
            print(f"  Improvement: {sub_rate/0.44:.1f}x subscriptions")
            print(f"\nFEATURE SELECTION METRICS:")
            print(f"  Avg Feature Toggles: {feature_toggles:.2f}")
            print(f"  Final Features Selected: {final_features_selected}")
            print(f"{'='*80}")

            # Store business metrics
            business_metrics['subscription_rate'].append(sub_rate)
            business_metrics['first_call_rate'].append(call_rate)
            business_metrics['avg_cost'].append(-np.mean([r for r in recent_rewards if r < 0]))

            # Store feature metrics
            feature_metrics['avg_toggles_per_episode'].append(feature_toggles)
            feature_metrics['avg_features_selected'].append(final_features_selected)

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f'agent_feature_selection_episode_{episode+1}.pkl')
            agent.save(checkpoint_path)

            # Save metrics
            metrics_path = os.path.join('logs', f'metrics_feature_selection_episode_{episode+1}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'technical': {k: [float(v) for v in vals] for k, vals in technical_metrics.items()},
                    'business': {k: [float(v) for v in vals] for k, vals in business_metrics.items()},
                    'feature_selection': {k: [float(v) for v in vals] for k, vals in feature_metrics.items()}
                }, f, indent=2)

    # Save final model
    agent.save(os.path.join(output_dir, 'agent_feature_selection_final.pkl'))

    # Save final metrics
    with open(os.path.join('logs', 'training_metrics_feature_selection_final.json'), 'w') as f:
        json.dump({
            'technical': {k: [float(v) for v in vals] for k, vals in technical_metrics.items()},
            'business': {k: [float(v) for v in vals] for k, vals in business_metrics.items()},
            'feature_selection': {k: [float(v) for v in vals] for k, vals in feature_metrics.items()}
        }, f, indent=2)

    # Plot training curves
    plot_training_curves(technical_metrics, feature_metrics, 'visualizations')

    print("\n" + "="*80)
    print("TRAINING COMPLETE - FEATURE SELECTION AGENT")
    print("="*80)
    print(f"Final Q-table size: {len(agent.q_table):,} states")
    print(f"Final epsilon: {agent.epsilon:.4f}")

    final_sub_rate = np.mean(technical_metrics['subscriptions'][-1000:]) * 100
    final_call_rate = np.mean(technical_metrics['first_calls'][-1000:]) * 100

    print(f"\nFinal Performance (last 1000 episodes):")
    print(f"  Subscription rate: {final_sub_rate:.2f}% (baseline: 0.44%)")
    print(f"  First call rate: {final_call_rate:.2f}% (baseline: 4.0%)")
    print(f"  Improvement: {final_sub_rate/0.44:.1f}x subscriptions")
    print("\nNext step: Run analyze_features.py to see which features matter!")
    print("="*80 + "\n")

    return agent, technical_metrics, business_metrics, feature_metrics


def plot_training_curves(metrics, feature_metrics, output_dir):
    """
    Plot training metrics including feature selection.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
    axes[0, 2].plot(metrics['epsilon_values'], color='purple', alpha=0.7)
    axes[0, 2].set_title('Exploration Rate (Epsilon)', fontsize=12)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon')
    axes[0, 2].grid(True, alpha=0.3)

    # Q-table growth
    axes[1, 0].plot(metrics['q_table_sizes'], color='orange', alpha=0.7)
    axes[1, 0].set_title('Q-Table Size (States Visited)', fontsize=12)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of States')
    axes[1, 0].grid(True, alpha=0.3)

    # Feature toggles (NEW)
    if feature_metrics['avg_toggles_per_episode']:
        axes[1, 1].plot(smooth(feature_metrics['avg_toggles_per_episode']), color='teal', alpha=0.7)
        axes[1, 1].set_title('Avg Feature Toggles per Episode', fontsize=12)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Toggles')
        axes[1, 1].grid(True, alpha=0.3)

    # Features selected (NEW)
    if feature_metrics['avg_features_selected']:
        axes[1, 2].plot(smooth(feature_metrics['avg_features_selected']), color='brown', alpha=0.7)
        axes[1, 2].axhline(y=16, color='r', linestyle='--', label='All features', linewidth=2)
        axes[1, 2].set_title('Features Selected at Decision Time', fontsize=12)
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Number of Features')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_feature_selection.png'), dpi=150)
    print(f"Training curves saved to: {output_dir}/training_curves_feature_selection.png")
    plt.close()


if __name__ == "__main__":
    # Run training
    agent, tech_metrics, bus_metrics, feat_metrics = train_agent(
        n_episodes=100000,
        log_interval=1000,
        save_interval=10000
    )

    print("\nTraining complete! Check:")
    print("  - checkpoints/ for saved models")
    print("  - logs/ for metrics")
    print("  - visualizations/ for plots")
    print("\nNext: python src/analyze_features.py")
