"""
Evaluation Script for Trained Feature Selection RL Agent

Evaluates agent on test set and generates business insights including:
1. Subscription rate improvement
2. Which features the agent learned to select
3. How feature selection impacts performance
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import sys
from collections import Counter

sys.path.append(str(Path(__file__).parent))

from environment_feature_selection import CRMFeatureSelectionEnv
from agent_feature_selection import QLearningAgentFeatureSelection


def evaluate_agent(agent_path='checkpoints/agent_feature_selection_final.pkl',
                   data_path='data/processed/crm_test.csv',
                   n_episodes=1000):
    """
    Evaluate trained feature selection agent on test set.

    Args:
        agent_path: Path to trained agent pickle file
        data_path: Path to test data CSV
        n_episodes: Number of test episodes to run

    Returns:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET - FEATURE SELECTION AGENT")
    print("="*80)

    # Load trained agent
    agent = QLearningAgentFeatureSelection()
    agent.load(agent_path)

    # Create test environment (no oversampling)
    env = CRMFeatureSelectionEnv(
        data_path=data_path,
        stats_path='data/processed/historical_stats.json',
        mode='test'
    )

    # Evaluation metrics
    results = {
        'subscriptions': [],
        'first_calls': [],
        'rewards': [],
        'steps': [],
        'feature_toggles': [],
        'n_features_selected': [],
        'selected_features': []
    }

    # Run episodes
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        state, info = env.reset()
        done, truncated = False, False

        episode_reward = 0
        episode_steps = 0
        episode_toggles = 0
        got_subscription = False
        got_first_call = False
        final_features = []
        final_n_features = 0

        while not (done or truncated):
            # Use greedy policy (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, step_info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Track feature toggles
            if step_info.get('action_type') == 'feature_toggle':
                episode_toggles += 1

            # Track achievements
            if step_info.get('subscribed', 0) == 1 and done:
                got_subscription = True
            if step_info.get('stage_reward', 0) == 15:
                got_first_call = True

            # Track final feature selection
            if done:
                final_features = step_info.get('active_features', [])
                final_n_features = step_info.get('n_active_features', 0)

            state = next_state

        results['subscriptions'].append(1 if got_subscription else 0)
        results['first_calls'].append(1 if got_first_call else 0)
        results['rewards'].append(episode_reward)
        results['steps'].append(episode_steps)
        results['feature_toggles'].append(episode_toggles)
        results['n_features_selected'].append(final_n_features)
        results['selected_features'].extend(final_features)

    # Calculate metrics
    sub_rate = np.mean(results['subscriptions']) * 100
    call_rate = np.mean(results['first_calls']) * 100
    avg_reward = np.mean(results['rewards'])
    avg_toggles = np.mean(results['feature_toggles'])
    avg_features = np.mean(results['n_features_selected'])

    # Feature selection analysis
    feature_counter = Counter(results['selected_features'])
    top_features = feature_counter.most_common(10)

    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(f"Episodes: {n_episodes}")

    print(f"\nBUSINESS METRICS:")
    print(f"  Subscription Rate: {sub_rate:.2f}% (baseline: 0.44%)")
    print(f"  First Call Rate: {call_rate:.2f}% (baseline: 4.0%)")
    print(f"  Improvement: {sub_rate/0.44:.1f}x subscriptions")

    print(f"\nTECHNICAL METRICS:")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Steps: {np.mean(results['steps']):.2f}")

    print(f"\nFEATURE SELECTION METRICS:")
    print(f"  Avg Feature Toggles: {avg_toggles:.2f}")
    print(f"  Avg Features Selected: {avg_features:.2f} / 16")
    print(f"  Feature Usage: {avg_features/16*100:.1f}%")
    print(f"  Data Collection Savings: {(16-avg_features)/16*100:.1f}%")

    print(f"\nTOP 10 MOST SELECTED FEATURES:")
    for rank, (feature, count) in enumerate(top_features, 1):
        percentage = (count / n_episodes) * 100
        print(f"  {rank}. {feature}: {count} times ({percentage:.1f}%)")

    print("="*80 + "\n")

    # Save results
    results_summary = {
        'subscription_rate': float(sub_rate),
        'first_call_rate': float(call_rate),
        'avg_reward': float(avg_reward),
        'baseline_sub_rate': 0.44,
        'baseline_call_rate': 4.0,
        'improvement_factor': float(sub_rate / 0.44),
        'avg_feature_toggles': float(avg_toggles),
        'avg_features_selected': float(avg_features),
        'feature_usage_percentage': float(avg_features / 16 * 100),
        'top_features': [
            {'feature': f, 'count': int(c), 'percentage': float(c/n_episodes*100)}
            for f, c in top_features
        ]
    }

    with open('logs/test_results_feature_selection.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"Results saved to: logs/test_results_feature_selection.json")
    print("\nNext: Run analyze_features.py for detailed feature analysis")

    return results_summary


def compare_with_baseline():
    """
    Compare feature selection agent with baseline agent.
    """
    print("\n" + "="*80)
    print("COMPARISON: BASELINE vs FEATURE SELECTION")
    print("="*80)

    # Load baseline results
    try:
        with open('logs/test_results.json', 'r') as f:
            baseline = json.load(f)

        # Load feature selection results
        with open('logs/test_results_feature_selection.json', 'r') as f:
            feature_sel = json.load(f)

        print(f"\n{'Metric':<30} {'Baseline':<15} {'Feature Selection':<20} {'Difference'}")
        print("-" * 80)

        print(f"{'Subscription Rate (%)':<30} {baseline['subscription_rate']:<15.2f} {feature_sel['subscription_rate']:<20.2f} {feature_sel['subscription_rate'] - baseline['subscription_rate']:+.2f}%")
        print(f"{'First Call Rate (%)':<30} {baseline['first_call_rate']:<15.2f} {feature_sel['first_call_rate']:<20.2f} {feature_sel['first_call_rate'] - baseline['first_call_rate']:+.2f}%")
        print(f"{'Improvement Factor':<30} {baseline['improvement_factor']:<15.2f}x {feature_sel['improvement_factor']:<20.2f}x")

        print(f"\n{'Features Used':<30} {'16 (100%)':<15} {feature_sel['avg_features_selected']:<20.2f} ({feature_sel['feature_usage_percentage']:.1f}%)")
        print(f"{'Data Collection Savings':<30} {'0%':<15} {100 - feature_sel['feature_usage_percentage']:<20.1f}%")

        print("\nKey Insights:")
        print(f"1. Feature selection uses {feature_sel['avg_features_selected']:.1f} features instead of all 16")
        print(f"2. Saves {100 - feature_sel['feature_usage_percentage']:.1f}% on data collection costs")
        print(f"3. Subscription rate: {feature_sel['subscription_rate']:.2f}% vs {baseline['subscription_rate']:.2f}% baseline")

        if feature_sel['subscription_rate'] > baseline['subscription_rate']:
            print(f"4. ✅ Feature selection IMPROVED performance by {feature_sel['subscription_rate'] - baseline['subscription_rate']:.2f}%")
        elif feature_sel['subscription_rate'] < baseline['subscription_rate']:
            print(f"4. ⚠️  Feature selection decreased performance by {baseline['subscription_rate'] - feature_sel['subscription_rate']:.2f}%")
        else:
            print(f"4. = Feature selection maintained same performance with fewer features")

    except FileNotFoundError:
        print("Baseline results not found. Run src/evaluate.py first.")

    print("="*80 + "\n")


if __name__ == "__main__":
    # Evaluate feature selection agent
    results = evaluate_agent()

    # Compare with baseline
    compare_with_baseline()
