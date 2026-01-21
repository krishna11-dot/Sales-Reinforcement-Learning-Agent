"""
Feature Importance Analysis Script

PURPOSE:
Answers the business question: "Which customer attributes drive subscriptions?"

This script analyzes a trained RL agent to discover:
1. Which features are most frequently selected?
2. What is the minimal effective feature set?
3. Which feature combinations lead to success?

FILE LOCATION: src/analyze_features.py

USAGE:
    python src/analyze_features.py
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter
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

# Import the feature selection agent (will be created next)
from agent_feature_selection import QLearningAgentFeatureSelection
from environment_feature_selection import CRMFeatureSelectionEnv


def analyze_feature_importance(agent_path='checkpoints/agent_feature_selection_final.pkl',
                                n_episodes=1000):
    """
    Analyze which features the trained agent learned to select.

    Args:
        agent_path: Path to trained agent pickle file
        n_episodes: Number of test episodes to run

    Returns:
        dict: Analysis results
    """
    logger.info("="*80)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Agent: {agent_path}")
    logger.info(f"Episodes: {n_episodes}")
    logger.info("="*80)

    # Load trained agent
    agent = QLearningAgentFeatureSelection(n_actions=22)
    agent.load(agent_path)

    logger.info(f"Agent loaded successfully")
    logger.debug(f"  Q-table size: {len(agent.q_table):,} states")
    logger.debug(f"  Episodes trained: {agent.episodes_trained:,}")

    # Load test environment
    env = CRMFeatureSelectionEnv(
        data_path='data/processed/crm_test.csv',
        stats_path='data/processed/historical_stats.json',
        mode='test'
    )

    # Feature names for reference
    feature_names = env.feature_names

    # Track results
    results = {
        'success_episodes': [],
        'failure_episodes': [],
        'all_episodes': []
    }

    feature_selection_counts = Counter()
    feature_selection_success = Counter()
    feature_selection_failure = Counter()

    logger.info("Running evaluation episodes...")

    for episode in range(n_episodes):
        if (episode + 1) % 100 == 0:
            logger.debug(f"  Progress: {episode+1}/{n_episodes}")

        state, _ = env.reset()
        done = False
        truncated = False

        episode_data = {
            'feature_toggles': 0,
            'final_features': None,
            'n_features': 0,
            'subscribed': 0,
            'reward': 0
        }

        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)

            episode_data['reward'] += reward

            if info['action_type'] == 'feature_toggle':
                episode_data['feature_toggles'] += 1
            else:
                # CRM action taken (terminal)
                episode_data['final_features'] = info['active_features']
                episode_data['n_features'] = info['n_active_features']
                episode_data['subscribed'] = info['subscribed']

            state = next_state

        # Categorize by outcome
        if episode_data['subscribed'] == 1:
            results['success_episodes'].append(episode_data)
            feature_selection_success.update(episode_data['final_features'])
        else:
            results['failure_episodes'].append(episode_data)
            feature_selection_failure.update(episode_data['final_features'])

        results['all_episodes'].append(episode_data)
        feature_selection_counts.update(episode_data['final_features'])

    # Analysis
    n_success = len(results['success_episodes'])
    n_failure = len(results['failure_episodes'])

    logger.info("="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Success episodes: {n_success} ({n_success/n_episodes*100:.1f}%)")
    logger.info(f"Failure episodes: {n_failure} ({n_failure/n_episodes*100:.1f}%)")

    # Feature importance ranking (success episodes)
    logger.info("="*80)
    logger.info("FEATURE IMPORTANCE (Success Episodes)")
    logger.info("="*80)
    logger.info(f"{'Rank':<6} {'Feature':<30} {'Frequency':<12} {'Percentage'}")
    logger.info("-" * 80)

    for rank, (feature, count) in enumerate(feature_selection_success.most_common(), 1):
        percentage = (count / n_success) * 100 if n_success > 0 else 0
        logger.debug(f"{rank:<6} {feature:<30} {count:<12} {percentage:.1f}%")

    # Average features used
    avg_features_success = np.mean([e['n_features'] for e in results['success_episodes']]) if n_success > 0 else 0
    avg_features_failure = np.mean([e['n_features'] for e in results['failure_episodes']]) if n_failure > 0 else 0
    avg_features_all = np.mean([e['n_features'] for e in results['all_episodes']])

    logger.info("="*80)
    logger.info("FEATURE SET SIZE")
    logger.info("="*80)
    logger.info(f"Average features used (Success):  {avg_features_success:.2f}")
    logger.debug(f"Average features used (Failure):  {avg_features_failure:.2f}")
    logger.debug(f"Average features used (Overall):  {avg_features_all:.2f}")

    logger.info(f"Insight: Agent learned to use ~{avg_features_success:.1f} features")
    logger.debug(f"         instead of all 16 features")

    # Most common feature combinations
    logger.info("="*80)
    logger.info("TOP FEATURE COMBINATIONS (Success)")
    logger.info("="*80)

    combo_counter = Counter()
    for episode in results['success_episodes']:
        if episode['final_features']:
            combo = tuple(sorted(episode['final_features']))
            combo_counter[combo] += 1

    logger.info(f"{'Rank':<6} {'Combination':<50} {'Count':<8} {'Percentage'}")
    logger.info("-" * 80)

    for rank, (combo, count) in enumerate(combo_counter.most_common(10), 1):
        percentage = (count / n_success) * 100 if n_success > 0 else 0
        combo_str = ', '.join(combo[:4])  # Show first 4 features
        if len(combo) > 4:
            combo_str += f" + {len(combo)-4} more"
        logger.debug(f"{rank:<6} {combo_str:<50} {count:<8} {percentage:.1f}%")

    # Feature toggle behavior
    avg_toggles_success = np.mean([e['feature_toggles'] for e in results['success_episodes']]) if n_success > 0 else 0
    avg_toggles_failure = np.mean([e['feature_toggles'] for e in results['failure_episodes']]) if n_failure > 0 else 0

    logger.info("="*80)
    logger.info("FEATURE TOGGLE BEHAVIOR")
    logger.info("="*80)
    logger.debug(f"Average toggles (Success): {avg_toggles_success:.2f}")
    logger.debug(f"Average toggles (Failure): {avg_toggles_failure:.2f}")

    # Performance metrics
    avg_reward_success = np.mean([e['reward'] for e in results['success_episodes']]) if n_success > 0 else 0
    avg_reward_failure = np.mean([e['reward'] for e in results['failure_episodes']]) if n_failure > 0 else 0

    logger.info("="*80)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*80)
    logger.info(f"Average reward (Success): {avg_reward_success:.2f}")
    logger.debug(f"Average reward (Failure): {avg_reward_failure:.2f}")

    # Save results
    results_summary = {
        'n_episodes': n_episodes,
        'n_success': n_success,
        'n_failure': n_failure,
        'success_rate': n_success / n_episodes * 100,
        'avg_features_success': avg_features_success,
        'avg_features_failure': avg_features_failure,
        'avg_features_all': avg_features_all,
        'top_features_success': [
            {'feature': f, 'count': c, 'percentage': c/n_success*100 if n_success > 0 else 0}
            for f, c in feature_selection_success.most_common(10)
        ],
        'top_combinations': [
            {'combo': list(combo), 'count': count, 'percentage': count/n_success*100 if n_success > 0 else 0}
            for combo, count in combo_counter.most_common(5)
        ]
    }

    output_path = Path('logs/feature_analysis_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    logger.info("="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)

    if n_success > 0:
        top_3_features = [f for f, c in feature_selection_success.most_common(3)]
        logger.info(f"1. Most important features: {', '.join(top_3_features)}")
        logger.info(f"2. Optimal feature count: ~{avg_features_success:.1f} features")
        logger.info(f"3. Agent learns to use {avg_features_success/16*100:.1f}% of available features")

        if avg_features_success < 16:
            savings = (16 - avg_features_success) / 16 * 100
            logger.info(f"4. Data collection savings: ~{savings:.0f}% fewer features needed")
    else:
        logger.warning("No successful episodes - agent may need more training")

    logger.info("="*80)

    return results_summary


def compare_feature_sets():
    """
    Compare performance of different feature subsets.

    Answers: "What happens if we only use top N features?"
    """
    logger.info("="*80)
    logger.info("FEATURE SET COMPARISON")
    logger.info("="*80)

    # This would require training multiple agents with different feature sets
    # For now, we analyze the learned feature selection patterns

    logger.info("This analysis shows which feature combinations work best")
    logger.info("Based on the trained agent's learned policy")

    # Would implement: Test agent with only top 3, top 5, top 10 features
    # Compare subscription rates for each configuration

    pass


if __name__ == "__main__":
    """
    Run feature importance analysis.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--agent', type=str,
                        default='checkpoints/agent_feature_selection_final.pkl',
                        help='Path to trained agent')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of evaluation episodes')

    args = parser.parse_args()

    # Run analysis
    results = analyze_feature_importance(
        agent_path=args.agent,
        n_episodes=args.episodes
    )

    logger.info("Feature analysis complete!")
    logger.info("Check logs/feature_analysis_results.json for detailed results")
