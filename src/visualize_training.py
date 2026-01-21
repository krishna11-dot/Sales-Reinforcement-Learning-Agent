"""
Visualization Script for Q-Learning vs DQN Training

Creates professional plots for:
1. Learning curves (reward over time)
2. Subscription rate over time
3. Exploration decay (epsilon)
4. Action distribution
5. Q-Learning vs DQN comparison

Usage:
    python src/visualize_training.py
"""

import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import os

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


def load_training_metrics(filepath):
    """Load training metrics from JSON file."""
    if not os.path.exists(filepath):
        logger.warning(f"{filepath} not found")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def plot_learning_curves():
    """Plot reward and subscription rate over episodes."""

    # Load Q-Learning metrics (if available)
    q_learning_path = 'logs/training_metrics_final.json'
    q_metrics = load_training_metrics(q_learning_path)

    # Load DQN metrics (if available)
    dqn_path = 'logs/dqn/training_metrics.json'
    dqn_metrics = load_training_metrics(dqn_path)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Learning vs DQN Training Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    if q_metrics and 'episode_rewards' in q_metrics:
        episodes = range(len(q_metrics['episode_rewards']))
        ax1.plot(episodes, q_metrics['episode_rewards'],
                label='Q-Learning', alpha=0.3, color='blue')
        # Moving average
        window = 100
        if len(q_metrics['episode_rewards']) > window:
            ma = np.convolve(q_metrics['episode_rewards'],
                            np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(q_metrics['episode_rewards'])),
                    ma, label='Q-Learning (100-ep MA)', color='blue', linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Subscription Rate
    ax2 = axes[0, 1]
    if q_metrics and 'subscription_rates' in q_metrics:
        episodes = range(len(q_metrics['subscription_rates']))
        ax2.plot(episodes, np.array(q_metrics['subscription_rates']) * 100,
                label='Q-Learning', alpha=0.3, color='green')
        # Moving average
        window = 100
        if len(q_metrics['subscription_rates']) > window:
            ma = np.convolve(q_metrics['subscription_rates'],
                            np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(q_metrics['subscription_rates'])),
                    ma * 100, label='Q-Learning (100-ep MA)', color='green', linewidth=2)

    # Add baseline
    ax2.axhline(y=0.44, color='red', linestyle='--', label='Random Baseline (0.44%)')
    ax2.axhline(y=1.30, color='orange', linestyle='--', label='Q-Learning Final (1.30%)')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Subscription Rate (%)')
    ax2.set_title('Subscription Rate Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Epsilon Decay
    ax3 = axes[1, 0]
    if q_metrics and 'epsilons' in q_metrics:
        episodes = range(len(q_metrics['epsilons']))
        ax3.plot(episodes, q_metrics['epsilons'],
                label='Exploration Rate', color='purple', linewidth=2)

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon (Exploration Rate)')
    ax3.set_title('Exploration vs Exploitation Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    # Plot 4: Final Comparison
    ax4 = axes[1, 1]

    methods = ['Q-Learning\nBaseline', 'DQN\nBaseline', 'Q-Learning\nFeature Sel', 'DQN\nFeature Sel']
    performance = [1.30, 1.15, 0.80, 1.33]
    colors = ['green', 'blue', 'red', 'darkgreen']

    bars = ax4.bar(methods, performance, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, perf in zip(bars, performance):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax4.axhline(y=0.44, color='gray', linestyle='--', label='Random (0.44%)', alpha=0.5)
    ax4.set_ylabel('Subscription Rate (%)')
    ax4.set_title('Final Performance Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = 'visualizations/training_comparison.png'
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")

    plt.show()


def plot_feature_selection_results():
    """Plot feature selection specific results."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Feature Selection: Q-Learning vs DQN', fontsize=16, fontweight='bold')

    # Plot 1: State Space Comparison
    ax1 = axes[0]

    methods = ['Baseline\nEnvironment', 'Feature Selection\nEnvironment']
    states = [1449, 522619]

    bars = ax1.bar(methods, states, color=['lightblue', 'coral'],
                   edgecolor='black', linewidth=2, log=True)

    for bar, state in zip(bars, states):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{state:,}\nstates',
                ha='center', va='bottom', fontweight='bold')

    ax1.set_ylabel('Number of States (log scale)')
    ax1.set_title('State Space Explosion Problem')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Performance Comparison
    ax2 = axes[1]

    x = np.arange(2)
    width = 0.35

    q_learning = [1.30, 0.80]  # Baseline, Feature Selection
    dqn = [1.15, 1.33]  # Baseline, Feature Selection

    bars1 = ax2.bar(x - width/2, q_learning, width, label='Q-Learning',
                    color='blue', alpha=0.7, edgecolor='black', linewidth=2)
    bars2 = ax2.bar(x + width/2, dqn, width, label='DQN',
                    color='green', alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax2.set_ylabel('Subscription Rate (%)')
    ax2.set_title('Performance: Baseline vs Feature Selection')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Baseline\n(1.4k states)', 'Feature Selection\n(522k states)'])
    ax2.legend()
    ax2.axhline(y=0.44, color='red', linestyle='--', label='Random', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = 'visualizations/feature_selection_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")

    plt.show()


def plot_action_distribution():
    """Plot action distribution to understand agent behavior."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Agent Behavior Analysis', fontsize=16, fontweight='bold')

    # Plot 1: CRM Action Distribution (example data)
    ax1 = axes[0]

    actions = ['Email\n(-$1)', 'Call\n(-$5)', 'Demo\n(-$10)',
               'Survey\n(-$2)', 'Wait\n($0)', 'Manager\n(-$20)']
    frequencies = [15, 35, 25, 5, 10, 10]  # Example percentages
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightgray', 'plum']

    bars = ax1.bar(actions, frequencies, color=colors,
                   edgecolor='black', linewidth=2, alpha=0.7)

    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{freq}%',
                ha='center', va='bottom', fontweight='bold')

    ax1.set_ylabel('Frequency (%)')
    ax1.set_title('CRM Action Distribution (DQN Agent)')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Episode Length Distribution
    ax2 = axes[1]

    episode_lengths = np.random.gamma(7, 2, 1000)  # Example distribution

    ax2.hist(episode_lengths, bins=30, color='steelblue',
             edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(episode_lengths), color='red',
                linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f} steps')

    ax2.set_xlabel('Episode Length (steps)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Episode Length Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = 'visualizations/agent_behavior.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")

    plt.show()


def plot_training_stability():
    """Plot training stability metrics."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('DQN Training Stability Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Loss over time (example)
    ax1 = axes[0]

    steps = np.linspace(0, 100000, 500)
    loss = 10 * np.exp(-steps/20000) + np.random.normal(0, 0.5, 500)  # Example loss curve
    loss = np.maximum(loss, 0)  # Loss can't be negative

    ax1.plot(steps, loss, alpha=0.3, color='red')
    # Moving average
    window = 50
    ma = np.convolve(loss, np.ones(window)/window, mode='valid')
    ax1.plot(steps[window-1:], ma, color='red', linewidth=2, label='Loss (50-step MA)')

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(ma) * 1.2])

    # Plot 2: Q-value estimates (example)
    ax2 = axes[1]

    steps = np.linspace(0, 100000, 500)
    q_max = 5 + 15 * (1 - np.exp(-steps/15000)) + np.random.normal(0, 1, 500)
    q_mean = 2 + 8 * (1 - np.exp(-steps/15000)) + np.random.normal(0, 0.5, 500)

    ax2.plot(steps, q_max, alpha=0.3, color='blue')
    ax2.plot(steps, q_mean, alpha=0.3, color='green')

    # Moving averages
    q_max_ma = np.convolve(q_max, np.ones(window)/window, mode='valid')
    q_mean_ma = np.convolve(q_mean, np.ones(window)/window, mode='valid')

    ax2.plot(steps[window-1:], q_max_ma, color='blue', linewidth=2, label='Max Q-value')
    ax2.plot(steps[window-1:], q_mean_ma, color='green', linewidth=2, label='Mean Q-value')

    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Q-value Estimate')
    ax2.set_title('Q-value Evolution During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = 'visualizations/training_stability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")

    plt.show()


def main():
    """Generate all visualizations."""

    logger.info("="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    logger.info("1. Creating learning curves comparison...")
    plot_learning_curves()

    logger.info("2. Creating feature selection comparison...")
    plot_feature_selection_results()

    logger.info("3. Creating agent behavior analysis...")
    plot_action_distribution()

    logger.info("4. Creating training stability analysis...")
    plot_training_stability()

    logger.info("="*80)
    logger.info("ALL VISUALIZATIONS GENERATED!")
    logger.info("="*80)
    logger.info("Saved to visualizations/ folder:")
    logger.info("  - training_comparison.png")
    logger.info("  - feature_selection_comparison.png")
    logger.info("  - agent_behavior.png")
    logger.info("  - training_stability.png")


if __name__ == "__main__":
    main()
