"""
Q-Learning Agent for CRM Sales Optimization with Feature Selection

This agent learns:
1. WHICH features to use (toggle features ON/OFF)
2. WHAT CRM action to take (Email, Call, Demo, etc.)

Key difference from baseline agent:
- n_actions = 22 instead of 6
- Actions 0-15: Toggle features (non-terminal)
- Actions 16-21: CRM actions (terminal)
"""

import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path


class QLearningAgentFeatureSelection:
    """
    Tabular Q-Learning agent with feature selection capability.

    State: 32-dimensional (16 feature mask + 16 customer features)
    Actions: 22 discrete actions (16 toggles + 6 CRM actions)

    Q-Learning Update Rule:
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(
        self,
        n_actions=22,  # Changed from 6 to 22
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    ):
        """
        Initialize Q-Learning agent for feature selection.

        Args:
            n_actions: 22 actions (16 toggles + 6 CRM)
            learning_rate: 0.1 = conservative updates
            discount_factor: 0.95 = value long-term rewards
            epsilon_start: 1.0 = start with 100% exploration
            epsilon_end: 0.01 = maintain 1% exploration
            epsilon_decay: 0.995 = reaches epsilon_end around episode 1000
        """
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: {state: [Q(s,a0), Q(s,a1), ..., Q(s,a21)]}
        # State keys are 32-dim tuples (rounded to 2 decimals)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # Tracking
        self.training_steps = 0
        self.episodes_trained = 0

        print("\n" + "="*80)
        print("Q-LEARNING AGENT WITH FEATURE SELECTION INITIALIZED")
        print("="*80)
        print(f"Actions: {n_actions} (16 feature toggles + 6 CRM actions)")
        print(f"Learning rate (alpha): {self.alpha}")
        print(f"Discount factor (gamma): {self.gamma}")
        print(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
        print(f"Expected epsilon=0.01 at episode: ~{int(np.log(epsilon_end/epsilon_start)/np.log(epsilon_decay))}")
        print("="*80 + "\n")

    def _discretize_state(self, state):
        """
        Convert continuous state to discrete key for Q-table.

        State is 32-dimensional:
        - [0-15]: Feature mask (already discrete 0/1)
        - [16-31]: Customer features (continuous, need rounding)

        Args:
            state: 32-dim continuous numpy array

        Returns:
            tuple: Hashable discrete state key
        """
        # Round each feature to 2 decimal places
        discrete_state = tuple(np.round(state, 2))
        return discrete_state

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: 32-dim continuous state
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: Integer 0-21
                0-15: Toggle features
                16-21: CRM actions
        """
        state_key = self._discretize_state(state)

        if training and np.random.rand() < self.epsilon:
            # Explore: Random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: Best known action
            q_values = self.q_table[state_key]

            # If all Q-values are equal (unseen state), random action
            if np.all(q_values == q_values[0]):
                return np.random.randint(self.n_actions)

            # Argmax with tie-breaking
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning rule.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Resulting state
            done: Episode finished?
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Future value (0 if episode done, otherwise max Q-value of next state)
        if done:
            max_next_q = 0  # No future value at terminal state
        else:
            max_next_q = np.max(self.q_table[next_state_key])

        # Target Q-value
        target = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[state_key][action] += self.alpha * (target - current_q)

        self.training_steps += 1

    def decay_epsilon(self):
        """
        Decay exploration rate.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def save(self, filepath):
        """
        Save agent state to file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes_trained': self.episodes_trained,
            'hyperparams': {
                'n_actions': self.n_actions,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"Agent saved to: {filepath}")
        print(f"  Q-table size: {len(self.q_table):,} states")
        print(f"  Episodes trained: {self.episodes_trained:,}")

    def load(self, filepath):
        """
        Load agent state from file.
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        # Restore Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.q_table.update(save_dict['q_table'])

        # Restore state
        self.epsilon = save_dict['epsilon']
        self.training_steps = save_dict['training_steps']
        self.episodes_trained = save_dict['episodes_trained']

        print(f"Agent loaded from: {filepath}")
        print(f"  Q-table size: {len(self.q_table):,} states")
        print(f"  Episodes trained: {self.episodes_trained:,}")
        print(f"  Current epsilon: {self.epsilon:.4f}")

    def get_q_table_stats(self):
        """
        Get Q-table statistics for analysis.
        """
        if len(self.q_table) == 0:
            return {
                'num_states': 0,
                'avg_max_q': 0,
                'min_max_q': 0,
                'max_max_q': 0
            }

        # Get max Q-value for each state
        max_q_values = [np.max(q_vals) for q_vals in self.q_table.values()]

        return {
            'num_states': len(self.q_table),
            'avg_max_q': np.mean(max_q_values),
            'min_max_q': np.min(max_q_values),
            'max_max_q': np.max(max_q_values),
            'std_max_q': np.std(max_q_values)
        }


if __name__ == "__main__":
    """
    Test agent functionality.
    """
    print("\n" + "="*80)
    print("TESTING FEATURE SELECTION AGENT")
    print("="*80)

    # Create agent
    agent = QLearningAgentFeatureSelection(
        n_actions=22,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    print("\n" + "="*80)
    print("TEST 1: State Discretization (32-dim)")
    print("="*80)

    # Test 32-dim state discretization
    state = np.random.rand(32).astype(np.float32)
    discrete = agent._discretize_state(state)
    print(f"State dimension: {state.shape}")
    print(f"Discretized state length: {len(discrete)}")

    print("\n" + "="*80)
    print("TEST 2: Action Selection (0-21)")
    print("="*80)

    actions = [agent.select_action(state, training=True) for _ in range(20)]
    print(f"20 training actions (epsilon-greedy): {actions}")
    print(f"Should include both toggles (0-15) and CRM actions (16-21)")

    # Count action types
    toggles = sum(1 for a in actions if a < 16)
    crm = sum(1 for a in actions if a >= 16)
    print(f"Toggles (0-15): {toggles}, CRM actions (16-21): {crm}")

    print("\n" + "="*80)
    print("TEST 3: Q-value Updates")
    print("="*80)

    # Simulate some updates
    for i in range(10):
        state = np.random.rand(32).astype(np.float32)
        action = i % 22
        reward = np.random.randn() * 10
        next_state = np.random.rand(32).astype(np.float32)
        done = (i == 9)

        agent.update(state, action, reward, next_state, done)

    stats = agent.get_q_table_stats()
    print(f"After 10 updates:")
    print(f"  States in Q-table: {stats['num_states']}")
    print(f"  Average max Q: {stats['avg_max_q']:.2f}")

    print("\n" + "="*80)
    print("Agent tests complete!")
    print("Ready for training!")
    print("="*80 + "\n")
