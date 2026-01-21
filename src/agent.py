"""
Q-Learning Agent for CRM Sales Optimization

NUANCE #5: State Discretization (Q-Table)
- Continuous states need discrete keys for Q-table lookup
- Trade-off: 2 decimals = precision vs learning speed
- Expected state space: 5,000-10,000 visited states

NUANCE #6: Hyperparameter Choices
- Alpha=0.1: Conservative for noisy data (228:1 imbalance)
- Gamma=0.95: Values multi-step paths (subscription takes 3-5 actions)
- Epsilon decay=0.995: Reaches 0.01 around episode 1000

INTERVIEW PREP: Be able to justify EACH hyperparameter value!
"""

import logging
import numpy as np
from collections import defaultdict
import pickle
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


class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.

    Q-Learning Update Rule:
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    NUANCE #9: Modular Design for Debugging
    - Each method is independent and testable
    - Can check Q-values, epsilon, action selection separately
    - If something breaks, can isolate which component failed

    INTERVIEW: "How would you debug if Q-values aren't updating?"
    ANSWER: "Test each component:
      1. Check environment rewards (non-zero?)
      2. Check alpha > 0
      3. Check state discretization (states being visited?)
      4. Check update formula implementation"
    """

    def __init__(
        self,
        n_actions=6,
        learning_rate=0.1,        # Alpha
        discount_factor=0.95,     # Gamma
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    ):
        """
        Initialize Q-Learning agent.

        NUANCE #6: Hyperparameter Rationale
        Args:
            n_actions: 6 sales actions
            learning_rate: 0.1 = conservative updates for noisy environment
            discount_factor: 0.95 = value long-term rewards (multi-step process)
            epsilon_start: 1.0 = start with 100% exploration
            epsilon_end: 0.01 = never go to pure exploitation (maintain 1% exploration)
            epsilon_decay: 0.995 = reaches epsilon_end around episode 1000
        """
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: {state: [Q(s,a0), Q(s,a1), ..., Q(s,a5)]}
        # defaultdict initializes unseen states with zeros
        # NUANCE #5: States will be discretized tuples (rounded to 2 decimals)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # Tracking for analysis
        self.training_steps = 0
        self.episodes_trained = 0

        logger.info("="*80)
        logger.info("Q-LEARNING AGENT INITIALIZED")
        logger.info("="*80)
        logger.info(f"Actions: {n_actions}")
        logger.debug(f"Learning rate (alpha): {self.alpha}")
        logger.debug(f"Discount factor (gamma): {self.gamma}")
        logger.debug(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
        logger.debug(f"Expected epsilon=0.01 at episode: ~{int(np.log(epsilon_end/epsilon_start)/np.log(epsilon_decay))}")
        logger.info("="*80)

    def _discretize_state(self, state):
        """
        Convert continuous state to discrete key for Q-table.

        NUANCE #5: State Discretization Trade-off
        - Round to 2 decimals: Balance precision vs learning speed
        - Too granular (3+ decimals): Sparse Q-table, slow learning
        - Too coarse (0-1 decimals): Lose information, poor decisions

        Example:
          Days_since=0.34729 -> 0.35 (bucket: "about 1/3 of max")
          Days_since=0.36123 -> 0.36 (different bucket)
          Difference: ~1-2 days (meaningful for customer engagement)

        Args:
            state: 16-dim continuous numpy array

        Returns:
            tuple: Hashable discrete state key
        """
        # Round each feature to 2 decimal places
        discrete_state = tuple(np.round(state, 2))
        return discrete_state

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        NUANCE #6: Exploration vs Exploitation
        - Training: Use epsilon-greedy (explore to learn)
        - Evaluation: Use greedy (exploit learned policy)

        Epsilon-greedy:
        - With probability epsilon: Random action (explore)
        - With probability 1-epsilon: Best action (exploit)

        Args:
            state: 16-dim continuous state
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: Integer 0-5
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

        Q-Learning Formula:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Components:
        - Q(s,a): Current Q-value estimate
        - r: Immediate reward
        - gamma * max_a' Q(s',a'): Discounted future value
        - target = r + gamma * max_a' Q(s',a')
        - error = target - Q(s,a)
        - Update: Q(s,a) <- Q(s,a) + alpha * error

        NUANCE #6: Why alpha=0.1?
        - Conservative updates (10% step toward new estimate)
        - Good for noisy environments (228:1 imbalance, sparse rewards)
        - Alternative alpha=0.5: Too fast, unstable with sparse rewards

        NUANCE #6: Why gamma=0.95?
        - Values future rewards highly (future worth 95% of immediate)
        - Good for multi-step tasks (subscription takes 3-5 actions)
        - Alternative gamma=0.5: Too shortsighted, ignores long-term value

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

        NUANCE #6: Epsilon Decay Strategy
        - Exponential decay: epsilon *= epsilon_decay
        - Starts high (1.0 = 100% exploration)
        - Gradually decreases (more exploitation as agent learns)
        - Minimum epsilon_end (0.01 = always maintain 1% exploration)

        Why not decay to 0?
        - Environment may be non-stationary (customer behavior changes)
        - Small exploration helps adapt to changes
        - 1% exploration is negligible cost for robustness

        Calculation:
        To reach epsilon=0.01 from epsilon=1.0 with decay=0.995:
        0.01 = 1.0 * (0.995)^t
        t = log(0.01) / log(0.995)
        t ≈ 919 episodes
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def save(self, filepath):
        """
        Save agent state to file.

        Saves:
        - Q-table (learned policy)
        - Epsilon (exploration state)
        - Training progress (steps, episodes)
        - Hyperparameters (for reproducibility)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'q_table': dict(self.q_table),  # Convert defaultdict to dict
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

        logger.info(f"Agent saved to: {filepath}")
        logger.debug(f"  Q-table size: {len(self.q_table):,} states")
        logger.debug(f"  Episodes trained: {self.episodes_trained:,}")

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

        logger.info(f"Agent loaded from: {filepath}")
        logger.debug(f"  Q-table size: {len(self.q_table):,} states")
        logger.debug(f"  Episodes trained: {self.episodes_trained:,}")
        logger.debug(f"  Current epsilon: {self.epsilon:.4f}")

    def get_q_table_stats(self):
        """
        Get Q-table statistics for analysis.

        Returns:
            dict: Q-table statistics
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

    INTERVIEW PREP: Understand each component!
    """
    logger.info("="*80)
    logger.info("TESTING Q-LEARNING AGENT")
    logger.info("="*80)

    # Create agent
    agent = QLearningAgent(
        n_actions=6,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    logger.info("="*80)
    logger.info("TEST 1: State Discretization")
    logger.info("="*80)

    # Test continuous state discretization
    state = np.array([
        5.123456, 12.987654, 3.456789, 0.7234567,
        0.342789, 0.156789, 0.987123, 0.123456,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0123456, 0.0456789, 2.789123
    ], dtype=np.float32)

    discrete = agent._discretize_state(state)
    logger.debug(f"Continuous state:\n{state}")
    logger.debug(f"Discretized state (2 decimals):\n{discrete}")

    logger.info("="*80)
    logger.info("TEST 2: Action Selection")
    logger.info("="*80)

    # Test epsilon-greedy
    logger.debug(f"Epsilon: {agent.epsilon}")
    actions = [agent.select_action(state, training=True) for _ in range(10)]
    logger.debug(f"10 training actions (epsilon-greedy): {actions}")
    logger.debug(f"Should be mostly random (epsilon=1.0)")

    # Set epsilon to 0 for testing exploitation
    agent.epsilon = 0.0
    actions = [agent.select_action(state, training=True) for _ in range(10)]
    logger.debug(f"10 actions with epsilon=0 (greedy): {actions}")
    logger.debug(f"Should be all same (best action)")

    # Reset epsilon
    agent.epsilon = 1.0

    logger.info("="*80)
    logger.info("TEST 3: Q-value Updates")
    logger.info("="*80)

    # Simulate some updates
    for i in range(5):
        state = np.random.rand(16).astype(np.float32)
        action = i % 6
        reward = np.random.randn() * 10
        next_state = np.random.rand(16).astype(np.float32)
        done = (i == 4)

        agent.update(state, action, reward, next_state, done)

    stats = agent.get_q_table_stats()
    logger.info(f"After 5 updates:")
    logger.debug(f"  States in Q-table: {stats['num_states']}")
    logger.debug(f"  Average max Q: {stats['avg_max_q']:.2f}")

    logger.info("="*80)
    logger.info("TEST 4: Epsilon Decay")
    logger.info("="*80)

    agent.epsilon = 1.0
    epsilons = [agent.epsilon]

    for i in range(1000):
        agent.decay_epsilon()
        if i in [99, 499, 999]:
            epsilons.append(agent.epsilon)
            logger.debug(f"Episode {i+1}: epsilon = {agent.epsilon:.4f}")

    logger.debug(f"Epsilon trajectory: 1.0 -> {epsilons[-1]:.4f}")
    logger.debug(f"Expected to reach ~0.01 around episode 1000")

    logger.info("="*80)
    logger.info("TEST 5: Save/Load")
    logger.info("="*80)

    # Save agent
    agent.save('checkpoints/test_agent.pkl')

    # Create new agent and load
    new_agent = QLearningAgent()
    new_agent.load('checkpoints/test_agent.pkl')

    logger.debug(f"Original Q-table size: {len(agent.q_table)}")
    logger.debug(f"Loaded Q-table size: {len(new_agent.q_table)}")
    logger.debug(f"Match: {len(agent.q_table) == len(new_agent.q_table)}")

    logger.info("Agent tests complete!")
    logger.info("Ready for training!")
    logger.info("="*80)
