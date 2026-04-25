import random
from collections import defaultdict
from scientist import Scientist
from oracle import Oracle
from planner import Planner
import oracle
import planner

class NexusAgent:
    def __init__(self):
        self.scientist = Scientist()
        self.oracle = Oracle()
        self.planner = Planner()

        # Q-learning
        self.Q = defaultdict(lambda: [0, 0, 0])  # actions: [-1, 0, +1]
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9

        self.prev_state = None
        self.prev_action = None
        self.prev_discrete = None

    # -------------------------------
    # Discretization
    # -------------------------------
    def discretize(self, state, predicted_peak, capacity):
        d = state.current_requests
        q = state.queue_length
        u = state.cpu_utilization

        demand_level = 0 if d < 500 else 1 if d < 1500 else 2
        queue_level  = 0 if q == 0 else 1 if q < 50 else 2
        util_level   = 0 if u < 0.7 else 1

        overload = 1 if predicted_peak > capacity else 0

        return (demand_level, queue_level, util_level, overload)

    # -------------------------------
    # Act
    # -------------------------------
    def act(self, state):
        # 1. Oracle
        predictions = self.oracle.predict_demand(state)
        peak = max(predictions)

        # 2. Scientist belief
        capacity = state.active_servers * self.scientist.capacity_per_server

        # 3. Planner
        base_action = self.planner.decide(state, predictions, self.scientist)

        # 4. RL state
        discrete = self.discretize(state, peak, capacity)

        # 5. epsilon-greedy
        if random.random() < self.epsilon:
            adj = random.choice([-1, 0, 1])
        else:
            q_vals = self.Q[discrete]
            adj = [-1, 0, 1][q_vals.index(max(q_vals))]

        final_action = base_action + adj
        final_action = max(-1, min(1, final_action))

        # store for learning
        self.prev_state = state
        self.prev_action = adj
        self.prev_discrete = discrete

        return final_action

    # -------------------------------
    # Learn
    # -------------------------------
    def learn(self, next_state, reward):
        if self.prev_state is None:
            return

        # update Scientist
        self.scientist.update(self.prev_state, next_state)

        # next state encoding
        predictions = self.oracle.predict_demand(next_state)
        peak = max(predictions)
        capacity = next_state.active_servers * self.scientist.capacity_per_server
        next_discrete = self.discretize(next_state, peak, capacity)

        # Q-learning update
        action_idx = [-1, 0, 1].index(self.prev_action)

        best_next = max(self.Q[next_discrete])

        self.Q[self.prev_discrete][action_idx] += self.alpha * (
            reward + self.gamma * best_next - self.Q[self.prev_discrete][action_idx]
        )