import numpy as np


def process_state_agent1(state):
    trend = state.current_requests - state.previous_requests

    # momentum (change in trend)
    if len(state.demand_history) >= 2:
        prev_trend = state.demand_history[-1] - state.demand_history[-2]
    else:
        prev_trend = 0

    momentum = trend - prev_trend

    return np.array([
        state.current_requests / 200.0,
        state.queue_length / 100.0,
        state.cpu_utilization,
        trend / 50.0,
        momentum / 50.0
    ])

def process_state_agent2(state):
    trend = state.current_requests - state.previous_requests

    return np.array([
        state.active_servers / 10.0,
        len(state.booting_queue) / 10.0,
        state.queue_length / 100.0,
        state.cpu_utilization,
        trend / 50.0
    ])

def compute_advantages(rewards):
    advantages = []
    rewards = [r * 5 for r in rewards]
    for t in range(len(rewards)):
        if t == 0:
            adv = rewards[t]
        else:
            adv = rewards[t] - rewards[t-1]

        advantages.append(adv)

    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages