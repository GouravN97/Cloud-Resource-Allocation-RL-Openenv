import json
import math

class Grader:
    def __init__(self, config=None):
        # Default weights (can tune later)
        self.w_sla = 0.5
        self.w_cost = 0.2
        self.w_stability = 0.3

        # Thresholds
        self.latency_threshold = 200  # ms
        self.queue_threshold = 50

        # Thrashing config
        self.scale_change_penalty = 0.05
        self.direction_flip_penalty = 0.2

        if config:
            self.__dict__.update(config)

    def grade_episode(self, episode_path):
        with open(episode_path, "r") as f:
            history = json.load(f)

        sla_score = self._compute_sla(history)
        cost_score = self._compute_cost(history)
        stability_score = self._compute_stability(history)

        total_reward = (
            self.w_sla * sla_score +
            self.w_cost * cost_score +
            self.w_stability * stability_score
        )

        return {
            "total_reward": total_reward,
            "sla": sla_score,
            "cost": cost_score,
            "stability": stability_score
        }

    # -------------------------
    # SLA COMPONENT
    # -------------------------
    def _compute_sla(self, history):
        penalty = 0

        for step in history:
            latency = step["metrics"]["latency"]
            queue = step["metrics"]["queue_length"]

            if latency > self.latency_threshold:
                penalty += (latency - self.latency_threshold) / self.latency_threshold

            if queue > self.queue_threshold:
                penalty += (queue - self.queue_threshold) / self.queue_threshold

        return math.exp(-penalty / len(history))

    # -------------------------
    # COST COMPONENT
    # -------------------------
    def _compute_cost(self, history):
        total_servers = 0

        for step in history:
            total_servers += step["state"]["active_servers"]

        avg_servers = total_servers / len(history)

        # Normalize (assume max reasonable = 100)
        return 1 - min(avg_servers / 100, 1)

    # -------------------------
    # STABILITY COMPONENT
    # -------------------------
    def _compute_stability(self, history):
        penalty = 0
        prev_servers = None
        prev_delta = None

        for step in history:
            curr_servers = step["state"]["active_servers"]

            if prev_servers is not None:
                delta = curr_servers - prev_servers

                # Penalize frequent scaling
                penalty += abs(delta) * self.scale_change_penalty

                # Penalize oscillation (direction flip)
                if prev_delta is not None:
                    if (delta > 0 and prev_delta < 0) or (delta < 0 and prev_delta > 0):
                        penalty += self.direction_flip_penalty

                prev_delta = delta

            prev_servers = curr_servers

        return math.exp(-penalty / len(history))
