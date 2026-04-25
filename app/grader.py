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

    def grade_episode(self, episode_path_or_history):
        if isinstance(episode_path_or_history, str):
            with open(episode_path_or_history, "r") as f:
                history = json.load(f)
        else:
            history = episode_path_or_history

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
            latency = None
            queue = None

            if isinstance(step, dict):
                latency = step.get("info", {}).get("latency")
                obs = step.get("observation")

                if obs is not None:
                    queue = getattr(obs, "queue_length", None)
                    if queue is None and isinstance(obs, dict):
                        queue = obs.get("queue_length")

            if latency is None or queue is None:
                raise ValueError("Unable to extract latency/queue from history step.")

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
            obs = step.get("observation")
            active_servers = None

            if obs is not None:
                active_servers = getattr(obs, "active_servers", None)
                if active_servers is None and isinstance(obs, dict):
                    active_servers = obs.get("active_servers")

            if active_servers is None:
                active_servers = step.get("state", {}).get("active_servers")

            if active_servers is None:
                raise ValueError("Unable to extract active_servers from history step.")

            total_servers += active_servers

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
            obs = step.get("observation")
            curr_servers = None

            if obs is not None:
                curr_servers = getattr(obs, "active_servers", None)
                if curr_servers is None and isinstance(obs, dict):
                    curr_servers = obs.get("active_servers")

            if curr_servers is None:
                curr_servers = step.get("state", {}).get("active_servers")

            if curr_servers is None:
                raise ValueError("Unable to extract active_servers from history step.")

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


def grade_episode(episode_history, config=None):
    """Grade an episode from history or from a JSON file path."""
    grader = Grader()
    return grader.grade_episode(episode_history)
