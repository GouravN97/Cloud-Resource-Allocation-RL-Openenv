import numpy as np

class Reflexion:
    def __init__(self):
        self.scale_bias = 0.0  # global bias

    def summarize_episode(self, states1, states2, rewards):
        avg_queue = np.mean([s[1] for s in states1])
        max_queue = max([s[1] for s in states1])
        avg_servers = np.mean([s[0] for s in states2])
        total_reward = sum(rewards)

        return {
            "avg_queue": avg_queue,
            "max_queue": max_queue,
            "avg_servers": avg_servers,
            "total_reward": total_reward
        }

    def build_prompt(self, summary):
        return f"""
You are analyzing a cloud autoscaling agent.

Episode stats:
- Avg queue: {summary['avg_queue']:.3f}
- Max queue: {summary['max_queue']:.3f}
- Avg servers: {summary['avg_servers']:.3f}
- Total reward: {summary['total_reward']:.3f}

Answer:
1. Was scaling too early, too late, or correct?
2. What is the main mistake?
3. Give ONE improvement rule.
"""

    def mock_llm(self, summary):
        max_q = summary["max_queue"]
        avg_q = summary["avg_queue"]
        avg_s = summary["avg_servers"]

        # detect late scaling
        if max_q > 0.3:
            return "Scaling too late. Increase servers earlier."

        # detect over-scaling
        elif avg_s > 0.5 and avg_q < 0.2:
            return "Scaling too early. Reduce unnecessary servers."

        # detect mild inefficiency
        elif avg_q > 0.2:
            return "Slight delay in scaling. Be more proactive."

        else:
            return "Scaling is efficient."

    def update_bias(self, response):
        if "too late" in response.lower():
            self.scale_bias += 0.2

        elif "too early" in response.lower():
            self.scale_bias -= 0.2

        elif "slight delay" in response.lower():
            self.scale_bias += 0.1

        # clamp
        self.scale_bias = max(-1.0, min(1.0, self.scale_bias))