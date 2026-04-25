class Planner:
    def decide(self, state, predictions, scientist):
        """
        Decide scaling action based on future demand vs capacity
        """

        active = state.active_servers

        # NOTE: add booting_servers later when available
        total_servers_future = active  

        capacity_per_server = scientist.capacity_per_server

        # 1. Estimate future capacity
        future_capacity = total_servers_future * capacity_per_server

        # 2. Worst-case demand
        peak_demand = max(predictions)

        # -----------------------------
        # 3. Decision logic
        # -----------------------------

        # SCALE UP (proactive)
        if peak_demand > future_capacity:
            return 1

        # SCALE DOWN (conservative)
        if (
            state.queue_length == 0 and
            peak_demand < 0.6 * future_capacity
        ):
            return -1

        # OTHERWISE HOLD
        return 0