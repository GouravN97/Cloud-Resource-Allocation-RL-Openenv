import textwrap

class Scientist:
    """
    Teacher Agent: Analyzes system health and discovers non-linear bottlenecks.
    Part of the 'World Modeling' reasoning chain.
    """
    def __init__(self):
        # Learned parameters
        self.capacity_per_server = 100.0
        self.congestion_threshold = 150.0

        # Learning rate
        self.alpha = 0.1

        self.system_prompt = textwrap.dedent("""
            You are the 'Scientist' agent in a MARL system for Cloud Resource Management.
            Your job is to diagnose system health and detect bottlenecks.
            
            Look for 'Congestion Collapse':
            - High queue length + dropping CPU utilization = The system is thrashing.
            - Low CPU utilization + High active servers = Under-utilization.
            
            Output strictly in JSON format.
        """)

    def get_prompt(self, obs):
        return f"Current Observation: {obs.model_dump()}"

    def update(self, prev_state, current_state, env):
        """
        Update internal beliefs and publish report to message bus
        """

        # -------------------------------
        # 0. Safety check
        # -------------------------------
        if current_state is None:
            return

        # -------------------------------
        # 1. Learn capacity per server
        # -------------------------------
        if prev_state is not None:
            servers = prev_state.active_servers

            if servers > 0:
                observed_capacity = prev_state.current_requests / servers

                self.capacity_per_server = (
                    (1 - self.alpha) * self.capacity_per_server
                    + self.alpha * observed_capacity
                )

        # -------------------------------
        # 2. Detect congestion collapse
        # -------------------------------
        if prev_state is not None:
            q_prev = prev_state.queue_length
            q_curr = current_state.queue_length

            u_prev = prev_state.cpu_utilization
            u_curr = current_state.cpu_utilization

            # congestion signal: queue ↑ but utilization ↓
            if q_curr > q_prev and u_curr < u_prev:
                self.congestion_threshold = (
                    (1 - self.alpha) * self.congestion_threshold
                    + self.alpha * q_curr
                )

        # -------------------------------
        # 3. Compute future capacity (FIX)
        # -------------------------------
        active = current_state.active_servers
        booting = len(current_state.booting_queue)

        future_servers = active + booting
        future_capacity = future_servers * self.capacity_per_server

        # -------------------------------
        # 4. Compute risk level
        # -------------------------------
        queue = current_state.queue_length
        cpu = current_state.cpu_utilization

        if queue > self.congestion_threshold:
            risk = "high"
        elif cpu > 0.8:
            risk = "medium"
        else:
            risk = "low"

        # -------------------------------
        # 5. Publish to message bus
        # -------------------------------
        env.update_communication("scientist_report", {
            "capacity_per_server": round(self.capacity_per_server, 2),
            "future_capacity": round(future_capacity, 2),
            "active_servers": active,
            "booting_servers": booting,
            "congestion_threshold": round(self.congestion_threshold, 2),
            "risk": risk
        })