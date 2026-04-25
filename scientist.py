class Scientist:
    def __init__(self):
        # Initial rough guesses (will be learned)
        self.capacity_per_server = 100.0
        self.congestion_threshold = 150.0

        # Smoothing factor for updates
        self.alpha = 0.1

    def update(self, prev_state, current_state):
        """
        Update internal beliefs using transition:
        prev_state → current_state
        """

        if prev_state is None:
            return

        # -------------------------------
        # 1. Estimate capacity per server
        # -------------------------------
        servers = prev_state.active_servers

        if servers > 0:
            observed_capacity = prev_state.current_requests / servers

            # exponential moving average
            self.capacity_per_server = (
                (1 - self.alpha) * self.capacity_per_server
                + self.alpha * observed_capacity
            )

        # -----------------------------------
        # 2. Detect congestion collapse point
        # -----------------------------------
        q_prev = prev_state.queue_length
        q_curr = current_state.queue_length

        u_prev = prev_state.cpu_utilization
        u_curr = current_state.cpu_utilization

        # Key signal:
        # queue increasing BUT utilization dropping → system choking
        if q_curr > q_prev and u_curr < u_prev:
            # update threshold conservatively
            self.congestion_threshold = (
                (1 - self.alpha) * self.congestion_threshold
                + self.alpha * q_curr
            )