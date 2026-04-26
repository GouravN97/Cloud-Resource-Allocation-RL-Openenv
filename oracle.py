import textwrap

class Oracle:
    def __init__(self):
        # how far ahead we predict (tunable later)
        self.horizon = 5
        
        self.system_prompt = textwrap.dedent("""
            You are the 'Oracle' agent in a MARL system for Cloud Resource Management.
            Your job is to predict future traffic patterns.
            
            Analyze the request history and trend:
            - If trend is sharply positive: Predict a spike.
            - If confidence is low: Be conservative.
            
            Output strictly in JSON format.
        """)

    def get_prompt(self, obs):
        return f"History: {obs.demand_history}, Current: {obs.current_requests}"

    def predict(self, state, env):
        """
        Predict future demand using demand_history
        and publish forecast to message bus
        """

        history = state.demand_history

        # -------------------------------
        # 1. Safety check
        # -------------------------------
        if len(history) < 6:
            # not enough data → fallback
            predicted_demand = state.current_requests
            trend = 0
            confidence = "low"
        else:
            # -------------------------------
            # 2. Compute trend using history
            # -------------------------------
            first_half = history[:len(history)//2]
            second_half = history[len(history)//2:]

            avg_past = sum(first_half) / len(first_half)
            avg_recent = sum(second_half) / len(second_half)

            trend = avg_recent - avg_past

            # -------------------------------
            # 3. Predict future demand
            # -------------------------------
            predicted_demand = state.current_requests + trend * self.horizon

            # -------------------------------
            # 4. Detect noise vs real spike
            # -------------------------------
            # measure variability
            diffs = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
            volatility = sum(diffs) / len(diffs)

            if abs(trend) > volatility:
                confidence = "high"
            else:
                confidence = "low"

        # clamp prediction (no negative demand)
        predicted_demand = max(0, predicted_demand)

        # -------------------------------
        # 5. Publish to message bus
        # -------------------------------
        env.update_communication("oracle_forecast", {
            "predicted_demand": round(predicted_demand, 2),
            "trend": round(trend, 2),
            "confidence": confidence
        })