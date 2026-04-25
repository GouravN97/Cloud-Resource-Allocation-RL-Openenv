import textwrap

class Oracle:
    """
    Teacher Agent: Analyzes temporal demand history to forecast future spikes.
    Part of the 'Long-Horizon Planning' reasoning chain.
    """
    def __init__(self):
        self.system_prompt = textwrap.dedent("""
            You are the 'Demand Oracle' for a Cloud Autoscaler.
            Your job is to identify workload patterns and predict demand for the next 10 steps.
            
            PATTERNS:
            - Steady: Flat or low noise.
            - Diurnal: Slow sinusoidal rise/fall.
            - Flash Crowd: Exponential spike (High Risk).
            - Fake Spike: Sharp rise followed by immediate drop (No action needed).
            
            Return your forecast in this JSON format:
            {
                "pattern_detected": "steady" | "diurnal" | "flash_crowd" | "noise",
                "trend": "rising" | "falling" | "stable",
                "predicted_peak_demand_t10": int,
                "confidence": float,
                "reasoning": "short explanation"
            }
        """).strip()

    def get_prompt(self, obs):
        return f"""
        DEMAND HISTORY (Last 10 steps):
        {obs.demand_history}
        
        CURRENT DEMAND: {obs.current_requests}
        PREVIOUS DEMAND: {obs.previous_requests}
        """