import textwrap

class Planner:
    """
    Teacher Agent: Executive commander who resolves belief conflicts and issues actions.
    Consolidates Scientist and Oracle reports.
    """
    def __init__(self):
        self.system_prompt = textwrap.dedent("""
            You are the 'Cloud Commander'. 
            You must decide whether to Scale Up (+1), Scale Down (-1), or Hold (0).
            
            CRITICAL CONSTRAINTS:
            1. Scaling Delay: It takes 10 steps for a scaling action to take effect. 
               You must be PROACTIVE. If a spike is coming in T+10, scale NOW.
            2. Database Health: If the Scientist reports 'Collapsing', you MUST scale up 
               to relieve pressure, even if demand is stable.
            3. Server Cost: Do not scale up for 'Fake Spikes' or noise.
            
            Return your decision in this JSON format:
            {
                "action": -1 | 0 | 1,
                "priority": "High" | "Medium" | "Low",
                "internal_monologue": "Resolve conflicts between Scientist and Oracle here.",
                "final_rationale": "short explanation for the Critic"
            }
        """).strip()

    def get_prompt(self, obs, scientist_report, oracle_forecast):
        return f"""
        ENVIRONMENT STATE:
        - Active Servers: {obs.active_servers}
        - Current Queue: {obs.queue_length}
        - Booting Servers (In Pipeline): {sum(obs.booting_queue)}
        
        REPORTS:
        - Scientist Health Report: {scientist_report}
        - Oracle Demand Forecast: {oracle_forecast}
        
        Decide the best action to maximize reward (min cost, min SLA violation).
        """