import textwrap

class Scientist:
    """
    Teacher Agent: Analyzes system health and discovers non-linear bottlenecks.
    Part of the 'World Modeling' reasoning chain.
    """
    def __init__(self):
        self.system_prompt = textwrap.dedent("""
            You are the 'System Scientist' for a Cloud Autoscaler.
            Your job is to analyze metrics and detect system health issues, specifically 'Congestion Collapse'.
            
            PHYSICS CONTEXT:
            - Normal Capacity: ~75-80 req/s per server.
            - Congestion Collapse: If the queue grows too large, backend database latency increases, 
              causing each server's effective capacity to drop quadratically.
            
            Analyze the relationship between Current Requests, CPU Utilization, and Queue Length.
            If Queue is rising but CPU is dropping or stagnant, the system is CHOKING.
            
            Return your report in this JSON format:
            {
                "health_status": "Healthy" | "Congested" | "Collapsing",
                "estimated_capacity_per_server": float,
                "bottleneck_detected": bool,
                "reasoning": "short explanation"
            }
        """).strip()

    def get_prompt(self, obs):
        return f"""
        CURRENT METRICS:
        - Active Servers: {obs.active_servers}
        - Current Requests: {obs.current_requests}
        - CPU Utilization: {obs.cpu_utilization:.2f}
        - Queue Length: {obs.queue_length}
        - Booting Servers (pending): {sum(obs.booting_queue)}
        """