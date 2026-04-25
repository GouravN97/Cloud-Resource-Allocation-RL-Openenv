import numpy as np
from typing import List
from app.models import TaskConfig, WorkloadPattern


class CloudSimulator:
    """
    Pure math engine for the cloud autoscaling MDP.

    Handles:
      - Pre-generating the full noisy workload sequence (deterministic via seed)
      - Applying the 1-step scaling delay
      - Queue dynamics: q_{t+1} = max(0, q_t + d_{t+1} - capacity_t)
      - CPU utilization: u_{t+1} = min(1.5, d_{t+1} / capacity_t)
      - Computing latency proxy L_t and instability for the reward

    Does NOT know about HTTP, FastAPI, rewards, or OpenEnv conventions.
    That is env.py's job.
    """

    def __init__(self, config: TaskConfig):
        self.config = config

        # Internal MDP state
        self.current_step: int = 0
        self.active_servers: int = config.initial_servers
        self.queue_length: int = 0
        self.current_requests: int = config.base_load
        self.previous_requests: int = config.base_load

        # Long-Horizon Planning: 10-step startup delay
        self.booting_queue: List[int] = [0] * self.config.scaling_delay
        
        # History for the 'Oracle' agent to perform forecasting
        self.demand_history: List[int] = [config.base_load] * 10

        # Pre-generate the entire workload at construction time
        self.workload_sequence: List[int] = self._generate_workload()

    # ------------------------------------------------------------------
    # Workload Generation
    # ------------------------------------------------------------------

    def _generate_workload(self) -> List[int]:
        """
        Generate the full noisy workload sequence once at reset time.

        Pattern options:
          steady     → flat base load with light Gaussian noise
          diurnal    → sine wave oscillation (low → peak → low)
          flash_crowd → calm baseline with sudden massive spikes
        """
        rng = np.random.default_rng(seed=self.config.seed)
        steps = self.config.max_steps
        base = self.config.base_load
        std = self.config.noise_std
        max_cap = self.config.max_servers * self.config.capacity_per_server

        pattern = self.config.workload_pattern

        if pattern == WorkloadPattern.steady:
            # Flat baseline — agent should maintain minimal servers
            base_sequence = np.full(steps, base, dtype=float)

        elif pattern == WorkloadPattern.diurnal:
            # Sine wave: rises from base to base + 4×base then drops back
            # Agent must anticipate peaks due to 1-step delay
            t = np.arange(steps)
            amplitude = 4 * base
            base_sequence = base + amplitude * np.sin(2 * np.pi * t / steps)
            base_sequence = np.maximum(base_sequence, 0.0)

        elif pattern == WorkloadPattern.flash_crowd:
            # Structure: calm → spike → recovery → spike → calm
            # Spike loads hit 90% of max capacity to test limits
            calm1    = int(steps * 0.25)
            spike1   = int(steps * 0.15)
            recovery = int(steps * 0.20)
            spike2   = int(steps * 0.15)
            calm2    = steps - calm1 - spike1 - recovery - spike2

            spike_load = max_cap * 0.9         # 90% of total max capacity
            calm2_load = base * 1.2            # slightly elevated after chaos

            base_sequence = np.array(
                [base]        * calm1    +
                [spike_load]  * spike1   +
                [base]        * recovery +
                [spike_load * 0.85] * spike2 +  # second spike is smaller
                [calm2_load]  * calm2,
                dtype=float
            )

        else:
            base_sequence = np.full(steps, base, dtype=float)

        # Add reproducible Gaussian noise and floor at 0
        noise = rng.normal(0.0, std, size=steps)
        noisy = np.clip(base_sequence + noise, 0.0, None)
        return noisy.astype(int).tolist()

    # ------------------------------------------------------------------
    # MDP Step
    # ------------------------------------------------------------------

    def step(self, scale_change: int) -> dict:
        """
        Apply one MDP transition step with 10-step delay and Service Dependency.

        Transition equations:
          booting_queue[t] = scale_change
          s_{t+1} = clip(s_t + booting_queue.pop(0), 1, S_max)
          capacity_t = s_t * capacity_per_server * db_health_factor
        """
        prev_servers = self.active_servers

        # 1. Long-Horizon Logic: Process the boot queue
        # The change initiated 'scaling_delay' steps ago takes effect NOW.
        self.booting_queue.append(scale_change)
        applied_change = self.booting_queue.pop(0)
        
        new_servers = prev_servers + applied_change
        new_servers = int(np.clip(new_servers, 1, self.config.max_servers))

        # 2. Demand & World Modeling (Congestion Collapse - Non-Linear)
        d_next = self.workload_sequence[self.current_step]
        
        # Simulate Database Health Factor (Service Dependency)
        # Non-linear "Tipping Point": Capacity is stable until queue > 150, 
        # then it collapses rapidly. This is what the 'Scientist' must discover.
        db_load_factor = 1.0
        if self.queue_length > 150:
            # Quadratic collapse: as queue grows, backend performance tanks
            overage = self.queue_length - 150
            db_load_factor = max(0.2, 1.0 - (overage / 500.0)**2)

        # Capacity is based on servers AND the non-linear health of the dependency
        capacity = new_servers * self.config.capacity_per_server * db_load_factor

        # 3. Queue & Resource Dynamics
        overflow  = max(0, d_next - capacity)
        new_queue = self.queue_length + d_next - capacity
        new_queue = int(np.clip(new_queue, 0, self.config.max_queue_length))

        # CPU utilization: can spike up to 2.0 during database congestion (waiting on I/O)
        cpu_util = round(min(2.0, d_next / max(capacity, 1)), 4)

        latency = self.queue_length + overflow
        instability = abs(new_servers - prev_servers)

        # Commit state
        self.demand_history.append(d_next)
        if len(self.demand_history) > 10:
            self.demand_history.pop(0)
            
        self.previous_requests = self.current_requests
        self.current_requests  = d_next
        self.active_servers    = new_servers
        self.queue_length      = new_queue
        self.current_step     += 1

        done = self.current_step >= self.config.max_steps

        return {
            "current_step":      self.current_step,
            "current_requests":  self.current_requests,
            "previous_requests": self.previous_requests,
            "demand_history":    self.demand_history.copy(),
            "active_servers":    self.active_servers,
            "booting_queue":     self.booting_queue.copy(),
            "cpu_utilization":   cpu_util,
            "queue_length":      self.queue_length,
            "latency":           latency,
            "overflow":          overflow,
            "instability":       instability,
            "capacity":          capacity,
            "db_load_factor":    round(db_load_factor, 2),
            "done":              done,
        }

    # ------------------------------------------------------------------
    # Read-only State Snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Snapshot for env.state() - MUST match step() physics exactly."""
        # Calculate db_load_factor to ensure 'Scientist' sees consistent capacity
        db_load_factor = 1.0
        if self.queue_length > 150:
            overage = self.queue_length - 150
            db_load_factor = max(0.2, 1.0 - (overage / 500.0)**2)
            
        capacity = self.active_servers * self.config.capacity_per_server * db_load_factor
        
        return {
            "current_step":      self.current_step,
            "current_requests":  self.current_requests,
            "previous_requests": self.previous_requests,
            "demand_history":    self.demand_history.copy(),
            "active_servers":    self.active_servers,
            "booting_queue":     self.booting_queue.copy(),
            "cpu_utilization":   round(min(2.0, self.current_requests / max(capacity, 1)), 4),
            "queue_length":      self.queue_length,
        }
