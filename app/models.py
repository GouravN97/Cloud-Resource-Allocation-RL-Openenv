from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List
from enum import Enum

class WorkloadPattern(str, Enum):
    steady = "steady"
    diurnal = "diurnal"
    flash_crowd = "flash_crowd"

class AutoscalerAction(BaseModel):
    scale_change: int = Field(
        ..., 
        ge=-1,
        le=1,
        description="Action to take: -1 (scale down), 0 (do nothing), or 1 (scale up)",
        json_schema_extra={"example": 1}
    )

class AutoscalerObservation(BaseModel):
    current_step: int = Field(..., ge=0, description="Current timestep")
    current_requests: int = Field(..., ge=0, description="Demand at current timestep")
    previous_requests: int = Field(..., ge=0, description="Demand at previous timestep")
    demand_history: List[int] = Field(..., description="Last 10 demand values")
    active_servers: int = Field(..., ge=1, description="Supply of active servers")
    booting_queue: List[int] = Field(..., description="The raw startup pipeline")
    cpu_utilization: float = Field(..., ge=0.0, le=2.0, description="Load per server")
    queue_length: int = Field(..., ge=0, description="Backlog of requests")

class StepResult(BaseModel):
    observation: AutoscalerObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class GraderBounds(BaseModel):
    L_max: float = Field(..., gt=0.0)
    S_max: float = Field(..., gt=0.0)
    I_max: float = Field(..., gt=0.0)
    L_SLA: float = Field(..., gt=0.0)

class GraderWeights(BaseModel):
    w_L: float = Field(..., ge=0.0, le=1.0)
    w_C: float = Field(..., ge=0.0, le=1.0)
    w_I: float = Field(..., ge=0.0, le=1.0)
    w_V: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_sum(self):
        total = self.w_L + self.w_C + self.w_I + self.w_V
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self

class TaskConfig(BaseModel):
    task_id: str
    description: str
    max_steps: int = Field(..., gt=0)
    seed: int
    workload_pattern: WorkloadPattern
    base_load: int = Field(..., ge=0)
    noise_std: float = Field(..., ge=0.0, le=200.0)
    max_servers: int = Field(..., gt=0)
    capacity_per_server: int = Field(..., gt=0)
    initial_servers: int = Field(default=1, gt=0)
    scaling_delay: int = Field(default=10, ge=0)
    max_queue_length: int = Field(default=1000, gt=0)
    grader_bounds: GraderBounds
    grader_weights: GraderWeights

    @model_validator(mode='after')
    def check_servers(self):
        if self.initial_servers > self.max_servers:
            raise ValueError("initial_servers cannot exceed max_servers")
        return self
