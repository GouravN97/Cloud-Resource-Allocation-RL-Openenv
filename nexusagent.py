import json
import os
from openai import OpenAI
from scientist import Scientist
from oracle import Oracle
from planner import Planner

class NexusAgent:
    def __init__(self, api_key=None, base_url=None):
        # HARDENING: Use environment variable for flexible deployment (vLLM, HF, etc.)
        self.base_url = base_url or os.getenv("INFERENCE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("HF_TOKEN") or "dummy_token"
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        
        self.scientist_agent = Scientist()
        self.oracle_agent = Oracle()
        self.planner_agent = Planner()

        # Gating State
        self.last_requests = None
        self.last_cpu = None
        self.steps_since_scale = 999
        self.steps_since_audit = 0

        self.STABLE_RESPONSE = {
            "scientist": {
                "health_status": "Healthy",
                "estimated_capacity_per_server": 75.0,
                "bottleneck_detected": False,
                "reasoning": "Gated: metrics stable"
            },
            "oracle": {
                "pattern_detected": "steady",
                "trend": "stable", 
                "predicted_peak_demand_t10": 0,
                "confidence": 0.9,
                "reasoning": "Gated: no trend break"
            },
            "planner": {
                "action": 0,
                "priority": "Low",
                "internal_monologue": "Gated: system nominal",
                "final_rationale": "No action needed"
            }
        }

    def should_reason(self, obs) -> tuple[bool, str]:
        self.steps_since_audit += 1

        # Gate 1: Forced audit every 10 steps
        if self.steps_since_audit >= 10:
            self.steps_since_audit = 0
            return True, "periodic_audit"

        # Gate 2: Anomaly detected
        if obs.queue_length > 10:
            return True, "queue_anomaly"

        # Gate 3: Trend break >10%
        if self.last_requests:
            delta = abs(obs.current_requests - self.last_requests)
            if delta / max(self.last_requests, 1) > 0.10:
                return True, "trend_break"

        # Gate 4: Action lock after scale
        if self.steps_since_scale < 4:
            return False, "action_lock"

        # Gate 5: Stable metrics
        if self.last_cpu:
            cpu_delta = abs(obs.cpu_utilization - self.last_cpu)
            if cpu_delta < 0.05 and obs.queue_length == 0:
                return False, "stable"

        return True, "default"

    def update_tracking(self, obs, action):
        self.last_requests = obs.current_requests
        self.last_cpu = obs.cpu_utilization
        if action != 0:
            self.steps_since_scale = 0
        else:
            self.steps_since_scale += 1

    def _call_llm(self, system_prompt, user_prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def act(self, obs):
        should_call, reason = self.should_reason(obs)
        
        full_log = {
            "step": obs.current_step,
            "gate_triggered": reason,
            "llm_called": should_call
        }

        if not should_call:
            res = self.STABLE_RESPONSE
            action = res["planner"]["action"]
            full_log.update(res)
        else:
            # 1. Scientist (with fallback)
            sci_res = self._call_llm(self.scientist_agent.system_prompt, self.scientist_agent.get_prompt(obs)) or self.STABLE_RESPONSE["scientist"]
            
            # 2. Oracle (with fallback)
            ora_res = self._call_llm(self.oracle_agent.system_prompt, self.oracle_agent.get_prompt(obs)) or self.STABLE_RESPONSE["oracle"]
            
            # 3. Planner (with fallback)
            plan_res = self._call_llm(self.planner_agent.system_prompt, self.planner_agent.get_prompt(obs, sci_res, ora_res)) or self.STABLE_RESPONSE["planner"]
            
            # 🛠️ ROBUST EXTRACTION: Handle strings and clamp range
            try:
                action = int(plan_res.get("action", 0))
            except:
                action = 0
            
            action = max(-1, min(1, action))
            
            full_log.update({
                "scientist": sci_res,
                "oracle": ora_res,
                "planner": plan_res
            })

        self.update_tracking(obs, action)
        return action, full_log