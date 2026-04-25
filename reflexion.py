import json
import os
import textwrap
from openai import OpenAI

class ReflexionCritic:
    """
    Teacher-Student Critic: Analyzes full trajectories to generate training data.
    Produces:
    1. Governance Rules (for the next episode prompt)
    2. Preference Pairs (for DPO/TRL training)
    """
    def __init__(self, api_key=None, base_url=None):
        self.base_url = base_url or os.getenv("INFERENCE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("HF_TOKEN") or "dummy_token"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = os.getenv("CRITIC_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    def analyze_trajectory(self, log_path="communication_log.json"):
        if not os.path.exists(log_path):
            print(f"[ERROR] No log found at {log_path}")
            return None

        with open(log_path, "r") as f:
            steps = json.load(f)

        # 🛠️ BUG FIX: Use top-level reward/queue keys and filter for LLM-reasoned steps
        failures = [
            s for s in steps 
            if s.get("llm_called", False) 
            and (s.get("reward", 0) < -0.5 or s.get("next_queue", 0) > 100)
        ]
        
        if not failures:
            print("[REFLEXION] No major failures detected in this trajectory.")
            return [], []

        print(f"[REFLEXION] Found {len(failures)} critical failure points for analysis.")
        
        preference_pairs = []
        for fail in failures[:5]:
            pair = self._generate_critique(fail)
            if pair:
                preference_pairs.append(pair)
        
        self._save_dataset(preference_pairs)
        
        # 📜 NEW: Generate governance rules from lessons learned
        rules = self.generate_governance_rules(preference_pairs)
        
        return preference_pairs, rules

    def generate_governance_rules(self, preference_pairs: list) -> list:
        """Extract lessons from preference pairs into injectable rules for the next episode."""
        rules = []
        for pair in preference_pairs:
            lesson = pair.get("lesson_learned")
            if lesson:
                rules.append(lesson)
        
        # Save for injection into next episode
        with open("governance_rules.json", "w") as f:
            json.dump(rules, f, indent=2)
        
        print(f"[REFLEXION] Generated {len(rules)} governance rules")
        return rules

    def _generate_critique(self, fail_step):
        # ... (rest of the method remains the same)
        """
        Calls a high-end LLM to generate the 'Correct' reasoning for a failure.
        """
        system_prompt = textwrap.dedent("""
            You are the 'Nexus Post-Mortem Critic'. 
            Your job is to analyze a failed scaling decision and provide the CORRECT reasoning.
            
            SCENARIO:
            The agent decided NOT to scale up, and the system collapsed shortly after.
            
            TASK:
            1. Identify which sub-agent failed (Scientist or Oracle).
            2. Write a 'Chosen' (Correct) reasoning trace.
            3. Write a 'Rejected' (The agent's actual bad) reasoning trace.
            
            Return in JSON format for DPO:
            {
                "prompt": "Full observation context...",
                "chosen_reasoning": "The correct expert analysis...",
                "rejected_reasoning": "The actual bad analysis...",
                "lesson_learned": "Short rule for the future"
            }
        """).strip()

        user_prompt = f"""
        FAILURE DATA:
        Step: {fail_step['step']}
        Observation: {fail_step.get('observation', 'N/A')}
        Scientist Thought: {fail_step.get('scientist', {}).get('reasoning', 'N/A')}
        Oracle Prediction: {fail_step.get('oracle', {}).get('reasoning', 'N/A')}
        Planner Action: {fail_step.get('planner', {}).get('action', 0)}
        """

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
            print(f"Critic LLM Error: {e}")
            return None

    def _save_dataset(self, pairs, path="trl_dataset.jsonl"):
        """Append the preference pairs to a TRL-compatible JSONL file."""
        with open(path, "a") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[REFLEXION] Appended {len(pairs)} pairs to {path}")

if __name__ == "__main__":
    critic = ReflexionCritic()
    critic.analyze_trajectory()