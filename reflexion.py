import json
import os
import textwrap
import shutil
from openai import OpenAI

class ReflexionCritic:
    """
    Teacher-Student Critic: Analyzes full trajectories to generate training data.
    """
    def __init__(self, api_key=None, base_url=None):
        self.base_url = base_url or os.getenv("INFERENCE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("HF_TOKEN") or "dummy_token"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = os.getenv("CRITIC_MODEL_NAME", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))

    def analyze_all_logs(self):
        """Disaster Recovery: Process all episode logs in the logs/ folder."""
        log_files = sorted([f for f in os.listdir("logs") if f.startswith("episode_")])
        
        for log_file in log_files:
            ep_id = log_file.split("_")[1].split(".")[0]
            print(f"\n[REFLEXION] Processing Episode {ep_id}...")
            
            with open(os.path.join("logs", log_file), "r") as f:
                steps = json.load(f)
            
            self.analyze_trajectory(steps, ep_id)

    def analyze_trajectory(self, steps, episode_id):
        # 🛠️ FAILURE DETECTION: Filter for LLM-reasoned steps with bad outcomes
        failures = [
            s for s in steps 
            if s.get("llm_called", False) 
            and (s.get("reward", 0) < -0.5 or s.get("next_queue", 0) > 100)
        ]
        
        # DISASTER 2: Minimum Dataset Guarantee
        # If no 'failures' found, take the 3 lowest-reward steps to ensure the model still learns.
        if len(failures) == 0:
            print(f"  [INFO] No critical failures found. Taking 3 lowest-reward steps.")
            failures = sorted(
                [s for s in steps if s.get("llm_called", False)], 
                key=lambda s: s.get("reward", 0)
            )[:3]

        if not failures:
            print("  [SKIP] No reasoning steps found in this episode.")
            return

        print(f"  [ANALYSIS] Critiquing {len(failures)} steps...")
        
        preference_pairs = []
        for fail in failures:
            pair = self._generate_critique(fail)
            if pair:
                preference_pairs.append(pair)
        
        self._save_dataset(preference_pairs)
        
        # DISASTER 2: Per-episode backup
        os.makedirs("backups", exist_ok=True)
        shutil.copy("trl_dataset.jsonl", f"backups/dataset_ep{episode_id}.jsonl")
        
        # Generate rules
        self.generate_governance_rules(preference_pairs)

    def generate_governance_rules(self, preference_pairs: list) -> list:
        rules = []
        for pair in preference_pairs:
            lesson = pair.get("lesson_learned")
            if lesson:
                rules.append(lesson)
        
        with open("governance_rules.json", "w") as f:
            json.dump(rules, f, indent=2)
        return rules

    def _generate_critique(self, fail_step):
        system_prompt = textwrap.dedent("""
            You are the 'Nexus Post-Mortem Critic'. Analyze and provide CORRECT reasoning.
            Return in JSON: {"prompt": "...", "chosen_reasoning": "...", "rejected_reasoning": "...", "lesson_learned": "..."}
        """).strip()

        user_prompt = f"FAILURE DATA:\n{json.dumps(fail_step, indent=2)}"

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
            print(f"  [ERROR] Critic LLM failed: {e}")
            return None

    def _save_dataset(self, pairs):
        with open("trl_dataset.jsonl", "a") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

if __name__ == "__main__":
    critic = ReflexionCritic()
    critic.analyze_all_logs()