import os
import time
import requests
import json

from app.models import AutoscalerObservation
from nexusagent import NexusAgent  # your agent

# --- ENV CONFIG ---
ENV_API_URL = "http://127.0.0.1:8000"


def run_episode(task_id, episode_num):
    print(f"\n[EPISODE {episode_num:02d}] Starting...", flush=True)
    agent = NexusAgent()
    step_count = 0
    total_reward = 0.0
    step_logs = []

    try:
        # 1. RESET ENV
        reset_res = requests.post(f"{ENV_API_URL}/reset", json={"task_id": task_id}, timeout=15)
        reset_res.raise_for_status()
        data = reset_res.json()
        session_id = data["session_id"]
        obs = AutoscalerObservation(**data["observation"])

        done = False
        while not done:
            step_count += 1
            action_val, full_log = agent.act(obs)
            
            # Initial Telemetry
            full_log["reward"] = 0.0
            full_log["queue_length"] = obs.queue_length
            full_log["current_requests"] = obs.current_requests

            # ENV STEP
            step_res = requests.post(
                f"{ENV_API_URL}/step",
                json={"session_id": session_id, "action": {"scale_change": action_val}},
                timeout=5
            )
            step_data = step_res.json()
            next_obs = AutoscalerObservation(**step_data["observation"])
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)

            # Enrich log
            full_log["reward"] = reward
            full_log["next_queue"] = next_obs.queue_length
            step_logs.append(full_log)

            total_reward += reward
            gate_info = f"[{full_log['gate_triggered']}]" if not full_log['llm_called'] else "[LLM]"
            print(f"  [S{step_count:02}] {gate_info:12} Q={next_obs.queue_length:3} R={reward:.2f}")

            obs = next_obs

        # DISASTER 1: Save per-episode log (Recoverable)
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/episode_{episode_num:02d}.json"
        with open(log_path, "w") as f:
            json.dump(step_logs, f, indent=2)
        
        print(f"[EPISODE {episode_num:02d}] DONE. Reward: {total_reward:.2f}. Saved to {log_path}")
        return total_reward

    except Exception as e:
        print(f"[ERROR] Episode {episode_num} failed: {e}")
        return None


if __name__ == "__main__":
    TASK = "hard"
    EPISODES = 30
    
    print(f"🚀 PROJECT NEXUS: DATA COLLECTION STARTING ({EPISODES} episodes)")
    
    for ep in range(1, EPISODES + 1):
        # Skip if already exists (Simple resume logic)
        if os.path.exists(f"logs/episode_{ep:02d}.json"):
            print(f"  [SKIP] Episode {ep} already exists.")
            continue
            
        run_episode(TASK, ep)
    
    # DISASTER 5: Shutdown Reminder
    print("\n" + "="*50)
    print("⚠️  SHUT DOWN YOUR HF ENDPOINT NOW TO SAVE CREDITS")
    print("="*50)
    print(f"Go to: https://ui.endpoints.huggingface.co/")
    
    # DISASTER 5: Dataset Check
    if os.path.exists("trl_dataset.jsonl"):
        lines = open("trl_dataset.jsonl").readlines()
        print(f"\n📊 DATASET STATUS: {len(lines)} pairs collected.")
        if len(lines) < 20:
            print("⚠️  WARNING: Dataset too small (<20 pairs). Run 10 more episodes!")
    else:
        print("\n⚠️  WARNING: trl_dataset.jsonl not found. Run reflexion.py next.")