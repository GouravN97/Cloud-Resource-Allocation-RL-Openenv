import os
import time
import requests
import json

from app.models import AutoscalerObservation
from nexusagent import NexusAgent  # your agent

# --- ENV CONFIG ---
ENV_API_URL = "http://127.0.0.1:8000"


def run_task(task_id):
    print(f"[START] task={task_id}", flush=True)

    agent = NexusAgent()

    step_count = 0
    total_reward = 0.0
    session_id = None
    step_logs = []  # Teacher-Student: Capture reasoning for training

    try:
        # 1. RESET ENV
        reset_res = requests.post(
            f"{ENV_API_URL}/reset",
            json={"task_id": task_id},
            timeout=15
        )
        reset_res.raise_for_status()

        data = reset_res.json()
        session_id = data["session_id"]

        obs_data = data["observation"]
        # FIXED: Removed obs_data.setdefault("previous_requests", 0) 
        # State should come correctly from environment/main.py
        obs = AutoscalerObservation(**obs_data)

        done = False

        while not done:
            step_count += 1

            # -------------------------
            # AGENT ACTION (with reasoning log)
            # -------------------------
            action_val, full_log = agent.act(obs)
            
            # TEACHER-STUDENT: Enrich log with telemetry for the Critic
            full_log["reward"] = 0.0 # Placeholder, will be updated after step
            full_log["queue_length"] = obs.queue_length
            full_log["current_requests"] = obs.current_requests
            step_logs.append(full_log)

            # -------------------------
            # ENV STEP
            # -------------------------
            step_res = requests.post(
                f"{ENV_API_URL}/step",
                json={
                    "session_id": session_id,
                    "action": {"scale_change": action_val}
                },
                timeout=5
            )

            if step_res.status_code != 200:
                print(f"[ERROR] Step failed: {step_res.status_code}", flush=True)
                break

            step_data = step_res.json()

            next_obs = AutoscalerObservation(**step_data["observation"])
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)

            # Update the log with the actual outcome
            step_logs[-1]["reward"] = reward
            step_logs[-1]["next_queue"] = next_obs.queue_length

            total_reward += reward

            # -------------------------
            # DEBUG PRINT
            # -------------------------
            gate_info = f"[{full_log['gate_triggered']}]" if not full_log['llm_called'] else "[LLM]"
            print(
                f"[STEP {step_count:02}] {gate_info:18} "
                f"Req={next_obs.current_requests:4} "
                f"Queue={next_obs.queue_length:3} "
                f"Action={action_val:2} "
                f"Rew={reward:.2f}",
                flush=True
            )

            obs = next_obs

        # -------------------------
        # PERSIST LOGS FOR REFLEXION
        # -------------------------
        with open("communication_log.json", "w") as f:
            json.dump(step_logs, f, indent=2)
        print(f"\n[DONE] Saved {len(step_logs)} reasoning steps to communication_log.json")

        # -------------------------
        # GRADER
        # -------------------------
        if session_id:
            grader_res = requests.get(
                f"{ENV_API_URL}/grader",
                params={"session_id": session_id},
                timeout=5
            )
            if grader_res.status_code == 200:
                result = grader_res.json()
                score = result.get("score", 0.0)
                print(f"[END] task={task_id} score={score:.3f} steps={step_count}")

    except Exception as e:
        print(f"[ERROR] Task failed: {e}", flush=True)


if __name__ == "__main__":
    print(f"# Connecting to ENV at {ENV_API_URL}", flush=True)

    # wait for env
    for i in range(10):
        try:
            requests.get(f"{ENV_API_URL}/health", timeout=2)
            print("# ENV is live", flush=True)
            break
        except:
            print("# waiting for env...", flush=True)
            time.sleep(2)

    # run HARD task for Teacher-Student data collection
    run_task("hard")