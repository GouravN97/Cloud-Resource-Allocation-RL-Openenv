import os
import time
import requests

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
        obs_data.setdefault("previous_requests", 0)
        obs = AutoscalerObservation(**obs_data)

        done = False

        while not done:
            step_count += 1

            # -------------------------
            # AGENT ACTION
            # -------------------------
            action_val = agent.act(obs)

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

            total_reward += reward

            # -------------------------
            # LEARNING
            # -------------------------
            agent.learn(next_obs, reward)

            # -------------------------
            # DEBUG PRINT
            # -------------------------
            print(
                f"[STEP {step_count}] "
                f"Req={next_obs.current_requests} "
                f"Queue={next_obs.queue_length} "
                f"Servers={next_obs.active_servers} "
                f"Action={action_val} "
                f"Reward={reward:.3f}",
                flush=True
            )

            obs = next_obs

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

    # run MEDIUM task
    run_task("medium")