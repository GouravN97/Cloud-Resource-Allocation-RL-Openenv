import requests
import random
import json
import os

ENV_API_URL = "http://127.0.0.1:8000"
TASK_ID = "hard"

def run_random_baseline():
    # 1. Reset
    res = requests.post(f"{ENV_API_URL}/reset", json={"task_id": TASK_ID})
    res.raise_for_status()
    data = res.json()
    session_id = data["session_id"]
    
    done = False
    step = 0
    total_reward = 0
    
    # 2. Loop
    while not done:
        step += 1
        action = random.choice([-1, 0, 1])
        
        step_res = requests.post(
            f"{ENV_API_URL}/step",
            json={"session_id": session_id, "action": {"scale_change": action}}
        )
        step_data = step_res.json()
        
        reward = step_data["reward"]
        done = step_data["done"]
        total_reward += reward
        
        obs = step_data["observation"]
        q = obs["queue_length"]
        s = obs["active_servers"]
        
        # print(f"  [S{step:02}] A: {action:2} | Q: {q:3} | S: {s:2} | R: {reward:.2f}")

    # 3. Grade
    grader_res = requests.get(f"{ENV_API_URL}/grader", params={"session_id": session_id})
    result = grader_res.json()
    
    # Use 'total_reward' instead of 'score'
    score = result.get("total_reward", 0.0)
    
    print(f"  [DONE] Score: {score:.3f}")
    
    return {
        "score": score,
        "total_reward": score,  # Map score to total_reward for the average calculation
        "sla": result.get("sla"),
        "cost": result.get("cost"),
        "stability": result.get("stability")
    }

if __name__ == "__main__":
    print(f" PROJECT NEXUS: Generating Stable Random Baseline (3 Runs)...")
    
    all_results = []
    for i in range(1, 4):
        print(f"  --- RUN {i}/3 ---")
        res = run_random_baseline()
        all_results.append(res)
    
    # Calculate Average
    avg_score = sum(r["score"] for r in all_results) / 3
    avg_reward = sum(r["total_reward"] for r in all_results) / 3
    
    final_output = {
        "task_id": TASK_ID,
        "avg_score": round(avg_score, 4),
        "avg_reward": round(avg_reward, 2),
        "runs": all_results
    }
    
    with open("baseline_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n" + "="*40)
    print(f"FINAL AVERAGE BASELINE SCORE: {avg_score:.4f}")
    print("="*40)
    print("Results saved to baseline_results.json")
