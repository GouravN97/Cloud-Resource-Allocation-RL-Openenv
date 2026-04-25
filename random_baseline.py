import requests
import random
import time

ENV_API_URL = "http://127.0.0.1:8000"
TASK_ID = "hard"

def run_random_baseline():
    print(f"🚀 Starting RANDOM BASELINE on task: {TASK_ID}")
    
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
        # Pick a random action: -1 (Scale Down), 0 (Hold), 1 (Scale Up)
        action = random.choice([-1, 0, 1])
        
        step_res = requests.post(
            f"{ENV_API_URL}/step",
            json={"session_id": session_id, "action": {"scale_change": action}}
        )
        step_data = step_res.json()
        
        reward = step_data["reward"]
        done = step_data["done"]
        total_reward += reward
        
        q = step_data["observation"]["queue_length"]
        s = step_data["observation"]["active_servers"]
        
        print(f"  [STEP {step:02}] Action: {action:2} | Queue: {q:3} | Servers: {s:2} | Reward: {reward:.2f}")

    # 3. Grade
    grader_res = requests.get(f"{ENV_API_URL}/grader", params={"session_id": session_id})
    score = grader_res.json()["score"]
    
    print("\n" + "="*30)
    print(f"RANDOM BASELINE SCORE: {score:.3f}")
    print(f"TOTAL REWARD: {total_reward:.2f}")
    print("="*30)
    print("Save this score! This is your 'Before' comparison.")

if __name__ == "__main__":
    run_random_baseline()
