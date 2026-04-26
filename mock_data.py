import json

# -------- helper functions --------

def action_to_text(a):
    if a == 1:
        return "scale up"
    elif a == -1:
        return "scale down"
    else:
        return "do nothing"


def build_prompt(obs):
    return f"""
You are a cloud autoscaling agent.

Current state:
Requests: {obs['current_requests']}
Queue: {obs['queue_length']}
Servers: {obs['active_servers']}
CPU: {obs['cpu_utilization']:.2f}

What should you do?
(-1 = scale down, 0 = do nothing, 1 = scale up)
"""


# -------- main pipeline --------

# load raw logs
with open("raw_logs.json", "r") as f:
    raw_data = json.load(f)

dataset = []

for step in raw_data:
    obs = step["observation"]
    action = step["action"]

    prompt = build_prompt(obs)

    chosen = action_to_text(action)

    # simple rejected logic (opposite action)
    if action == 1:
        rejected = "scale down"
    elif action == -1:
        rejected = "scale up"
    else:
        rejected = "scale up"

    dataset.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })


# save TRL dataset
with open("trl_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("✅ TRL dataset created: trl_dataset.json")