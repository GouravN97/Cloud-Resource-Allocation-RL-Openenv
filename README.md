---
title: Cloud Resource Allocation RL OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# Cloud Resource Allocation RL (OpenEnv)

🔗 **[Hugging Face Space (Live Environment)](https://huggingface.co/spaces/gouravnayak97/MARL_cloud_resource_allocation)**
🔗 **[Read our Hackathon Writeup (Blog.md)](./Blog.md)**

An RL-driven autoscaling system for handling **flash-crowd traffic** in cloud services where wrong scaling decisions cause queue explosions, SLA violations, and unnecessary server spend.

## The Problem: Why Flash Crowds Are Hard

Real cloud systems cannot scale instantly. During a flash crowd:

- **Server boot delay** means new instances are requested now but become usable a few steps later.
- **Database bottlenecks** can amplify latency even when app servers are available.
- **Queue growth** can become nonlinear under sustained overload.
- **Naive policies fail both ways**: under-scaling causes SLA pain, over-scaling burns money.

This project models that reality as an MDP and trains policies that make robust autoscaling decisions under uncertainty.

## Environment Design

The environment state includes demand, active servers, booting servers, utilization, and queue length. At each step, the agent chooses a scaling action in `{-1, 0, +1}`.

### Queue Metric (`Q`) and Latency Pressure

- `Q = queue_length` is a first-class signal of system stress.
- Reward includes **queue-aware latency shaping**, not just raw latency:
  - `L_hat = min(1.0, (latency + 0.5 * queue_pressure) / L_max)`
  - `queue_pressure = queue_length / L_max`
- This discourages “riding the SLA boundary” while queue silently accumulates.

### Server Cost Modeling

- Cost term scales with active server count:
  - `C_hat = min(1.0, active_servers / S_max)`
- A **waste penalty** is added when utilization is too low (`< 20%`) to prevent over-provisioning.

### Why the Reward Is Hack-Proof (4 Penalties)

The final reward is the negative weighted sum of four normalized penalties:

1. **Latency + Queue Pressure (`L_hat`)**
  Stops policies from ignoring queue buildup until it is too late.
2. **Cost + Waste Penalty (`C_hat`)**
  Stops the trivial “always scale up” exploit.
3. **Instability (`I_hat`)**
  Penalizes jitter / scale thrashing.
4. **SLA Violation (`V_hat`)**
  Hard penalty when latency crosses SLA threshold.

Reward form:

`reward = -(w_L * L_hat + w_C * C_hat + w_I * I_hat + w_V * V_hat)`

## Results

### Baseline vs Trained Agent

- **Random baseline** (`random_baseline.py`): approximately `**-45.0`** total reward.
- **Trained agent**: expected to be significantly better (target zone often near `**-26.0`**, depending on run and seed).

> Replace the trained-agent value below with your final measured number from evaluation:
>
> - **Final comparison**: Random `~ -45.0` vs Trained `~ <ADD_FINAL_SCORE>`

## Required Plots (For README + Notebook)

Create a `plots/` folder and commit both plot images (PNG/JPG) so they render in GitHub.

### Plot 1: Training Loss Curve

- **What it shows:** The model actually learned during fine-tuning.
- **X-axis:** Training Step
- **Y-axis:** Training Loss
- **Judge signal:** Line starts higher and trends downward.
- Optional: include Weights & Biases run link in README.

Recommended file name: `plots/training_loss.png`

Figure 1 caption (example):  
**Figure 1:** Training loss decreases over fine-tuning steps, indicating stable convergence.

### Plot 2: Reward Comparison (Most Important)

- **What it shows:** The trained model performs better in the actual environment.
- **X-axis:** Episode Number (e.g., 1..10)
- **Y-axis:** Total Reward
- **Line 1:** Random baseline (**red/dotted**)
- **Line 2:** Trained model (**green/solid**)
- **Judge signal:** Trained line consistently above baseline (less negative reward).

Recommended file name: `plots/reward_comparison.png`

Figure 2 caption (example):  
**Figure 2:** Across evaluation episodes, the trained model consistently outperforms the random baseline in total reward.

## README Plot Embeds

Add these after committing plot files:

Training Loss Curve

*Figure 1: Training loss decreases over fine-tuning steps, indicating stable convergence.*

Reward Comparison

*Figure 2: Trained model achieves higher (less negative) total reward than the random baseline across episodes.*

## Exact Checklist for Person 3

📈 **Plot 1: The "We Actually Trained It" Graph**

- What it is: Training Loss Curve.
- X-Axis: Training Step
- Y-Axis: Training Loss
- What the Judges want to see: A line that starts high and goes down, proving the model actually learned the dataset.
- Pro-Tip: If using Weights & Biases (Wandb), include the link to the run in the README.

📊 **Plot 2: The "We Actually Solved the Problem" Graph (Most Important)**

- What it is: A comparison plot of Total Reward.
- X-Axis: Episode Number (e.g., 1 to 10)
- Y-Axis: Total Reward (e.g., 0 to -100)
- What the Judges want to see: Two lines on the SAME graph.
  - Line 1 (Red/Dotted): The Random Baseline (hovering near -45.0 or worse).
  - Line 2 (Green/Solid): The Trained Student Model (should be much higher, e.g., near -26.0).

🏷️ **Mandatory Formatting Rules**

- Labels: Both axes MUST be clearly labeled.
- Export Format: Save plots as `.png` or `.jpg`, commit them to the repo, and show them in the README.
- Captions: Every plot in the README needs a one-sentence caption explaining what it means.

## Repro: Baseline + Evaluation

1. Start the environment server.
2. Run baseline:
  - `python random_baseline.py`
3. Run trained agent evaluation episodes (Person 3 script/notebook).
4. Save both plots into `plots/`.
5. Update README trained score and plot links.

## Repository Structure (Current)

- `app/` - environment, simulator, models, grader
- `random_baseline.py` - random policy benchmark
- `run.py` - episode runner / data collection entrypoint
- `marl_agent.py` - lightweight policy/value network definitions