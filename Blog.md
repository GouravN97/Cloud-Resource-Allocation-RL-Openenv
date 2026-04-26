# Project Nexus: Taming Cloud Flash Crowds with Multi-Agent Reinforcement Learning

## 🌪️ The Problem we address: The "Flash Crowd" scenario that traditional cloud servers cannot handle...
Scaling cloud infrastructure isn't as simple as "if CPU > 80%, add a server." In the real world, cloud systems face **Flash Crowds**—sudden, massive spikes in traffic (e.g., a viral moment or a Black Friday sale). 

When a flash crowd hits, two deadly things happen:
1. **Boot Delays:** It takes time for new servers to turn on. If you wait until traffic is high to scale up, it's already too late. A queue forms, and users experience massive lag.
2. **Non-Linear Congestion Collapse:** As the queue grows, the backend database starts thrashing. Capacity doesn't just degrade smoothly; it collapses non-linearly.

Standard autoscalers fail here because they are reactive. To survive, a system needs to be **predictive and preventative**. We wanted to see if a Large Language Model (LLM) could learn to navigate this chaotic environment better than standard heuristics.

---

## 🏗️ Environment Innovation & Hackathon Themes
We built **Cloud-RL**, a custom OpenEnv-compliant environment designed to simulate real-world Site Reliability Engineering (SRE) nightmares. Our environment was built specifically to tackle two of the core Hackathon themes:

### 🤝 Theme #1: Multi-Agent Interactions
Instead of a standard single-agent approach, we designed a **Multi-Agent Teacher-Student framework** that forces cooperation and belief modeling in a partially observable setting:
*   🔬 **The Scientist:** Analyzes the system for "Congestion Collapse" (e.g., detecting when the queue is rising but CPU utilization is dropping, indicating thrashing).
*   🔮 **The Oracle:** Analyzes the history of incoming requests to forecast future flash crowds.
*   🧠 **The Planner:** Takes the reports from the Scientist and Oracle and decides whether to scale up, scale down, or hold, knowing that any action has a strict **10-step delay** before taking effect.

This architecture drives **emergent strategic behavior**. The Planner must learn to model the beliefs of the Oracle (e.g., ignoring a traffic spike if the Oracle has low confidence) and trust the Scientist (e.g., scaling up immediately if the Scientist detects backend collapse).

### 🌍 Theme #3: World Modeling (Professional Tasks)
Cloud-RL forces the LLM to do "real hard work" in a highly dynamic professional setting. The environment features a **persistent, non-linear world model**. The capacity of the servers is not static; if the queue crosses a certain threshold (150 requests), the database begins to thrash, and the capacity per server drops quadratically. The agent must maintain a consistent internal state and orchestrate multi-step workflows to prevent this collapse, pushing it far beyond shallow next-token reasoning. 


---

## 🎯 Designing a "Hack-Proof" Reward Signal
Reinforcement Learning agents are notorious for "Reward Hacking" (e.g., spinning up 1,000 servers just to guarantee zero latency, wasting thousands of dollars). 

To ensure the agent actually learned to balance the system, we designed a rigorous, 4-pillar reward function:
1. **Queue Pressure Penalty:** We don't just penalize SLA violations; we penalize the agent smoothly simply for letting a queue form.
2. **The Waste Penalty:** If the agent over-provisions servers and utilization drops below 20%, it receives a heavy cost penalty.
3. **Instability Penalty:** The agent is penalized for "thrashing" (rapidly scaling up and down).
4. **Hard SLA Violations:** A strict penalty if latency crosses the absolute threshold.

An agent cannot exploit this reward. The *only* way to get a high score is to accurately predict traffic, scale up just in time, and scale down immediately when the spike passes.

---

## 📊 Results & Training
We collected trajectories of the Expert LLM (Qwen 2.5) surviving extreme "Hard" tasks with 10-step boot delays. We then used Hugging Face TRL and Unsloth to distill this reasoning into a smaller, highly efficient Student model.

### Baseline vs. Trained Performance
*(Placeholder for Graph 1: A line graph comparing the Random Baseline score vs. the Trained Student score across 10 episodes)*
* **Figure 1:** The Trained Student model drastically outperforms the baseline, proving it learned to proactively manage the queue.

### Training Convergence
<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/ede6f559-cbde-476f-8d04-46656127fff4" />

* **Figure 2:** The loss curve demonstrates stable convergence during the TRL fine-tuning phase.

⚠️ Challenges We Faced (As a Team of First-Time Builders)

Building Project Nexus as a team of three without deep prior experience in large-scale RL systems forced us to confront several hard, practical challenges:

🧩 1. Designing a Realistic Yet Tractable Environment

One of the hardest problems wasn’t training the agent—it was defining the world correctly.
We had to strike a balance between:

Realism (non-linear congestion, delayed scaling, partial observability)
Simplicity (keeping the environment learnable within limited compute)

Early versions were either:

Too simple → trivial policies worked
Too complex → the agent failed to learn anything meaningful

Finding the “learning sweet spot” took multiple redesigns of the simulator.

🧠 2. Multi-Agent Coordination is Non-Trivial

Our Teacher-Agent setup (Scientist, Oracle, Planner) looked clean on paper, but in practice:

Agents produced conflicting signals
The Planner struggled to weigh trust vs uncertainty
Small prompt or reward changes caused unstable behavior

We underestimated how difficult belief modeling and coordination are, especially in partially observable systems.

⏱️ 3. Delayed Actions Made Learning Much Harder

The 10-step boot delay introduced a long credit assignment problem:

Actions taken now only affect the system much later
Naive policies failed because rewards felt “disconnected” from actions

This made debugging extremely difficult:

Was the policy bad?
Or was the feedback just delayed?

Understanding this required careful logging and visualization.

🧪 4. Reward Function Tuning Took Significant Iteration

Designing a “hack-proof” reward turned out to be an iterative process:

Early agents found loopholes (e.g., over-scaling or oscillating)
Small coefficient changes led to completely different behaviors

We learned that:

Reward design is not just a component—it is the problem in RL.

💻 5. Compute Constraints on Google Colab

Working within Colab’s limitations created constant friction:

Session timeouts interrupted long training runs
Limited GPU memory constrained model size and batch size
Data I/O bottlenecks slowed iteration cycles

This forced us to:

Optimize aggressively
Work with smaller, distilled models
Prioritize experimentation efficiency over scale
🔍 6. Debugging RL Systems is Inherently Difficult

Unlike traditional programs:

There is no clear “correct output”
Failures are often silent or gradual

We frequently faced:

Agents that looked like they were learning—but weren’t
Metrics that improved temporarily, then collapsed

This required building custom metrics, visualizations, and sanity checks just to understand what was happening.

🤝 7. Coordination as a Small Team

As a team of three, we also had to manage:

Parallel work on environment, agents, and training
Integration issues between independently built components
Rapid iteration under hackathon time pressure

We learned quickly that:

Clear interfaces and modular design were essential
Frequent syncs mattered more than perfect planning
💡 Key Takeaway

The biggest lesson wasn’t just about RL or LLMs—it was about systems thinking under constraints.
Every component (environment, reward, agents, infrastructure) is tightly coupled, and small decisions cascade into large effects.


## 🚀 Conclusion
Project Nexus proves that Multi-Agent LLM reasoning can be successfully distilled into a fast, reactive agent capable of managing highly non-linear cloud infrastructure crises. By moving away from reactive threshold rules to predictive RL, we can build cloud systems that never go down during a viral moment.
