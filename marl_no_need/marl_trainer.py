from marl_agent import PolicyNetwork, ValueNetwork
from marl_utils import compute_advantages, process_state_agent1, process_state_agent2
from reflexion import Reflexion

class MARLTrainer:
    def __init__(self):
        self.agent1 = PolicyNetwork(input_dim=5)
        self.agent2 = PolicyNetwork(input_dim=5)
        self.best_reward = -float("inf")
        self.value_fn = ValueNetwork(input_dim=5)
        self.reflexion = Reflexion()

    def combine_actions(self, a1, a2):
        # agent2 decides whether to act
        if a2 <= 0:
            return 0
        return a1

    def run_episode(self, env, config):
        state = env.reset(config)

        states1 = []
        states2 = []
        actions1 = []
        actions2 = []
        rewards = []

        done = False

        while not done:
            s1 = process_state_agent1(state)
            s2 = process_state_agent2(state)

            a1, _, idx1 = self.agent1.sample_action(s1)
            a2, _, idx2 = self.agent2.sample_action(s2)

            final_action = a1 * (1 if a2 > 0 else 0)

            # apply reflexion bias
            if self.reflexion.scale_bias > 0:
                final_action = max(final_action, 1)
            elif self.reflexion.scale_bias < 0:
                final_action = min(final_action, -1)

            result = env.step(type("A", (), {"scale_change": final_action})())

            next_state = result.observation
            reward = result.reward
            done = result.done

            states1.append(s1)
            states2.append(s2)
            actions1.append(idx1)
            actions2.append(idx2)
            rewards.append(reward)

            state = next_state

        return states1, states2, actions1, actions2, rewards

    def train(self, env, config, episodes=100):
        for ep in range(episodes):
            states1, states2, a1s, a2s, rewards = self.run_episode(env, config)

            total_reward = sum(rewards)
            # only learn from good episodes
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                print(f"Episode {ep} new best: {total_reward:.3f}")

            elif total_reward < self.best_reward - 0.15:
                print(f"Episode {ep} skipped (bad): {total_reward:.3f}")
                continue

            advantages = compute_advantages(rewards)


            for t in range(len(states1)):
                G = advantages[t]   # discounted return

                V = self.value_fn.predict(states1[t])
                adv = G - V

                # update value function
                self.value_fn.update(states1[t], G)

                self.agent1.update(states1[t], a1s[t], adv)
                self.agent2.update(states2[t], a2s[t], adv)

            # Reflexion
            summary = self.reflexion.summarize_episode(states1, states2, rewards)
            prompt = self.reflexion.build_prompt(summary)
            # use mock LLM for now
            response = self.reflexion.mock_llm(summary)
            self.reflexion.update_bias(response)
            print(f"[REFLEXION] {response} | bias={self.reflexion.scale_bias:.2f}")


            print(f"Episode {ep} | Total Reward: {sum(rewards):.3f}")