import numpy as np
from collections import defaultdict
import random

class DAPOAgent:
    def __init__(self, state_space, action_space, gamma=0.99, alpha_Q=0.1, alpha_pi=0.01, 
                 epsilon=0.1, G=5, epsilon_low=0.2, epsilon_high=0.5, buffer_size=32, 
                 beta_entropy=0.01, overlong_threshold=10, overlong_penalty=-0.1):
        """
        初始化 DAPO 代理，结合 Q-table 和 π_θ-table。
        """
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.alpha_Q = alpha_Q
        self.alpha_pi = alpha_pi
        self.epsilon = epsilon
        self.G = G
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.buffer_size = buffer_size
        self.beta_entropy = beta_entropy
        self.overlong_threshold = overlong_threshold
        self.overlong_penalty = overlong_penalty

        # 初始化 Q-table 和 π_θ-table
        self.Q = defaultdict(lambda: np.zeros(action_space))
        self.pi_theta = defaultdict(lambda: np.ones(action_space) / action_space)
        self.pi_theta_old = defaultdict(lambda: np.ones(action_space) / action_space)

        # 动态采样缓冲区
        self.buffer = []

    def choose_action(self, state):
        """
        使用混合策略选择动作。
        """
        if random.random() < self.epsilon:
            probs = self.pi_theta[state]
            # 确保概率有效
            if np.any(np.isnan(probs)) or np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0):
                probs = np.ones(self.action_space) / self.action_space
            return np.random.choice(self.action_space, p=probs)
        else:
            return np.argmax(self.Q[state])

    def sample_actions(self, state):
        """
        采样 G 个动作（带 ε-greedy 探索）。
        """
        actions = []
        for _ in range(self.G):
            if random.random() < self.epsilon:
                action = np.random.randint(self.action_space)
            else:
                probs = self.pi_theta_old[state]
                # 确保概率有效
                if np.any(np.isnan(probs)) or np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0):
                    probs = np.ones(self.action_space) / self.action_space
                action = np.random.choice(self.action_space, p=probs)
            actions.append(action)
        return actions

    def compute_advantage(self, rewards):
        """
        计算组内归一化的优势（论文 Equation 9）。
        """
        if len(rewards) == 0:
            return []
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1.0
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        return advantages

    def compute_entropy_gradient(self, probs):
        """
        计算熵正则化的梯度：∂H/∂π_θ(a|s)。
        """
        # 确保概率严格正且和为 1
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= np.sum(probs)
        entropy_grad = np.zeros_like(probs)
        for a in range(len(probs)):
            entropy_grad[a] = - (np.log(probs[a]) + 1)
        return entropy_grad

    def update(self, state, action, reward, next_state, sequence_length):
        """
        更新 Q-table 和 π_θ-table。
        """
        # 1. 采样 G 个动作
        sampled_actions = self.sample_actions(state)

        # 2. 计算采样动作的“奖励”（使用 Q-table 预估）
        rewards = []
        for a_i in sampled_actions:
            if a_i == action:
                r_i = reward + self.gamma * np.max(self.Q[next_state])
            else:
                r_i = self.Q[state][a_i]
            
            # 应用 Overlong Reward Shaping
            if sequence_length > self.overlong_threshold:
                r_i += self.overlong_penalty * (sequence_length - self.overlong_threshold)
            rewards.append(r_i)

        # 3. 动态采样：过滤掉奖励全相同的样本
        if len(set(rewards)) > 1:
            self.buffer.append((state, sampled_actions, rewards, next_state))
        else:
            # 如果奖励全相同，添加一个默认样本以避免缓冲区为空
            self.buffer.append((state, sampled_actions, rewards, next_state))

        # 4. 检查缓冲区大小
        if len(self.buffer) < self.buffer_size:
            return

        # 5. 更新 Q-table 和 π_θ-table
        for buf_state, buf_actions, buf_rewards, buf_next_state in self.buffer:
            # 更新 Q-table（仅针对实际动作）
            actual_action = action
            actual_reward = reward + self.gamma * np.max(self.Q[buf_next_state])
            self.Q[buf_state][actual_action] += self.alpha_Q * (
                actual_reward - self.Q[buf_state][actual_action]
            )

            # 计算优势
            advantages = self.compute_advantage(buf_rewards)

            # 更新 π_θ-table（针对所有采样的动作）
            self.pi_theta_old[buf_state] = self.pi_theta[buf_state].copy()
            for idx, a_i in enumerate(buf_actions):
                r_ratio = self.pi_theta[buf_state][a_i] / (self.pi_theta_old[buf_state][a_i] + 1e-10)
                clipped_ratio = np.clip(r_ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
                
                # DAPO 更新项
                update_term = min(r_ratio * advantages[idx], clipped_ratio * advantages[idx])

                # 熵正则化
                entropy_grad = self.compute_entropy_gradient(self.pi_theta[buf_state])
                update_term += self.beta_entropy * entropy_grad[a_i]

                # 更新 π_θ(a_i|s)，并确保不产生负值
                self.pi_theta[buf_state][a_i] += self.alpha_pi * update_term
                self.pi_theta[buf_state][a_i] = max(self.pi_theta[buf_state][a_i], 1e-10)

            # 标准化 π_θ(·|s)
            total = np.sum(self.pi_theta[buf_state])
            if total > 0:
                self.pi_theta[buf_state] /= total
            else:
                # 如果总和为 0，恢复为均匀分布
                self.pi_theta[buf_state] = np.ones(self.action_space) / self.action_space

        # 清空缓冲区
        self.buffer = []

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = 0
        self.goal = size - 1
        self.action_space = 4

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        next_state = self.state
        if action == 0 and self.state >= self.size:
            next_state = self.state - self.size
        elif action == 1 and self.state < self.size * (self.size - 1):
            next_state = self.state + self.size
        elif action == 2 and self.state % self.size > 0:
            next_state = self.state - 1
        elif action == 3 and self.state % self.size < self.size - 1:
            next_state = self.state + 1

        reward = -1
        done = False
        if next_state == self.goal:
            reward = 10
            done = True

        self.state = next_state
        return next_state, reward, done

def train_and_test():
    env = GridWorld(size=5)
    agent = DAPOAgent(state_space=25, action_space=4, gamma=0.99, alpha_Q=0.1, alpha_pi=0.01, 
                      epsilon=0.1, G=5, epsilon_low=0.2, epsilon_high=0.5, buffer_size=32, 
                      beta_entropy=0.01, overlong_threshold=10, overlong_penalty=-0.1)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        sequence_length = 0
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            sequence_length += 1

            agent.update(state, action, reward, next_state, sequence_length)

            total_reward += reward
            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    state = env.reset()
    done = False
    total_reward = 0
    print("\nTesting:")
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        print(f"State: {state}, Action: {action}, Reward: {reward}")
    print(f"Test Total Reward: {total_reward}")

if __name__ == "__main__":
    train_and_test()