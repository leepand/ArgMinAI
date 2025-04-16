from diskcache import Cache
import numpy as np
import random
import pickle

class DAPOAgent:
    def __init__(self, gamma=0.99, alpha_Q=0.1, alpha_pi=0.01, 
                 epsilon=0.1, G=5, epsilon_low=0.2, epsilon_high=0.5, buffer_size=3, 
                 beta_entropy=0.01, overlong_threshold=10, overlong_penalty=-0.1, 
                 model_cache=None, default_action=None):
        """
        初始化 DAPO 代理，添加 default_action 参数。
        """
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
        self.default_action = default_action
        self._model_cache = model_cache if model_cache is not None else Cache("dapo")

    def _is_serializable(self, obj):
        """检查对象是否可序列化"""
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, TypeError):
            return False

    def _serialize_action(self, action):
        """将动作转换为可序列化的格式"""
        if self._is_serializable(action):
            return action
        return str(action)

    def _deserialize_action(self, action):
        """将序列化的动作还原"""
        return action

    def add_action(self, action, actions, action_to_index, index_to_action, next_action_index):
        """
        添加新动作，验证可序列化性。
        """
        serialized_action = self._serialize_action(action)
        if serialized_action not in actions:
            if not self._is_serializable(serialized_action):
                raise ValueError(f"Action {action} cannot be serialized for caching")
            actions.add(serialized_action)
            action_to_index[serialized_action] = next_action_index
            index_to_action[next_action_index] = serialized_action
            next_action_index += 1
        return actions, action_to_index, index_to_action, next_action_index

    def get_action_index(self, action, actions, action_to_index, index_to_action, next_action_index):
        """
        获取动作索引，使用序列化格式。
        """
        serialized_action = self._serialize_action(action)
        if serialized_action not in actions:
            actions, action_to_index, index_to_action, next_action_index = self.add_action(
                serialized_action, actions, action_to_index, index_to_action, next_action_index
            )
        return action_to_index[serialized_action], actions, action_to_index, index_to_action, next_action_index
    
    def get_action_space_size(self, actions):
        """
        获取当前动作空间大小。
        """
        return len(actions)

    def _resize_state_arrays(self, state, old_size, new_size, Q, pi_theta, pi_theta_old):
        """
        调整状态相关数组的大小以匹配新的动作空间大小。
        
        参数：
        - state: 要调整的状态
        - old_size: 原动作空间大小
        - new_size: 新动作空间大小
        - Q, pi_theta, pi_theta_old: 要调整的表格
        """
        if state in Q:
            if len(Q[state]) < new_size:
                new_Q = np.zeros(new_size)
                new_Q[:old_size] = Q[state]
                Q[state] = new_Q
        else:
            Q[state] = np.zeros(new_size)

        if state in pi_theta:
            if len(pi_theta[state]) < new_size:
                new_pi = np.ones(new_size) / new_size
                new_pi[:old_size] = pi_theta[state] * (old_size / new_size)  # 重新归一化
                pi_theta[state] = new_pi / np.sum(new_pi)
        else:
            pi_theta[state] = np.ones(new_size) / new_size

        if state in pi_theta_old:
            if len(pi_theta_old[state]) < new_size:
                new_pi_old = np.ones(new_size) / new_size
                new_pi_old[:old_size] = pi_theta_old[state] * (old_size / new_size)
                pi_theta_old[state] = new_pi_old / np.sum(new_pi_old)
        else:
            pi_theta_old[state] = pi_theta[state].copy()

    def choose_action(self, state, default_action=None, model_id=None):
        """
        使用混合策略选择动作，处理空动作空间。
        """
        (
            Q, pi_theta, pi_theta_old, actions, action_to_index, 
            index_to_action, next_action_index, buffer
        ) = self.load_from_cache(model_id=model_id)

        if Q is None: Q = {}
        if pi_theta is None: pi_theta = {}
        if pi_theta_old is None: pi_theta_old = {}
        if actions is None:
            actions = set()
            action_to_index = {}
            index_to_action = {}
            next_action_index = 0
        if buffer is None: buffer = []

        action_space_size = self.get_action_space_size(actions)
        effective_default = default_action if default_action is not None else self.default_action
        if action_space_size == 0:
            if effective_default is None:
                raise ValueError("Action space is empty and no default action provided")
            actions, action_to_index, index_to_action, next_action_index = self.add_action(
                effective_default, actions, action_to_index, index_to_action, next_action_index
            )
            self.save_to_cache(model_id=model_id, model_data={
                "Q": Q, "pi_theta": pi_theta, "pi_theta_old": pi_theta_old,
                "actions": actions, "action_to_index": action_to_index,
                "index_to_action": index_to_action, "next_action_index": next_action_index,
                "buffer": buffer
            })
            return self._deserialize_action(effective_default)

        if state not in Q:
            Q[state] = np.zeros(action_space_size)
            pi_theta[state] = np.ones(action_space_size) / action_space_size
            pi_theta_old[state] = pi_theta[state].copy()

        probs = pi_theta[state]
        if np.any(np.isnan(probs)) or np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0):
            probs = np.ones(action_space_size) / action_space_size
            pi_theta[state] = probs

        if random.random() < self.epsilon:
            action_index = np.random.choice(len(probs), p=probs)
        else:
            action_index = np.argmax(Q[state]) if len(Q[state]) > 0 else np.random.randint(action_space_size)

        action = index_to_action[action_index]
        self.save_to_cache(model_id=model_id, model_data={
            "Q": Q, "pi_theta": pi_theta, "pi_theta_old": pi_theta_old,
            "actions": actions, "action_to_index": action_to_index,
            "index_to_action": index_to_action, "next_action_index": next_action_index,
            "buffer": buffer
        })
        return self._deserialize_action(action)
    
    def sample_actions(self, state, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index):
        """
        采样 G 个动作（带 ε-greedy 探索）。
        """
        action_space_size = self.get_action_space_size(actions)
        if action_space_size == 0:
            default_action = self.default_action if self.default_action is not None else 0
            actions, action_to_index, index_to_action, next_action_index = self.add_action(
                default_action, actions, action_to_index, index_to_action, next_action_index
            )
            return [default_action] * self.G, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index

        if state not in pi_theta:
            pi_theta[state] = np.ones(action_space_size) / action_space_size
            pi_theta_old[state] = pi_theta[state].copy()

        sampled_actions = []
        for _ in range(self.G):
            if random.random() < self.epsilon:
                action_index = np.random.randint(action_space_size)
            else:
                probs = pi_theta_old[state]
                if np.any(np.isnan(probs)) or np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0):
                    probs = np.ones(action_space_size) / action_space_size
                action_index = np.random.choice(len(probs), p=probs)
            sampled_actions.append(index_to_action[action_index])
        return sampled_actions, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index

    def compute_advantage(self, rewards):
        """
        计算优势值，处理空列表和零标准差。
        """
        if len(rewards) == 0:
            return []
        rewards = np.array(rewards)
        if np.all(rewards == rewards[0]):
            return np.zeros_like(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        if std_reward < 1e-10:
            std_reward = 1.0
        advantages = (rewards - mean_reward) / std_reward
        return advantages.tolist()

    def compute_entropy_gradient(self, probs):
        """
        计算熵正则化的梯度。
        """
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= np.sum(probs)
        entropy_grad = np.zeros_like(probs)
        for a in range(len(probs)):
            entropy_grad[a] = - (np.log(probs[a]) + 1)
        return entropy_grad

    def update(self, state, action, reward, next_state, sequence_length, model_id):
        """
        更新 Q-table 和 π_θ-table，确保新动作正确扩展数据结构。
        """
        (
            Q, pi_theta, pi_theta_old, actions, action_to_index, 
            index_to_action, next_action_index, buffer
        ) = self.load_from_cache(model_id=model_id)

        if Q is None: Q = {}
        if pi_theta is None: pi_theta = {}
        if pi_theta_old is None: pi_theta_old = {}
        if actions is None:
            actions = set()
            action_to_index = {}
            index_to_action = {}
            next_action_index = 0
        if buffer is None: buffer = []

        # 获取当前动作的索引
        old_action_space_size = self.get_action_space_size(actions)
        action_index, actions, action_to_index, index_to_action, next_action_index = self.get_action_index(
            action, actions, action_to_index, index_to_action, next_action_index
        )
        new_action_space_size = self.get_action_space_size(actions)

        # 如果动作空间大小增加，调整 Q, pi_theta, pi_theta_old
        if new_action_space_size > old_action_space_size:
            for s in list(Q.keys()) + [state, next_state]:
                self._resize_state_arrays(s, old_action_space_size, new_action_space_size, Q, pi_theta, pi_theta_old)

        # 确保当前状态和下一状态已初始化
        if state not in Q:
            Q[state] = np.zeros(new_action_space_size)
            pi_theta[state] = np.ones(new_action_space_size) / new_action_space_size
            pi_theta_old[state] = pi_theta[state].copy()
        if next_state not in Q:
            Q[next_state] = np.zeros(new_action_space_size)
            pi_theta[next_state] = np.ones(new_action_space_size) / new_action_space_size
            pi_theta_old[next_state] = pi_theta[next_state].copy()

        # 采样动作
        sampled_actions, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index = self.sample_actions(
            state, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index
        )
        sampled_action_indices = [
            self.get_action_index(a, actions, action_to_index, index_to_action, next_action_index)[0]
            for a in sampled_actions
        ]

        # 确保当前动作包含在采样动作中
        serialized_action = self._serialize_action(action)
        if action_index not in sampled_action_indices:
            sampled_actions.append(serialized_action)
            sampled_action_indices.append(action_index)
            
        # 更新 Q-table（仅为当前交互）
        actual_reward = reward + self.gamma * np.max(Q[next_state])
        Q[state][action_index] += self.alpha_Q * (
            actual_reward - Q[state][action_index]
        )

        # 计算奖励
        rewards = []
        for a_i in sampled_action_indices:
            r_i = Q[state][a_i] if len(Q[state]) > a_i else 0.0
            #if a_i == action_index:
            #    # r_i = reward + self.gamma * np.max(Q[next_state])
            #    Q_old = Q[state][a_i] if len(Q[state]) > a_i else 0.0
            #3    r_i_new = reward + self.gamma * np.max(Q[next_state])
            #    r_i 
            #else:
            #    r_i = Q[state][a_i] if len(Q[state]) > a_i else 0.0
            if sequence_length > self.overlong_threshold:
                r_i += self.overlong_penalty * (sequence_length - self.overlong_threshold)
            rewards.append(r_i)

        # 更新缓冲区
        if len(set(rewards)) >= 1:
            buffer.append((state, sampled_action_indices, rewards, next_state))
        if len(buffer) < self.buffer_size:
            self.save_to_cache(model_id=model_id, model_data={
                "Q": Q, "pi_theta": pi_theta, "pi_theta_old": pi_theta_old,
                "actions": actions, "action_to_index": action_to_index,
                "index_to_action": index_to_action, "next_action_index": next_action_index,
                "buffer": buffer
            })
            return Q, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index, buffer

        # 更新 Q-table 和 π_θ-table
        for buf_state, buf_action_indices, buf_rewards, buf_next_state in buffer:
            #actual_reward = reward + self.gamma * np.max(Q[buf_next_state])
            #Q[buf_state][action_index] += self.alpha_Q * (
            #    actual_reward - Q[buf_state][action_index]
            #)

            advantages = self.compute_advantage(buf_rewards)
            pi_theta_old[buf_state] = pi_theta[buf_state].copy()
            for idx, a_i in enumerate(buf_action_indices):
                r_ratio = pi_theta[buf_state][a_i] / (pi_theta_old[buf_state][a_i] + 1e-10)
                clipped_ratio = np.clip(r_ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
                update_term = min(r_ratio * advantages[idx], clipped_ratio * advantages[idx])
                entropy_grad = self.compute_entropy_gradient(pi_theta[buf_state])
                update_term += self.beta_entropy * entropy_grad[a_i]
                pi_theta[buf_state][a_i] = max(pi_theta[buf_state][a_i] + self.alpha_pi * update_term, 1e-10)

            # 归一化策略分布
            total = np.sum(pi_theta[buf_state])
            if total > 0:
                pi_theta[buf_state] /= total
            else:
                pi_theta[buf_state] = np.ones(new_action_space_size) / new_action_space_size

        buffer = []
        self.save_to_cache(model_id=model_id, model_data={
            "Q": Q, "pi_theta": pi_theta, "pi_theta_old": pi_theta_old,
            "actions": actions, "action_to_index": action_to_index,
            "index_to_action": index_to_action, "next_action_index": next_action_index,
            "buffer": buffer
        })
        return Q, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index, buffer

    def save_to_cache(self, model_id="dapo", model_data={}):
        """
        将代理的状态保存到缓存。
        """
        Q = model_data.get("Q", {})
        pi_theta = model_data.get("pi_theta", {})
        pi_theta_old = model_data.get("pi_theta_old", {})
        actions = model_data.get("actions", set())
        action_to_index = model_data.get("action_to_index", {})
        index_to_action = model_data.get("index_to_action", {})
        next_action_index = model_data.get("next_action_index", 0)
        buffer = model_data.get("buffer", [])

        Q = Q if Q is not None else {}
        pi_theta = pi_theta if pi_theta is not None else {}
        pi_theta_old = pi_theta_old if pi_theta_old is not None else {}

        Q_serializable = {state: Q[state].tolist() for state in Q}
        pi_theta_serializable = {state: pi_theta[state].tolist() for state in pi_theta}
        pi_theta_old_serializable = {state: pi_theta_old[state].tolist() for state in pi_theta_old}

        actions_data = {
            "actions": list(actions),
            "action_to_index": action_to_index,
            "index_to_action": index_to_action,
            "next_action_index": next_action_index
        }

        params = {
            "Q": Q_serializable,
            "pi_theta": pi_theta_serializable,
            "pi_theta_old": pi_theta_old_serializable,
            "actions_data": actions_data
        }
        buffer_data = {
            "buffer": buffer
        }

        self._model_cache.save_model(model_id, "params", params)
        self._model_cache.save_model(model_id, "buffer", buffer_data)

    def load_from_cache(self, model_id="dapo"):
        """
        从缓存加载代理的状态。
        """
        params = self._model_cache.get_model(model_id, "params")
        buffer_data = self._model_cache.get_model(model_id, "buffer")

        Q_serializable = params.get("Q") if params else None
        pi_theta_serializable = params.get("pi_theta") if params else None
        pi_theta_old_serializable = params.get("pi_theta_old") if params else None
        actions_data = params.get("actions_data") if params else None
        _buffer = buffer_data.get("buffer") if buffer_data else None
        #print(params,"params")

        Q = None
        pi_theta = None
        pi_theta_old = None
        actions = None
        action_to_index = None
        index_to_action = None
        next_action_index = None
        buffer = None

        if Q_serializable is not None:
            Q = {state: np.array(values) for state, values in Q_serializable.items()}
        if pi_theta_serializable is not None:
            pi_theta = {state: np.array(values) for state, values in pi_theta_serializable.items()}
        if pi_theta_old_serializable is not None:
            pi_theta_old = {state: np.array(values) for state, values in pi_theta_old_serializable.items()}
        if actions_data is not None:
            actions = set(actions_data["actions"])
            action_to_index = actions_data["action_to_index"]
            index_to_action = actions_data["index_to_action"]
            next_action_index = actions_data["next_action_index"]
        if _buffer is not None:
            buffer = _buffer

        return Q, pi_theta, pi_theta_old, actions, action_to_index, index_to_action, next_action_index, buffer

# 测试代码
"""from diskcache import Cache
cache = Cache("dapo")
from .model_cache import ModelManager
mgr=ModelManager(cache)
agent=DAPOAgent(
        gamma=0.99,
        alpha_Q=0.1,
        alpha_pi=0.01,
        epsilon=10.1,
        G=5,
        epsilon_low=0.2,
        epsilon_high=0.5,
        buffer_size=7,
        beta_entropy=0.01,
        overlong_threshold=10,
        overlong_penalty=-0.1,
        model_cache=mgr,
    )

state=(0,1)
for i in range(10):
    x=agent.choose_action(state,1,"test5")
    print(x)
#agent.load_from_cache("test2")
action=4
next_state=(1,2)
sequence_length=0
reward=10.9
model_id="test5"
agent.update(state, action, reward, next_state, sequence_length, model_id)"""