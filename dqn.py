import random
from collections import deque

import numpy as np
import tensorflow as tf
import gymnasium as gym

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense


class DQN:
    def __init__(
        self,
        state_size,
        action_size,
        memory_capacity=10000,
        batch_size=32,
        gamma=0.95,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_interval=200,
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)

        # 经验池：容量满后，自动删除最旧的经验。
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size

        # 学习参数。
        self.gamma = gamma
        self.learning_rate = learning_rate

        # ε-greedy 探索参数。
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 每进行多少次梯度更新，同步一次目标网络。
        self.target_update_interval = target_update_interval
        self.train_steps = 0

        # 在线网络：用于选择动作，并通过训练更新。
        self.model = self.create_model()

        # 目标网络：用于计算相对稳定的训练目标。
        self.target_model = self.create_model()
        self.update_target_model()

    def create_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss="mse",
        )

        return model

    def update_target_model(self):
        """将在线网络的参数复制到目标网络。"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """保存一次状态转移；done 表示环境真正终止。"""
        state = np.asarray(
            state, dtype=np.float32
        ).reshape(self.state_size).copy()

        next_state = np.asarray(
            next_state, dtype=np.float32
        ).reshape(self.state_size).copy()

        self.memory.append((
            state,
            int(action),
            float(reward),
            next_state,
            bool(done),
        ))

    def act(self, state, training=True):
        """训练时允许随机探索；评估时选择 Q 值最大的动作。"""
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_size))

        state = np.asarray(
            state, dtype=np.float32
        ).reshape(1, self.state_size)

        q_values = self.model(state, training=False).numpy()

        return int(np.argmax(q_values[0]))

    def replay(self):
        """从经验池随机抽取一个批次，更新在线网络。"""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = np.stack([item[0] for item in batch])
        actions = np.asarray(
            [item[1] for item in batch], dtype=np.int32
        )
        rewards = np.asarray(
            [item[2] for item in batch], dtype=np.float32
        )
        next_states = np.stack([item[3] for item in batch])
        dones = np.asarray(
            [item[4] for item in batch], dtype=np.float32
        )

        # 目标网络估计下一状态下每个动作的 Q 值。
        next_q_values = self.target_model(
            next_states, training=False
        ).numpy()

        # 每条经验对应的 TD 目标：
        # 非终止状态：reward + gamma * max Q(next_state)
        # 终止状态：reward
        td_targets = (
            rewards
            + (1.0 - dones)
            * self.gamma
            * np.max(next_q_values, axis=1)
        )

        # 先复制当前预测，只替换实际执行动作对应的目标值。
        targets = self.model(
            states, training=False
        ).numpy().copy()

        rows = np.arange(self.batch_size)
        targets[rows, actions] = td_targets

        # 用一个批次完成一次梯度更新。
        loss = self.model.train_on_batch(states, targets)

        self.train_steps += 1

        if self.train_steps % self.target_update_interval == 0:
            self.update_target_model()

        return float(np.asarray(loss))

    def decay_epsilon(self):
        """每个训练回合结束后，减少随机探索概率。"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )


def main():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)

    agent = DQN(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
    )

    episodes = 500
    recent_rewards = deque(maxlen=20)

    try:
        for episode in range(1, episodes + 1):
            # 仅首次 reset 设置种子，后续沿用随机数序列。
            state, _ = env.reset(
                seed=seed if episode == 1 else None
            )

            total_reward = 0.0

            while True:
                action = agent.act(state)

                next_state, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                # terminated：任务真正终止。
                # truncated：达到时间上限等外部限制。
                # 两者都结束当前回合，但这里仅 terminated
                # 阻止 TD 目标使用下一状态的 Q 值。
                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done=terminated,
                )

                agent.replay()

                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break

            agent.decay_epsilon()
            recent_rewards.append(total_reward)

            if episode == 1 or episode % 10 == 0:
                print(
                    f"回合 {episode:3d}/{episodes} | "
                    f"奖励 {total_reward:6.1f} | "
                    f"最近20回合平均 {np.mean(recent_rewards):6.1f} | "
                    f"epsilon {agent.epsilon:.3f}"
                )

        # 评估阶段关闭随机探索，也不再训练。
        test_rewards = []

        for _ in range(10):
            state, _ = env.reset()
            total_reward = 0.0

            while True:
                action = agent.act(state, training=False)

                state, reward, terminated, truncated, _ = (
                    env.step(action)
                )
                total_reward += reward

                if terminated or truncated:
                    break

            test_rewards.append(total_reward)

        print(
            "\n评估10回合的平均奖励："
            f"{np.mean(test_rewards):.1f}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()