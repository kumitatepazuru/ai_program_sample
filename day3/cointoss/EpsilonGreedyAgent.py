import random
import numpy as np


class EpsilonGreedyAgent:
    def __init__(self, epsilon):
        """EpsilonGreedyAgent.__init__
Class EpsilonGreedyAgentが外部から呼ばれたときに発火する関数です。

引数
epsilon:探索を行う確率を指定します。
    exp. 0.2 => 20%の確率で探索をし、残りの80%で探索結果を活用します。"""

        self.epsilon = epsilon
        self.V = []

    def policy(self):
        """EpsilonGreedyAgent.policy
Epsilon-Greedy法の実装です。"""

        coins = range(len(self.V))
        if random.random() < self.epsilon:
            # 探索
            return random.choice(coins)
        else:
            # 活用
            return np.argmax(self.V)

    def play(self, env):
        """EpsilonGreedyAgent.play
実際にコイントスをプレイします。

引数
env:環境です。CoinTossクラスを渡します。"""

        # 見積もりを初期化します。
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards
