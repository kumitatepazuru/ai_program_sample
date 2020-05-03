import math
import time
from collections import defaultdict
import gym, cuitools
from ELAgent import ELAgent
from frozen_lake_util import show_q_value
from tqdm.gui import tqdm


class MonteCarloAgent(ELAgent):
    """このファイルはday3のcode3のmonte carloの実装部分のプログラムを一部改変し、読みやすくし、ファイルを分けたものです。

具体的には？
render引数をTrueにすると学習過程がわかるようになりました。
tqdmを使いプログレスバーが表示できるようになりました。
説明が各関数に入っています。

関数の説明
monte_carlo.MonteCarloAgent
実行をします。"""

    def __init__(self, epsilon=0.1):
        """ELAgent.ELAgentを参照してください。"""
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):
        """MonteCarloAgent.learn
env:環境のデータが格納された変数です。
episode_count:エピソード回数を指定します。default:1000
gamma:割引率を指定します。default:0.9
render:画面に様子を表示するかどうか設定します。default:False
report_interval:ログを保存する間隔を設定します。default:50"""
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))

        for e in tqdm(range(episode_count)):
            s = env.reset()
            done = False
            # Play until the end of episode.
            experience = []
            while not done:
                if render:
                    cuitools.reset()
                    env.render()
                    time.sleep(0.01)
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else:
                self.log(reward)

            # Evaluate each state, action.
            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                # Calculate discounted future reward of s.
                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1  # count of s, a pair
                alpha = 1 / N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                pass
            #     self.show_reward_log(episode=e)


def train():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
