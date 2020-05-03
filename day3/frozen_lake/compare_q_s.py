import time
from multiprocessing import Pool
from collections import defaultdict

import cuitools
import gym
from ELAgent import ELAgent
from frozen_lake_util import show_q_value
from tqdm.gui import tqdm


class CompareAgent(ELAgent):
    """このファイルはday3のcode3のSARSAの実装部分のプログラムを一部改変し、読みやすくし、ファイルを分けたものです。

            具体的には？
            render引数をTrueにすると学習過程がわかるようになりました。
            tqdmを使いプログレスバーが表示できるようになりました。
            説明が各関数に入っています。

            関数の説明
            compare_q_s.CompareAgent
            実行をします。"""

    def __init__(self, q_learning=True, epsilon=0.33):
        self.q_learning = q_learning
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        """CompareAgent.learn
                        env:環境のデータが格納された変数です。
                        episode_count:エピソード回数を指定します。default:1000
                        gamma:割引率を指定します。default:0.9
                        render:画面に様子を表示するかどうか設定します。default:False
                        report_interval:ログを保存する間隔を設定します。default:50"""
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))
        for e in tqdm(range(episode_count)):
            s = env.reset()
            done = False
            a = self.policy(s, actions)
            while not done:
                if render:
                    cuitools.reset()
                    env.render()
                    time.sleep(0.01)

                n_state, reward, done, info = env.step(a)

                if done and reward == 0:
                    reward = -0.5  # Reward as penalty

                n_action = self.policy(n_state, actions)

                if self.q_learning:
                    gain = reward + gamma * max(self.Q[n_state])
                else:
                    gain = reward + gamma * self.Q[n_state][n_action]

                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state

                if self.q_learning:
                    a = self.policy(s, actions)
                else:
                    a = n_action
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                pass
            #     self.show_reward_log(episode=e)


def train(q_learning):
    env = gym.make("FrozenLakeEasy-v0")
    agent = CompareAgent(q_learning=q_learning)
    agent.learn(env, episode_count=100000)
    return dict(agent.Q)


if __name__ == "__main__":
    with Pool() as pool:
        results = pool.map(train, ([True, False]))
        for r in results:
            show_q_value(r)
