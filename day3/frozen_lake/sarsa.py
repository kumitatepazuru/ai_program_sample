import time
from collections import defaultdict

import cuitools
import gym
from ELAgent import ELAgent
from frozen_lake_util import show_q_value
from tqdm.gui import tqdm


class SARSAAgent(ELAgent):
    """このファイルはday3のcode3のSARSAの実装部分のプログラムを一部改変し、読みやすくし、ファイルを分けたものです。

        具体的には？
        render引数をTrueにすると学習過程がわかるようになりました。
        tqdmを使いプログレスバーが表示できるようになりました。
        説明が各関数に入っています。

        関数の説明
        sarsa.SARSAAgent
        実行をします。"""

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        """SARSAAgent.learn
                env:環境のデータが格納された変数です。
                episode_count:エピソード回数を指定します。default:1000
                gamma:割引率を指定します。default:0.9
                render:画面に様子を表示するかどうか設定します。default:False
                report_interval:ログを保存する間隔を設定します。default:50"""
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
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

                n_action = self.policy(n_state, actions)  # On-policy
                gain = reward + gamma * self.Q[n_state][n_action]
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                a = n_action
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                pass
                # self.show_reward_log(episode=e)


def train():
    agent = SARSAAgent()
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
