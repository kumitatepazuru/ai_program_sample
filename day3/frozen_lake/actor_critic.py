import time

import cuitools
import numpy as np
import gym
from ELAgent import ELAgent
from frozen_lake_util import show_q_value
from tqdm.gui import tqdm


class Actor(ELAgent):

    def __init__(self, env):
        super().__init__(epsilon=-1)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        self.actions = list(range(env.action_space.n))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s):
        a = np.random.choice(self.actions, 1,
                             p=self.softmax(self.Q[s]))
        return a[0]


class Critic():

    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)


class ActorCritic():
    """このファイルはday3のcode3のActorCriticの実装部分のプログラムを一部改変し、読みやすくし、ファイルを分けたものです。

        具体的には？
        render引数をTrueにすると学習過程がわかるようになりました。
        tqdmを使いプログレスバーが表示できるようになりました。
        説明が各関数に入っています。

        関数の説明
        actor_critic.ActorCritic
        実行をします。"""

    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        """actor_critic.train
                env:環境のデータが格納された変数です。
                episode_count:エピソード回数を指定します。default:1000
                gamma:割引率を指定します。default:0.9
                render:画面に様子を表示するかどうか設定します。default:False
                report_interval:ログを保存する間隔を設定します。default:50"""
        actor = self.actor_class(env)
        critic = self.critic_class(env)

        actor.init_log()
        for e in tqdm(range(episode_count)):
            s = env.reset()
            done = False
            while not done:
                if render:
                    cuitools.reset()
                    env.render()
                    time.sleep(0.01)
                a = actor.policy(s)
                n_state, reward, done, info = env.step(a)

                gain = reward + gamma * critic.V[n_state]
                estimated = critic.V[s]
                td = gain - estimated
                actor.Q[s][a] += learning_rate * td
                critic.V[s] += learning_rate * td
                s = n_state

            else:
                actor.log(reward)

            if e != 0 and e % report_interval == 0:
                pass
                # actor.show_reward_log(episode=e)

        return actor, critic


def train():
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("FrozenLakeEasy-v0")
    actor, critic = trainer.train(env, episode_count=100000)
    show_q_value(actor.Q)
    actor.show_reward_log()


if __name__ == "__main__":
    train()
