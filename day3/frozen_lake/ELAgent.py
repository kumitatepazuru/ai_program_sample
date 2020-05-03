import numpy as np
import matplotlib.pyplot as plt


class ELAgent:
    """ELAgent.ELAgent
実装するエージェントのベースとなるクラスと環境を扱うためのクラスです。"""
    def __init__(self, epsilon):
        """ELAgent.__init__
Class ELAgentが外部から呼ばれたときに発火する関数です。

引数
epsilon:探索を行う確率を指定します。
    exp. 0.2 => 20%の確率で探索をし、残りの80%で探索結果を活用します。"""
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s, actions):
        """ELAgent.policy
Epsilon-Greedy法の実装です。

引数
s:状態が格納されているデータが渡されます。
actions:AIが行ける場所が格納されているデータを渡されます。"""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        """ELAgent.init_log
エージェントが獲得した報酬の記録を初期化します。"""
        self.reward_log = []

    def log(self, reward):
        """ELAgent.log
報酬の記録をします。"""
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        """ELAgent.show_reward_log
記録した報酬の可視化を行います。

引数
interval:記録されている報酬の表示の間隔を指定します。default:50
episode:episodeが指定されていた場合は指定されていた報酬の結果を表示します。
    exp. 15 => 15番目の報酬の結果を表示します。"""
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()
