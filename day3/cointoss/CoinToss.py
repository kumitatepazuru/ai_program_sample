import random


class CoinToss:
    def __init__(self, head_probs, max_episode_steps=30):
        """CoinToss.__init__
Class CoinTossが外部から呼ばれたときに発火する関数です。

引数
head_probs:各コインの表が出る確率を指定します。(list)
    exp. [0.1,0.8,0.3] => コインの数:3, 表が出る確率:10% 80% 30%
        [0.3,0.52,0.41] => コインの数:3, 表が出る確率:30% 52% 41%

max_episode_steps:コイントスの最大試行回数を設定します(int)"""

        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        """CoinToss.__len__
class CoinToss内で組み込み関数len()が実行されたときに発火される関数です。"""

        return len(self.head_probs)

    def reset(self):
        """CoinToss.reset
コイントスのカウントを0に戻す関数です。"""

        self.toss_count = 0

    def step(self, action):
        """CoinToss.step
実際にコイントスをやります。

引数
action:この引数で指定されたコイントスを__init__のhead_probsで設定した確率で実行します。"""

        final = self.max_episode_steps - 1
        if self.toss_count > final:
            # max_episode_stepsで指定されたコインよりも多く投げようとしたときにエラーを出します。
            raise Exception("English: The step count exceed maximum."
                            "Use the reset () function to reset the environment.\n"
                            "日本語 : コイントスの最大試行回数を超えています。"
                            "reset()関数を使い環境をリセットさせてください。")
        else:
            if self.toss_count == final:
                # 上限に達したか確認し、もし達していれば終了します。
                done = True
            else:
                done = False

        if action > len(self.head_probs) - 1:
            # actionで指定されたコインが存在しない場合にエラーを出します
            raise Exception("English: The No.{} coin doesn't\n"
                            "日本語 : 指定された{}番目のコインは存在しません。".format(action, action))
        else:
            head_prob = self.head_probs[action]

            # 実際にコイントスをします。rewardは報酬です。
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done
