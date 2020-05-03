import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gym
from gym.envs.registration import register

register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})  # is_slipperyをTrueにするとAIが滑ります(?)


def show_q_value(Q):
    """frozen_lake_util.show_q_value
    Show Q-values for FrozenLake-v0.
    To show each action's evaluation,
    a state is shown as 3 x 3 matrix like following.
    FrozenLake-v 0のQ値を表示します。
    各アクションの評価を表示するには次のように3×3の行列で表されます。

    +---+---+---+
    |   | u |   |  u: up value 上昇値
    | l | m | r |  l: left value 左の値, r: right value 右の値, m: mean value 平均値
    |   | d |   |  d: down value ダウン値
    +---+---+---+

    迷路の内容
    +---+---+---+---+
    |STR|   |   |   | STR:スタート位置
    |   |HLE|   |HLE| HLE: 大穴
    |   |   |   |HLE| GOL:ゴール位置
    |HLE|   |   |GOL|
    +---+---+---+---+

    引数
    Q:行動価値が格納されたデータです。
    """
    env = gym.make("FrozenLake-v0")
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = np.zeros((q_nrow, q_ncol))

    for r in range(nrow):
        for c in range(ncol):
            s = r * nrow + c
            state_exist = False
            if isinstance(Q, dict) and s in Q:
                state_exist = True
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]:
                state_exist = True

            if state_exist:
                # At the display map, the vertical index is reversed.
                _r = 1 + (nrow - 1 - r) * state_size
                _c = 1 + c * state_size
                reward_map[_r][_c - 1] = Q[s][0]  # LEFT = 0
                reward_map[_r - 1][_c] = Q[s][1]  # DOWN = 1
                reward_map[_r][_c + 1] = Q[s][2]  # RIGHT = 2
                reward_map[_r + 1][_c] = Q[s][3]  # UP = 3
                reward_map[_r][_c] = np.mean(Q[s])  # Center

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
               vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()
