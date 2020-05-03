from CoinToss import CoinToss
from EpsilonGreedyAgent import EpsilonGreedyAgent
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


def main():
    """このファイルはday3のcode3のコイントスの実装部分のプログラムを一部改変し、読みやすくし、ファイルを分けたものです。

具体的には？
プログラムのスレッド化をし、実行を早くしました。（とてつもなく重いです）
tqdmを使いプログレスバーが表示できるようになりました。
説明が各関数に入っています。

関数の説明
main.main
実行をします。

変数の説明
env:CoinTossのクラスが格納されます。len(env)でコインの枚数がわかります。初期値:[0.1, 0.5, 0.1, 0.9, 0.1]
epsilons:EpsilonGreedyAgentで使う探索の確率がリスト形式で格納されます。for文によって5回実行されます。（初期値）
    初期値:0.0, 0.1, 0.2, 0.5, 0.8
game_steps:実行のステップ数が格納されます。初期値:range(10, 310, 10)
result:結果が格納されます。最後にmatplotlibにより解析されてグラフになります。初期値:{}
agent:EpsilonGreedyAgentのクラスが格納されます。
means:resultの平均値が格納されます。最後にmatplotlibにより解析されてグラフになります。初期値:[]
※他の変数は触接的には関係ないので省きます。
"""

    env = CoinToss(list(map(lambda n:n*0.1,range(0,10))))
    epsilons = list(map(lambda n:n*0.01,range(0,10)))
    print("epsilons:"+" ".join(list(map(lambda n:str(n),epsilons))))

    game_steps = list(range(10, 10000, 10))
    result = {}
    for e in tqdm(epsilons):
        agent = EpsilonGreedyAgent(epsilon=e)
        means = []
        thread_data = []
        with ProcessPoolExecutor() as executor:
            for i in game_steps:
                env.max_episode_steps = i
                thread_data.append(executor.submit(agent.play,env))
            tqdm_g = tqdm(thread_data)
            for s in tqdm_g:
                tqdm_g.set_description("loading... epsilons:"+str(e))
                rewards = s.result()
                means.append(np.mean(rewards))
        result["epsilon={}".format(e)] = means

    result["coin toss count"] = game_steps
    result = pd.DataFrame(result)
    result.set_index("coin toss count", drop=True, inplace=True)
    result.plot.line(figsize=(10, 5))
    plt.show()


if __name__ == '__main__':
    main()
