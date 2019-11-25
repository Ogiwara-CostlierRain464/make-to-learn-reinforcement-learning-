import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gym
from collections import namedtuple
from matplotlib import animation
from IPython.display import display
from JSAnimation.IPython_display import display_animation
import warnings

# PyTorch内のwarnを非表示に
warnings.filterwarnings("ignore")


def display_frames_as_gif(frames):
    plt.figure(
        figsize=(
            frames[0].shape[1] / 72,
            frames[0].shape[0] / 72
        ),
    )
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(),
        animate,
        frames=len(frames),
        interval=50
    )

    display(display_animation(anim, default_mode=50))


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)

ENV = "CartPole-v0"
# 時間割引率
GAMMA = 0.99
# 1試行のstep数
MAX_STEPS = 200
# 最大試行回数
NUM_EPISODES = 500
BATCH_SIZE = 32
CAPACITY = 10000


# 経験を保存するメモリクラス
class ReplayMemory:
    def __init__(self, capacity):
        # メモリのサイズ
        self.capacity = capacity
        # 経験を保存する
        self.memory = []
        # 保存するindexを表す
        self.index = 0

    def push(self, state, action, state_next, reward):
        """transition = (state, action, state_next, reward)をメモリに保存する"""

        # メモリが満タンでない時は足す
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        """batch_size分だけ、ランダムに保存内容を取り出す"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """関数lenに対して、現在の変数memoryの長さを返す"""
        return len(self.memory)


class Brain:
    def __init__(self, num_states, num_actions):
        # CartPoleの行動分岐数(2)を取得
        self.num_actions = num_actions
        # 経験を記録するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)
        # Make NN
        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(num_states, 32))  # 4 -> 32
        self.model.add_module("relu1", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(32, 32))
        self.model.add_module("relu2", nn.ReLU())
        self.model.add_module("fc3", nn.Linear(32, num_actions))
        print(self.model)

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        """Experience Replayでネットワークの結合パラメータを学習"""

        # -----
        # 1. メモリサイズの確認
        # -----
        # メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < BATCH_SIZE:
            return

        # -----
        # 2. make mini batch
        # -----
        # 2.1 メモリからミニバッチ分布のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        batch = Transition(*zip(*transitions))

        # 2.3 各変数の要素をミニバッチに対応する形に変形
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        # -----
        # 3. 教師信号となるQ(s_t, a_t)値を求める
        # -----
        # 3.1 ネットワークを推論モードに切り替え
        self.model.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意
        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state))
        )

        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(
            non_final_next_states
        ).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # -----
        # 4. 結合パラメータの更新
        # -----
        # 4.1 ネットワークを訓練モードに変える
        self.model.train()

        # 4.2 損失関数を計算する
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 4.3 結合パラメータを更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        """現在の状態に応じて、行動を決定する"""
        # ε-greedy法で徐々に最適行動のみを採用する
        # episodeがだんだん大きくなるので、epsilonはだんだん小さくなる
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            # 微分は推論では必要ない
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            # 0,1の行動をランダムに返す「
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        """Q関数を更新する"""
        self.brain.replay()

    def get_action(self, state, episode):
        """行動を決定する"""
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        """memoryオブジェクトにargsを保存する"""
        self.brain.memory.push(state, action, state_next, reward)


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        # 10試行分のたち続けたstep数を格納し、平均ステップ数を出力に利用
        episode_10_list = np.zeros(10)
        # 195step以上連続でたち続けた試行数
        complete_episodes = 0
        # 最後の試行フラグ
        episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)

            # flattenにするだけ
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                if episode_final is True:
                    frames.append(self.env.render(mode="rgb_array"))

                # 行動を求める
                action = self.agent.get_action(state, episode)
                observation_next, _, done, _ = self.env.step(action.item())

                if done:
                    state_next = None

                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])

                        complete_episodes = complete_episodes + 1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()

                state = state_next

                if done:
                    print("{0} Episode: Finished after {1} steps: 10試行の平均step数 = {2}"
                          .format(episode, step + 1, episode_10_list.mean()))
                    break

            if episode_final is True:
                # 描画
                break

            if complete_episodes >= 10:
                print("10回連続成功")
                episode_final = True


cartpole_env = Environment()
cartpole_env.run()
