import numpy as np
import matplotlib.pyplot as plt
import gym
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def display_frames_as_git(frames):
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


# region CONSTS
ENV = "CartPole-v0"
NUM_DIZITIZED = 6
GAMMA = 0.99
ETA = 0.5
MAX_STEPS = 200
NUM_EPISODES = 1000


# endregion


class Agent:
    """棒突き台そのもの"""

    def __init__(self, num_states, num_actions):
        # Agentが行動を決定するための頭脳を作成
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(
            observation, action, reward, observation_next
        )

    def get_action(self, observation, step):
        """行動の決定"""
        action = self.brain.decide_action(observation, step)
        return action


class Brain:
    def __init__(self, num_states, num_actions):
        # CartPoleの行動(左右に押す) の2を取得
        self.num_actions = num_actions
        # 連続一様分布から、ランダムなQテーブルを作成。
        # 行数は状態を分割数^(4変数)にデジタル変換した値、列数は行動数を表す
        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(NUM_DIZITIZED ** num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        """観測した状態(連続値)を離散値にデジタル変換する閾値を求める"""
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        """観測したobservation状態を、離散値に変換する"""
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED)),
        ]
        return sum([x * (NUM_DIZITIZED ** i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
                                      ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        """ε-greedy法で徐々に最適行動のみを採用する"""
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0,1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.argmax(self.q_table[state][:])
        return action


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        # CartPoleが持つ状態変数の数(P81の)である4を示す
        num_states = self.env.observation_space.shape[0]
        # CartPoleが持つ行動の数(左右)である2を示す
        num_actions = self.env.action_space.n

        self.agent = Agent(num_states, num_actions)

    def run(self):
        # 195step以上連続で立ち続けた試行数
        complete_episodes = 0
        is_episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            for step in range(MAX_STEPS):
                if is_episode_final is True:
                    frames.append(self.env.render(mode="rgb_array"))

                # 行動を求める
                action = self.agent.get_action(observation, episode)

                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, _, done, _ = self.env.step(action)

                # 報酬を与える
                # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                if done:
                    if step < 195:
                        # 途中でこけたら罰則として報酬-1を与える
                        # 195step以上連続でたち続けた試行数をリセット
                        reward = -1
                        complete_episodes = 0

                    else:
                        # 立ったまま終了時は報酬1を与える
                        # 連続記録を更新
                        reward = 1
                        complete_episodes += 1
                else:
                    # 途中の報酬は0
                    reward = 0

                # step+1の状態observation_nextを用いて、Q関数を更新する
                self.agent.update_Q_function(observation, action, reward, observation_next)

                observation = observation_next

                if done:
                    print("{0} Episode: Finished after {1} time steps".format(episode, step + 1))
                    break

            if is_episode_final is True:
                display_frames_as_git(frames)
                break

            if complete_episodes >= 10:
                print("10回連続成功")
                is_episode_final = True


cartpole_env = Environment()
cartpole_env.run()
