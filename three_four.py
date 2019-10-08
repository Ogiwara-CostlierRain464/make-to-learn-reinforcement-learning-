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
        dpi=72
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
        action = self.brain.deceide_action(observation, step)
        return action


class Brain:
    def __init__(self, num_states, num_actions):
        # CartPoleの行動(左右に押す) の2を取得
        self.num_actions = num_actions
        # Qテーブルを作成。
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

