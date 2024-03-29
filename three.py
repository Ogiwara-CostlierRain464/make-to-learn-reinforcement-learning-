import numpy as np
import matplotlib.pyplot as plt
import gym

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def display_gif(frames):
    plt.figure(
        figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
        dpi=72
    )
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save("movie_cartpole.mp4")
    display(display_animation(anim, default_mode="loop"))


frames = []
env = gym.make("CartPole-v0")
observation = env.reset()

for step in range(0,200):
    frames.append(env.render(mode="rgb_array"))
    action = np.random.choice(2)

    observation, reward, done, info = env.step(action)