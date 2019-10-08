import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

plt.plot([1, 1], [0, 1], color="red", linewidth=2)
plt.plot([1, 2], [2, 2], color="red", linewidth=2)
plt.plot([2, 2], [2, 1], color="red", linewidth=2)
plt.plot([2, 3], [1, 1], color="red", linewidth=2)

plt.text(0.5, 2.5, "S0", size=14, ha="center")
plt.text(1.5, 2.5, "S1", size=14, ha="center")
plt.text(2.5, 2.5, "S2", size=14, ha="center")
plt.text(0.5, 1.5, "S3", size=14, ha="center")
plt.text(1.5, 1.5, "S4", size=14, ha="center")
plt.text(2.5, 1.5, "S5", size=14, ha="center")
plt.text(0.5, 0.5, "S6", size=14, ha="center")
plt.text(1.5, 0.5, "S7", size=14, ha="center")
plt.text(2.5, 0.5, "S8", size=14, ha="center")
plt.text(0.5, 2.3, "START", ha="center")
plt.text(2.5, 0.3, "GOAL", ha="center")

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="off", right="off", left="off", labelleft="off")

line, = ax.plot([0.5], [2.5], marker="o", color="g", markersize=60)

plt.show()

theta_0 = np.array([
    [np.nan, 1, 1, np.nan],
    [np.nan, 1, np.nan, 1],
    [np.nan, np.nan, 1, 1],
    [1, 1, 1, np.nan],
    [np.nan, np.nan, 1, 1],
    [1, np.nan, np.nan, np.nan],
    [1, np.nan, np.nan, np.nan],
    [1, 1, np.nan, np.nan],
])


def softmax(theta):
    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    exp_theta = np.exp(beta * theta)

    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])

    pi = np.nan_to_num(pi)

    return pi


# 1step移動後の状態sを求める関数を定義
def get_next_s(pi, s):
    direction = ["up", "right", "down", "left"]

    # pi[s,:]の確率にしたがって、directionが選択される
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        action = 0
        s_next = s - 3
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 3
    else:
        action = 3
        s_next = s - 1

    return [action, s_next]


# 迷路内をエージェントがゴールするまで移動させる関数の定義
def goal_maze(pi):
    s = 0
    s_history = [[0, np.nan]]

    while 1:
        [action, next_s] = get_next_s(pi, s)
        s_history[-1][1] = action
        s_history.append([next_s, np.nan])

        # if goal
        if next_s == 8:
            break
        else:
            s = next_s

    return s_history


def update_theta(theta, pi, s_history):
    eta = 0.1
    t = len(s_history)

    [m, n] = theta.shape
    delta_theta = theta.copy()

    for i in range(0, m):
        for j in range(0, n):
            if not (np.isnan(theta[i, j])):
                sa_i = [SA for SA in s_history if SA[0] == i]
                sa_ij = [SA for SA in s_history if SA == [i, j]]

                n_i = len(sa_i)
                n_ij = len(sa_ij)
                delta_theta[i, j] = (n_ij - pi[i, j] * n_i) / t

    n_theta = theta + eta * delta_theta

    return n_theta


pi_0 = softmax(theta_0)
state_history = goal_maze(pi_0)

new_theta = update_theta(theta_0, pi_0, state_history)
pin = softmax(new_theta)
print(pin)
