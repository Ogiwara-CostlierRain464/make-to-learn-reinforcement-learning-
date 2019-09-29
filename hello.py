import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5))
ax = plt.gca()

plt.plot([1,1],[0,1],color="red",linewidth=2)
plt.plot([1,2],[2,2],color="red",linewidth=2)
plt.plot([2,2],[2,1],color="red",linewidth=2)
plt.plot([2,3],[1,1],color="red",linewidth=2)

plt.text(0.5,2.5,"S0",size=14,ha="center")
plt.text(1.5,2.5,"S1",size=14,ha="center")
plt.text(2.5,2.5,"S2",size=14,ha="center")
plt.text(0.5,1.5,"S3",size=14,ha="center")
plt.text(1.5,1.5,"S4",size=14,ha="center")
plt.text(2.5,1.5,"S5",size=14,ha="center")
plt.text(0.5,0.5,"S6",size=14,ha="center")
plt.text(1.5,0.5,"S7",size=14,ha="center")
plt.text(2.5,0.5,"S8",size=14,ha="center")
plt.text(0.5,2.3,"START",ha="center")
plt.text(2.5,0.3,"GOAL",ha="center")

ax.set_xlim(0,3)
ax.set_ylim(0,3)
plt.tick_params(axis="both",which="both",bottom="off",top="off",
                labelbottom="off",right="off",left="off",labelleft="off")

line, = ax.plot([0.5],[2.5],marker="o",color="g",markersize=60)


plt.show()
'''
theta_0 = np.array([
    [np.nan, 1, 1, np.nan],
    [np.nan, 1, 1, np.nan, 1],
    [np.nan, np.nan, 1, 1],
    [1, 1, 1, np.nan],
    [np.nan, np.nan, 1, 1],
    [1, np.nan, np.nan, np.nan],
    [1, np.nan, np.nan, np.nan],
    [1, 1, np.nan, np.nan],
])


def simple_convert_into_pi_from_theta(theta):

    (m, n) = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

    pi = np.nan_to_num(pi)

    return pi


pi_0 = simple_convert_into_pi_from_theta(theta_0)
'''