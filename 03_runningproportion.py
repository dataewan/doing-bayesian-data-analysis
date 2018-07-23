import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

# Goal, toss a coin N times and compute the running proportion of heads.

# Specify the number of flips
N = 500

# say that a 1 is a heads
flips = np.random.choice(a=(0, 1), p=(0.5, 0.5), size=N, replace=True)

r = np.cumsum(flips)
n = np.linspace(1, N, N)

runningproportion = r / n


fig, ax = plt.subplots(1)
ax.plot(n, runningproportion)
ax.set_ylim(0, 1)
ax.set_xlim(1, N)
ax.axhline(0.5)
ax.set_xlabel("Number of flips")
ax.set_ylabel("Proportion of heads")

fig.savefig("figures/03_runningproportion.png")
plt.close()
