# Adapt the code from section 3.5.2 (integralofdesnity.py) to plot the density
# function of:
# p(x) = 6x (1 - x)
# over the interval x \in [0, 1]


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

xlow = 0
xhigh = 1

dx = 0.02

x = np.arange(start=xlow, stop=xhigh, step=dx)
y = 6 * x * (1 - x)
area = np.sum(dx * y)


fig, ax = plt.subplots(1)
ax.plot(x, y)
ax.stem(x, y, markerfmt=' ')

ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')
ax.set_title("Normal probability density")

ax.text(0.8, 1.5, '$\Delta x = {}$'.format(dx))
ax.text(0.2, 1.5, '$\sum_x \Delta x p(x) = {:3f}$'.format(area))

fig.savefig("figures/03.2_ex.png")
plt.close()
