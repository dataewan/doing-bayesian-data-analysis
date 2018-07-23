import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

meanval = 0.0
sdval = 0.2
xlow = meanval - 3 * sdval
xhigh = meanval + 3 * sdval

dx = 0.02

x = np.arange(start=xlow, stop=xhigh, step=dx)
y = (1/(sdval*np.sqrt(2*np.pi))) * np.exp(-0.5 * (((x - meanval) / sdval))**2)
area = np.sum(dx * y)


fig, ax = plt.subplots(1)
ax.plot(x, y)
ax.stem(x, y, markerfmt=' ')

ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')
ax.set_title("Normal probability density")

ax.text(-0.6, 1.7, '$\mu = {}$'.format(meanval))
ax.text(-0.6, 1.5, '$\sigma = {}$'.format(sdval))
ax.text(0.2, 1.7, '$\Delta x = {}$'.format(dx))
ax.text(0.2, 1.5, '$\sum_x \Delta x p(x) = {:3f}$'.format(area))

fig.savefig("figures/03_integral.png")
plt.close()
