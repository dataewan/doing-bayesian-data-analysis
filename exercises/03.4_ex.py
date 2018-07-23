# Adapt the integral code from 3.5.2 to determine the probability mass under
# the normal curve between plusminus 1 SD of the mean.

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

meanval = 0.0
sdval = 0.2
# find values between -1 and 1 standard deviations of the mean value.
xlow = meanval - sdval
xhigh = meanval + sdval

dx = 0.02

x = np.arange(start=xlow, stop=xhigh, step=dx)
y = (1 / (sdval * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (((x - meanval) / sdval)) ** 2)
area = np.sum(dx * y)


fig, ax = plt.subplots(1)
ax.plot(x, y)
ax.stem(x, y, markerfmt=" ")

ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")
ax.set_title("Normal probability density")

ax.text(-0.1, 1.7, "$\mu = {}$".format(meanval))
ax.text(-0.1, 1.5, "$\sigma = {}$".format(sdval))
ax.text(0.1, 1.7, "$\Delta x = {}$".format(dx))
ax.text(0.1, 1.5, "$\sum_x \Delta x p(x) = {:3f}$".format(area))

fig.savefig("figures/03.4_ex.png")
plt.close()


# around 68% of values are within 1 standard deviation of the mean. This is the
# expected result. Now the next part of the excercise:

# Now use the normal curve to describe the following belief. Suppose you
# believe that women’s heights follow a bell-shaped distribution, centered at
# 162cm with about two-thirds of all women having heights between 147cm and
# 177cm.

# This is described by a normal distribution with a mean of 1.62m.
# The difference between mean and lower bound is 0.15m
# The difference between mean and upper bound is 0.15m (fancy that!)
# The stanard deviation of this distribution should be 0.15m
