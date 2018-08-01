import numpy as np
from matplotlib import pyplot as plt

nThetaVals = 20

Theta = np.linspace(
    start=1 / (nThetaVals + 1), stop=(nThetaVals) / (nThetaVals + 1), num=nThetaVals
)

# Make a triangular distribution to describe the prior
pTheta = np.minimum(Theta, 1 - Theta)
pTheta = pTheta / np.sum(pTheta)

data = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# data = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

nHeads = np.sum(data)
nTails = len(data) - nHeads

# Compute the likelihood of the data for each value of theta
pDataGivenTheta = np.power(Theta, nHeads) * np.power(1 - Theta, nTails)

# Compute the posterior
pData = np.sum(pDataGivenTheta * pTheta)

# This is bayes rule
pThetaGivenData = pDataGivenTheta * pTheta / pData

plt.figure(figsize=(12, 11))

fig, ax = plt.subplots(nrows=3, ncols=1, sharex="all")
plt.xlim(0, 1)
plt.xlabel("$\\theta$")


ax[0].stem(Theta, pTheta, markerfmt=" ")
ax[0].set_ylabel("$P(\\theta)$")

ax[1].stem(Theta, pDataGivenTheta, markerfmt=" ")
ax[1].set_ylabel("Likelihood")

ax[2].stem(Theta, pThetaGivenData, markerfmt=" ")
ax[2].set_ylabel("Posterior")

ax[2].text(0.6, np.max(pThetaGivenData) / 3, "P(D) = %g" % pData)


fig.savefig("figures/04_bayesupdate.png")
plt.close()
