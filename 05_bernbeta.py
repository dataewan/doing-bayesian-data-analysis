import sys
import numpy as np
from scipy.stats import beta
from scipy.special import beta as beta_func
import matplotlib.pyplot as plt
from HDIofICDF import HDIofICDF

plt.style.use("seaborn-darkgrid")


def BernBeta(priorShape, dataVec, credMass=0.95, saveGraph=False):
    """Bayesian updating for Bernouli likelihood and beta prior.
    Input arguments:
        priorShape
            vectorof parameter values for the prior beta distribution
        dataVec
            vector of 1 and 0
        credMass
            the probability mass of the equal tailed credible interval

    Output:
        postShape
            vector of the parameter values for the posterior beta distribution

    Graphics:
        Creates a three panel graph of prior, likelihood, and posterior with
        highest posterior density interval

    Example of use:
        post_shape = bernBeta(priorShape=[1, 1], dataVec=[1, 0, 0, 1, 1]"""

    # check for errors in the input arguments:
    if len(priorShape) != 2:
        sys.exit("prior shape must have two components")

    if any([i < 0 for i in priorShape]):
        sys.exit("priorShape components must be positive")

    if any([i != 0 and i != 1 for i in dataVec]):
        sys.exit("dataVec must only contain 0 or 1")

    if credMass <= 0 or credMass >= 1:
        sys.exit("credMass must be between 0 and 1")

    # rename the prior shape parameters for convenience
    a = priorShape[0]
    b = priorShape[1]

    # create summary values of the data:
    z = sum(dataVec)  # number of 1 in datavec
    N = len(dataVec)  # length of datavec

    # compute the posterior shape parameters
    postShape = [a + z, b + N - z]

    # compute the evidence p(D)
    pData = beta_func(z + a, N - z + b) / beta_func(a, b)

    # Now plot everything
    # Construct a grid of theta values, used for graphing
    bin_width = 0.005

    theta = np.arange(bin_width / 2, 1 - (bin_width / 2) + bin_width, bin_width)

    # Compute the likelihood of the data at each value of theta.
    p_theta = beta.pdf(theta, a, b)

    # Compute the likelihood of the data at each value of theta.
    p_theta_given_theta = theta ** z * (1 - theta) ** (N - z)

    # Computer the posterior at each value of theta.
    post_a = a + z
    post_b = b + N - z

    p_theta_given_data = beta.pdf(theta, a + z, b + N - z)

    # Determine the limits of the highest density interval
    intervals = HDIofICDF(beta, credMass, a=postShape[0], b=postShape[1])

    # Plot the results

    plt.figure(figsize=(12, 12))

    plt.subplots_adjust(hspace=0.7)

    locx = 0.05

    plt.subplot(3, 1, 1)
    plt.plot(theta, p_theta)
    plt.xlim(0, 1)
    plt.ylim(0, np.max(p_theta) * 1.2)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$P(\theta|D)$")
    plt.title("Posterior")

    locy = np.linspace(0, np.max(p_theta_given_theta), 5)

    plt.text(locx, locy[1], r"beta($\theta$, {}, {})".format(post_a, post_b))

    plt.text(locx, locy[2], "P(D) = %g" % pData)

    plt.text(locx, locy[3], "Intervals = %.3f - %.3f" % (intervals[0], intervals[1]))

    plt.fill_between(
        theta,
        0,
        p_theta_given_data,
        where=np.logical_and(theta > intervals[0], theta < intervals[1]),
        color="blue",
        alpha=0.3,
    )

    return intervals


data_vec = np.repeat([1, 0], [11, 3])
intervals = BernBeta(priorShape=[100, 100], dataVec=data_vec)
plt.savefig("figure_5.2.png")
plt.show()
