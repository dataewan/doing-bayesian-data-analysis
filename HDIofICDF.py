"""
This program finds the HDI of a probability density function that is specified
mathematically in python
"""

from scipy.optimize import fmin
from scipy.stats import *


def HDIofICDF(dist_name, credMass=0.95, **args):
    # freeze distribution with given arguments
    distri = dist_name(**args)

    # Initial guess for HDIlowTailPr
    incredMass = 1.0 - credMass

    def intervalWidth(lowTailPr):
        return distri.ppf(credMass + lowTailPr) - distri.ppf(lowTailPr)

    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]

    return distri.ppf([HDIlowTailPr, credMass + HDIlowTailPr])
