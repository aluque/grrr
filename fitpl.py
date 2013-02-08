""" Fits power laws with cutoffs, i.e. dependencies such as 
   y = a * x^beta * exp(-x/x0).
"""
from numpy import *
from scipy.optimize import leastsq

def fitpl(x, y, p0=None):
    flt = y > 0
    x, y = x[flt], y[flt]

    logx = log(x)
    logy = log(y)

    def _cpl(p):
        return (p[0] + p[1] * logx - x / p[2]) - logy

    if p0 is None:
        p0 = array([1.0, -1.0, 1.0])

    p = leastsq(_cpl, p0)
    return p


def applypl(p, x):
    return exp(p[0] + p[1] * log(x) - x / p[2])
