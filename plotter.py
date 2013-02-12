""" Routines to plot the simulation status. """

import logging

from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm

import fitpl

numpy_histogram = histogram
logging.basicConfig(format='[%(asctime)s] %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.DEBUG)

FIGURE_SET = set()

def figure(func):
    def _f(*args, **kwargs):
        pylab.figure(func.func_name)
        FIGURE_SET.add(func.func_name)
        logging.debug("Updating figure '{}'".format(func.func_name))
        func(*args, **kwargs)

    _f.__doc__ = func.__doc__
    return _f

def save_all():
    for fig in FIGURE_SET:
        pylab.figure(fig)
        fname = '{}.pdf'.format(fig)
        logging.info("Saving figure '{}' into {}".format(fig, fname))
        pylab.savefig(fname)


@figure
def phases(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    pylab.plot(co.c * sim.p[:, 2] / co.eV, 
               sim.r[:, 2] - sim.U0 * sim.TIME, 'o',
               c=color, mew=0, ms=2.0)
    pylab.xlabel("$p_z$ [eV/c]")
    pylab.ylabel("$z - ut$ [m]")


@figure
def histogram(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    bins = logspace(5.5, 9, 100)
    h, a = numpy_histogram(sim.eng / co.eV, bins=bins, density=True)
    am = 0.5 * (a[1:] + a[:-1])
    flt = h > 0
    pylab.plot(am[flt], h[flt], 'o', mew=0, ms=4, c=color)

    pl, _ = fitpl.fitpl(am[flt], h[flt], p0=array([1e-6, -1.0, 7.5e6 * co.eV]))

    pylab.plot(am[flt], fitpl.applypl(pl, am[flt]), c=color, alpha=0.3, lw=3.0)
    logging.info("alpha = {1:.4g}; cutoff = {2:.3g} eV".format(*pl))

    pylab.xlabel("$E$ [eV]")
    pylab.ylabel("f(E) [1/eV]")
    pylab.loglog()


@figure
def front(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    bins = linspace(amin(sim.xi), amax(sim.xi), 100)
    h, a = numpy_histogram(sim.xi, bins=bins, density=True)
    h *= sim.nparticles

    am = 0.5 * (a[1:] + a[:-1])

    flt = h > 0
    pylab.plot(am[flt], h[flt], lw=1.5, c=color)

    pylab.xlabel("$z - ut$ [m]")
    pylab.ylabel("$\sigma$ [1/m]")


@figure
def field(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    pylab.plot(sim.front_xi, 
               sim.front_ez / (co.kilo / co.centi), 
               lw=1.5, c=color)
    
    pylab.xlabel("$z - ut$ [m]")
    pylab.ylabel("$E_z$ [kV / cm]")
