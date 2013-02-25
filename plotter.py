""" Routines to plot the simulation status. """

import logging
import sys

from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm

from inout import IOContainer
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
def momentum(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    pylab.plot(co.c * sim.p[:, 1] / co.eV, 
               co.c * sim.p[:, 2] / co.eV,
               'o', c=color, mew=0, ms=2.0)
    pylab.xlabel("$p_y$ [eV/c]")
    pylab.ylabel("$p_z$ [eV/c]")


@figure
def distributions(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    pylab.plot(co.c * sim.p[self.xi > 0, 1] / co.eV, 
               co.c * sim.p[self.xi > 0, 2] / co.eV,
               'o', c='#990000', mew=0, ms=2.0)

    pylab.plot(co.c * sim.p[self.xi < 0, 1] / co.eV, 
               co.c * sim.p[self.xi < 0, 2] / co.eV,
               'o', c='#0000bb', mew=0, ms=2.0)

    pylab.xlabel("$p_y$ [eV/c]")
    pylab.ylabel("$p_z$ [eV/c]")


@figure
def histogram(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    bins = logspace(5.5, 9, 100)
    #beam = (sim.p[:, 2]**2 / (sim.p[:, 0]**2 + sim.p[:, 1]**2)) > 1

    am, h = sim.spectrum()

    flt = h > 0
    pylab.plot(am[flt], h[flt], 'o', mew=0, ms=4, c=color)

    pl, _ = fitpl.fitpl(am[flt], h[flt], p0=array([1e-6, -1.0, 7.5e6 * co.eV]))

    pylab.plot(am[flt], fitpl.applypl(pl, am[flt]), c=color, alpha=0.3, lw=3.0)
    logging.info("alpha = {1:.4g}; cutoff = {2:.3g} eV".format(*pl))
    savetxt('histogram.dat', c_[am[flt], h[flt], h[flt] / am[flt]])

    pylab.xlabel("$E$ [eV]")
    pylab.ylabel("f(E) [1/eV]")
    pylab.loglog()


@figure
def age_histogram(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0

    bins = logspace(0, 4, 100) * co.nano
    h, a = numpy_histogram(age[sim.xi > 0], bins=bins, density=True)
    am = 0.5 * (a[1:] + a[:-1])
    
    pylab.plot(am / co.nano, h, 'o', mew=0, ms=4, c=color)
    pylab.xlabel(r"Age, $\tau$ [ns]")
    pylab.ylabel(r"$f(\tau)$ [1/eV]")


@figure
def age_angle(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    theta = arccos(sim.p[:, 2] / sqrt(sim.p[:, 0]**2 + sim.p[:, 1]**2))

    pylab.plot(age[sim.xi > 0] / co.nano, theta[sim.xi > 0], 
               'o', mew=0, ms=2, c=color)
    pylab.xlabel(r"Age, $\tau$ [ns]")
    pylab.ylabel(r"$\theta$")


@figure
def energy_angle(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    theta = arccos(sim.p[:, 2] / sqrt(sim.p[:, 0]**2 + sim.p[:, 1]**2))

    pylab.plot(sim.eng / co.eV, cos(theta), 
               'o', mew=0, ms=2, c=color)
    pylab.xlabel(r"Energy [eV]")
    pylab.ylabel(r"$\cos(\theta$)")




@figure
def front(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    bins = linspace(amin(sim.xi), amax(sim.xi), 100)
    h, a = numpy_histogram(sim.xi, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])

    flt = h > 0
    pylab.plot(am[flt], h[flt], 'o', mew=0, ms=4, c=color)

    pylab.xlabel("$z - ut$ [m]")
    pylab.ylabel("$\sigma$ [1/m]")


@figure
def edge(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    bins = linspace(amin(sim.xi), amax(sim.xi), 500)
    h, a = numpy_histogram(sim.xi, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])
    imax = argmax(h)
    ammax = am[imax]

    xi = am[am > ammax] - ammax
    n = h[am > ammax]

    a, b = simple_regression(xi[n>0], log(n[n>0]))
    pylab.plot(xi, n, 'o', mew=0, ms=5.0, c=color)
    pylab.plot(xi, exp(xi * a + b), lw=2.0, alpha=0.3, c=color)

    logging.info("delta = {:.4g} m".format(-1 / a))

    pylab.semilogy()
    pylab.xlabel("$z - ut$ [m]")
    pylab.ylabel("$n$")


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


@figure
def selfcons_field(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    pylab.plot(sim.zfcells, 
               sim.ez / (co.kilo / co.centi), 
               lw=1.5, c=color)
    
    pylab.xlabel("$z - ut$ [m]")
    pylab.ylabel("$E_z$ [kV / cm]")


@figure
def selfcons_field_xi(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    pylab.plot(sim.zfcells - sim.TIME * sim.U0, 
               sim.ez / (co.kilo / co.centi), 
               lw=1.5, c=color)
    
    pylab.xlabel("$z - ut$ [m]")
    pylab.ylabel("$E_z$ [kV / cm]")



@figure
def age(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(age[sim.xi > 0] / co.nano, 
               sim.eng[sim.xi > 0] / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel("$K$ [eV]")


@figure
def xi0(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(age / co.nano, 
               sim.r0[:, 2] - sim.U0 * sim.t0,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel("Initial $z - ut$ [m]")


@figure
def age_dxi(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    xi0 = sim.r0[:, 2] - sim.U0 * sim.t0
    xi  = sim.r[:, 2] - sim.U0 * sim.TIME

    pylab.plot(age / co.nano, 
               abs(xi - xi0),
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel(r"$|\xi - \xi_0|$ [m]")


@figure
def engxi0(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(sim.r0[:, 2] - sim.U0 * sim.t0,
               sim.eng / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Initial $z - ut$ [m]")
    pylab.ylabel("Energy [eV]")


@figure
def eng0(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(age / co.nano, 
               sim.eng0 / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel("Initial $K$ [eV]")


@figure
def engeng0(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(sim.eng0 / co.eV,
               sim.eng / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Initial $K$ [eV]")
    pylab.ylabel("Present $K$ [eV]")


@figure
def deng(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(age / co.nano,
               (sim.eng - sim.eng0) / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel("$\Delta K$ [eV]")


@figure
def phase_trajectory(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(age / co.nano, 
               sim.r[:, 2] - sim.U0 * sim.TIME,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel("$z - ut$ [m]")


def front_location(front):
    pylab.figure('front location')

    pylab.plot(front[:, 0] / co.nano, front[:, 1], 'o', c='b')

    u, b = simple_regression(front[:, 0], front[:, 1])
    beta = u / co.c
    gamma = 1 / sqrt(1 - beta**2)
    K = co.electron_mass * co.c**2 * (gamma - 1)

    pylab.plot(front[:, 0] / co.nano, u * front[:, 0] + b, lw=2.0, c='k')
    print("Front velocity u = {:g} m/s \n"
          "\t[c - u = {:g} m/s, beta = {}, 1 - beta = {}, K(u) = {:g} eV]"
          .format(u, co.c - u, beta, 1 - beta, K / co.eV))


    pylab.xlabel('Time [ns]')
    pylab.ylabel('Location [m]')


def simple_regression(xi, yi):
    A = ones((len(yi), 2), dtype=float)
    A[:, 0] = xi[:]
    
    r = linalg.lstsq(A, yi)
    return r[0][0], r[0][1]


def main():
    from optparse import OptionParser
    me = sys.modules[__name__]

    parser = OptionParser()
    (opts, args) = parser.parse_args()

    ioc = IOContainer()
    ioc.open(args[0])

    plots = args[1:]
    if not plots:
        plots = ['histogram', 'phases', 'age', 
                 'selfcons_field', 'front', 'momentum']

    front = []

    for step in ioc:
        ioc.load(step)
        for plot in plots:
            func = getattr(me, plot)
            func(ioc)

        front.append([ioc.TIME, ioc.centroid()[2]])

    front_location(array(front))

    pylab.show()

if __name__ == '__main__':
    main()
