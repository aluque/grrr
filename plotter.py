""" Routines to plot the simulation status. """

import logging
import sys

from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm

from inout import IOContainer, M, MC2
import fitpl
from scipy.optimize import leastsq

numpy_histogram = histogram
logging.basicConfig(format='[%(asctime)s] %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.DEBUG)


FIGURE_SET = set()

def figure(func):
    def _f(*args, **kwargs):
        pylab.figure(func.__name__)
        FIGURE_SET.add(func.__name__)
        logging.debug("Updating figure '{}'".format(func.__name__))
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
def p2(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    pylab.plot(sim.TIME - sim.t0[sim.t0==0], 
               co.c * co.c * sum(sim.p[sim.t0 == 0, :]**2, axis=1) 
               / co.eV / co.eV, 
               'o',
               c=color, mew=0, ms=2.0)
    pylab.ylabel("$p^2$")
    pylab.xlabel("$z$ [m]")


@figure
def pz(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    pylab.plot(sim.TIME - sim.t0[sim.t0==0], 
               co.c * sim.p[sim.t0 == 0, 2] / co.eV, 
               'o',
               c=color, mew=0, ms=2.0)
    pylab.ylabel("$p^2$")
    pylab.xlabel("$z$ [m]")


@figure
def momentum(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    flt = sim.t0 == 0
    pylab.plot(co.c * sim.p[flt, 1] / co.eV, 
               co.c * sim.p[flt, 2] / co.eV,
               'o', c=color, mew=0, ms=2.0)
    pylab.xlabel("$p_y$ [eV/c]")
    pylab.ylabel("$p_z$ [eV/c]")



@figure
def location(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    flt = (sim.t0 == 0)

    logging.info("nprimaries = {:d}".format(sum(flt)))

    rc = sim.centroid(filter=flt)
    r = sim.r[flt, :] - rc[newaxis, :]
 
    pylab.plot(r[:, 0], 
               r[:, 2],
               'o', c=color, mew=0, ms=2.0)
    pylab.xlabel("$x$ [m]")
    pylab.ylabel(r"$\xi$ [m]")


@figure
def beta_front(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    v = (sim.r[:, 2] - 5) / sim.TIME / co.c
    zmin, zmax = amin(v), amax(v)

    bins = linspace(zmin, zmax, 60)
    h, a = histogram(v, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])
    p0 = array([1.0, a[argmax(h)], std(v)])

    def residua(p):
        return (reldiff(p, am[h > 0]) - h[h > 0]) / sqrt(h[h > 0])

    p, _ = leastsq(residua, p0)

    pylab.plot(am, h, 's', mew=0, ms=5.0, c=color)
    pylab.plot(am, reldiff(p, am), lw=1.0, alpha=0.5, c=color)


@figure
def rtau(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    xi = abs(sim.r[:, 2] - co.c * sim.TIME - 5)
    beta_avg = (sim.r[:, 2] - 5) / (co.c * sim.TIME)

    pylab.plot(xi, 
               sim.TIME * sim.tau / co.nano,
               'o', c=color, mew=0, ms=2.0)

    # pylab.plot(xi, 
    #            sim.TIME * sqrt(1 - beta_avg**2) / co.nano,
    #            's', c=color, mew=0, ms=1.0, alpha=0.6)

    pylab.xlabel(r"$z$ [m]")
    pylab.ylabel(r"$\tau$ [ns]")


@figure
def eng_tau(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    pylab.plot(sim.eng / co.eV, 
               sim.tau / co.nano,
               'o', c=color, mew=0, ms=2.0)


    pylab.xlabel(r"$K$ [eV]")
    pylab.ylabel(r"$\tau$ [ns]")


@figure
def collisions_tau(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    pylab.plot(sim.nelastic, 
               sim.TIME * sim.tau / co.nano,
               'o', c=color, mew=0, ms=2.0)


    pylab.xlabel(r"# collisions")
    pylab.ylabel(r"$\tau$ [ns]")



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
def primary_dist(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    pylab.plot(co.c * sim.p[sim.t0 == 0, 1] / co.eV, 
               co.c * sim.p[sim.t0 == 0, 2] / co.eV,
               'o', c='#990000', mew=0, ms=2.0)

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
def tau_angle(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    costheta = sim.p[:, 2] / sqrt(sim.p[:, 0]**2 + sim.p[:, 1]**2)

    pylab.plot(sim.tau / co.nano, costheta, 
               'o', mew=0, ms=2, c=color)
    pylab.xlabel(r"$\tau$ [ns]")
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
def primary_front(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    flt = (sim.t0 == 0)
    rc = sim.centroid(filter=flt)
    rstd = sim.rstd(filter=flt)

    xi = sim.r[flt, 2] - rc[2]

    bins = linspace(-60, 10, 71)
    h, a = numpy_histogram(xi, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])
    pylab.plot(am, 1 / sqrt(2 * pi) / rstd[2] * exp(-am**2 / (2 * rstd[2]**2)),
               lw=1.0, alpha=0.7, c=color)

    flt = h > 0
    pylab.plot(am[flt], h[flt], 'o', mew=0, ms=4.0, c=color)


    pylab.xlabel("$z - ut$ [m]")
    pylab.ylabel("$n$ [1/m]")


@figure
def primary_pz(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    #dpz = max(sim.p[sim.t0 == 0, 2]) - sim.p[sim.t0 == 0, 2]
    pz = co.c * sim.p[sim.t0 == 0, 2] / co.eV

    print(("Distribution of pz: avg = {:g}, var = {:g}, std = {:g}"
          .format(average(pz), var(pz), std(pz))))

    bins = linspace(nanmin(pz), nanmax(pz), 100)
    h, a = numpy_histogram(pz, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])

    flt = h > 0
    pylab.plot(am[flt], h[flt], lw=1.5, c=color)

    pylab.xlabel("$p_z$ [eV/c]")
    pylab.ylabel("$n [c/eV]$")


@figure
def primary_p2(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    #dpz = max(sim.p[sim.t0 == 0, 2]) - sim.p[sim.t0 == 0, 2]
    p2 = sum((co.c * sim.p[sim.t0 == 0, :] / co.eV)**2, axis=1)

    bins = linspace(nanmin(p2), nanmax(p2), 100)
    h, a = numpy_histogram(p2, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])

    flt = h > 0
    pylab.plot(am[flt], h[flt], lw=1.5, c=color)

    pylab.xlabel("$p^2$ [eV/c]")
    pylab.ylabel("$n [c/eV]$")


@figure
def primary_py(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    py = co.c * sim.p[sim.t0 == 0, 1] / co.eV

    bins = linspace(amin(py), amax(py), 100)
    h, a = numpy_histogram(py, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])

    flt = h > 0
    pylab.plot(am[flt], h[flt], lw=1.5, c=color)

    pylab.xlabel("$p_y$ [eV/c]")
    pylab.ylabel("$n [c/eV]$")


@figure
def primary_v_hist(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    flt = sim.t0 == 0
    v = sim.p[flt, 2] / (co.c * co.electron_mass 
                         * sqrt(1 + sum(sim.p[flt, :]**2, axis=1) 
                                 / (M * MC2)))

    bins = linspace(0.995, 1.0, 100)
    h, a = numpy_histogram(v, bins=bins, density=True)

    am = 0.5 * (a[1:] + a[:-1])

    flt = h > 0
    pylab.plot(am[flt], h[flt], lw=1.5, c=color)

    pylab.xlabel("$v$ [c]")
    pylab.ylabel("$n [c/eV]$")


@figure
def primary_v(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    flt = sim.t0 == 0
    v = sim.p[flt, :] / (co.c * co.electron_mass 
                         * sqrt(1 + sum(sim.p[flt, :]**2, axis=1)[:, newaxis]
                                 / (M * MC2)))

    pylab.plot(v[:, 1], v[:, 2], 'o', mew=0, ms=4.0, c=color)

    pylab.xlabel("$v_y$ [c]")
    pylab.ylabel("$v_z$ [c]$")



@figure
def collisions(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)

    eng = sim.eng[sim.t0 == 0]

    pylab.plot(sim.nionizing, 
                abs(sim.EB) * (sim.r[sim.t0 == 0, 2] - 5) - eng / co.eV, 
               'o', mew=0, ms=4.0, c=color)

    pylab.xlabel("# inelastic collisions")
    pylab.ylabel(r"$\Delta p_z$ [eV/c]")



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
def age_pz(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(age / co.nano, 
               co.c * sim.p[:, 2] / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel("$p_z$ [eV/c]")


@figure
def age_p(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    age = sim.TIME - sim.t0
    pylab.plot(age / co.nano, 
               co.c * sqrt(sum(sim.p**2, axis=1)) / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("Age [ns]")
    pylab.ylabel("$p_z$ [eV/c]")


@figure
def dz_p(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    dz = sim.r[:, 2] - sim.r0[:, 2]
    pylab.plot(dz, 
               co.c * sqrt(sum(sim.p**2, axis=1)) / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("$\Delta z$ [m]")
    pylab.ylabel("$|p|$ [eV/c]")


@figure
def dz_pz(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    dz = sim.r[:, 2] - sim.r0[:, 2]
    pylab.plot(dz, 
               co.c * sim.p[:, 2] / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.xlabel("$\Delta z$ [m]")
    pylab.ylabel("$|p|$ [eV/c]")


@figure
def p_pz(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    pylab.plot(co.c * sqrt(sum(sim.p**2, axis=1)) / co.eV, 
               co.c * sim.p[:, 2] / co.eV,
               'o', c=color, mew=0, ms=2.0)

    pylab.ylabel("$p_z$ [m]")
    pylab.xlabel("$|p|$ [eV/c]")


@figure
def v_p(sim, tfraction=None):
    if tfraction is None:
        tfraction = sim.tfraction

    color = cm.jet(tfraction)
    v = sim.p[:, 2] / (co.c * co.electron_mass 
                       * sqrt(1 + sum(sim.p[:, :]**2, axis=1) 
                                / (M * MC2)))
    vp = sim.p[:, 2] / (co.c * co.electron_mass 
                       * sqrt(1 + sim.p[:, 2]**2
                                / (M * MC2)))
    pylab.plot(co.c * sim.p[:, 2] / co.eV,
               1 - v,
               'o', c=color, mew=0, ms=2.0)

    pylab.plot(co.c * sim.p[:, 2] / co.eV,
               1 - vp,
               'o', mew=0, ms=2.0, c="#999999", alpha=0.7)

    pylab.ylabel(r"$1 - \beta$")
    pylab.xlabel(r"$p_z$ [eV/c]")


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
    print(("Front velocity u = {:g} m/s \n"
          "\t[c - u = {:g} m/s, beta = {}, 1 - beta = {}, K(u) = {:g} eV]"
          .format(u, co.c - u, beta, 1 - beta, K / co.eV)))


    pylab.xlabel('Time [ns]')
    pylab.ylabel('Location [m]')


def simple_regression(xi, yi):
    A = ones((len(yi), 2), dtype=float)
    A[:, 0] = xi[:]
    
    r = linalg.lstsq(A, yi)
    return r[0][0], r[0][1]


def reldiff(p, z):
    a, v, sigma = p
    return a * (exp(-(z - v)**2 / (2 * (1 - v * z)**2 * sigma**2))
                *  (1 - v**2) * sigma**2
                / (1 - v * z))**2


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
