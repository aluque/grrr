import sys
from collections import namedtuple

from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm

import grrr
from plotfield import evaluate_fields, plot_field
import fitpl

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2

Stats = namedtuple('Stats', ['max_energy', 'avg_energy'])

EB =  -0 * co.kilo / co.centi
B0 =  20 * co.micro
E0 =  -20 * co.kilo / co.centi
L  =  1000
BETA = 1 - 0.001182
U0  =  BETA * co.c
THETA = pi / 16

fdist = open("dist.dat", "w")

def main():
    grrr.set_parameter('L' , L)
    grrr.set_parameter('B0', B0)
    grrr.set_parameter('E0', E0)
    grrr.set_parameter('U0', U0)
    grrr.set_parameter('EB',  EB)
    grrr.set_parameter('THETA' , THETA)
    grrr.set_parameter('EBWIDTH', 4)
    grrr.set_emfield_func('pulse')

    grrr.list_clear()
    init_list(0, 10, 10000 * co.kilo * co.eV, 1000)

    run(init_hooks=[init_output],
        inner_hooks=[output],
        finish_hooks=[pylab.show])


def emfunc_const(t, r, e, b):
    e[0], e[1], e[2] = 0.0, 0.0, EB;
    b[0], b[1], b[2] = B0, 0.0, 0.0;
    

def run(init_hooks=[], inner_hooks=[], finish_hooks=[]):
    dt = 0.0025 * co.nano
    final_t = 5000 * co.nano
    output_dt = 20 * co.nano
    output_n = int(output_dt / dt)
    max_particles = 2000

    t = 0
    purge_factor = 0.5

    for f in init_hooks:
        f()


    while (t <= final_t):
        t = grrr.list_step_n_with_purging(t, dt, output_n, max_particles,
                                          purge_factor)

        print("[{0:.2f} ns]: {1:d} particles ({2:g} weighted)"\
                  .format(t / co.nano, grrr.particle_count.value,
                          grrr.particle_weight() * grrr.particle_count.value))
        for f in inner_hooks:
            f(t, final_t)

    for f in finish_hooks:
        f()


def stats():
    p = grrr.particles_p()
    r = grrr.particles_r()
    eng = grrr.particles_energy(p)
    
    try:
        s = Stats(max_energy=amax(eng), 
                  avg_energy=average(eng))
    except ValueError:
        s = Stats(max_energy=0.0, 
                  avg_energy=0.0)
    return s

def init_list(zmin, zmax, emax, n):
    """ Inits a list with n particles randomly located from -zmin to zmin
    and top energy emax. """
    U0 = MC2 + emax

    z = random.uniform(zmin, zmax, size=n)
    q = random.uniform(0, 1, size=n)

    r = [array([0.0, 0.0, z0]) for z0 in z]
    p = [array([0, 0, q0 * sqrt(U0**2 - M2C4) / co.c]) for q0 in q]

    for r0, p0 in zip(r, p):
        grrr.create_particle(grrr.ELECTRON, r0, p0)


def init_output():
    print("Initializing output...")
    y = linspace(-5 * L, 5 * L, 400)
    z = linspace(-300, 50, 700)
    e, b = evaluate_fields(0, y, z)

    pylab.figure('trajectories')
    plot_field(y, z, e, b)

    print("Done.")
    # pylab.figure('trajectories / pz')
    # plot_field(y, z, e, b)
    

def output(t, final_t):
    """ A lot of figures to analyze the results. """
    if grrr.particle_count.value == 0:
        return

    p = grrr.particles_p()
    r = grrr.particles_r()
    eng = grrr.particles_energy(p[p[:, 2] > 0])
    #u = co.c / cos(grrr.get_parameter('THETA'))
    u = grrr.get_parameter('U0')

    color = cm.jet(t / final_t)

    pylab.figure('phases')
    pylab.plot(co.c * p[:, 2] / co.eV, r[:, 2] - u * t, 'o',
               c=color, mew=0, ms=2.0)
    pylab.xlabel("$p_z$ [eV/c]")
    pylab.ylabel("$z - ut$ [m]")
    #pylab.semilogx()

    pylab.figure('histogram')
    bins = logspace(5.5, 9, 100)
    h, a = histogram(eng / co.eV, bins=bins, density=True)
    am = 0.5 * (a[1:] + a[:-1])
    flt = h > 0
    pylab.plot(am[flt], h[flt], 'o', mew=0, ms=4, c=color)

    pl, _ = fitpl.fitpl(am[flt], h[flt], p0=array([1e-6, -1.0, 7.5e6 * co.eV]))
    print("alpha = {1:.4g}; cutoff = {2:.3g} eV".format(*pl))

    pylab.plot(am[flt], fitpl.applypl(pl, am[flt]), c=color, alpha=0.3, lw=3.0)
    fdist.write("{0:g} {1:.4g} {2:.3g} {3:g}\n"\
                    .format(t, pl[1], pl[2], 
                            grrr.particle_weight() * grrr.particle_count.value))
    fdist.flush()

    savetxt('histogram.dat', c_[am[flt], h[flt], h[flt] / am[flt]])

    pylab.xlabel("$E$ [eV]")
    pylab.ylabel("f(E) [1/eV]")
    pylab.loglog()

    pylab.figure('momentum')
    pt = sqrt(p[:, 0]**2 + p[:, 1]**2)
    pylab.plot(co.c * p[:, 1] / co.eV, co.c * abs(p[:, 2]) / co.eV, 
               'o', mew=0, ms=2, c=color)
    pylab.xlabel("$p_\perp$ [eV/c]")
    pylab.ylabel("$p_z$ [eV/c]")
    pylab.loglog()

    pylab.figure('trajectories')
    rt = sqrt(r[:, 0]**2 + r[:, 1]**2)
    pylab.plot(r[:, 1], r[:, 2] - u * t, 'o', mew=0, ms=2, c=color)
    pylab.xlabel("$r_\perp$ [m]")
    pylab.ylabel("$r_z - ut$ [m]")

    pylab.figure('transversal r')
    rt = sqrt(r[:, 0]**2 + r[:, 1]**2)
    pylab.plot(r[:, 1], co.c * p[:, 2] / co.eV, 
               'o', mew=0, ms=2, c=color)
    pylab.xlabel("$r_\perp$ [m]")
    pylab.ylabel("$p_z$ [eV/c]")
    #pylab.semilogy()

    # pylab.figure('trajectories / pz')
    # s = co.c * abs(p[:, 2]) / (2e8 * co.eV)
    # c = cm.jet(s)
    # rt = sqrt(r[:, 0]**2 + r[:, 1]**2)
    # pylab.scatter(r[:, 1], r[:, 2] - u * t, faceted=False, s=4 * s, c=c)
    # pylab.xlabel("$r_\perp$ [m]")
    # pylab.ylabel("$r_z - ut$ [m]")



if __name__ == '__main__':
    main()
