from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm

import grrr

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2


def main():
    grrr.set_parameter('EBIAS', 0.0)
    grrr.set_parameter('L' , 1000.0)
    grrr.set_parameter('B0', 0.0 * 20 * co.micro)
    grrr.set_parameter('E0', -26 * co.mega)
    grrr.set_parameter('EB',  0 * co.kilo / co.centi)
    grrr.set_parameter('EBWIDTH', 1)

    dt = 0.5 * co.nano
    final_t = 2000 * co.nano
    output_dt = 10 * co.nano
    output_n = int(output_dt / dt)
    particles_max = 50000

    init_list(0.14, 0.16, 100 * co.kilo * co.eV, 10000)
    
    t = 0
    weight = 1.0
    purge_factor = 0.5

    while (t <= final_t):
        t = grrr.list_step_n(t, dt, output_n)
        if grrr.particle_count.value > particles_max:
            oldn = grrr.particle_count.value
            grrr.list_purge(purge_factor)
            print("            Purging {} -> {}"
                  .format(oldn, grrr.particle_count.value))
            weight /= purge_factor

        print("[{0:.2f} ns]: {1:d} particles ({2:g} weighted)"\
                  .format(t / co.nano, grrr.particle_count.value,
                          weight * grrr.particle_count.value))
        output(t, final_t)

    pylab.show()


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


def output(t, final_t):
    """ A lot of figures to analyze the results. """

    p = grrr.particles_p()
    r = grrr.particles_r()
    eng = grrr.particles_energy(p)
    color = cm.jet(t / final_t)

    pylab.figure('phases')
    pylab.plot(co.c * abs(p[:, 2]) / co.eV, r[:, 2] - co.c * t, 'o',
               c=color, mew=0, ms=2.0)
    pylab.xlabel("$p_z$ [eV/c]")
    pylab.ylabel("$z - ct$ [m]")
    pylab.semilogx()

    pylab.figure('histogram')
    bins = logspace(5.5, 9, 100)
    h, a = histogram(eng / co.eV, bins=bins, density=True)
    am = 0.5 * (a[1:] + a[:-1])
    flt = h > 0
    pylab.plot(am[flt], h[flt], 'o', mew=0, ms=4, c=color)
    pylab.xlabel("$E$ [eV]")
    pylab.ylabel("f(E) [1/eV]")
    pylab.loglog()

    pylab.figure('momentum')
    pt = sqrt(p[:, 0]**2 + p[:, 1]**2)
    pylab.plot(co.c * abs(pt) / co.eV, co.c * abs(p[:, 2]) / co.eV, 
               'o', mew=0, ms=2, c=color)
    pylab.xlabel("$p_\perp$ [eV/c]")
    pylab.ylabel("$p_z$ [eV/c]")
    pylab.loglog()

    pylab.figure('trajectories')
    rt = sqrt(r[:, 0]**2 + r[:, 1]**2)
    pylab.plot(rt, r[:, 2] - co.c * t, 'o', mew=0, ms=2, c=color)
    pylab.xlabel("$r_\perp$ [m]")
    pylab.ylabel("$r_z - ct$ [m]")

    pylab.figure('transversal r')
    rt = sqrt(r[:, 0]**2 + r[:, 1]**2)
    pylab.plot(rt, abs(co.c * p[:, 2] / co.eV), 
               'o', mew=0, ms=2, c=color)
    pylab.xlabel("$r_\perp$ [m]")
    pylab.ylabel("$p_z$ [eV/c]")
    pylab.semilogy()

if __name__ == '__main__':
    main()
