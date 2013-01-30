from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm

import grrr

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2


def init_list(zmax, emax, n):
    U0 = MC2 + emax

    z = random.uniform(-zmax, 0, size=n)
    q = random.uniform(0, 1, size=n)

    r = [array([0.0, 0.0, z0]) for z0 in z]
    p = [array([0, 0, q0 * sqrt(U0**2 - M2C4) / co.c]) for q0 in q]

    for r0, p0 in zip(r, p):
        grrr.create_particle(grrr.ELECTRON, r0, p0)


def output(t, final_t):
    p = grrr.particles_p()
    r = grrr.particles_r()
    eng = grrr.particles_energy(p)

    pylab.figure('phases')
    pylab.plot(co.c * p[:, 2] / co.eV, r[:, 2] - co.c * t, 'o',
               c=cm.jet(t / final_t), mew=0, ms=2.0)
    pylab.xlabel("$p_z$ [eV/c]")
    pylab.ylabel("$z - ct$ [m]")
    pylab.semilogx()

    pylab.figure('histogram')
    bins = logspace(5.5, 9, 100)
    h, a = histogram(eng / co.eV, bins=bins, density=True)
    am = 0.5 * (a[1:] + a[:-1])
    flt = h > 0
    pylab.plot(am[flt], h[flt], 'o', mew=0, ms=4, c=cm.jet(t / final_t))
    pylab.xlabel("$E$ [eV]")
    pylab.ylabel("f(E) [1/eV]")
    pylab.loglog()

    pylab.figure('momentum')
    pt = sqrt(p[:, 0]**2 + p[:, 1]**2)
    pylab.plot(co.c * abs(pt) / co.eV, co.c * abs(p[:, 2]) / co.eV, 
               'o', mew=0, ms=2, c=cm.jet(t / final_t))
    pylab.xlabel("$p_\perp$ [eV/c]")
    pylab.ylabel("$p_z$ [eV/c]")
    pylab.loglog()


def main():
    grrr.set_parameter('E0', 10 * co.kilo / co.centi)
    grrr.set_parameter('EBIAS', 0.5)
    grrr.set_parameter('L' , 30.0)
    grrr.set_parameter('B0', 20 * co.micro)

    dt = 0.5 * co.nano
    final_t = 1.0 * co.micro
    output_dt = 10 * co.nano
    output_n = int(output_dt / dt)
    particles_max = 5000

    init_list(50, 1 * co.mega * co.eV, 5000)
    
    t = 0
    weight = 1.0
    purge_factor = 0.5

    while (t <= final_t):
        t = grrr.list_step_n(t, dt, output_n)
        if grrr.particle_count.value > particles_max:
            grrr.list_purge(purge_factor)
            weight /= purge_factor

        print("[{0:.2f} ns]: {1:d} particles ({2:g} weighted)"\
                  .format(t / co.nano, grrr.particle_count.value,
                          weight * grrr.particle_count.value))
        output(t, final_t)

    pylab.show()


if __name__ == '__main__':
    main()
