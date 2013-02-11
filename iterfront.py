from ctypes import *

from numpy import *
import scipy.constants as co
import grrr
from main import init_list, run, init_output, output

import pylab
from matplotlib import cm

EB =  -0 * co.kilo / co.centi
B0 =  20 * co.micro
E0 =  -10 * co.kilo / co.centi
L  =  30
BETA = 1 - 0.001182 * 6
U0  =  BETA * co.c
THETA = 0.0
NINTERP = 100
XI = linspace(-L / 2, L / 2, NINTERP + 1)


def main():
    grrr.set_parameter('L' , L)
    grrr.set_parameter('B0', B0)
    grrr.set_parameter('E0', E0)
    grrr.set_parameter('U0', U0)
    grrr.set_parameter('EB',  EB)
    grrr.set_parameter('THETA' , THETA)
    grrr.set_parameter('EBWIDTH', 4)
    grrr.set_emfield_func('front')


    grrr.list_clear()
    init_list(0, 10, 10000 * co.kilo * co.eV, 1000)
    init_front()
    t = 0
    n = 25
    init_output()

    for i in xrange(n):
        grrr.particle_weight(1.0)
        oldt = t

        t = run(start_t=t,
                init_hooks=[],
                inner_hooks=[],
                finish_hooks=[])

        update_front(t, (t - oldt) * n, i)
        output(t, (t - oldt) * n)
    pylab.show()


def init_front():
    efield = where(XI >= 0, EB + E0, EB)
    grrr.set_front(XI, efield)


def update_front(t, final_t, i):
    u = grrr.get_parameter('U0')
    r = grrr.particles_r()
    color = cm.jet(t / final_t)

    zp = r[:, 2] - u * t
    h, a = histogram(zp, bins=XI, density=True)
    cumh = cumsum(h)
    efield = EB + E0 * r_[0, cumh] / cumh[-1]
    grrr.set_front(XI, efield)

    savetxt("front_%.3d.dat" % i, c_[XI, r_[0.0, h], efield])

    pylab.figure('field')
    pylab.plot(XI, efield, lw=1.5, c=color)

    pylab.figure('density')
    pylab.plot(XI, r_[0.0, h], lw=1.5, c=color)

    
if __name__ == '__main__':
    main()
