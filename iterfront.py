from ctypes import *

from numpy import *
import scipy.constants as co
from scipy.interpolate import interp1d
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
    n = 10
    init_output()
    grrr.particle_weight(1.0)

    counter = Counter()
    for i in xrange(n):
        oldt = t

        t = run(start_t=t,
                init_hooks=[],
                inner_hooks=[counter],
                finish_hooks=[])

        growth, b = simple_regression(counter.t[-n:], log(counter.n[-n:]))
        print("growth rate = {:g}/ns".format(growth * co.nano))
        # pylab.figure('growth')
        # pylab.plot(counter.t[-n:], counter.n[-n:], 'o')
        # pylab.plot(counter.t[-n:], exp(growth * counter.t[-n:] + b), lw=1.8)
        # pylab.show()

        update_front(t, (t - oldt) * n, i, growth)
        output(t, (t - oldt) * n)
    pylab.show()


def init_front():
    efield = where(XI >= 0, EB + E0, EB)
    grrr.set_front(XI, efield)

class Counter(object):
    def __init__(self):
        self._t = []
        self._n = []

    @property
    def t(self):
        return array(self._t)

    @property
    def n(self):
        return array(self._n)

    def __call__(self, t, final_t):
        self._t.append(t)
        self._n.append(grrr.particle_count.value * grrr.particle_weight())


def update_front(t, final_t, i, growth):
    global E0, EB

    u = grrr.get_parameter('U0')
    r = grrr.particles_r()
    color = cm.jet(t / final_t)

    zp = r[:, 2] - u * t
    h, a = histogram(zp, bins=XI, density=False)
    h *= grrr.particle_weight()

    dt = 0.0025 * co.nano
    dxi = XI[1] - XI[0]

    # Lets count the positives inside each box in xi.
    # First we obtain the positives in their boxes
    zf, nc = grrr.charge_density(return_faces=True)

    # We now count the charges to the left of a face
    cumn = r_[0.0, cumsum(nc)]
    
    # And interpolate linearly into the cell boundaries of the front array
    dn = interp1d(zf, cumn, fill_value=0, bounds_error=False)(XI + u * t)

    # to obtain the particles inside each box of xi we now make differences
    npos = diff(dn)

    cumh = cumsum(npos - h)

    alpha = 1.0 * co.nano
    E0 = E0 * (1 - alpha * growth)
    efield = EB + E0 * r_[0, cumh] / cumh[-1]
    print("E0 = {:g}".format(E0))

    grrr.set_front(XI, efield)

    savetxt("front_%.3d.dat" % i, c_[XI, r_[0.0, h], efield])
    
    xim = 0.5 * (XI[1:] + XI[:-1])

    z, charge = grrr.charge_density()
    pylab.figure('charge')
    pylab.plot(z, charge, lw=1.5, c=color)

    # fi = grrr.count_collisions(100, t, dt) / dxi
    # npos = r_[dxi * cumsum(fi[::-1])[::-1] / U0, 0.0]
    # nposm = 0.5 * (npos[1:] + npos[:-1])

    # pylab.figure('ions')
    # pylab.plot(xim, nposm, lw=1.5, c=color)

    # pylab.figure('collisions')
    # pylab.plot(xim, fi, lw=1.5, c=color)

    pylab.figure('field')
    pylab.plot(XI, efield, lw=1.5, c=color)

    pylab.figure('ion density')
    pylab.plot(xim, npos, lw=1.5, c=color)

    pylab.figure('electron density')
    pylab.plot(xim, h, lw=1.5, c=color)


def simple_regression(xi, yi):
    A = ones((len(yi), 2), dtype=float)
    A[:, 0] = xi[:]
    
    r = linalg.lstsq(A, yi)
    return r[0][0], r[0][1]
    
if __name__ == '__main__':
    main()
