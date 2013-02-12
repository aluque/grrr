from ctypes import *

from numpy import *
import scipy.constants as co
from scipy.interpolate import interp1d
import grrr
from runner import Runner

import pylab
from matplotlib import cm

NINTERP = 100
XI = linspace(-L / 2, L / 2, NINTERP + 1)
EB =  -0 * co.kilo / co.centi
E0 =  -10 * co.kilo / co.centi


def main():
    runner = Runner()

    runner.B0 =  20 * co.micro
    runner.L  =  30
    runner.BETA = 1 - 0.001182 * 6
    runner.U0  =  BETA * co.c
    runner.THETA = 0.0

    runner.list_clear()
    runner.init_list(0, 10, 10000 * co.kilo * co.eV, 1000)

    init_front(runner)

    counter = Counter(runner)

    runner.inner_hooks.append(counter)
    
    for i in xrange(n):
        runner(100 * co.nano)

        growth, b = simple_regression(counter.t[-n:], log(counter.n[-n:]))
        print("growth rate = {:g}/ns".format(growth * co.nano))

        update_front(t, (t - oldt) * n, i, growth)
        #output(t, (t - oldt) * n)

    pylab.show()


def init_front(runner):
    efield = where(XI >= 0, EB + E0, EB)
    runner.set_front(XI, efield)


class Counter(object):
    def __init__(self, runner):
        self.runner = runner
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
        self._n.append(runner.nparticles)


def update_front(runner, growth):
    global E0, EB
    runner.prepare_data()

    h, a = histogram(runner.xi, bins=XI, density=True)
    h *= runner.nparticles

    dt = 0.0025 * co.nano
    dxi = XI[1] - XI[0]

    # Lets count the positives inside each box in xi.
    # First we obtain the positives in their boxes
    zf, nc = grrr.charge_density(return_faces=True)

    # We now count the charges to the left of a face
    cumn = r_[0.0, cumsum(runner.charge)]
    
    # And interpolate linearly into the cell boundaries of the front array
    ipolator = interp1d(runner.zfcells, cumn, 
                  fill_value=0, bounds_error=False)
    dn = ipolator(XI + runner.U0 * runner.TIME)

    # to obtain the particles inside each box of xi we now make differences
    npos = diff(dn)

    cumh = cumsum(npos - h)

    alpha = 1.0 * co.nano
    E0 = E0 * (1 - alpha * growth)
    efield = EB + E0 * r_[0, cumh] / cumh[-1]
    print("E0 = {:g}".format(E0))

    runner.set_front(XI, efield)


def foo():

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
