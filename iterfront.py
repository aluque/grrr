import logging

from numpy import *
import scipy.constants as co
from scipy.interpolate import interp1d
import grrr
from runner import Runner
import plotter
import pylab
from matplotlib import cm

logging.basicConfig(format='[%(asctime)s] %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.DEBUG)

NINTERP = 400
L = 100
XI = linspace(-L / 2, L / 2, NINTERP + 1)
EB =  -0 * co.kilo / co.centi
E0 =  -10 * co.kilo / co.centi
BETA = 1 - 0.001182 * 4


def main():
    runner = Runner()

    runner.E0    = E0
    runner.EB    = EB
    runner.B0    =  20 * co.micro
    runner.L     =  30
    runner.U0    =  BETA * co.c
    runner.THETA = 0.0

    runner.list_clear()
    runner.init_list(0, 10, 10000 * co.kilo * co.eV, 2500)
    runner.set_emfield_func('front')

    init_front(runner)

    counter = Counter(runner)

    runner.inner_hooks.append(counter)
    n = 10

    runner.prepare_data(tfraction=0.0)
    plotter.phases(runner)
    plotter.front(runner)
    plotter.field(runner)

    runner.output_n = 1000
    runner.max_particles = 5000

    for i in xrange(n):
        runner(10 * co.nano)

        growth, b = simple_regression(counter.t[-10:], log(counter.n[-10:]))
        logging.info("growth rate = {:g} /ns".format(growth * co.nano))

        update_front(runner, growth)
        tfraction = float(i + 1) / n

        runner.prepare_data(tfraction)
        plotter.phases(runner)
        plotter.histogram(runner)
        plotter.front(runner)
        plotter.field(runner)

    plotter.save_all()
    pylab.show()


def init_front(runner):
    efield = where(XI >= 0, EB + E0, EB)
    runner.set_front(XI, efield)
    

class Counter(object):
    def __init__(self, runner):
        self.runner = runner
        self._t = []
        self._n = []
        self.fp = open("counter.dat", "w")

    @property
    def t(self):
        return array(self._t)

    @property
    def n(self):
        return array(self._n)

    def __call__(self, runner):
        self._t.append(runner.TIME)
        self._n.append(runner.nparticles)
        self.fp.write("{} {}\n".format(runner.TIME, runner.nparticles))
        self.fp.flush()


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

    alpha = 20.0 * co.nano
    E0 = E0 * (1 - alpha * growth)
    efield = EB + E0 * r_[0, cumh] / cumh[-1]
    logging.info("E0 = {:g} kV/cm".format(E0 / (co.kilo / co.centi)))

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
