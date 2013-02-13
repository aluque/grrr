import logging

from numpy import *
import scipy.constants as co
from runner import Runner
import plotter
import pylab
from misc import Counter

logging.basicConfig(format='[%(asctime)s] %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.DEBUG)

EB =  -10 * co.kilo / co.centi
E0 =  -0 * co.kilo / co.centi
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
    runner.particle_weight(1e9);
    runner.init_list(100, 110, 10000 * co.kilo * co.eV, 1000)
    runner.set_emfield_func('selfcons')

    counter = Counter(runner)

    runner.inner_hooks.append(counter)
    n = 50

    runner.prepare_data(tfraction=0.0)
    plotter.phases(runner)
    plotter.selfcons_field(runner)

    runner.output_n = 4000
    runner.max_particles = 5000

    for i in xrange(n):
        runner(50 * co.nano)

        tfraction = float(i + 1) / n

        runner.prepare_data(tfraction)
        plotter.phases(runner)
        plotter.histogram(runner)
        plotter.selfcons_field(runner)

    plotter.save_all()
    pylab.show()


    
if __name__ == '__main__':
    main()
