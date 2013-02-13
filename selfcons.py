import sys
import logging

from numpy import *
import scipy.constants as co
import pylab

from misc import Counter
from runner import Runner
import plotter

logging.basicConfig(format='[%(asctime)s] %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.DEBUG)

EB =  -10 * co.kilo / co.centi
BETA = 1 - 0.001182 * 4


def main():
    runner = Runner()

    runner.EB    = EB
    runner.U0    = co.c

    runner.list_clear()
    runner.particle_weight(1e9);
    runner.init_list(100, 110, 10000 * co.kilo * co.eV, 1000)
    runner.set_emfield_func('selfcons')

    counter = Counter(runner)

    runner.save_to(sys.argv[1])

    runner.inner_hooks.append(counter)

    runner.prepare_data(tfraction=0.0)
    plotter.phases(runner)
    plotter.selfcons_field(runner)

    runner.output_n = 4000
    runner.max_particles = 5000

    n = 250
    for i in xrange(n):
        runner(50 * co.nano)

        tfraction = float(i + 1) / n

        runner.prepare_data(tfraction)
        runner.save()

        plotter.phases(runner)
        plotter.histogram(runner)
        plotter.selfcons_field(runner)

    plotter.save_all()
    pylab.show()


    
if __name__ == '__main__':
    main()
