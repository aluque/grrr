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

E0 =  0 * co.kilo / co.centi
EB =  -6 * co.kilo / co.centi
L = 10000
B0 = 0.0

def main():
    runner = Runner()

    runner.B0    = B0
    runner.E0    = E0
    runner.EB    = EB
    runner.U0    = co.c
    runner.L     = L

    runner.list_clear()
    runner.particle_weight(1e9);
    runner.init_list(0, 10, 10000 * co.kilo * co.eV, 1000)
    runner.set_emfield_func('const')

    counter = Counter(runner)

    runner.save_to(sys.argv[1])

    runner.inner_hooks.append(counter)

    runner.prepare_data(tfraction=0.0)
    plotter.phases(runner)

    runner.output_n = 4000
    runner.max_particles = 5000

    n = 250
    for i in range(n):
        runner(50 * co.nano)

        tfraction = float(i + 1) / n

        runner.prepare_data(tfraction)
        runner.save()

        plotter.phases(runner)
        plotter.histogram(runner)

    plotter.save_all()
    pylab.show()


    
if __name__ == '__main__':
    main()
