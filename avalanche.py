from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm

import grrr
from main import run, init_list

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2

#EB =  -5 * co.kilo / co.centi
B0 =  0 * co.micro
E0 =   0 * co.kilo / co.centi

lt, lz, ln = [], [], []

def main():
    grrr.set_parameter('B0', B0)
    grrr.set_parameter('E0', E0)
    grrr.set_emfield_func('const')

    EB = -linspace(4, 25, 16) * co.kilo / co.centi
    L = empty_like(EB)

    for i, iEB in enumerate(EB):
        grrr.set_parameter('EB',  iEB) 
        run(init_hooks=[init],
            inner_hooks=[track],
            finish_hooks=[])

        t, z, n = (array(x) for x in (lt, lz, ln))

        a, b = simple_regression(z, log(n))
        L[i] = 1 / a
        print("L = {} m".format(1 / a))

    savetxt("avalanche_lengths.dat", c_[-EB / (co.kilo / co.centi), L]) 
    pylab.plot(-EB / (co.kilo / co.centi), L, 'o')
    pylab.show()

def simple_regression(xi, yi):
    A = ones((len(yi), 2), dtype=float)
    A[:, 0] = xi[:]
    
    r = linalg.lstsq(A, yi)
    return r[0][0], r[0][1]

def init():
    global ln, lz, lt

    lt, lz, ln = [], [], []
    grrr.list_clear()
    init_list(0, 0, 1000 * co.kilo * co.eV, 10)


def track(t, final_t):
    global ln, lz, lt

    ln.append(grrr.particle_count.value * grrr.particle_weight())
    lt.append(t)

    r = grrr.particles_r()
    
    lz.append(average(r[:, 2]))

    pylab.figure('energy')
    eng = grrr.particles_energy()
    pylab.plot(r[:, 2], eng / co.eV, 'o', mew=0.0, ms=3.0, c='b')

if __name__ == '__main__':
    main()
    
