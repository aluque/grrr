# This is a thin python wrapper around libgrrr, which contains the
# hard MC computations.  Everything is built around the ctypes 
# module in the python stdlib.

from numpy import *
from ctypes import *
import scipy.constants as co

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2

# The library object
grrr = cdll.LoadLibrary("libgrrr.so")

# The partcle types
ELECTRON, PROTON, PHOTON = 0, 1, 2

# Vectors are stored as 3-arrays
c_vector3 = c_double * 3

# This is the particle data type.  It is binary-compatible with its C 
# counterpart.  That means that changes in grrr.h must be translated here
class PARTICLE(Structure):
    pass

PARTICLE._fields_ = [('ptype', c_int),
                     ('r', c_double *3),
                     ('p', c_double *3),
                     ('charge', c_int),
                     ('mass', c_int),
                     ('prev', POINTER(PARTICLE)),
                     ('next', POINTER(PARTICLE))]

# Definition of the argument types to the exported functions
grrr.particle_init.argtypes = [c_int]
grrr.particle_init.restype = POINTER(PARTICLE)
grrr.particle_delete.argtypes = [POINTER(PARTICLE)]
grrr.particle_append.argtypes = [POINTER(PARTICLE)]
grrr.list_step.argtypes = [c_double, c_double]
grrr.list_step_n.argtypes = [c_double, c_double, c_int]
grrr.list_step_n.restype = c_double
grrr.list_purge.argtypes = [c_double]
grrr.list_dump.argtypes = [c_char_p]
grrr.electromagnetic_interf_field.argtypes = [c_double, c_vector3, 
                                              c_vector3, c_vector3]
grrr.electromagnetic_wave_field.argtypes = [c_double, c_vector3, 
                                            c_vector3, c_vector3]

# These functions are useful outside this module.  The rest is buried in
# the grrr object here.
list_step = grrr.list_step
list_step_n = grrr.list_step_n
list_purge = grrr.list_purge
list_dump = grrr.list_dump


# Exported variables describing the particle list
particle_head = POINTER(PARTICLE).in_dll(grrr, 'particle_head')
particle_tail = POINTER(PARTICLE).in_dll(grrr, 'particle_tail')
particle_count = c_int.in_dll(grrr, 'particle_count')

t = 0

def set_parameter(name, value, ctype=c_double):
    """ Sets one parameter with the given value.
    Note that this function may be used to change some other variable, 
    something that may result in unpleasant consequences.  Do not.
    """
    var = ctype.in_dll(grrr, name)
    var.value = value



def create_particle(ptype, r, p):
    """ Creates a particle of type ptype and momentum and position puts 
    it in the particle pool. """
    P = grrr.particle_init(ptype)
    P.contents.r[:] = r[:]
    P.contents.p[:] = p[:]
    grrr.particle_append(P)


def iter_particles():
    """ Iterate over all the particles. """
    ppart = particle_head
    while ppart:
        yield ppart.contents
        ppart = ppart.contents.next


def particles_r():
    """ Returns an array with shape (NPARTICLES, 3) with all the particle
    locations. """
    r = empty((particle_count.value, 3))
    for i, part in enumerate(iter_particles()):
        r[i, :] = part.r[:]

    return r


def particles_p():
    """ Returns an array with shape (NPARTICLES, 3) with all the particle
    momenta. """
    p = empty((particle_count.value, 3))
    for i, part in enumerate(iter_particles()):
        p[i, :] = part.p[:]

    return p


def particles_energy(p=None):
    """ Returns an array the kinetic energies of the particles.  If
    p is None, calls particles_p. """
    if p is None:
        p = particles_p()

    return sqrt(co.c**2 * sum(p**2, axis=1) + M2C4) - MC2
