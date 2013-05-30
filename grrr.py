# This is a thin python wrapper around libgrrr, which contains the
# hard MC computations.  Everything is built around the ctypes 
# module in the python stdlib.
import logging
import logger

from numpy import *
from ctypes import *
import scipy.constants as co

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2

# The library object
grrr = cdll.LoadLibrary("libgrrr.so")
grrr.grrr_init()
logging.debug('libgrrr.so loaded and initialized.')

# The partcle types
ELECTRON, PROTON, PHOTON = 0, 1, 2

# Vectors are stored as 3-arrays
c_vector3 = c_double * 3

# The functions that return em fields have this signature
c_emfunc = CFUNCTYPE(None, c_double, POINTER(c_double), POINTER(c_double), 
                     POINTER(c_double))
c_emfunc_p = POINTER(c_emfunc)

# This is the particle data type.  It is binary-compatible with its C 
# counterpart.  That means that changes in grrr.h must be translated here
class PARTICLE(Structure):
    pass

PARTICLE._fields_ = [('ptype', c_int),
                     ('id', c_int),
                     ('r', c_double *3),
                     ('p', c_double *3),
                     ('tau', c_double),
                     ('dr', c_double *3),
                     ('dp', c_double *3),
                     ('dtau', c_double),
                     ('charge', c_int),
                     ('mass', c_int),
                     ('thermal', c_int),
                     ('t0', c_double),
                     ('r0', c_double *3),
                     ('p0', c_double *3),
                     ('nelastic', c_int),
                     ('nionizing', c_int),
                     ('prev', POINTER(PARTICLE)),
                     ('next', POINTER(PARTICLE))]

# Definition of the argument types to the exported functions
grrr.particle_init.argtypes = [c_int]
grrr.particle_init.restype = POINTER(PARTICLE)
grrr.particle_delete.argtypes = [POINTER(PARTICLE), c_int]
grrr.particle_append.argtypes = [POINTER(PARTICLE), c_int]
grrr.list_step.argtypes = [c_double]
grrr.list_step_n.argtypes = [c_double, c_int]
grrr.list_step_n.restype = c_double
grrr.list_step_n_with_purging.argtypes = [c_double, c_int, 
                                          c_int, c_double]
grrr.list_step_n_with_purging.restype = c_double
grrr.list_purge.argtypes = [c_double]
grrr.list_dump.argtypes = [c_char_p]
grrr.list_clear.argtypes = []
grrr.emfield_eval.argtypes = [c_double, c_vector3, c_vector3, c_vector3]
grrr.emfield_eval.restype = None
grrr.set_emfield_front.argtypes = [c_double, c_int, POINTER(c_double)]
grrr.set_emfield_front.restype = None
grrr.count_collisions.argtypes = [c_int, c_double, c_double, POINTER(c_double)]
grrr.count_collisions.restype = None

# grrr.electromagnetic_interf_field.argtypes = [c_double, c_vector3, 
#                                               c_vector3, c_vector3]
# grrr.electromagnetic_wave_field.argtypes = [c_double, c_vector3, 
#                                             c_vector3, c_vector3]
grrr.total_fd.argtypes = [c_double]
grrr.total_fd.restype = c_double


# These functions are useful outside this module.  The rest is buried in
# the grrr object here.
list_step = grrr.list_step
list_step_n = grrr.list_step_n
list_step_n_with_purging = grrr.list_step_n_with_purging
list_purge = grrr.list_purge
list_dump = grrr.list_dump
list_clear = grrr.list_clear
total_fd = grrr.total_fd
emfield_eval = grrr.emfield_eval
set_emfield_front = grrr.set_emfield_front

# Exported variables describing the particle list
particle_head = POINTER(PARTICLE).in_dll(grrr, 'particle_head')
particle_tail = POINTER(PARTICLE).in_dll(grrr, 'particle_tail')
particle_count = c_int.in_dll(grrr, 'particle_count')

t = 0

emfield_func = c_emfunc_p.in_dll(grrr, 'emfield_func')

# This is to keep a reference to a callback function.  Otherwise
# it will be gc'ed and we will run into memory corruption issues.
_emfield_func = None


def set_parameter(name, value, ctype=c_double):
    """ Sets one parameter with the given value.
    Note that this function may be used to change some other variable, 
    something that may result in unpleasant consequences.  Do not.
    """
    var = ctype.in_dll(grrr, name)
    var.value = value


def get_parameter(name, ctype=c_double):
    """ Gets the parameter with the given name.
    """
    var = ctype.in_dll(grrr, name)
    return var.value


def set_emfield_func(f):
    # See above why we have to use a global variable here.
    global _emfield_func
    if hasattr(f, '__call__'):
        _emfield_func = c_emfunc(f)
        grrr.set_emfield_callback(_emfield_func)
    else:
        newfunc = c_emfunc.in_dll(grrr, 'emfield_' + f)
        emfield_func.contents = newfunc

    

def particle_weight(value=None):
    """ Gets/sets the particle weight. """
    var = c_double.in_dll(grrr, 'particle_weight')
    if value is not None:
        var.value = value

    return var.value


def only_primaries(value=True):
    var = c_int.in_dll(grrr, 'ONLY_PRIMARIES')
    if value is not None:
        var.value = value

    return var.value


def create_particle(ptype, r, p, track=True):
    """ Creates a particle of type ptype and momentum and position puts 
    it in the particle pool. """
    P = grrr.particle_init(ptype)
    P.contents.r[:] = r[:]
    P.contents.p[:] = p[:]
    grrr.particle_append(P, track)


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


def particles_t0():
    """ Returns an array with shape (NPARTICLES,) with the creation time
    of each particle. """
    b = empty((particle_count.value))
    for i, part in enumerate(iter_particles()):
        b[i] = part.t0

    return b


def particles_tau():
    """ Returns an array with shape (NPARTICLES,) with the proper time
    of each particle. """
    b = empty((particle_count.value))
    for i, part in enumerate(iter_particles()):
        b[i] = part.tau

    return b


def particles_id():
    """ Returns an array with shape (NPARTICLES,) with the id
    of each particle. """
    b = empty((particle_count.value), dtype='i')
    for i, part in enumerate(iter_particles()):
        b[i] = part.id

    return b


def particles_r0():
    """ Returns an array with shape (NPARTICLES, 3) with all the particle
    initial locations. """
    r = empty((particle_count.value, 3))
    for i, part in enumerate(iter_particles()):
        r[i, :] = part.r0[:]

    return r


def particles_p0():
    """ Returns an array with shape (NPARTICLES, 3) with the creation momenta
    of the particles. """
    p = empty((particle_count.value, 3))
    for i, part in enumerate(iter_particles()):
        p[i, :] = part.p0[:]

    return p


def particles_collisions():
    """ Returns two arrays with shape (NPARTICLES,) with the number of
    elastic and ionizing collisions that a particle has experienced in
    his lifetime. """
    nelastic = empty((particle_count.value,))
    nionizing = empty((particle_count.value,))

    for i, part in enumerate(iter_particles()):
        nelastic[i] = part.nelastic
        nionizing[i] = part.nionizing

    return nelastic, nionizing


def particles_energy(p=None):
    """ Returns an array the kinetic energies of the particles.  If
    p is None, calls particles_p. """
    if p is None:
        p = particles_p()

    return sqrt(co.c**2 * sum(p**2, axis=1) + M2C4) - MC2


def set_front(xi, efield):
    """ Sets the front shape for emfield_front from xi and the electric field.
    """

    logging.debug("Setting a new field profile.")

    if (xi[0] + xi[-1] > 1e-4):
        raise ValueError("xi must go from -L / 2 to L / 2")

    n = len(xi)

    if (len(efield) != n):
        raise ValueError("efield and xi must have the same length")

    L = xi[-1] - xi[0]

    c_ndouble = c_double * n
    c_front = c_ndouble()
    for i, ef in enumerate(efield):
        c_front[i] = ef

    set_emfield_front(L, n - 1, c_front)


def count_collisions(trials, t, dt):
    ninterp = c_int.in_dll(grrr, 'NINTERP').value

    c_ndouble = c_double * ninterp
    c_count = c_ndouble()
    
    grrr.count_collisions(trials, t, dt, c_count)
    a = empty((ninterp,))
    a[:] = c_count[:]

    return a


def charge_density(return_faces=False):
    ncells = c_int.in_dll(grrr, 'NCELLS').value
    cell_dz = c_double.in_dll(grrr, 'CELL_DZ').value

    faces = linspace(0, ncells * cell_dz, ncells + 1)
    centers = 0.5 * (faces[1:] + faces[:-1])

    charge = empty((ncells, ))

    c_ndouble = c_double * ncells

    c_charge =  c_ndouble.in_dll(grrr, 'fixedcharge')
    charge[:] = c_charge[:]

    c_charge =  c_ndouble.in_dll(grrr, 'mobilecharge')
    charge[:] += c_charge[:]

    if return_faces:
        return faces, charge
    else:
        return centers, charge


def selfcons_field(return_faces=False):
    ncells = c_int.in_dll(grrr, 'NCELLS').value
    cell_dz = c_double.in_dll(grrr, 'CELL_DZ').value

    faces = linspace(0, ncells * cell_dz, ncells + 1)
    centers = 0.5 * (faces[1:] + faces[:-1])

    c_ndouble = c_double * (ncells + 1)
    c_ez =  c_ndouble.in_dll(grrr, 'ez')

    ez = empty((ncells + 1, ))
    ez[:] = c_ez[:]

    if return_faces:
        return faces, ez
    else:
        return centers, ez

