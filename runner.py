# This module provides a pythonic, high-level object class to run grrr
# simulations.
import logging

from numpy import *
import scipy.constants as co
import grrr
from inout import IOContainer

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2

logging.basicConfig(format='[%(asctime)s] %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.DEBUG)

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None 

    def __call__(cls,*args,**kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


def parameter_property(name):
    """ Creates a property that gets/sets a given parameter. """
    def fget(self):
        return grrr.get_parameter(name)

    def fset(self, value):
        grrr.set_parameter(name, value)
    
    fget.__doc__ = "The parameter {} in libgrrr.".format(name)

    return property(fget, fset) 


class Runner(IOContainer):
    """ A high-level class to run grrr simulations.  Note that this
    is a singleton, since you can't have more that one grrr simulation
    in the same process (libgrrr is non-reentrant). """

    __metaclass__ = Singleton

    def __init__(self):
        self.init_hooks = []
        self.inner_hooks = [self.print_status]
        self.finish_hooks = []

        # Here we use some reasonable default values
        self.dt = 0.0025 * co.nano
        self.output_dt = 20 * co.nano
        self.output_n = int(self.output_dt / self.dt)
        self.max_particles = 2500
        self.purge_factor = 0.5

    def __getattr__(self, name):
        """ The parameters not included in an instance are directly
        passed to the global variables of libgrrr. """
        if name.isupper():
            return grrr.get_parameter(name)


    def __setattr__(self, name, value):
        if name.isupper():
            grrr.set_parameter(name, value)
        else:
            object.__setattr__(self, name, value)

    
    @property
    def nparticles(self):
        """ The number of (weighted) particles in the simulation. """
        return grrr.particle_count.value * grrr.particle_weight()

    @property
    def nsuper(self):
        return grrr.particle_count.value


    def set_front(self, xi, ez):
        self.front_xi = xi
        self.front_ez = ez
        grrr.set_front(self.front_xi, self.front_ez)


    set_emfield_func = staticmethod(grrr.set_emfield_func)
    list_clear       = staticmethod(grrr.list_clear)
    particles_p      = staticmethod(grrr.particles_p)
    particles_r      = staticmethod(grrr.particles_r)
    particles_t0     = staticmethod(grrr.particles_t0)
    particles_p0     = staticmethod(grrr.particles_p0)
    particles_r0     = staticmethod(grrr.particles_r0)
    particles_energy = staticmethod(grrr.particles_energy)
    particle_weight  = staticmethod(grrr.particle_weight)
    charge_density   = staticmethod(grrr.charge_density)
    selfcons_field   = staticmethod(grrr.selfcons_field)


    def init_list(self, zmin, zmax, emax, n):
        """ Inits a list with n particles randomly located from -zmin to zmin
        and top energy emax. """
        logging.debug("Initializing particle list [n = {}]".format(n))
        E0 = MC2 + emax

        z = random.uniform(zmin, zmax, size=n)
        q = random.uniform(0, 1, size=n)

        r = [array([0.0, 0.0, z0]) for z0 in z]
        p = [array([0, 0, q0 * sqrt(E0**2 - M2C4) / co.c]) for q0 in q]

        for r0, p0 in zip(r, p):
            grrr.create_particle(grrr.ELECTRON, r0, p0)

    
    def __call__(self, duration):
        """ Runs the simulation for a specified time 'duration'.
        """
        logging.debug("Simulating {:g} ns".format(duration / co.nano))
        for f in self.init_hooks:
            f(self)

        self.init_time = self.TIME
        self.end_time  = self.init_time + duration

        while self.TIME < self.end_time - 1e-5 * co.nano:
            grrr.list_step_n_with_purging(self.dt, 
                                          self.output_n, 
                                          self.max_particles,
                                          self.purge_factor)
            for f in self.inner_hooks:
                f(self)

        for f in self.finish_hooks:
            f(self)

        
    def prepare_data(self, tfraction=None):
        """ Prepares the data for the inner hooks, which usually
        are plotting functions that need r, p, eng etc. """
        self.r = self.particles_r()
        self.p = self.particles_p()
        self.r0 = self.particles_r0()
        self.p0 = self.particles_p0()
        self.eng = self.particles_energy(self.p)
        self.eng0 = self.particles_energy(self.p0)
        self.xi = self.r[:, 2] - self.U0 * self.TIME
        self.zfcells, self.charge = self.charge_density(return_faces=True)
        self.zccells, self.ez = self.selfcons_field()
        self.t0 = self.particles_t0()

        if tfraction is None:
            tfraction = ((self.TIME - self.init_time) 
                         / self.end_time - self.init_time)

        self.tfraction = tfraction


    @staticmethod
    def print_status(sim):
        logging.info("[{0:.2f} ns]: {1:g} particles ({2:d} superparticles)"\
                         .format(sim.TIME / co.nano, 
                                 sim.nparticles,
                                 sim.nsuper))
