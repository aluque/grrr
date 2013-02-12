# This module provides a pythonic, high-level object class to run grrr
# simulations.

from numpy import *
import scipy.constants as co
import grrr

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2

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


class Runner(object):
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
        self.output_n = int(self.output_dt / dt)
        self.max_particles = 2500

    def __getattr__(self, name):
        """ The parameters not included in an instance are directly
        passed to the global variables of libgrrr. """
        if name.isupper():
            return grrr.get_parameter(name)
        else:
            return object.__getattr__(name)


    def __setattr__(self, name, value):
        if name.isupper():
            grrr.set_parameter(name, value)
        else:
            object.__setattr__(name, value)

    
    @property
    def nparticles(self):
        """ The number of (weighted) particles in the simulation. """
        return grrr.particle_count.value * grrr.particle_weight()

    @property
    def nsuper(self)
        return grrr.particle_count.value

        
    list_clear = grrr.list_clear
    particles_p = grrr.particles_p
    particles_r = grrr.particles_r
    particles_energy = grrr.particles_energy

    
    def init_list(self, zmin, zmax, emax, n):
        """ Inits a list with n particles randomly located from -zmin to zmin
        and top energy emax. """
        U0 = MC2 + emax

        z = random.uniform(zmin, zmax, size=n)
        q = random.uniform(0, 1, size=n)

        r = [array([0.0, 0.0, z0]) for z0 in z]
        p = [array([0, 0, q0 * sqrt(self.U0**2 - M2C4) / co.c]) for q0 in q]

        for r0, p0 in zip(r, p):
            grrr.create_particle(grrr.ELECTRON, r0, p0)

    
    def __call__(self, duration):
        for f in init_hooks:
            f()

        self.init_time = self.TIME
        self.end_time = self.init_time + duration

        while (self.TIME <= self.end_time):
            grrr.list_step_n_with_purging(self.dt, 
                                          self.output_n, 
                                          self.max_particles,
                                          self.purge_factor)
            for f in inner_hooks:
                f(t, final_t)

        for f in finish_hooks:
            f()
        

    def print_status(self):
        print("[{0:.2f} ns]: {1:d} particles ({2:g} superparticles)"\
                  .format(self.TIME / co.nano, 
                          self.nparticles,
                          self.nsuper))
