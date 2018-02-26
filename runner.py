# This module provides a pythonic, high-level object class to run grrr
# simulations.
import sys
import os
import logging
import logger
# Conflicting names with numpy
import random as pyrandom

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



class Runner(IOContainer):
    """ A high-level class to run grrr simulations.  Note that 
    libgrrr is non-reentrant, so different instances of this class
    will interfere with each other. """

    def __init__(self, fname, ofile=None):
        super(Runner, self).__init__()
        self.file_load(fname)

        if ofile is None:
            base, ext = os.path.splitext(fname)
            ofile = base + '.h5'

        if self.full_output:
            self.save_to(ofile)


            
    def init_simulation(self):
        """ Sets the parameters into the libgrrr-space and initializes the
        particle list. """
        for k in self.param_names:
            if k.isupper():
                grrr.set_parameter(k, getattr(self, k))
            logging.debug("Set parameter {} = {}".format(k, getattr(self, k)))

        self.only_primaries(self.track_only_primaries)
        self.set_delete_at_wall(self.delete_at_wall)
        self.set_first_wall_only(self.first_wall_only)
        self.set_emfield_func(self.emfield)
        for z in self.z_wall:
            self.add_wall(z)
            logging.debug("New wall at {}".format(z))
        
        self.list_clear()
        self.particle_weight(self.init_particle_weight)
        self.init_list(self.init_particle_z, self.init_particle_z,
                       self.init_particle_energy, self.init_particles)
        
        self.output_n = int(self.output_dt / self.dt)
        grrr.set_parameter('TIME', 0.0)
        grrr.all_time_max_particles.value = 0

        if isinstance(self.random_seed, int):
            grrr.set_random_seed(self.random_seed)
        elif self.random_seed == 'generate':
            sr = pyrandom.SystemRandom()
            grrr.set_random_seed(int(sr.randint(0, 2**32 - 1)))


    def run(self):
        """ Runs the simulation. """
        self.init_simulation()
        
        while self.TIME <= self.end_time:
            tfraction = self.TIME / self.end_time
            if self.full_output:
                self.prepare_data(tfraction)
                self.save()

            self.advance(self.output_dt)
            if (self.nparticles > self.max_particles_cutoff
                or self.nparticles == 0):
                break

        if self.avalanches_output:
            with open(self.avalanches_output, 'a') as faval:
                faval.write("%d\n" % grrr.all_time_max_particles.value)

        
        if not self.full_output:
            return
        
        tfraction = self.TIME / self.end_time
        self.prepare_data(tfraction)
        self.save()

        rc = self.crossings_r()
        pc = self.crossings_p()
        tc = self.crossings_t()
        wc = self.crossings_wall()
        pid = self.crossings_id()

        self.save_crossings(wc, rc, pc, tc, pid)


    def advance(self, duration):
        """ Runs the simulation for a specified time 'duration'.
        """
        logging.debug("Simulating {:g} ns".format(duration / co.nano))

        grrr.list_step_n_with_purging(self.dt, 
                                      self.output_n, 
                                      self.max_particles,
                                      self.purge_factor)
        self.print_status()


    @property
    def nparticles(self):
        """ The number of (weighted) particles in the simulation. """
        return grrr.particle_count.value * grrr.particle_weight()

    @property
    def nsuper(self):
        return grrr.particle_count.value

    @property
    def TIME(self):
        return grrr.get_parameter('TIME')


    def set_front(self, xi, ez):
        self.front_xi = xi
        self.front_ez = ez
        grrr.set_front(self.front_xi, self.front_ez)


    add_wall             = staticmethod(grrr.add_wall)
    only_primaries       = staticmethod(grrr.only_primaries)
    set_delete_at_wall   = staticmethod(grrr.set_delete_at_wall)
    set_first_wall_only  = staticmethod(grrr.set_first_wall_only)
    set_emfield_func     = staticmethod(grrr.set_emfield_func)
    list_clear           = staticmethod(grrr.list_clear)
    particles_id         = staticmethod(grrr.particles_id)
    particles_p          = staticmethod(grrr.particles_p)
    particles_r          = staticmethod(grrr.particles_r)
    particles_tau        = staticmethod(grrr.particles_tau)
    particles_t0         = staticmethod(grrr.particles_t0)
    particles_p0         = staticmethod(grrr.particles_p0)
    particles_r0         = staticmethod(grrr.particles_r0)
    particles_collisions = staticmethod(grrr.particles_collisions)
    particles_energy     = staticmethod(grrr.particles_energy)
    particle_weight      = staticmethod(grrr.particle_weight)
    charge_density       = staticmethod(grrr.charge_density)
    selfcons_field       = staticmethod(grrr.selfcons_field)
    crossings_r          = staticmethod(grrr.crossings_r)
    crossings_p          = staticmethod(grrr.crossings_p)
    crossings_t          = staticmethod(grrr.crossings_t)
    crossings_wall       = staticmethod(grrr.crossings_wall)
    crossings_id         = staticmethod(grrr.crossings_id)

    def init_list(self, zmin, zmax, emax, n):
        """ Inits a list with n particles randomly located from -zmin to zmin
        and top energy emax. """
        logging.debug("Initializing particle list [n = {}]".format(n))
        E0 = MC2 + emax

        z = random.uniform(zmin, zmax, size=n)
        if self.randomize_init_energy:
            q = random.uniform(0, 1, size=n)
        else:
            q = ones(n)

        r = [array([0.0, 0.0, z0]) for z0 in z]
        p = [array([0, 0, q0 * sqrt(E0**2 - M2C4) / co.c]) for q0 in q]

        for r0, p0 in zip(r, p):
            grrr.create_particle(grrr.ELECTRON, r0, p0)

    
        
    def prepare_data(self, tfraction=None):
        """ Prepares the data for the inner hooks, which usually
        are plotting functions that need r, p, eng etc. """
        self.r = self.particles_r()
        self.p = self.particles_p()
        self.tau = self.particles_tau()
        self.r0 = self.particles_r0()
        self.p0 = self.particles_p0()
        self.eng = self.particles_energy(self.p)
        self.eng0 = self.particles_energy(self.p0)
        self.nelastic, self.nionizing = self.particles_collisions()
        self.xi = self.r[:, 2] - self.U0 * self.TIME
        self.zfcells, self.charge = self.charge_density(return_faces=True)
        self.zccells, self.ez = self.selfcons_field()
        self.t0 = self.particles_t0()
        self.id = self.particles_id()

        if tfraction is None:
            tfraction = ((self.TIME - self.init_time) 
                         / self.end_time - self.init_time)

        self.tfraction = tfraction


    def print_status(self):
        logging.debug("[{0:.2f} ns]: {1:g} particles ({2:d} superparticles)"\
                         .format(self.TIME / co.nano, 
                                 self.nparticles,
                                 self.nsuper))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="The input file (.yaml/.json/.h5)")
    parser.add_argument("-o", "--ofile", 
                        help="Output file (.h5)",
                        default=None)
    parser.add_argument("-N", type=int,
                        help="Run the simulation N times",
                        default=1)

    args = parser.parse_args()

    runner = Runner(args.input, ofile=args.ofile)
    for i in range(args.N):
        runner.run()
