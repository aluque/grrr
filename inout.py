import sys
import time
import logging
from param import ParamContainer

from numpy import *
import scipy.constants as co
import h5py

from param import ParamContainer, param, positive, contained_in

M = co.electron_mass
MC2 = co.electron_mass * co.c**2
M2C4 = MC2**2

class IOContainer(ParamContainer):
    SAVED_ATTRS = ['id', 'r', 'p', 'eng', 'zfcells', 'r0', 'p0', 't0', 'tau', 
                   'zccells', 'charge', 'ez', 'nelastic', 'nionizing']

    @param(default=0.0)
    def E0(s):
        return float(s)

    @param(default=0.0)
    def EB(s):
        """ Electric field behind the front. """
        return float(s)

    @param(default=co.c)
    def U0(s):
        """ Front velocity (if not self-consistently calculated) """
        return float(s)

    @param(default=0.0)
    def B0(s):
        """ The geomagnetic field. """
        return float(s)

    @param(positive, default=3.0)
    def L(s):
        return float(s)

    @param(default=[])
    def z_wall(s):
        try:
            return [float(s)]
        except TypeError:
            return [float(x) for x in s]
    

    @param(positive, default=0.0025 * co.nano)
    def dt(s):
        """ The timestep of the simulation."""
        return float(s)

    @param(positive, default=250 * co.nano)
    def end_time(s):
        """ The end time of the simulation. """
        return float(s)

    @param(positive, default=20*co.nano)
    def output_dt(s):
        """ The timestep for output data. """
        return float(s)

    @param(positive, default=1.0)
    def init_particle_weight(s):
        """ Initial weight of the superparticles. """
        return float(s)

    @param(default=1.0)
    def init_particle_energy(s):
        """ Initial energy of the particles (in eV). """
        return float(s)

    @param(default=True)
    def randomize_init_energy(s):
        return bool(s)

    @param(default=0.0)
    def init_particle_z(s):
        """ Initial z-location of the particles. """
        return float(s)

    @param(positive, default=2500)
    def max_particles(s):
        """ Max. number of superparticles (will purge when exceeded)."""
        return int(s)

    @param(positive, default=0.75)
    def purge_factor(s):
        """ Purging factor when max_particles is reached. """
        return float(s)

    @param(contained_in({'static', 'wave', 'const', 'pulse',
                         'interf', 'dipole', 'eval', 'front',
                         'selfcons'}), default='selfcons')
    def emfield(s):
        return s


    @param(default=False)
    def track_only_primaries(s):
        return bool(s)

    @param(default=False)
    def delete_at_wall(s):
        return bool(s)

    @param(default=False)
    def first_wall_only(s):
        return bool(s)
    
    @param(positive, default=1000)
    def init_particles(s):
        """ Initial number of particles. """
        return int(s)

    @param(positive, default=2**32)
    def max_particles_cutoff(s):
        """ Stop the simulation if we reach this number of particles. """
        return int(s)

    @param(default=None)
    def avalanches_output(s):
        """ Write in this file whether we reached the avalanche threshold. """
        return s

    @param(default=True)
    def full_output(s):
        """ Write the full output file? """
        return bool(s)
    
    
    def save_to(self, fname):
        """ Sets the file to save data to. """
        self.root = h5py.File(fname, "w")
        self.fname = fname
        self.itimestep = 0
        self.steps = self.root.create_group('steps')

        self.h5_dump(self.root)

        self.root.flush()
        logging.info("File '{}' open for writing.".format(fname))


    def open(self, fname, mode='r'):
        """ Opens a file to read."""
        self.root = h5py.File(fname, mode)
        self.fname = fname
        self.steps = self.root['steps']
        self.h5_load(self.root)
        logging.info("File '{}' open for reading."
                     .format(fname, self._user_, self._host_, self._ctime_))
        logging.info("This simulation was run by {}@{} on {}."
                     .format(self._user_, self._host_, self._ctime_))
        logging.info("The command was '{}'.".format(self._command_))

        self.nsteps = self.root.attrs['nsteps']


    def save(self):
        """ Saves the current status as a timestep. """
        gid = "%.5d" % self.itimestep
        
        g = self.steps.create_group(gid)
        g.attrs['_timestamp_'] = time.time()
        g.attrs['TIME'] = self.TIME
        g.attrs['istep'] = self.itimestep
        g.attrs['particle_weight'] = self.particle_weight()
        g.attrs['nparticles'] = self.nparticles

        for var in self.SAVED_ATTRS:
            value = getattr(self, var)
            g.create_dataset(var, data=value, compression='gzip')
            
        self.itimestep += 1

        self.root.attrs['nsteps'] = self.itimestep
        self.root.flush()
        logging.info("Timestep {} [TIME={}] saved.".format(gid, self.TIME))


    def save_crossings(self, wall, r, p, t, pid):
        """ Save the r, p, t of the crossings in a single group at the end of 
        the simulation. """
        g = self.root.create_group('crossings')

        g.create_dataset('wall', data=wall, compression='gzip')
        g.create_dataset('r', data=r, compression='gzip')
        g.create_dataset('p', data=p, compression='gzip')
        g.create_dataset('t', data=t, compression='gzip')
        g.create_dataset('id', data=pid, compression='gzip')


    def load(self, gid):
        """ Load a given timestep. """
        if gid == "latest":
            gid = self.latest()

        g = self.steps[gid]
        self.TIME = g.attrs['TIME']
        self.istep = g.attrs['istep']
        self.particle_weight = g.attrs['particle_weight']
        self.nparticles = g.attrs['nparticles']

        for var in self.SAVED_ATTRS:
            value = array(g[var])
            setattr(self, var, value)

        self.tfraction = float(self.istep) / self.nsteps
        self.xi = self.r[:, 2] - self.U0 * self.TIME
        self.eng0 = sqrt(co.c**2 * sum(self.p0**2, axis=1) + M2C4) - MC2
        logging.info("Timestep {} [TIME={}] read.".format(gid, self.TIME))


    def latest(self):
        """ Returns the id of the latest timestep. """
        return sorted(self.steps.keys())[-1]


    def spectrum(self, bins=None, density=True):
        """ Calculates the energy spectrum of the data.  Returns
        energies, spectrum density. """

        if bins is None:
            bins = logspace(5.5, 9, 100)

        beam = (self.p[:, 2]**2 / (self.p[:, 0]**2 + self.p[:, 1]**2)) > 1
        h, a = histogram(self.eng[beam] / co.eV, 
                         bins=bins, density=density)
        am = 0.5 * (a[1:] + a[:-1])
        
        return am, h


    def centroid(self, filter=slice(None)):
        """ Locates the centroid of the particles by averaging their locations
        """
        return average(self.r[filter, :], axis=0)


    def rstd(self, filter=slice(None)):
        """ Finds the variance of the locations. """
        return std(self.r[filter, :], axis=0)


    def __iter__(self):
        """ Allows us to iterate over the saved timesteps of the object. """
        for gid in list(self.steps.keys()):
            yield gid
