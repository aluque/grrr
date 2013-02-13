import sys
import time
import logging

from numpy import *
import h5py

logging.basicConfig(format='[%(asctime)s] %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    level=logging.DEBUG)

class IOContainer(object):
    SAVED_PARAMS = ['E0', 'EB', 'B0', 'L', 'THETA', 'KTH',
                    'NCELLS', 'U0', 'CELL_DZ',
                    'dt', 'output_dt', 'output_n',
                    'max_particles', 'purge_factor']

    SAVED_ATTRS = ['r', 'p', 'eng', 'zfcells', 'zccells', 'charge', 'ez']

    def save_to(self, fname):
        """ Sets the file to save data to. """
        self.root = h5py.File(fname, "w")
        self.fname = fname
        self.itimestep = 0
        self.steps = self.root.create_group('steps')

        # We always write at least these metadata
        self.root.attrs['command'] = ' '.join(sys.argv)
        self.root.attrs['timestamp'] = time.time()
        self.root.attrs['ctime'] = time.ctime()
        self.root.attrs['nsteps'] = 0

        for param in self.SAVED_PARAMS:
            value = getattr(self, param)
            self.root.attrs[param] = value

        self.root.flush()
        logging.info("File '{}' open for writing.".format(fname))

    def open(self, fname):
        """ Opens a file to read."""
        self.root = h5py.File(fname, "r")
        self.fname = fname
        self.steps = self.root['steps']

        for param in self.SAVED_PARAMS + ['nsteps', 'ctime', 'command']:
            value = self.root.attrs[param]
            setattr(self, param, value)

        self.nsteps = self.root.attrs['nsteps']
        logging.info("File '{}' open for reading.".format(fname))


    def save(self):
        """ Saves the current status as a timestep. """
        gid = "%.5d" % self.itimestep
        
        g = self.steps.create_group(gid)
        g.attrs['timestamp'] = time.time()
        g.attrs['TIME'] = self.TIME
        g.attrs['istep'] = self.itimestep

        for var in self.SAVED_ATTRS:
            value = getattr(self, var)
            g.create_dataset(var, data=value, compression='gzip')
            
        self.itimestep += 1
        self.root.attrs['nsteps'] = self.itimestep
        self.root.flush()
        logging.info("Timestep {} [TIME={}] saved.".format(gid, self.TIME))


    def load(self, gid):
        """ Load a given timestep. """
        g = self.steps[gid]
        self.TIME = g.attrs['TIME']
        self.istep = g.attrs['istep']

        for var in self.SAVED_ATTRS:
            value = array(g[var])
            setattr(self, var, value)

        self.tfraction = float(self.istep) / self.nsteps

        logging.info("Timestep {} [TIME={}] read.".format(gid, self.TIME))


    def __iter__(self):
        """ Allows us to iterate over the saved timesteps of the object. """
        for gid in self.steps.keys():
            yield gid
