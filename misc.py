from numpy import *

class Counter(object):
    def __init__(self, runner):
        self.runner = runner
        self._t = []
        self._n = []
        self.fp = open("counter.dat", "w")

    @property
    def t(self):
        return array(self._t)

    @property
    def n(self):
        return array(self._n)

    def __call__(self, runner):
        self._t.append(runner.TIME)
        self._n.append(runner.nparticles)
        self.fp.write("{} {}\n".format(runner.TIME, runner.nparticles))
        self.fp.flush()
