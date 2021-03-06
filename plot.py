""" A nice interface to plot any variable in a simulation. """
import logging
import logger

from numpy import *
import scipy.constants as co
import pylab
from matplotlib import cm
from matplotlib.colors import LogNorm

from inout import IOContainer, M, MC2

class Variable(object):
    def __init__(self, func, doc=None, name="", units="?"):
        self.func = func
        self.name = name
        self.units = units
        self.doc = doc

    def __call__(self, sim, flt):
        r = self.func(sim)
        # Apply the filter only if not a scalar
        return r if isscalar(r) else r[flt]
        

VARIABLES = {}
def variable(**kwargs):
    """ A decorator to define variables from functions. """
    def deco(f):
        kwargs.update({'doc': f.__doc__})
        VARIABLES[f.__name__] = Variable(f, **kwargs)
        return f
    return deco


class Filter(object):
    def __init__(self, func, doc="", name=""):
        self.func = func
        self.name = name
        
    def __call__(self, sim):
        return self.func(sim)


FILTERS = {}
def filter(**kwargs):
    """ A decorator to define filters from functions. """
    def deco(f):
        kwargs.update({'doc': f.__doc__})
        try:
            FILTERS[kwargs['name']] = Filter(f, **kwargs)
        except KeyError:
            FILTERS[f.__name__] = Filter(f, **kwargs)

        return f
    return deco


class Plot(object):
    """ This is an abstract class for plotting commands.  You have to
    sub-class this if you want useful functionality. """
    def __init__(self, filters=[], logx=False, logy=False,
                 makefig=True):
        self.color = 'k'
        self.filters = filters
        self.logx = logx
        self.logy = logy
        self.makefig = makefig

    def init(self):
        pass

    def update(self, sim):
        pass

    def finish(self):
        ax = pylab.gca()
        if self.logx:
            ax.set_xscale('log')

        if self.logy:
            ax.set_yscale('log')


    def set_color(self, sim):
        self.color = cm.jet(sim.tfraction)

    def combine_filters(self, sim):
        if not self.filters:
            return s_[:]

        r = self.filters[0](sim)
        for f in self.filters[1:]:
            r = logical_and(r, f(sim))
        
        return r


class PlotXY(Plot):
    """ The class for XY scatter plots. """
    def __init__(self, x, y, joined=False, **kwargs):
        super(PlotXY, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.figure = "{} vs {}".format(x.name, y.name)
        self.joined = joined


    def update(self, sim):
        if self.makefig:
            pylab.figure(self.figure)
        self.set_color(sim)
        flt = self.combine_filters(sim)

        x = self.x(sim, flt)
        y = self.y(sim, flt)

        x, y = congruent(x, y)

        if not self.joined:
            pylab.plot(x, y, 'o', c=self.color, mew=0, mec=self.color,
                       ms=2.0, zorder=10)
        else:
            pylab.plot(x, y, lw=1.5, c=self.color, 
                       zorder=10)


    def finish(self):
        super(PlotXY, self).finish()
        pylab.xlabel("{} [{}]".format(self.x.name, self.x.units))
        pylab.ylabel("{} [{}]".format(self.y.name, self.y.units))


class PlotXYZ(Plot):
    """ The class for XYZ scatter plots (Z is the color). """
    def __init__(self, x, y, z, logz=False, **kwargs):
        super(PlotXYZ, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z
        self.logz = logz

        self.figure = "{} vs {} vs {}".format(x.name, y.name, z.name)
        self.norm = None

    def update(self, sim):
        if self.makefig:
            pylab.figure(self.figure)

        flt = self.combine_filters(sim)

        x = self.x(sim, flt)
        y = self.y(sim, flt)
        z = self.z(sim, flt)
        if self.logz:
            norm = LogNorm()
        else:
            norm = None
        
        pylab.scatter(x, y, faceted=False, s=7.0, c=z, norm=norm)


    def finish(self):
        super(PlotXYZ, self).finish()
        pylab.xlabel("{} [{}]".format(self.x.name, self.x.units))
        pylab.ylabel("{} [{}]".format(self.y.name, self.y.units))

        cbar = pylab.colorbar()
        cbar.set_label("{} [{}]".format(self.z.name, self.z.units))


class PlotHistogram(Plot):
    """ The class for histogram plots. """
    def __init__(self, x, bins=60, joined=True, **kwargs):
        super(PlotHistogram, self).__init__(**kwargs)
        self.x = x
        self.figure = "{}".format(x.name)
        self.bins = bins
        self.joined = joined


    def update(self, sim):
        if self.makefig:
            pylab.figure(self.figure)

        self.set_color(sim)
        flt = self.combine_filters(sim)

        x = self.x(sim, flt)

        try:
            bins = linspace(amin(x), amax(x), self.bins)
        except ValueError:
            # If we start all th particles with the same energy we cannot
            # build a reasonable histogram for that timestep.  In that
            # case we simply do not plot anything
            logging.warn("Skipping histogram for this time")
            return

        self.h, self.a = histogram(x, bins=bins, density=True)
        self.am = 0.5 * (self.a[1:] + self.a[:-1])

        if not self.joined:
            pylab.plot(self.am, self.h, 'o', c=self.color, mew=0, ms=4.0)
        else:
            nz = self.h > 0
            pylab.plot(self.am[nz], self.h[nz], c=self.color, lw=1.75)


    def finish(self):
        super(PlotHistogram, self).finish()
        pylab.xlabel("{} [{}]".format(self.x.name, self.x.units))
        pylab.ylabel("Density [1/({})]".format(self.x.units))



class PlotEvolution(Plot):
    """ The class for plots of a scalar vs time. """
    def __init__(self, x, y, xfunc=None, yfunc=None, joined=True, **kwargs):
        super(PlotEvolution, self).__init__(**kwargs)
        self.x = x
        self.y = y

        self.figure = "{}".format(x.name)
        self.joined = joined
        self.xdata = []
        self.ydata = []
        dfunc = {'std': std, 'avg': average, 'min': amin, 'max': amax,
                 'len': len}

        self.xfunc = dfunc[xfunc] if xfunc else lambda x: x
        self.yfunc = dfunc[yfunc] if yfunc else lambda x: x


    def update(self, sim):
        if self.makefig:
            pylab.figure(self.figure)

        flt = self.combine_filters(sim)
        
        x = self.xfunc(self.x(sim, flt))
        y = self.yfunc(self.y(sim, flt))

        self.xdata.append(x)
        self.ydata.append(y)


    def finish(self):
        super(PlotEvolution, self).finish()

        if not self.joined:
            pylab.plot(array(self.xdata), array(self.ydata), 'o', mew=0, ms=4.0)
        else:
            pylab.plot(array(self.xdata), array(self.ydata), lw=1.5)

        pylab.xlabel("{} [{}]".format(self.x.name, self.x.units))
        pylab.ylabel("{} [{}]".format(self.y.name, self.y.units))


def iter_steps(s):
    """ Iterate over timesteps according to a string that can be
    e.g. '00034..47' or '00087'. """
    
    try:
        fro, to = s.split('..')
    except ValueError:
        yield s
        return

    l = len(fro)
    if ',' in to:
        to, step = to.split(',')
    else:
        step = 1

    ifro, ito, istep = int(fro), int(to), int(step)
    for i in range(ifro, ito, istep):
        yield ("%.*d" % (l, i))



def main():
    import argparse
    def xy(args):
        return PlotXY(VARIABLES[args.x], VARIABLES[args.y],
                      filters=[FILTERS[f] for f in args.filter],
                      joined=args.joined,
                      logx=args.logx, logy=args.logy)

    def xyz(args):
        return PlotXYZ(VARIABLES[args.x], VARIABLES[args.y], VARIABLES[args.z],
                      filters=[FILTERS[f] for f in args.filter],
                      logx=args.logx, logy=args.logy, logz=args.logz)

    def evol(args):
        return PlotEvolution(VARIABLES[args.x], VARIABLES[args.y],
                             filters=[FILTERS[f] for f in args.filter],
                             joined=args.joined,
                             xfunc=args.xfunc, yfunc=args.yfunc,
                             logx=args.logx, logy=args.logy)

    def hist(args):
        return PlotHistogram(VARIABLES[args.x],
                             filters=[FILTERS[f] for f in args.filter],
                             bins=args.bins,
                             joined=args.joined,
                             logx=args.logx, logy=args.logy)


    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", help="Input file")
    parser.add_argument("--steps", "-s",
                        help="Step/steps to plot (empty for all)",
                        default=None)

    parser.add_argument("--logx", action='store_true',
                        help="Logarithmic scale on X")
    parser.add_argument("--logy", action='store_true',
                        help="Logarithmic scale on Y")
    parser.add_argument("--logz", action='store_true',
                        help="Logarithmic scale on Z")

    parser.add_argument("-f", "--filter", 
                        help="Add a particle filter",
                        action='append', default=[], 
                        choices=list(FILTERS.keys()))


    subparsers = parser.add_subparsers()
    
    parser_xy = subparsers.add_parser("xy", help="A XY scatter plot")
    parser_xy.add_argument("x", choices=list(VARIABLES.keys()))
    parser_xy.add_argument("y", choices=list(VARIABLES.keys()))
    parser_xy.add_argument("--joined", 
                           help="Uses lines instead of dots",
                           action='store_true', default=False)
    parser_xy.set_defaults(func=xy)

    parser_xy = subparsers.add_parser("xyz", help="A XYZ scatter plot")
    parser_xy.add_argument("x", choices=list(VARIABLES.keys()))
    parser_xy.add_argument("y", choices=list(VARIABLES.keys()))
    parser_xy.add_argument("z", choices=list(VARIABLES.keys()))
    parser_xy.set_defaults(func=xyz)


    parser_hist = subparsers.add_parser("hist", help="A Histogram")
    parser_hist.add_argument("x", choices=list(VARIABLES.keys()))
    parser_hist.add_argument("--joined", 
                             help="Join the histogram points with lines",
                             action='store_true', default=False)
    parser_hist.add_argument("--bins", 
                             help="Number of histogram bins",
                             type=int, default=60)
    parser_hist.set_defaults(func=hist)


    parser_evol = subparsers.add_parser("evol", help="An evolution plot")
    parser_evol.add_argument("x", choices=list(VARIABLES.keys()))
    parser_evol.add_argument("y", choices=list(VARIABLES.keys()))
    parser_evol.add_argument("--xfunc", choices=['avg', 'min', 'max', 'std',
                                                 'len'],
                             default=None)
    parser_evol.add_argument("--yfunc", choices=['avg', 'min', 'max', 'std',
                                                 'len'],
                             default=None)
    parser_evol.add_argument("--joined", 
                           help="Uses lines instead of dots",
                           action='store_true', default=False)
    parser_evol.set_defaults(func=evol)



    args = parser.parse_args()
    plot = args.func(args)

    ioc = IOContainer()
    ioc.open(args.ifile)

    plot.init()
    steps = ioc if args.steps is None else iter_steps(args.steps)
    for step in steps:
        ioc.load(step)
        plot.update(ioc)

    plot.finish()
    pylab.show()

    
def congruent(x, y):
    """ Makes two variables congruent, i.e. if they are both vectors does
    nothing but it transforms (scalar, vector) or (vector, scalar) to 
    (vector, vector). """
    
    if not isscalar(x) and not isscalar(y):
        return x, y

    if isscalar(x) and not isscalar(y):
        return x + zeros_like(y), y

    if not isscalar(x) and isscalar(y):
        return x, y + zeros_like(x)

    # if both are scalar we also do nothing.
    return x, y



#
# Here we have the list of possible variables
#
@variable(name="$p_z$", units="eV/c")
def pz(sim):
    return co.c * sim.p[:, 2] / co.eV


@variable(name="$p_z/mc$", units="")
def pzmc(sim):
    return sim.p[:, 2] / (M * co.c)

@variable(name=r"$\theta_{min}$", units="")
def thetamin(sim):
    Z_AIR = (0.8 * 7 + 0.2 * 8)
    P0 = Z_AIR**(1./3.) / (4 * pi * 183.8)

    return abs(P0 / (sim.p[:, 2] / (M * co.c)))

@variable(name="$p_x$", units="eV/c")
def px(sim):
    return co.c * sim.p[:, 0] / co.eV

@variable(name="$p_y$", units="eV/c")
def py(sim):
    return co.c * sim.p[:, 1] / co.eV

@variable(name="$p_{0z}$", units="eV/c")
def p0z(sim):
    return co.c * sim.p0[:, 2] / co.eV

@variable(name="$p_{0x}$", units="eV/c")
def p0x(sim):
    return co.c * sim.p0[:, 0] / co.eV

@variable(name="$p_{0y}$", units="eV/c")
def p0y(sim):
    return co.c * sim.p0[:, 1] / co.eV

@variable(name="$\Delta p_z$", units="eV/c")
def dpz(sim):
    return pz(sim) - p0z(sim)

@variable(name="$\Delta p_x$", units="eV/c")
def dpx(sim):
    return px(sim) - p0x(sim)

@variable(name="$\Delta p_y$", units="eV/c")
def dpy(sim):
    return py(sim) - p0y(sim)

@variable(name="$z$", units="m")
def z(sim):
    return sim.r[:, 2]

@variable(name="$\Delta z$", units="m")
def dz(sim):
    return sim.r[:, 2] - sim.r0[:, 2]

@variable(name=r"$\theta$", units="")
def theta(sim):
    return arctan2(sqrt(sim.p[:, 0]**2 + sim.p[:, 1]**2), sim.p[:, 2])

@variable(name=r"$\theta_0$", units="")
def theta0(sim):
    return arctan2(sqrt(sim.p0[:, 0]**2 + sim.p0[:, 1]**2), sim.p0[:, 2])

@variable(name=r"$\Delta\theta$", units="")
def dtheta(sim):
    return theta(sim) - theta0(sim)

@variable(name="$x$", units="m")
def x(sim):
    return sim.r[:, 0]

@variable(name="$y$", units="m")
def y(sim):
    return sim.r[:, 1]

@variable(name="$r$", units="m")
def r(sim):
    return sqrt(sum(sim.r[:, (0, 1)]**2, axis=1))

@variable(name="$\gamma$", units="")
def gamma(sim):
    return sqrt(1 + sum(sim.p**2, axis=1) / (M * MC2))

@variable(name="$\gamma_0$", units="")
def gamma0(sim):
    return sqrt(1 + sum(sim.p0**2, axis=1) / (M * MC2))

@variable(name=r"$\beta_z$", units="")
def betaz(sim):
    return sim.p[:, 2] / gamma(sim) / (M * co.c)

@variable(name=r"$\beta_z-\beta_{bulk}$", units="")
def dbetaz(sim):
    return betaz(sim) - sim.U0 / co.c

@variable(name=r"$\beta_x$", units="")
def betax(sim):
    return sim.p[:, 0] / gamma(sim) / (M * co.c)

@variable(name=r"$\beta_y$", units="")
def betay(sim):
    return sim.p[:, 1] / gamma(sim) / (M * co.c)

@variable(name=r"$\beta_{0z}$", units="")
def beta0z(sim):
    return sim.p0[:, 2] / gamma0(sim) / (M * co.c)

@variable(name=r"$\beta_{0x}$", units="")
def beta0x(sim):
    return sim.p0[:, 0] / gamma0(sim) / (M * co.c)

@variable(name="$p^2$", units="(eV/c)$^\mathdefault{2}$")
def p2(sim):
    return sum(sim.p**2, axis=1) * (co.c / co.eV)**2

@variable(name="$p$", units="eV/c")
def pabs(sim):
    return sqrt(sum(sim.p**2, axis=1)) * (co.c / co.eV)

@variable(name="$z - ut$", units="m")
def xi(sim):
    return sim.r[:, 2] - sim.U0 * sim.TIME  # - sim.init_particle_z

@variable(name="$z - ct$", units="m")
def zct(sim):
    return sim.r[:, 2] - co.c * sim.TIME - sim.init_particle_z

@variable(name="$z_0 - ut_0$", units="m")
def xi0(sim):
    return sim.r0[:, 2] - sim.U0 * sim.t0

@variable(name="$-\Delta xi$", units="m")
def dxi(sim):
    return xi0(sim) - xi(sim)

@variable(name="$z - z_{bulk}$", units="m")
def xi_centroid(sim):
    return sim.r[:, 2] - sim.centroid()[2]

@variable(name="$K$", units="eV")
def energy(sim):
    return sim.eng / co.eV

@variable(name="$\rangle F \langle$", units="eV/m")
def average_force(sim):
    dp = sqrt(sum(sim.p**2, axis=1)) - sqrt(sum(sim.p0**2, axis=1))
    return (dp / (sim.TIME - sim.t0)) / co.eV

@variable(name="$K_0$", units="eV")
def energy0(sim):
    return sim.eng0 / co.eV

@variable(name="$t_0$", units="ns")
def t0(sim):
    return sim.t0 / co.nano

@variable(name=r"$\tau$", units="ns")
def tau(sim):
    return sim.tau / co.nano

@variable(name="$t - t_0$", units="ns")
def age(sim):
    return (sim.TIME - sim.t0) / co.nano

@variable(name="$z - z_0$", units="m")
def deltaz(sim):
    return (sim.r[:, 2] - sim.r0[:, 2])

@variable(name="$z$", units="m")
def zf(sim):
    return sim.zfcells

@variable(name="$z$", units="m")
def zc(sim):
    return sim.zccells

@variable(name="$q$", units="e/m$^\mathdefault{2}$")
def charge(sim):
    return sim.charge

@variable(name=r"$\xi$", units="m")
def xif(sim):
    return sim.zfcells - sim.TIME * sim.U0 # - sim.init_particle_z

@variable(name="$E$", units="kV/cm")
def field(sim):
    return sim.ez / (co.kilo / co.centi)

@variable(name="$t$", units="ns")
def t(sim):
    return sim.TIME / co.nano

@variable(name="$N$", units="")
def nparticles(sim):
    return sim.particle_weight * len(sim.eng)

@variable(name=r"$\langle z \rangle$", units="m")
def centroid(sim):
    return sim.centroid()[2]

@variable(name=r"$\langle \xi \rangle$", units="m")
def xicentroid(sim):
    return sim.centroid()[2] - sim.TIME * sim.U0

@variable(name=r"$\langle z \rangle$", units="m")
def shifted_centroid(sim):
    return sim.centroid()[2] - sim.init_particle_z

@variable(name="#elastic", units="")
def nelastic(sim):
    return sim.nelastic

@variable(name="#ionizing", units="")
def nionizing(sim):
    return sim.nionizing

@variable(name=r"$\langle \beta_z \rangle$", units="")
def average_betaz(sim):
    return average(sim.p[:, 2] / gamma(sim) / (M * co.c))

# These are Lorentz-transformed quantities to the co-moving frame.
@variable(name=r"$z'$", units="m")
def zprime(sim):
    gamma = 1 / sqrt(1 - (sim.U0 / co.c)**2)
    dz = sim.r[:, 2] - sim.init_particle_z
    return gamma * (dz - sim.U0 * sim.TIME)

@variable(name=r"$z'$", units="m")
def z0prime(sim):
    gamma = 1 / sqrt(1 - (sim.U0 / co.c)**2)
    dz = sim.r0[:, 2] - sim.init_particle_z
    return gamma * (dz - sim.U0 * sim.t0)

# These are Lorentz-transformed quantities to the co-moving frame.
@variable(name=r"$\Delta z'$", units="m")
def dzprime(sim):
    gamma = 1 / sqrt(1 - (sim.U0 / co.c)**2)
    dz0 = sim.r0[:, 2] - sim.init_particle_z
    dz1 = sim.r[:, 2] - sim.init_particle_z
    return gamma * ((dz1 - sim.U0 * sim.TIME) - (dz0 - sim.U0 * sim.t0))


# Remember that in a Lorentz-transformation the new time becomes dependent on
# on the position, so not now not all particles have the same time.
@variable(name=r"$t'$", units="ns")
def tprime(sim):
    gamma = 1 / sqrt(1 - (sim.U0 / co.c)**2)
    dz = sim.r[:, 2] - sim.init_particle_z
    return gamma * (sim.TIME - sim.U0 *  dz / (co.c**2)) / co.nano


@variable(name=r"$\Delta t'$", units="ns")
def dtprime(sim):
    gamma = 1 / sqrt(1 - (sim.U0 / co.c)**2)
    dz0 = sim.r0[:, 2] - sim.init_particle_z
    dz1 = sim.r[:, 2] - sim.init_particle_z
    return gamma * ((sim.TIME - sim.U0 *  dz1 / (co.c**2))
                    - (sim.t0 - sim.U0 *  dz0 / (co.c**2))) / co.nano


@variable(name=r"$\xi'$", units="m")
def xifprime(sim):
    gamma = 1 / sqrt(1 - (sim.U0 / co.c)**2)
    xif = sim.zfcells - sim.init_particle_z - sim.U0 * sim.TIME
    return gamma * xif



#
# Here we have the list of possible filters
#
@filter(name="primaries")
def primaries(sim):
    return sim.t0 == 0


@filter(name="nonprimaries")
def primaries(sim):
    return sim.t0 != 0


@filter(name="hi1mev")
def hi1mev(sim):
    return sim.eng >= 1 * co.mega * co.eV


@filter(name="hi10mev")
def hi10mev(sim):
    return sim.eng >= 10 * co.mega * co.eV


@filter(name="hi50mev")
def hi50mev(sim):
    return sim.eng >= 50 * co.mega * co.eV


@filter(name="lo1mev")
def lo1mev(sim):
    return sim.eng < 1 * co.mega * co.eV


@filter(name="lo10mev")
def lo10mev(sim):
    return sim.eng < 10 * co.mega * co.eV


@filter(name="lo50mev")
def lo50mev(sim):
    return sim.eng < 50 * co.mega * co.eV



if __name__ == '__main__':
    main()
