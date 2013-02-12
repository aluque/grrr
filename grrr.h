#ifndef _GRRR_H_

enum particle_type {electron=0, proton, photon};

const int particle_charge[3] = {-1, 1, 0};
const int particle_mass[3] = {1, 1, 0};

typedef struct particle_t {
  enum particle_type ptype;

  /* The momentum and location. */
  double r[3], p[3];
  
  /* The charge and mass is set as a multiple of the elementary charge
     and the electron mass. */
  int charge, mass;

  /* We store all the particles in a doubly-linked list. */
  struct particle_t *prev, *next;

} particle_t;


/* Signature for the functions that calculate em-fields. */
typedef void (*emfield_func_t)(double, double *, double *, double *);

#define TRUE  1
#define FALSE 0

#define PI 3.141592653589793

/* Electron mass. */
#define M 9.10938291e-31

/* Speed of light. */
#define C 299792458.0
#define C2 (C * C)

/* Electron rest energy. */
#define MC2 (M * C2)
#define M2C4 (MC2 * MC2)

/* Classical electron radius. */
#define RE 2.8179403267e-15
#define RE2 (RE * RE)

/* Planck's constant. */
#define HBAR 1.0545717253362894e-34
#define HBAR2 (HBAR * HBAR)

/* Some constants appearing in the Coulomb scattering formula. */
#define COMPTON_WAVELENGTH (HBAR / (M * C))
#define COULOMB_A (183.8 * COMPTON_WAVELENGTH * pow(AIR_Z, -1.0 / 3.0))
#define COULOMB_A2 (COULOMB_A * COULOMB_A)
#define COULOMB_B (0.25 * HBAR2 / COULOMB_A2)

/* Elementary charge and eV, keV, MeV. */
#define ELEMENTARY_CHARGE 1.602176565e-19
#define EV ELEMENTARY_CHARGE
#define KEV (1e3 * EV)
#define MEV (1e6 * EV)


/* Atmospheric air density. */
#define AIR_DENSITY 2.6867805e+25

/* Average atomic number of air molecules. */
#define AIR_Z (0.8 * 14 + 0.2 * 16)

/* Average ionization energy of air. */
#define AIR_IONIZATION (85.7 * EV)

/* The prefactor in Bethe's ionization formula. */
#define BETHE_PREFACTOR (2 * PI * AIR_DENSITY * AIR_Z * RE2 * MC2)

/* The prefactor in the Coulomg scattering formula. */
#define COULOMB_PREFACTOR (0.25 * AIR_DENSITY * (AIR_Z * AIR_Z * RE2))
#define BSTRAHLUNG_A (1.2744e-28)
#define BSTRAHLUNG_B (-1.0027e-27 * MEV)

#define X 0
#define Y 1
#define Z 2

#define DOT(_X, _Y) ((_X)[0]*(_Y)[0] + (_X)[1]*(_Y)[1] + (_X)[2]*(_Y)[2]) 
#define NORM2(_X) (DOT(_X, _X))
#define ENERGY(_P) (sqrt(NORM2(_P) * C2 + M2C4))
#define KINENERGY(_P) (ENERGY(_P) - MC2)


/* Some useful macros. */
#define warning(...) do{				\
    fprintf (stderr, "%s: Warning: ", invok_name);	\
    fprintf (stderr, ## __VA_ARGS__);			\
  } while(0)

#define fatal(...) do{					\
    fprintf (stderr, "%s: Fatal error: ", invok_name);	\
    fprintf (stderr, ## __VA_ARGS__);			\
    exit(-1);						\
  } while(0)

#define particle_printf_str "{r = [%g, %g, %g], p = [%g, %g, %g]}"
#define particle_printf_args(G_) (G_)->r[X], (G_)->r[Y], (G_)->r[Z], \
    (G_)->p[X], (G_)->p[Y], (G_)->p[Z]
#define vector_printf_str "{%g, %g, %g}"
#define vector_printf_args(G_) (G_)[X], (G_)[Y], (G_)[Z]


/* Function declarations. */
/* grrr.c */
void set_emfield_callback(emfield_func_t ef);
particle_t *particle_init(int ptype);
void particle_delete(particle_t *part, int track);
void particle_append(particle_t *part, int track);
void list_clear(void);

double total_fd(double K);
int drpdt(particle_t *part, double t, double *r, const double *p, 
	  double *dr, double *dp, double h);
int rk4(particle_t *part, double t, double dt, int update);
int ionizing_collision(particle_t *part, double dt, double *K1, double *K2);
int elastic_collision(particle_t *part, double dt, double *theta);
void ionizing_momenta(const double *p, double K1, double K2, 
		      double *p1, double *p2);
void elastic_momentum(double *p, double theta, double *pnew);
particle_t *timestep(particle_t *part, double t, double dt);
void list_step(double dt);
void list_step_n(double dt, int n);
void list_step_n_with_purging(double dt, int n, 
				int max_particles, double fraction);
void list_purge(double fraction);
void list_dump(char *fname);
void count_collisions(int trials, double t, double dt, double *values);

/* misc.c */
void *xmalloc (size_t size);
void *xrealloc (void *ptr, size_t size);
void *xcalloc (size_t count, size_t size);


#define _GRRR_H_
#endif
