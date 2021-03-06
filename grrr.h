#ifndef _GRRR_H_

enum particle_type {electron=0, proton, photon};

const int particle_charge[3] = {-1, 1, 0};
const int particle_mass[3] = {1, 1, 0};

typedef struct particle_t {
  enum particle_type ptype;

  /* An unique identified fot the particle. */
  int id;

  /* The momentum, location. */
  double r[3], p[3];
  
  /* We also integrate the particle's proper time.  This is not needed for
     the simulation but it may be useful for the analysis. */
  double tau;

  /* The RK4 algorithm is much easier if we also store here the derivatives
     of r, p and tau. */
  double dr[3], dp[3], dtau;

  /* The charge and mass is set as a multiple of the elementary charge
     and the electron mass. */
  int charge, mass;

  /* We flag a particle as thermalized to ignore it and delete it later. */
  int thermal;

  /* For the anaylisis it is useful to record the creation time and initial
     r and p of particles. */
  double t0, r0[3], p0[3];
  
  /* At some point, I also find useful to cound the number of collision
     that each particle has undergone. */
  int nelastic, nionizing;

  /* The rightmost wall that this particle (or any of its ascendents) has
     crossed. */
  int rightmost_wall;
  
  /* We store all the particles in a doubly-linked list. */
  struct particle_t *prev, *next;

} particle_t;


/* A structure to store wall crossings. */
typedef struct crossing_t {
  enum particle_type ptype;

  /* An unique identified fot the particle. */
  int id;

  /* The id of the wall that has been crossed. */
  int wall;

  /* We store the particle's position and momentum on crossing the wall. */
  double t, r[3], p[3];
  struct crossing_t *next;

} crossing_t;

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
#define HPLANK    6.62606896e-34
#define HBAR      1.0545717253362894e-34
#define HBAR2     (HBAR * HBAR)

/* Some constants appearing in the Coulomb scattering formula. */
#define COMPTON_WAVELENGTH (HPLANK / (M * C))
#define COULOMB_A (183.8 * COMPTON_WAVELENGTH * pow(AIR_Z, -1.0 / 3.0))
#define COULOMB_A2 (COULOMB_A * COULOMB_A)
#define COULOMB_B (0.25 * HBAR2 / COULOMB_A2)
#define COULOMB_P02  (HBAR2 / COULOMB_A2)
#define COULOMB_P0 (sqrt(COULOMB_P02))
#define COULOMB_M 1000

/* Elementary charge and eV, keV, MeV. */
#define ELEMENTARY_CHARGE 1.602176565e-19
#define EV ELEMENTARY_CHARGE
#define KEV (1e3 * EV)
#define MEV (1e6 * EV)

#define EPSILON_0 8.854187817620389e-12

/* Density of atoms in atmospheric air. */
#define AIR_DENSITY (2 * 2.6881960532935158e+25)

/* Average atomic number of atoms in air molecules. */
#define AIR_Z (0.8 * 7 + 0.2 * 8)

/* Average ionization energy of air. */
#define AIR_IONIZATION (85.7 * EV)

/* The prefactor in Bethe's ionization formula. */
#define BETHE_PREFACTOR (2 * PI * AIR_DENSITY * AIR_Z * RE2 * MC2)

/* The prefactor in the Coulomg scattering formula. */
#define COULOMB_PREFACTOR (0.25 * AIR_DENSITY * (AIR_Z * AIR_Z * RE2))

/* The bremmstrahlung linear approx.  We divide by two because the
   calculation assumes that we will multiply by the density of air
   molecules, but we actually multiply by density of atoms. */
#define BSTRAHLUNG_A (1.2744e-28 / 2.0)
#define BSTRAHLUNG_B (-1.0027e-27 * MEV / 2.0)

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
void add_wall(double z);
particle_t *particle_init(int ptype);
void particle_delete(particle_t *part, int track);
void particle_append(particle_t *part, int track);
void particle_birth(particle_t *part);
void count_mobile(particle_t *plist);
void solve_ez(void);
void list_clear(void);
void list_erase(particle_t *plist);

double total_fd(double K);
int drpdt(particle_t *part, double t, double *r, const double *p, 
	  double *dr, double *dp, double *dtau, double h);
void drpdt_all(particle_t *plist, double t, double dt);
int rk4_single(particle_t *part, double t, double dt, int update);
void rk4(double t, double dt);
particle_t *rkstep(particle_t *plist0, particle_t *plist1, double rkfactor);
void drop_thermal(void);
int ionizing_collision(particle_t *part, double dt, double *K1, double *K2);
int elastic_collision(particle_t *part, double dt, double *theta);
void elastic_diffusion(particle_t *part, double dt, double *theta);

void ionizing_momenta(const double *p, double K1, double K2, 
		      double *p1, double *p2);
void elastic_momentum(double *p, double theta, double *pnew);
particle_t *timestep(particle_t *part, double t, double dt);
void sync_list_step(double dt);
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
