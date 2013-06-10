/*** 
   This is the main module of the GRanada Relativistic Runaway (GRRR)
   simulator.
   
   Alejandro Luque Estepa, 2013
***/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "grrr.h"

static double truncated_bethe_fd(double gamma, double gamma2, double beta2);
static double bremsstrahlung_fd(double gamma);
static double cross(const double *a, const double *b, double *r);
static void rotate(double theta, const double *v, double *r);
static int argmax(const double *p, int sign, int n);
static void perp_unit_vectors(const double p[3], double pabs, double up[3],
			      double a[3], double b[3]);
static double moller_differential(double gamma, double gamma2, double beta2,
				  double Kp);
static double coulomb_differential(double gamma2, double beta2, double p2, 
				   double rtheta);
static void particle_track_charge(particle_t *part, int change);
static double rnd_gauss(double mu, double sigma);
static double ranf (void);

/* Several pre-defined em fields. */
void emfield_static    (double t, double *r, double *e, double *b);
void emfield_wave      (double t, double *r, double *e, double *b);
void emfield_const     (double t, double *r, double *e, double *b);
void emfield_pulse     (double t, double *r, double *e, double *b);
void emfield_interf    (double t, double *r, double *e, double *b);
void emfield_dipole    (double t, double *r, double *e, double *b);
void emfield_eval      (double t, double *r, double *e, double *b);
void emfield_front     (double t, double *r, double *e, double *b);
void emfield_selfcons  (double t, double *r, double *e, double *b);

void set_emfield_front(double l, int ninterp, double *values);


/* We will keep a global pointer to the beginning of the particle list
   b.c. the first particle may get thermal and dissapear. */
particle_t *particle_head = NULL, *particle_tail = NULL;

/* We also have a counter of the total number of particles. */
int particle_count = 0;

/* When we use super-particles, we track the particle weight. */
double particle_weight = 1.0;

/* Name of this program. */
const char *invok_name = "grrr";

/* These are parameters.  Later they have to be set at run-time. */
double E0 = 7.0e5;
double U0 = 0.0;
double THETA = PI / 4;
double EB = 0.0;
double EBWIDTH = 0.0;
double B0 = 20e-6;
double KTH = 0.01 * MEV;
double GAMMATH;
double L = 3.0;
int NINTERP = 0;
double *INTERP_VALUES = NULL;

/* If this flag is set we ignore all secondary particles. */
int ONLY_PRIMARIES = 0;

/* This is an unphysical cutoff to try to model more easily. */
double FD_CUTOFF = 100000 * MEV;

/* We will track the creation/destruction of particles to keep track
   of the charge density. */
#define __NCELLS 32768
const int NCELLS = __NCELLS;
const double CELL_DZ = 0.25;

/* fixedcharge counts the number of "fixed" charge-carriers inside a cell.
   That includes ions and thermalized electrons, which, for our purposes
   are also considered immobile. */
double fixedcharge[__NCELLS];

/* mobilecharge counts the number of mobile charges inside each cell. */
double mobilecharge[__NCELLS];

/* And this contains the collective ez, calculated in ez_integrate.
   Note that ez is evaluated in the cell boundaries and therefore
   contains NCELLS+1 numbers. */
double ez[__NCELLS + 1];

/* We track_time inside the C module with this global.  Note that t is only
 read or changed in the high-level functions list_step* (actually, only
 in list_step(...), since the others call this one).  Thus you can use
 lower-level functions without side-effects. */
double TIME = 0;

/* The function that computes em fields at any time in any point. */
emfield_func_t emfield_func = &emfield_static;

void call_emfield(emfield_func_t ef);

void 
grrr_init(void)
/* Initializations must come here. */
{
  int i;
  for (i = 0; i < NCELLS; i++) {
    fixedcharge[i] = 0.0;
  }
  TIME = 0;
}


void
set_emfield_callback(emfield_func_t ef)
/* Sets a function to calculate the em fields. */
{
  emfield_func = ef;
}

void
emfield_eval(double t, double *r, double *e, double *b)

/* Evaluates the electric field at a given point.  This is needed only
 if you want to evaluates em fields directly from python.  I am unable
 to do this using only ctypes.*/
{
  (*emfield_func) (t, r, e, b);
}

particle_t*
particle_init(int ptype)
/* Initializes a particle of the provided type.  The rest of the elements
   are set to 0. */
{
  particle_t *part;
  static int nid = 0;

  part = xcalloc(1, sizeof(particle_t));

  part->ptype = ptype;
  part->id = nid++;
  part->charge = particle_charge[ptype];
  part->mass = particle_mass[ptype];
  part->next = NULL;
  part->thermal = FALSE;
  
  return part;
}

void
particle_delete(particle_t *part, int track)
/* Deletes a particle taking care of the prev/next links and freeing
   memory. */
{
  if (part->prev != NULL) {
    part->prev->next = part->next;
  } else {
    /* We are at the beginning of the list and we have to update the
       pointer to the first particle. */
    particle_head = part->next;
  }

  if (part->next != NULL) {
    part->next->prev = part->prev;
  } else {
    /* We are at the end of the list. */
    particle_tail = part->prev;
  }

  if (track) particle_track_charge(part, -1);
  free(part);

  particle_count--;
}

void
particle_append(particle_t *part, int track)
/* Appends a particle to the end of the list. */
{
  part->next = NULL;
  part->prev = particle_tail;
  if (particle_tail != NULL) {
    particle_tail->next = part;
  }
  particle_tail = part;

  /* If there were no particles on the list, we put newpart as the head. */
  if (particle_head == NULL) {
    particle_head = part;
  }
  
  if (track) particle_track_charge(part, 1);
  particle_count++;
}

void
particle_birth(particle_t *part)
/* Sets the current t, r, p as the t0, r0 and p0 of a newly created particle */
{
  int i;

  part->t0 = TIME;
  part->tau = 0;

  part->nelastic = 0;
  part->nionizing = 0;

  for (i = 0; i < 3; i++) {
    part->r0[i] = part->r[i];
    part->p0[i] = part->p[i];    
  }
}


static void
particle_track_charge(particle_t *part, int change)
/* Tracks the creation / destruction of a particle into the charge array.
   change must be 1 when you create one particle, -1 when you destroy it. */
{
  int n;

  n = (int) floor(part->r[Z] / CELL_DZ);
  if (n < 0 || n >= NCELLS) {
    fprintf (stderr, 
	     "%s: Warning: particle at z=%g m outside tracking range.\n",
	     invok_name, part->r[Z]);
    return;
  }

  fixedcharge[n] += -particle_weight * part->charge * change;
}

void
count_mobile(particle_t *plist)
/* Counts the number of mobile particles inside each cells.  Updates
   mobilecharge. */
{
  particle_t *part;
  int n;

  memset(mobilecharge, 0, NCELLS * sizeof(double));
  
  for (part = plist; part; part = part->next) {
    n = (int) floor(part->r[Z] / CELL_DZ);
    mobilecharge[n] += particle_weight * part->charge;
  }
}

void
solve_ez(void)
/* Solves ez from fixedcharge and mobilecharge.  The boundary condition
   here is always ez->EB for z->infinity.  If you want a different b.c.
   you have to add a constant to ez.
*/
{
  int i;
  double field = EB;

  for (i = 0; i < NCELLS + 1; i++) {
    ez[NCELLS - i] = field;
    if (NCELLS - 1 - i >= 0) {
      field -= ((ELEMENTARY_CHARGE / EPSILON_0)
		* (mobilecharge[NCELLS - 1 - i] + fixedcharge[NCELLS - 1 - i]));
    }
  }
}

void
list_clear(void)
/* Deletes all particles from the list and releases their memory. */
{
  while (particle_head != NULL) {
    particle_delete(particle_head, FALSE);
  }
}

void
list_erase(particle_t *plist)
/* Erases all particles in a list, but does not change the main list. */
{
  particle_t *part, *partnext;

  for (part = plist; part; part = partnext) {
    partnext = part->next;
    free(part);
  }
}

void
emfield_static(double t, double *r, double *e, double *b)
/* Calculates the electromagnetic field at a given time and location.
   Returns it into the e and b pointers. */
{
  b[X] = b[Z] = 0.0;
  b[Y] = B0;

  e[X] = e[Y] = 0.0;
  e[Z] = -E0 * cos(2 * PI * r[Z] / L) + EB; 
}


void
emfield_wave(double t, double *r, double *e, double *b)
/* Calculates the electromagnetic field at a given time and location.
   Returns it into the e and b pointers. */
{
  double cosphi;

  cosphi = cos(2 * PI * (r[Z] - C * t) / L);

  e[X] = 0.0;
  e[Y] = E0 * cosphi;
  e[Z] = 0.0;
  //e[Z] = -EB * exp(-(r[X] * r[X] + r[Y] * r[Y]) / (EBWIDTH*EBWIDTH));

  b[X] = -E0 * cosphi / C;
  b[Y] = 0.0;
  b[Z] = 0.0;

}


void
emfield_const(double t, double *r, double *e, double *b)
{
  e[X] = 0.0;
  e[Y] = 0.0; 
  e[Z] = EB;

  b[X] = B0;
  b[Y] = 0.0;
  b[Z] = 0.0;
}


void
emfield_pulse(double t, double *r, double *e, double *b)
{
  e[X] = 0.0;
  e[Y] = 0.0; 
  b[X] = 0.0;
  b[Y] = 0.0;
  b[Z] = 0.0;

  e[Z] =  (r[Z] - U0 * t > L || r[Z] - U0 * t < 0)? E0: EB;
}

void
emfield_front(double t, double *r, double *e, double *b)
{
  double xi, x, dx, e1, e2;
  int n; 

  e[X] = 0.0;
  e[Y] = 0.0; 
  b[X] = 0.0;
  b[Y] = 0.0;
  b[Z] = 0.0;
  
  xi = r[Z] - U0 * t;
  if(!isfinite(xi)) {
    fprintf(stderr, "%s: xi is NaN encountered in emfield_front.\n", 
	    invok_name);
    exit(-1);
  }

  if (xi <= -L / 2) {
    e[Z] = INTERP_VALUES[0];
    return;
  }

  if (xi >= L / 2) {
    e[Z] = INTERP_VALUES[NINTERP];
    return;
  }

  x = NINTERP * (xi + L / 2) / L;
  n = (int) floor(x);
  dx = x - n;

  e1 = INTERP_VALUES[n];
  e2 = INTERP_VALUES[n + 1];

  e[Z] = e1 + dx * (e2 - e1);
}


void
emfield_selfcons(double t, double *r, double *e, double *b)
{
  double x, dx;
  int n; 

  e[X] = 0.0;
  e[Y] = 0.0; 
  b[X] = 0.0;
  b[Y] = 0.0;
  b[Z] = 0.0;
  
  if (r[Z] < 0) {
    e[Z] = ez[0];
    return;
  }

  if (r[Z] >= NCELLS * CELL_DZ) {
    e[Z] = ez[NCELLS];
    return;
  }
  x = r[Z] / CELL_DZ;
  n = (int) floor(x);
  dx = x - n;

  e[Z] = ez[n] + dx * (ez[n + 1] - ez[n]);
}


void 
set_emfield_front(double l, int ninterp, double *values)
/* Sets the shape of the front that will be interpolated from in
   emfield_front.
   * l is the width of the front
   * ninterp is the number of interpolating segments from -l/2 to l/2.
   * values is an array with at least ninterp + 1 values. */
{
  int i;

  L = l;
  NINTERP = ninterp;
  if (INTERP_VALUES != NULL) {
    free(INTERP_VALUES);
  }

  INTERP_VALUES = xmalloc((ninterp + 1) * sizeof(double));
  memcpy(INTERP_VALUES, values, sizeof(double) * (ninterp + 1));
}


void 
count_collisions(int trials, double t, double dt, double *values)
/* Counts the number of ionizing collisions minus thermalizations
   per unit time in the front.
   It counts the number of collisions during a time dt and
   returns an average over n trials. */
{
  int i, n;
  double xi, x, dn, theta;
  particle_t *part;

  memset(values, 0, NINTERP * sizeof(double));
  dn = 1.0 / trials / dt;

  /* First the ionizing collisions. */
  for (part = particle_head; part; part = part->next) {
    int elastic;

    xi = part->r[Z] - U0 * t;
    if (xi < -L / 2 || xi > L / 2) continue;
	
    x = NINTERP * (xi + L / 2) / L;
    n = (int) floor(x);

    elastic = elastic_collision(part, dt, &theta);
    if (elastic) {
      elastic_momentum(part->p, theta, part->p);
    }

    if (rk4_single(part, t, dt, FALSE)) {
      values[n] -= 1.0 / dt;
    }

    for (i = 0; i < trials; i++) {
      if (ionizing_collision(part, dt, NULL, NULL)) {
	values[n] += dn;
      }
    }
  }
}

void
emfield_interf(double t, double *r, double *e, double *b)
/* Calculates the electromagnetic field at a given time and location.
   Returns it into the e and b pointers. */
{
  double rprime[3], et[3], bt[3], e1[3], e2[3], b1[3], b2[3];
  int i;

  rotate(THETA, r, rprime);

  emfield_wave(t, rprime, et, bt);
  rotate(-THETA, et, e1);
  rotate(-THETA, bt, b1);

  rotate(-THETA, r, rprime);
  emfield_wave(t, rprime, et, bt);
  rotate(THETA, et, e2);
  rotate(THETA, bt, b2);

  for (i = 0; i < 3; i++) {
    e[i] = e1[i] - e2[i];
    b[i] = b1[i] - b2[i];
  }

  e[Z] += EB;
  b[X] += B0;

  // b[X] = b[Y] = b[Z] = 0;

}

void
emfield_celestin(double t, double *r, double *e, double *b)
/* Calculates the electromagnetic field at a given time and location.
   Returns it into the e and b pointers. */
{
  double a, a2;
  int i;

  a2 = NORM2(r);
  a = sqrt(a2);
  
  for (i = 0; i < 3; i++) {
    b[i] = 0;
    e[i] = (r[i] / a) * E0 / a;
  }
  
}

void
emfield_dipole(double t, double *r, double *e, double *b)
/* The EM field created by a dipole at r=0,0,0 */
{
  // Here a is the abs of r, but the r var is already taken.
  double a, a2, a3;
  double l, l2, l3;
  double sinphi, cosphi;

  a2 = NORM2(r);
  a = sqrt(a2);
  a3 = a * a2;

  l = L / a;
  l2 = l * l;
  l3 = l * l2;

  /* The prototype of sincos does not appear in math.h.
  sincos((a - C * t) / L, &sinphi, &cosphi);
  */
  sinphi = sin((a - C * t) / L);
  cosphi = cos((a - C * t) / L);

  e[X] = (E0 / a2) * ((-r[X] * r[Z] * l + 3 * r[X] * r[Z] * l3) * cosphi
		      + ( 3 * r[X] * r[Z] * l2) * sinphi);
  e[Y] = (E0 / a2) * ((-r[Y] * r[Z] * l + 3 * r[Y] * r[Z] * l3) * cosphi
		      + ( 3 * r[Y] * r[Z] * l2) * sinphi);
  e[Z] = (E0 / a2) * (((r[X] * r[X] + r[Y] * r[Y]) * l 
		        + 3 * (r[Z] * r[Z] + a2)  * l3) * cosphi
		      + ( 3 * (r[Z] * r[Z] + a2) * l2) * sinphi);

  b[X] =  (E0 / a / C) * l * r[Y] * (cosphi - l * sinphi);
  b[Y] = -(E0 / a / C) * l * r[X] * (cosphi - l * sinphi);
  b[Z] = 0.0;

}


static double
truncated_bethe_fd(double gamma, double gamma2, double beta2)
{
  double A, T1, T2, T3, T5, T6;

  A = BETHE_PREFACTOR / beta2;
  T1 = log(M2C4 * (gamma2 - 1) * (gamma - 1) / (AIR_IONIZATION*AIR_IONIZATION));
  T1 -= log(MC2 * (gamma - 1) / KTH / 2);
  
  T2 = -(1 + 2 / gamma - 1 / gamma2) * log(2);
  T2 -= -((1 + 2 / gamma - 1 / gamma2)
	  * log(2 * (MC2 * (gamma - 1) - KTH)
		/ (MC2 * (gamma - 1))));

  T3 = 1 / gamma2;

  T5 = - (1 - KTH / (MC2 * (gamma - 1) - KTH));
  T6 = + KTH*KTH / (2 * M2C4 * gamma2);

  /* The total force on the particle is */
  return  A * (T1 + T2 + T3 + T5 + T6);
}

static double
moller_differential(double gamma, double gamma2, double beta2,
		    double Kp)
/* The moller differential cross-section. */
{
  double A, T1, T2, T3;

  A = BETHE_PREFACTOR / beta2;

  T1 = ((gamma - 1)*(gamma - 1) * M2C4 
	/ (Kp*Kp * pow(MC2 * (gamma - 1) - Kp, 2)));
  T2 = -((2 * gamma2 + 2 * gamma - 1) 
	 / (Kp * (MC2 * (gamma - 1) - Kp) * gamma2));
  T3 = 1. / (M2C4 * gamma2);

  return  A * (T1 + T2 + T3);
}


static double
coulomb_differential(double gamma2, double beta2, double p2, 
		     double theta)
/* The formula for the elastic coulomb scattering. */
{
  double A, T, sin2;
  
  sin2 = sin(theta / 2);
  sin2 *= sin2;

  A = COULOMB_PREFACTOR / (beta2 * beta2) / gamma2;
  T = (1. - beta2 * sin2) / pow(sin2 + COULOMB_B / p2, 2);
  
  /* The 2 pi sin(theta) comes from
     d\Omega = 2 pi sin(theta) dtheta. */
  return  A * T * 2 * PI * sin(theta);
}


static double
bremsstrahlung_fd(double gamma)
/* The stopping power of Bremsstrahlung radiation.  We use a linear
   approximation that works quite well for energies above 10 MeV.
   For lower energies Bremsstrahlung is anyway negligible against
   collisional stopping. */
{
  double K;

  K = MC2 * (gamma - 1);
  return AIR_DENSITY * (BSTRAHLUNG_A * K + BSTRAHLUNG_B);
}


static double
cross(const double *a, const double *b, double *r)
{
  /* {-az by + ay bz, az bx - ax bz, -ay bx + ax by} */
  r[X] = -a[Z] * b[Y] + a[Y] * b[Z];
  r[Y] =  a[Z] * b[X] - a[X] * b[Z];
  r[Z] = -a[Y] * b[X] + a[X] * b[Y];
}


static void
rotate(double theta, const double *v, double *r)
/* Rotates an the vector v an angle theta in the (y, z) plane. */
{
  r[X] = v[X];

  r[Y] = v[Y] * cos(theta) - v[Z] * sin(theta);
  r[Z] = v[Y] * sin(theta) + v[Z] * cos(theta);
}


double 
total_fd(double K)
/* This is for debugging only: check that we have the correct fd, 
   including the Bremsstrahlung term. */
{
  double gamma, gamma2, beta2, fd;

  if (K >= FD_CUTOFF) {
    /* This introduces a constant Fd above a certain energy FD_CUTOFF.
       Although non-physical this is simple to analyze and provides a
       "zero-order" model. */
    K = FD_CUTOFF;
  }

  gamma = 1 + K / MC2;
  gamma2 = gamma * gamma;

  beta2 = 1 - 1 / gamma2;
  
  fd = truncated_bethe_fd(gamma, gamma2, beta2);
  fd += bremsstrahlung_fd(gamma);

  return fd;
}


int
drpdt(particle_t *part, double t, double *r, const double *p, 
      double *dr, double *dp, double *dtau, double h)
/* Computes the derivatives of r, p and tau for the particle *part.
   Note that part->p and part->r are ignored.  The particle is only needed
   for the charge. 
   The result is multipled by h (use h=1.0) if you want the actual derivatives.
   Returns 0 if succesful, 1 if the particle has been thermalized.
*/
{
  double p2, gamma2, gamma, beta2, fd;
  double e[3], b[3], mf[3];
  int i;

  p2 = NORM2(p);

  gamma2 = 1. + p2 / (MC2 * M);
  gamma = sqrt(gamma2);

  /* We already know the derivative of the proper time. */
  *dtau = h / gamma;

  if (gamma < GAMMATH) {
    /* The particle has been thermalized.  No sense in continuing. */
    return 1;
  }

  if(gamma < 2 * GAMMATH - 1) {
    /* In this case, there can be no collisions where the secondary
       energy is > Kth. The substracted term is zero.
       Note: In a reimplementation it is more efficient to simply
       ignore the substracted terms and not rely on the identity that
       they are 0 when gamma = 2 * gammath - 1. */
    gamma = 2 * GAMMATH - 1;
    gamma2 = gamma * gamma;
  }

  beta2 = 1 - 1 / gamma2;
  
  if (gamma <= 1 + FD_CUTOFF / MC2) {
    fd  = truncated_bethe_fd(gamma, gamma2, beta2);
    fd += bremsstrahlung_fd(gamma);
  } else {
    /* We are above FD_CUTOFF: Use the FD corresponding to FD_CUTOFF. */
    double gamma_, gamma2_, beta2_;
    gamma_ = 1 + FD_CUTOFF / MC2;
    gamma2_ = gamma_ * gamma_;
    beta2_ = 1 - 1 / gamma2_;
    
    fd  = truncated_bethe_fd(gamma_, gamma2_, beta2_);
    fd += bremsstrahlung_fd(gamma_);    
  }

  (*emfield_func) (t, r, e, b);
  
  for(i = 0; i < 3; i++){
    /* The derivative of r is calculated from the relativistic velocity. */
    dr[i] = p[i] / gamma / M;
  }

  /* The magnetic force is F = v x B. */
  cross(dr, b, mf);

  for(i = 0; i < 3; i++){
    double lorentz;
    lorentz = part->charge * ELEMENTARY_CHARGE * (e[i] + mf[i]);

    dp[i] = -fd * p[i] / sqrt(p2) + lorentz;

    dp[i] *= h;
    dr[i] *= h;      
  }

  return 0;
}

void
drpdt_all(particle_t *plist, double t, double dt)
/* Calculates dp and dr for all particles in plist */
{
  particle_t *part;
  int thermal;

  if (emfield_func == &emfield_selfcons) {
    count_mobile(plist);
    solve_ez();
  }

  for (part = plist; part; part = part->next) {
    part->thermal = drpdt(part, t, part->r, part->p, 
			  part->dr, part->dp, &(part->dtau), dt);
  }
}

int
rk4_single(particle_t *part, double t, double dt, int update)
/* Updates the r and p of particle *part by a time dt using a 4th order 
   Runge-Kutta solver. 
   If update is 0, just checks if the particle is being thermalized, without
   updating momenta or position.
*/
{
  double dr1[3], dr2[3], dr3[3], dr4[3];
  double dp1[3], dp2[3], dp3[3], dp4[3];
  double dtau1, dtau2, dtau3, dtau4;
  double r[3], p[3], tau;
  int i, thermal;

  thermal = drpdt(part, t, part->r, part->p, dr1, dp1, &dtau1, dt);
  if(thermal) return 1;

  for (i = 0; i < 3; i++) {
    r[i] = part->r[i] + dr1[i] / 2;
    p[i] = part->p[i] + dp1[i] / 2;
  }
  tau = part->tau + dtau1 / 2;

 
  thermal = drpdt(part, t + 0.5 * dt, r, p, dr2, dp2, &dtau2, dt);
  if(thermal) return 1;

  for (i = 0; i < 3; i++) {
    r[i] = part->r[i] + dr2[i] / 2;
    p[i] = part->p[i] + dp2[i] / 2;
  }
  tau = part->tau + dtau2 / 2;


  thermal = drpdt(part, t + 0.5 * dt, r, p, dr3, dp3, &dtau3, dt);
  if(thermal) return 1;
 

  for (i = 0; i < 3; i++) {
    r[i] = part->r[i] + dr3[i];
    p[i] = part->p[i] + dp3[i];
  }
  tau = part->tau + dtau3;

  thermal = drpdt(part, t + dt, r, p, dr4, dp4, &dtau4, dt);
  if(thermal) return 1;

  if (!update) return 0;

  for (i = 0; i < 3; i++) {
    part->r[i] = part->r[i] + dr1[i] / 6 + dr2[i] / 3 + dr3[i] / 3 + dr4[i] / 6;
    part->p[i] = part->p[i] + dp1[i] / 6 + dp2[i] / 3 + dp3[i] / 3 + dp4[i] / 6;
  }
  part->tau = part->tau + dtau1 / 6 + dtau2 / 3 + dtau3 / 3 + dtau4 / 6;

  return 0;
}

void
rk4(double t, double dt)
/* Implements a full RK4 step of all the particles in the list. */
{
  particle_t *plist1, *plist2, *plist3;
  particle_t *part1, *part2, *part3;
  particle_t *part;
  int i;

  drpdt_all(particle_head, t, dt);
  plist1 = rkstep(particle_head, particle_head, 0.5);

  drpdt_all(plist1, t + 0.5 * dt, dt);
  plist2 = rkstep(particle_head, plist1, 0.5);
  
  drpdt_all(plist2, t + 0.5 * dt, dt);
  plist3 = rkstep(particle_head, plist2, 1.0);

  drpdt_all(plist3, t + dt, dt);
  
  for(part = particle_head, part1 = plist1, part2 = plist2, part3 = plist3; 
      part; 
      part = part->next, 
	part1 = part1->next, 
	part2 = part2->next, 
	part3 = part3->next) {

    for (i = 0; i < 3; i++) {
      part->r[i] = (part->r[i] + part->dr[i] / 6 + part1->dr[i] / 3 + 
		    part2->dr[i] / 3 + part3->dr[i] / 6);

      part->p[i] = (part->p[i] + part->dp[i] / 6 + part1->dp[i] / 3 + 
		    part2->dp[i] / 3 + part3->dp[i] / 6);
    }
    part->tau = (part->tau + part->dtau / 6 + part1->dtau / 3 + 
		 part2->dtau / 3 + part3->dtau / 6);

    part->thermal = (part->thermal || part1->thermal || part2->thermal ||
		     part3->thermal);
  }

  list_erase(plist1);
  list_erase(plist2);
  list_erase(plist3);
}



particle_t *
rkstep(particle_t *plist0, particle_t *plist1, double rkfactor)
/* Performs a Runke-Kutta inner step.  It reads from plist0 and plist1
   and creates a new particle list.  This new list is not doubly linked 
   and does not touch particle_count, particle_head, etc.

   For the new particles, we take position/momenta from plist0 and derivatives
   from plist1; they can be the same.
*/
{
  particle_t *part0, *part1, *newlist = NULL, *tail = NULL, *newpart;
  int i;

  for(part0 = plist0, part1 = plist1; part0; 
      part0 = part0->next, part1 = part1->next) {
    newpart = particle_init(part0->ptype);

    for (i = 0; i < 3; i++) {
      newpart->r[i] = part0->r[i] + rkfactor * part1->dr[i];
      newpart->p[i] = part0->p[i] + rkfactor * part1->dp[i];
    }
    newpart->tau = part0->tau + rkfactor * part1->dtau;

    if (newlist == NULL) {
      newlist = newpart;
      tail = newpart;
    } else {
      if (tail != NULL) {
	tail->next = newpart;
	tail = newpart;
      }
    }
  }

  return newlist;
}

      
void
drop_thermal(void)
/* Removes from the list all particles that have been thermalized. */
{
  particle_t *part, *newpart;

  for (part = particle_head; part; part = newpart) {
    newpart = part->next;
    if (part->thermal) {
      particle_delete(part, TRUE);
    }
  }
}


int
ionizing_collision(particle_t *part, double dt, double *K1, double *K2)
/* Checks if the electron expriences a Moller collision with scattered
   energy larger than Kth.  If it does, returns the K1, K2 of the
   two resulting particles. */
{
  double p2, gamma2, gamma, beta2, v, K, Kp, W, P;
  double pmax;

  p2 = NORM2(part->p);
  gamma2 = 1. + p2 / (MC2 * M);
  gamma = sqrt(gamma2);
  beta2 = 1 - 1 / gamma2;

  K = MC2 * (gamma - 1);

  if (K / 2 < KTH) {
    return 0;
  }

  v = C * sqrt(beta2);
  Kp = KTH + rand()  * (K / 2 - KTH) /  RAND_MAX;
  W = rand() / (K / 2 - KTH) / RAND_MAX;

  P    = v * dt * moller_differential(gamma, gamma2, beta2, Kp);
  pmax = v * dt * moller_differential(gamma, gamma2, beta2, KTH);
  /* if(pmax > 1.0 / (K / 2 - KTH)) { */
  /*   fprintf(stderr, "%s: error: pmax > 1/DK\n", invok_name); */
  /*   fprintf(stderr, "max(P) = %g,\t1/DK = %g\n", */
  /* 	    pmax, */
  /* 	    1.0 / (K / 2 - KTH)); */
  /* } */

  if (W > P) {
    /* No collision. */
    return 0;
  } else {
    /* KABOOM! */
    if (K2 != NULL) *K1 = K - Kp;
    if (K1 != NULL) *K2 = Kp;
    return 1;
  }
}

static int
argmax(const double *p, int sign, int n)
/* Finds the index of the largest component of p. */
{
  int q;
  if (n == 1) return 0;

  q = 1 + argmax(p + 1, sign, n - 1);

  if(sign * p[0] > sign * p[q]) {
    return 0;
  } else {
    return q;
  }
}

void
ionizing_momenta(const double *p, double K1, double K2, double *p1, double *p2)
/* Finds the p1, p2 of a relativistic collision where the incident
   momentum is p and the scattered energies are K1 and K2.
  WARNING: This routine DOES NOT WORK if p is equal to either p1 or p2.
*/
{
  double E1, E2, E12, E22, psq, pabs, p1sq, p2sq, p1perp, p2perp;
  double u[3], up[3], a[3], b[3], theta;
  int i;

  /* We first find the absolute value of the momenta from the 
     energy-momentum relation. */
  E1 = K1 + MC2;
  E12 = E1 * E1;
  E2 = K2 + MC2;
  E22 = E2 * E2;

  psq = NORM2(p);
  pabs = sqrt(psq);
 
  p1sq = (E12 - M2C4) / C2;
  p2sq = (E22 - M2C4) / C2;

  /* Now from energy conservation we can find the proyection of p1 and p2
     on the direction of p. */
  for (i = 0; i < 3; i++) {
    p1[i] = p[i] * (E12 - E22 + C2 * psq) / (2 * C2 * psq);
    p2[i] = p[i] * (E22 - E12 + C2 * psq) / (2 * C2 * psq);
  }
  
  /* The perpendicular components are obtained from Pythagoras. */
  p1perp = sqrt(p1sq - NORM2(p1));
  p2perp = sqrt(p2sq - NORM2(p2));

  perp_unit_vectors(p, pabs, up, a, b);

  /* Now we generate a random angle */
  theta = 2 * PI * rand() / RAND_MAX;

  for (i = 0; i < 3; i++) {
    u[i] = a[i] * cos(theta) + b[i] * sin(theta);
    p1[i] += u[i] * p1perp;
    p2[i] -= u[i] * p2perp;
  }
}

void
elastic_momentum(double *p, double theta, double *pnew)
/* Calculates the new momentum after an elastic collision. 
   Note that pnew CAN be the same as p if you want to throw away p.
*/
{
  double psq, pabs, phi, up[3], a[3], b[3];
  int i;

  /* TODO: repeated calculation!. */
  psq = NORM2(p);
  pabs = sqrt(psq);

  perp_unit_vectors(p, pabs, up, a, b);
  
  phi = 2 * PI * rand() / RAND_MAX;
  for (i = 0; i < 3; i++) {
    pnew[i] = ( up[i] * pabs * cos(theta)
	       + a[i] * pabs * sin(theta) * cos(phi)
	       + b[i] * pabs * sin(theta) * sin(phi));
  }
}

static void
perp_unit_vectors(const double p[3], double pabs, double up[3], 
		  double a[3], double b[3])
{
  /* This is an algorithm to break the rotational 
     symmetry around p.  For that we construct two unit vectors 
     perpendicular to p. First we build a vector that is not parallel to p. */
  double t, dotaup, aabs;
  int i, imax, imin;

  for (i = 0; i < 3; i++) {
    a[i] = p[i];
  }

  /* This is a trick to build a vector that we are sure is NOT parallel to
     a (unless a = {0, 0, 0} of course. */
  imax = argmax(a, 1, 3);
  imin = argmax(a, -1, 3);

  t = -a[imax];
  a[imax] = a[imin];
  a[imin] = t;
    
  /* Now we remove the p component. */
  for (i = 0; i < 3; i++) {    
    up[i] = p[i] / pabs;
  }

  dotaup = DOT(a, up);
  for (i = 0; i < 3; i++) {    
    a[i] = a[i] - dotaup * up[i];
  }
  
  /* And unitarize */
  aabs = sqrt(NORM2(a));
  for (i = 0; i < 3; i++) {
    a[i] = a[i] / aabs;
  }

  /* Finally we get the second vector with a cross product */
  cross(a, up, b);
}


int
elastic_collision(particle_t *part, double dt, double *theta)
/* Checks if the particle undergoes an elastic collision dirung dt.
   If it does, return the theta of the collision (i.e. the change in
   the direction of p. 
   Starting from Sun Jun  9 20:30:38 2013, To allow MC sampling
   with a reasonable dt, we must avoid very small-angle scattering.
   In air with energies 100 MeV I get around 10^18 small-angle
   scatterings per second.  With these numbers it's unrealistic
   to simulate even 100 ns.  Dwyer claims to do it but I am now doubting
   it; possibly he made the same mistake as I did believing that a
   cutoff at low angles was not so important.  However, many
   small things may add to something significant and small-angle
   scattering contributes a factor ~log(p) to the diffusion.
   Long story short: we here sample only scatterings larger than
   M * theta_min, where M ~= 100 and theta_min = p0 / p << 1.
   [M = COULOMB_M]
*/
{
  /* TODO: We have already calculated these things for the same
     particle.  Do not repeat calculations! */

  double p2, gamma2, gamma, beta2, v;
  double rtheta, P, W, theta_min, r, theta_t, ptheta_t;

  p2 = NORM2(part->p);
  gamma2 = 1. + p2 / (MC2 * M);
  gamma = sqrt(gamma2);
  beta2 = 1 - 1 / gamma2;
  v = C * sqrt(beta2);
  theta_min = sqrt(COULOMB_P02 / p2);

  theta_t = COULOMB_M * theta_min;
  if (theta_t >= PI) {
    /* All collisions are accounted for by the diffusion term this only
       happens for very low energies. */
    return 0;
  }

  ptheta_t = v * dt * coulomb_differential(gamma2, beta2, p2, theta_t);
  if (ptheta_t > 1.0 / (PI - theta_t)) {
    fprintf(stderr, "%s: elastic_collision: ptheta_t > 1 / (PI - theta_t).\n", 
	    invok_name);
    fprintf(stderr, "%s: ptheta_t = %g, p = %g eV/c\n", 
	    invok_name, ptheta_t, sqrt(p2) * C / EV);
    fprintf(stderr, "%s: Probably this means that your dt is too large.\n",
	    invok_name);
    exit(-1);
  }

  rtheta = theta_t + (PI - theta_t) * ranf();
  W = ranf() / (PI - theta_t);

  P = v * dt * coulomb_differential(gamma2, beta2, p2, rtheta);

  if (W > P) {
    /* No collision. */
    return 0;
  } else {
    *theta = rtheta;
    return 1;
  }
}

void
elastic_diffusion(particle_t *part, double dt, double *theta)
/* Takes into account the many small-angle scattering collisions
   to calculate a total, gaussian theta.  See elastic_collision
   above for an explanation on why we separate small-angle and
   large angle collisions into to terms.
   Here we consider the many collisions with theta < M * theta_min.
   Since theta_min << PI, here we considered sin(theta) = theta. */
{
  /* TODO: We have already calculated these things for the same
     particle.  Do not repeat calculations! */

  double p2, gamma2, gamma, beta2, v;
  double dtheta2;

  p2 = NORM2(part->p);
  gamma2 = 1. + p2 / (MC2 * M);
  gamma = sqrt(gamma2);
  beta2 = 1 - 1 / gamma2;
  v = C * sqrt(beta2);

  dtheta2 = (v * dt * 16 * PI * COULOMB_PREFACTOR / gamma2 / (beta2 * beta2)
	     * log(COULOMB_M));

  *theta = rnd_gauss(0.0, sqrt(dtheta2));
}

particle_t *
timestep(particle_t *part, double t, double dt)
/* Integrates the evolution equations for a time dt.
   If new particles are created appended to the ->next field
   of the particle. 
   Returns the nest particle to advance (may be the same *part if it has not 
   been advanced)
   occurs).
*/
{
  int collides, elastic, thermal;
  particle_t *newpart;
  double K1, K2;

  collides = ionizing_collision(part, dt, &K1, &K2);

  if(!collides) {
    double theta;
    elastic = elastic_collision(part, dt, &theta);
    if (elastic) {
      elastic_momentum(part->p, theta, part->p);
      part->nelastic++;
    }

    newpart = part->next;
    thermal = rk4_single(part, t, dt, TRUE);
    if(thermal) {
      particle_delete(part, TRUE);      
    }
    return newpart;

  } else {
    double p[3];

    newpart = particle_init(part->ptype);

    memcpy(newpart->r, part->r, 3 * sizeof(double));
    memcpy(p, part->p, 3 * sizeof(double));

    ionizing_momenta(p, K1, K2, part->p, newpart->p);
    part->nionizing++;

    particle_append(newpart, TRUE);
    particle_birth(newpart);
    return part;
  }
}

void
perform_ionizing_collision(particle_t *part, double dt)
/* Checks if a particle undergoes an ionizing collision.  If it does, adds
   the newly created particle to the particle list.  This new particle will
   also be checked for ionizing collisions later. */
{
  int collides;
  particle_t *newpart;
  double K1, K2, p[3];

  collides = ionizing_collision(part, dt, &K1, &K2);
  if (!collides) return;

  newpart = particle_init(part->ptype);
  
  memcpy(newpart->r, part->r, 3 * sizeof(double));
  memcpy(p, part->p, 3 * sizeof(double));

  ionizing_momenta(p, K1, K2, part->p, newpart->p);
  part->nionizing++;

  if (ONLY_PRIMARIES) {
    free(newpart);
    return;
  }

  particle_append(newpart, TRUE);
  particle_birth(newpart);
}


void
perform_elastic_collision(particle_t *part, double dt)
/* Checks if the particle undergoes an elastic collision; if it does,
   update it momentum vector. */
{
  int elastic;
  double theta;
  elastic = elastic_collision(part, dt, &theta);

  if (elastic) {
    elastic_momentum(part->p, theta, part->p);
    part->nelastic++;
  }
  
  /* We always add an equivalent "elastic collision" that accounts for the
     many small-angle scatterings during dt. */
  elastic_diffusion(part, dt, &theta);
  elastic_momentum(part->p, theta, part->p);
}



void
sync_list_step(double dt)
/* Applies a synchronized time-step over the complete particle list. 
   In the steps, some
   particles may dissapear and others might be newly minted. */
{
  particle_t *part;

  GAMMATH = 1 + KTH / MC2;

  /* First let's check the collisions. */
  for (part = particle_head; part; part = part->next) {
    perform_ionizing_collision(part, dt);
    perform_elastic_collision(part, dt);
  }

  /* Update r and p with a 4th order Runge-Kutta. */
  rk4(TIME, dt);

  /* Drop the thermalized particles. */
  drop_thermal();

  TIME += dt;
}

void
list_step(double dt)
/* Applies a time-step over the complete particle list. In the steps, some
   particles may dissapear and others might be newly minted. */
{
  particle_t *part;

  GAMMATH = 1 + KTH / MC2;

  part = particle_head;
  while (part) {
    part = timestep(part, TIME, dt);
  }  

  TIME += dt;
}

void
list_step_n(double dt, int n)
/* Performs n time-steps over the full list. Returns the final time. */
{
  int i;

  for(i = 0; i < n; i++) {
    sync_list_step(dt);
  }
}

void
list_step_n_with_purging(double dt, int n, 
			 int max_particles, double fraction)
/* Performs n time-steps over the full list. Returns the final time. */
{
  int i;

  for(i = 0; i < n; i++) {
    sync_list_step(dt);
    if(particle_count > max_particles) {
      list_purge(fraction);
    }
  }
}


void
list_purge(double fraction)
/* Randomly removes some particles in the list, leaving a fraction 'fraction'
   of them. */
{
  particle_t *part, *next;

  part = particle_head;
  while (part) {
    next = part->next;
    if (rand() * 1.0 / RAND_MAX > fraction) {
      particle_delete(part, FALSE);
    }
    part = next;
  }  

  particle_weight /= fraction;
}


void
list_dump(char *fname)
/* Saves the r and p of the particles into a file.  This is used mostly
   for debugging; you should better use better I/O routines. */
{
  FILE *fp;
  particle_t *part;

  fp = fopen(fname, "w");
  if (fp == NULL) {
    fprintf (stderr, "%s: Cannot open file %s", invok_name, fname);
    exit (1);
  }

  for (part=particle_head; part; part = part->next) {
    fprintf(fp, "%g %g %g %g %g %g\n", 
	    part->r[0], part->r[1], part->r[2],
	    part->p[0], part->p[1], part->p[2]);
  }
  
  fclose(fp);
}  


/* Returns a random number with a gaussian distribution centered around
   mu with width sigma. */
static double
rnd_gauss(double mu, double sigma)
{
  double x1, x2, w;
  static double y1, y2;
  static int has_more = FALSE;

  
  if (has_more) {
    has_more = FALSE;
    return mu + y2 * sigma;
  }

   
  do {
    x1 = 2.0 * ranf() - 1.0;
    x2 = 2.0 * ranf() - 1.0;
    w = x1 * x1 + x2 * x2;
  } while (w >= 1.0);
  
  w = sqrt ((-2.0 * log (w)) / w);
  y1 = x1 * w;
  y2 = x2 * w;

  has_more = TRUE;

  return mu + y1 * sigma;
}


#define AM (1.0 / RAND_MAX)

/* Returns a random number uniformly distributed in [0, 1] */
static double
ranf (void)
{
  return rand() * AM;
}
  
