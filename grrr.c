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
static double cross(const double *a, const double *b, double *r);
static int argmax(const double *p, int sign, int n);
static void electromagnetic_field(double t, const double *r, double *e, 
				  double *b);
static void electromagnetic_wave_field(double t, const double *r, double *e, 
				       double *b);
static void electromagnetic_dipole_field(double t, const double *r, double *e, 
					 double *b);
static double moller_differential(double gamma, double gamma2, double beta2,
				  double Kp);


/* We will keep a global pointer to the beginning of the particle list
   b.c. the first particle may get thermal and dissapear. */
particle_t *particle_head = NULL, *particle_tail = NULL;

/* We also have a counter of the total number of particles. */
int particle_count = 0;

/* Name of this program. */
const char *invok_name = "grrr";

/* These are parameters.  Later they have to be set at run-time. */
double E0 = 7.0e5;
double EB = 0.0;
double EBIAS = 0.0;
double B0 = 20e-6;
double KTH = 0.0549351 * MEV;
double GAMMATH;
double L = 3.0;

particle_t*
particle_init(int ptype)
/* Initializes a particle of the provided type.  The rest of the elements
   are set to 0. */
{
  particle_t *part;
  part = xcalloc(1, sizeof(particle_t));

  part->ptype = ptype;
  part->charge = particle_charge[ptype];
  part->mass = particle_mass[ptype];
  
  return part;
}

void
particle_delete(particle_t *part)
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

  free(part);

  particle_count--;
}

void
particle_append(particle_t *part)
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

  particle_count++;
}



static void
electromagnetic_field(double t, const double *r, double *e, double *b)
/* Calculates the electromagnetic field at a given time and location.
   Returns it into the e and b pointers. */
{
  b[X] = b[Z] = 0.0;
  b[Y] = B0;

  e[X] = e[Y] = 0.0;
  e[Z] = -E0 * (EBIAS + (1.0 - EBIAS) * cos(2 * PI * (r[Z] - C * t) / L)); 
}


static void
electromagnetic_wave_field(double t, const double *r, double *e, double *b)
/* Calculates the electromagnetic field at a given time and location.
   Returns it into the e and b pointers. */
{
  double cosphi;

  cosphi = cos(2 * PI * (r[Z] - C * t) / L);

  e[X] = E0 * cosphi;
  e[Y] = 0.0;
  e[Z] = -EB;

  b[X] = B0;
  b[Y] = E0 * cosphi / C;
  b[Z] = 0.0;

}


static void
electromagnetic_dipole_field(double t, const double *r, double *e, double *b)
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
cross(const double *a, const double *b, double *r)
{
  /* {-az by + ay bz, az bx - ax bz, -ay bx + ax by} */
  r[X] = -a[Z] * b[Y] + a[Y] * b[Z];
  r[Y] =  a[Z] * b[X] - a[X] * b[Z];
  r[Z] = -a[Y] * b[X] - a[X] * b[Y];
}


int
drpdt(particle_t *part, double t, const double *r, const double *p, 
      double *dr, double *dp, double h)
/* Computes the derivatives of r and p for the particle *part.
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
  
  fd = truncated_bethe_fd(gamma, gamma2, beta2);

  electromagnetic_wave_field(t, r, e, b);
  
  for(i = 0; i < 3; i++){
    /* The derivative of r is calculated from the relativistic velocity. */
    dr[i] = p[i] / gamma / M;
  }

  /* The magnetic force is F = v x B. */
  cross(dr, b, mf);

  for(i = 0; i < 3; i++){
    double lorentz;
    /* The stopping. */
    lorentz = part->charge * ELEMENTARY_CHARGE * (e[i] + mf[i]);

    dp[i] = -fd * p[i] / sqrt(p2) + lorentz;

    dp[i] *= h;
    dr[i] *= h;      
  }

  return 0;
}


int
rk4(particle_t *part, double t, double dt)
/* Updates the r and p of particle *part by a time dt using a 4th order 
   Runge-Kutta solver. */
{
  double dr1[3], dr2[3], dr3[3], dr4[3];
  double dp1[3], dp2[3], dp3[3], dp4[3];
  double r[3], p[3];
  int i, thermal;

  thermal = drpdt(part, t, part->r, part->p, dr1, dp1, dt);
  if(thermal) return 1;

  for (i = 0; i < 3; i++) {
    r[i] = part->r[i] + dr1[i] / 2;
    p[i] = part->p[i] + dp1[i] / 2;
  }

 
  thermal = drpdt(part, t + 0.5 * dt, r, p, dr2, dp2, dt);
  if(thermal) return 1;

  for (i = 0; i < 3; i++) {
    r[i] = part->r[i] + dr2[i] / 2;
    p[i] = part->p[i] + dp2[i] / 2;
  }

  thermal = drpdt(part, t + 0.5 * dt, r, p, dr3, dp3, dt);
  if(thermal) return 1;
 

  for (i = 0; i < 3; i++) {
    r[i] = part->r[i] + dr3[i];
    p[i] = part->p[i] + dp3[i];
  }

  thermal = drpdt(part, t + dt, r, p, dr4, dp4, dt);
  if(thermal) return 1;

  for (i = 0; i < 3; i++) {
    part->r[i] = part->r[i] + dr1[i] / 6 + dr2[i] / 3 + dr3[i] / 3 + dr4[i] / 6;
    part->p[i] = part->p[i] + dp1[i] / 6 + dp2[i] / 3 + dp3[i] / 3 + dp4[i] / 6;
  }

  return 0;
}


int
collision(particle_t *part, double dt, double *K1, double *K2)
/* Checks if the electron expriences a Moller collision with scattered
   energy larger than Kth.  If it does, returns the K1, K2 of the
   two resulting particles. */
{
  double p2, gamma2, gamma, beta2, v, K, Kp, W, P;

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

  P = v * dt * moller_differential(gamma, gamma2, beta2, Kp);

  if (W > P) {
    /* No collision. */
    return 0;
  } else {
    /* KABOOM! */
    *K2 = K - Kp;
    *K1 = Kp;
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
momenta(const double *p, double K1, double K2, double *p1, double *p2)
/* Finds the p1, p2 of a relativistic collision where the incident
   momentum is p and the scattered energies are K1 and K2.
  WARNING: This routine DOES NOT WORK if p is equal to either p1 or p2.
*/
{
  double E1, E2, E12, E22, psq, pabs, p1sq, p2sq, p1perp, p2perp;
  double a[3], b[3], u[3], up[3], t, dotaup, aabs, theta;
  int i, imin, imax;

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

  /* Now it comes the hard part: we have to break the rotational 
     symmetry around p.  For that we construct two unit vectors 
     perpendicular to p. First we build a vector that is not parallel to p. */
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
  
  /* Now we generate a random angle */
  theta = 2 * PI * rand() / RAND_MAX;

  for (i = 0; i < 3; i++) {
    u[i] = a[i] * cos(theta) + b[i] * sin(theta);
    p1[i] += u[i] * p1perp;
    p2[i] -= u[i] * p2perp;
  }

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
  int collides, thermal;
  particle_t *newpart;
  double K1, K2;

  collides = collision(part, dt, &K1, &K2);

  if(!collides) {
    newpart = part->next;
    thermal = rk4(part, t, dt);
    if(thermal) {
      particle_delete(part);      
    }
    return newpart;

  } else {
    double p[3];

    newpart = particle_init(part->ptype);

    memcpy(newpart->r, part->r, 3 * sizeof(double));
    memcpy(p, part->p, 3 * sizeof(double));

    momenta(p, K1, K2, part->p, newpart->p);

    particle_append(newpart);
    return part;
  }
}


void
list_step(double t, double dt)
/* Applies a time-step over the complete particle list. In the steps, some
   particles may dissapear and others might be newly minted. */
{
  particle_t *part;

  GAMMATH = 1 + KTH / MC2;

  part = particle_head;
  while (part) {
    // printf("%d " particle_printf_str "\n", i++, particle_printf_args(part));
    part = timestep(part, t, dt);
  }  
}

double
list_step_n(double t, double dt, int n)
/* Performs n time-steps over the full list. Returns the final time. */
{
  int i;

  for(i = 0; i < n; i++) {
    list_step(t, dt);
    t += dt;
  }

  return t;
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
      particle_delete(part);
    }
    part = next;
  }  
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
  
