/**
 * spacetime.h
 * 
 * Functions for handling spacetime geometry and general relativity calculations.
 */

#ifndef SPACETIME_H
#define SPACETIME_H

#include <stdbool.h>
#include "blackhole_types.h"

/**
 * Initialize a black hole with the specified parameters
 * 
 * @param blackhole Pointer to the BlackHoleParams struct to initialize
 * @param mass Mass of the black hole in solar masses
 * @param spin Dimensionless spin parameter (0 for Schwarzschild, 0-0.998 for Kerr)
 * @param charge Electric charge (Q)
 */
void initialize_black_hole_params(BlackHoleParams* blackhole, double mass, double spin, double charge);

/**
 * Calculate the Schwarzschild metric tensor components at a given radius
 * 
 * @param r Radius from the black hole center (in Schwarzschild coordinates)
 * @param blackhole Pointer to black hole parameters
 * @return The Schwarzschild metric components
 */
SchwarzschildMetric calculate_schwarzschild_metric(double r, const BlackHoleParams* blackhole);

/**
 * Calculate the Kerr metric tensor components at given coordinates
 * 
 * @param r Radial coordinate
 * @param theta Angular coordinate
 * @param blackhole Black hole parameters
 * @return The Kerr metric components
 */
KerrMetric calculate_kerr_metric(double r, double theta, const BlackHoleParams* blackhole);

/**
 * Calculate the black hole metric based on type (Schwarzschild or Kerr)
 * 
 * @param r Radial coordinate
 * @param theta Angular coordinate
 * @param blackhole Black hole parameters
 * @return The appropriate metric (Schwarzschild or Kerr)
 */
BlackHoleMetric calculate_metric(double r, double theta, const BlackHoleParams* blackhole);

/**
 * Calculate the Christoffel symbols for the metric
 * 
 * @param r Radial coordinate
 * @param theta Angular coordinate
 * @param blackhole Black hole parameters
 * @param christoffel Output array for Christoffel symbols
 */
void calculate_christoffel_symbols(double r, double theta, const BlackHoleParams* blackhole, double christoffel[4][4][4]);

/**
 * Calculate the Kerr metric components at a given position in Boyer-Lindquist coordinates
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param metric Output metric tensor components
 * @return 0 if successful, error code otherwise
 */
int calculate_kerr_metric_bl(const double position[4], double a, double M, KerrMetric* metric);

/**
 * Calculate the inverse Kerr metric components
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param inv_metric Output inverse metric tensor components
 * @return 0 if successful, error code otherwise
 */
int calculate_inverse_kerr_metric(const double position[4], double a, double M, KerrMetric* inv_metric);

/**
 * Calculate Kerr Christoffel symbols at a given position
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param christoffel Output array for Christoffel symbols (4x4x4 array)
 * @return 0 if successful, error code otherwise
 */
int calculate_kerr_christoffel(const double position[4], double a, double M, double christoffel[4][4][4]);

/**
 * Calculate the ISCO (Innermost Stable Circular Orbit) radius for a Kerr black hole
 * 
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param prograde Whether to calculate for prograde (true) or retrograde (false) orbits
 * @return The ISCO radius in geometrized units (GM/c²)
 */
double calculate_kerr_isco(double a, double M, bool prograde);

/**
 * Calculate the event horizon radius for a Kerr black hole
 * 
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @return The event horizon radius in geometrized units
 */
double calculate_kerr_event_horizon(double a, double M);

/**
 * Calculate the ergosphere radius at a given polar angle for a Kerr black hole
 * 
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param theta Polar angle
 * @return The ergosphere radius at the given angle
 */
double calculate_kerr_ergosphere(double a, double M, double theta);

/**
 * Calculate the relativistic velocity components for frame dragging
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param velocity Output array for velocity components (v^r, v^theta, v^phi)
 * @return 0 if successful, error code otherwise
 */
int calculate_frame_dragging(const double position[4], double a, double M, double velocity[3]);

/**
 * Calculate the Kerr geodesic equations for a given position, velocity, and black hole parameters
 * This computes the rate of change of velocity components (accelerations)
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param velocity The velocity components (v^t, v^r, v^theta, v^phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param acceleration Output array for accelerations (a^t, a^r, a^theta, a^phi)
 * @return 0 if successful, error code otherwise
 */
int calculate_kerr_geodesic(
    const double position[4], 
    const double velocity[4], 
    double a, 
    double M, 
    double acceleration[4]);

/**
 * Compute geodesic equation right-hand side for numerical integration
 * 
 * @param position 4D position array (t, r, theta, phi)
 * @param velocity 4D velocity array
 * @param blackhole Black hole parameters
 * @param acceleration Output 4D acceleration array
 */
void geodesic_equation(const double position[4], const double velocity[4], 
                       const BlackHoleParams* blackhole, double acceleration[4]);

/**
 * Calculate the time dilation factor at a given radius
 * 
 * @param r Radius from the black hole center
 * @param blackhole Black hole parameters
 * @return Time dilation factor (dt_infinity / dt_local)
 */
double calculate_time_dilation(double r, const BlackHoleParams* blackhole);

/**
 * Calculate the effective potential for a particle in orbit around the black hole
 * 
 * @param r Radius from the black hole
 * @param l Angular momentum
 * @param blackhole Black hole parameters
 * @return Effective potential value
 */
double calculate_effective_potential(double r, double l, const BlackHoleParams* blackhole);

/**
 * Calculate the ergosphere radius at a given angle theta
 * 
 * @param theta Angular coordinate
 * @param blackhole Black hole parameters
 * @return Ergosphere radius
 */
double calculate_ergosphere_radius(double theta, const BlackHoleParams* blackhole);

/**
 * Get the ISCO (Innermost Stable Circular Orbit) radius
 * 
 * @param blackhole Black hole parameters
 * @return ISCO radius
 */
double get_isco_radius(const BlackHoleParams* blackhole);

#endif /* SPACETIME_H */ 