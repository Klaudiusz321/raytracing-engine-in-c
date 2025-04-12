/**
 * math_util.h
 * 
 * Mathematical utilities for the black hole physics engine.
 */

#ifndef MATH_UTIL_H
#define MATH_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include "blackhole_types.h" // Include this for Vector3D and Vector4D definitions

/* Mathematical constants */
#define BH_PI 3.14159265358979323846
#define BH_EPSILON 1.0e-10
#define SPEED_OF_LIGHT 299792458.0         // m/s
#define GRAVITATIONAL_CONSTANT 6.67430e-11   // m^3 kg^-1 s^-2
#define SOLAR_MASS 1.989e30                // kg
#define SCHWARZSCHILD_RADIUS_SUN 2950.0     // m
#define TWO_PI 6.28318530717958647692
#define BH_TWO_PI TWO_PI  /* Alias for compatibility */
#define HALF_PI 1.57079632679489661923
#define DEG_TO_RAD (BH_PI / 180.0)
#define RAD_TO_DEG (180.0 / BH_PI)
#define G_OVER_C_SQUARED 7.425e-28 /* m/kg, for converting mass to distance */

/* SIMD optimizations flag - use this to toggle SIMD operations at compile time */
#ifndef USE_SIMD
#define USE_SIMD 1
#endif

/* Vector definitions */

/**
 * Create a 3D vector with the given components.
 */
Vector3D vector3D_create(double x, double y, double z);

/**
 * Create a 4D vector from components
 * 
 * @param t Time component
 * @param x X component
 * @param y Y component
 * @param z Z component
 * @return The created vector
 */
Vector4D vector4D_create(double t, double x, double y, double z);

/**
 * Add two 3D vectors.
 */
Vector3D vector3D_add(const Vector3D a, const Vector3D b);

/**
 * Subtract two 3D vectors.
 */
Vector3D vector3D_sub(const Vector3D a, const Vector3D b);

/**
 * Scale a 3D vector by a scalar value.
 */
Vector3D vector3D_scale(const Vector3D v, double scale);

/**
 * Calculate the dot product of two 3D vectors.
 */
double vector3D_dot(const Vector3D a, const Vector3D b);

/**
 * Calculate the cross product of two 3D vectors.
 */
Vector3D vector3D_cross(const Vector3D a, const Vector3D b);

/**
 * Calculate the length of a 3D vector.
 */
double vector3D_length(const Vector3D v);

/**
 * Normalize a 3D vector.
 */
Vector3D vector3D_normalize(const Vector3D v);

/**
 * Add two 4D vectors.
 */
Vector4D vector4D_add(const Vector4D a, const Vector4D b);

/**
 * Subtract two 4D vectors.
 */
Vector4D vector4D_sub(const Vector4D a, const Vector4D b);

/**
 * Scale a 4D vector by a scalar value.
 */
Vector4D vector4D_scale(const Vector4D v, double scale);

/**
 * Calculate the dot product of two 4D vectors.
 */
double vector4D_dot(const Vector4D a, const Vector4D b);

/**
 * Convert Cartesian coordinates to spherical coordinates.
 * 
 * @param cartesian Input Cartesian coordinates (x, y, z)
 * @param spherical Output spherical coordinates (r, theta, phi)
 */
void cartesian_to_spherical(const Vector3D* cartesian, Vector3D* spherical);

/**
 * Convert spherical coordinates to Cartesian coordinates.
 * 
 * @param spherical Input spherical coordinates (r, theta, phi)
 * @param cartesian Output Cartesian coordinates (x, y, z)
 */
void spherical_to_cartesian(const Vector3D* spherical, Vector3D* cartesian);

/**
 * Convert coordinates to Boyer-Lindquist coordinates.
 * 
 * @param x Cartesian x coordinate
 * @param y Cartesian y coordinate
 * @param z Cartesian z coordinate
 * @param a Black hole spin parameter
 * @param r Boyer-Lindquist radial coordinate (output)
 * @param theta Boyer-Lindquist theta coordinate (output)
 * @param phi Boyer-Lindquist phi coordinate (output)
 */
void cartesian_to_boyer_lindquist(double x, double y, double z, double a, double* r, double* theta, double* phi);

/**
 * Convert Boyer-Lindquist coordinates to Cartesian coordinates.
 * 
 * @param r Boyer-Lindquist radial coordinate
 * @param theta Boyer-Lindquist theta coordinate
 * @param phi Boyer-Lindquist phi coordinate
 * @param a Black hole spin parameter
 * @param x Cartesian x coordinate (output)
 * @param y Cartesian y coordinate (output)
 * @param z Cartesian z coordinate (output)
 */
void boyer_lindquist_to_cartesian(double r, double theta, double phi, double a, double* x, double* y, double* z);

/* Differential equation solvers */

/**
 * Function pointer type for ODE right-hand side functions.
 * 
 * @param t Current time
 * @param y Current state vector
 * @param dydt Output array for derivatives
 * @param params Additional parameters
 */
typedef void (*ODEFunction)(double t, const double y[], double dydt[], void *params);
/**
 * Function pointer type for second-order ODE right-hand side functions (for leapfrog integration).
 * 
 * @param t Current time
 * @param x Current position array
 * @param v Current velocity array
 * @param a Output acceleration array
 * @param params Additional parameters
 */
typedef void (*ODEFunctionSecondOrder)(
    double t,
    double* x,
    double* v,
    double* a,
    void* params
);

/**
 * Integrate a system of first-order ODEs using 4th order Runge-Kutta method
 * 
 * @param f Function that computes the right-hand side of the ODE system
 * @param y Initial state vector, will be updated with the result
 * @param n Dimension of the system
 * @param t Current time
 * @param h Step size
 * @param params Additional parameters for the ODE function
 */
void rk4_integrate(
    ODEFunction f,
    double* y,
    int n,
    double t,
    double h,
    void* params);

/**
 * Integrate a system of first-order ODEs using adaptive step size Runge-Kutta-Fehlberg (RKF45)
 * 
 * @param f Function that computes the right-hand side of the ODE system
 * @param y Initial state vector, will be updated with the result
 * @param n Dimension of the system
 * @param t Current time, will be updated with new time
 * @param h_try Step size to try
 * @param h_next Output parameter for the next suggested step size
 * @param eps_rel Relative error tolerance
 * @param params Additional parameters for the ODE function
 * @return 0 if step succeeded, 1 if step failed (retry with smaller step)
 */
int rkf45_integrate(
    ODEFunction f,
    double y[],
    int n,
    double* t,
    double h_try,
    double* h_next,
    double eps_rel,
    void* params);

/**
 * Leapfrog integrator for N-body simulations and geodesic tracing
 * 
 * @param f Function that computes the right-hand side of the ODE system
 * @param x Position array
 * @param v Velocity array
 * @param n Dimension of the system
 * @param t Current time
 * @param dt Time step
 * @param params Additional parameters
 */
void leapfrog_integrate(
    ODEFunctionSecondOrder f,
    double* x,
    double* v,
    int n,
    double t,
    double dt,
    void* params);

/* Color utilities */

/**
 * Convert temperature to RGB color based on blackbody radiation.
 * 
 * @param temperature Temperature in Kelvin
 * @param rgb Output array for RGB values (each 0.0-1.0)
 */
void temperature_to_rgb(double temperature, double rgb[3]);

/**
 * Apply redshift to RGB color
 * 
 * @param rgb Original RGB color (modified in-place)
 * @param redshift Redshift factor (z value)
 */
void apply_redshift_to_rgb(double rgb[3], double redshift);

/**
 * Calculate visible wavelength doppler shift
 * 
 * @param wavelength Original wavelength in nanometers
 * @param beta Velocity as fraction of speed of light (v/c)
 * @param cos_angle Cosine of angle between velocity and observation direction
 * @return Shifted wavelength in nanometers
 */
double doppler_shift_wavelength(double wavelength, double beta, double cos_angle);

/**
 * Utility function to clamp a value between min and max values
 */
double clamp(double value, double min, double max);

#ifdef __cplusplus
}
#endif

#endif /* MATH_UTIL_H */ 