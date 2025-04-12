/**
 * math_util.c
 * 
 * Implementation of mathematical utilities for the black hole physics engine.
 */

#include "../include/math_util.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Add SIMD headers based on platform
#ifdef __SSE4_1__
#include <smmintrin.h>  // SSE4.1
#define HAVE_SIMD 1
#elif defined(__SSE3__)
#include <pmmintrin.h>  // SSE3
#define HAVE_SIMD 1
#elif defined(__SSE2__)
#include <emmintrin.h>  // SSE2
#define HAVE_SIMD 1
#endif

// SIMD optimized vector operations
Vector3D vector3D_add(const Vector3D a, const Vector3D b) {
    Vector3D result;
    
#ifdef HAVE_SIMD
    __m128d va = _mm_loadu_pd(&a.x);
    __m128d vb = _mm_loadu_pd(&b.x);
    __m128d vr = _mm_add_pd(va, vb);
    _mm_storeu_pd(&result.x, vr);
    result.z = a.z + b.z;
#else
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
#endif
    
    return result;
}

Vector3D vector3D_sub(const Vector3D a, const Vector3D b) {
    Vector3D result;
    
#ifdef HAVE_SIMD
    __m128d va = _mm_loadu_pd(&a.x);
    __m128d vb = _mm_loadu_pd(&b.x);
    __m128d vr = _mm_sub_pd(va, vb);
    _mm_storeu_pd(&result.x, vr);
    result.z = a.z - b.z;
#else
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
#endif
    
    return result;
}

Vector3D vector3D_scale(const Vector3D v, double scale) {
    Vector3D result;
    
#ifdef HAVE_SIMD
    __m128d vv = _mm_loadu_pd(&v.x);
    __m128d vs = _mm_set1_pd(scale);
    __m128d vr = _mm_mul_pd(vv, vs);
    _mm_storeu_pd(&result.x, vr);
    result.z = v.z * scale;
#else
    result.x = v.x * scale;
    result.y = v.y * scale;
    result.z = v.z * scale;
#endif
    
    return result;
}

double vector3D_dot(const Vector3D a, const Vector3D b) {
#ifdef HAVE_SIMD
    __m128d va = _mm_loadu_pd(&a.x);
    __m128d vb = _mm_loadu_pd(&b.x);
    __m128d mult = _mm_mul_pd(va, vb);
    
    // Extract values and sum them
    double xy[2];
    _mm_storeu_pd(xy, mult);
    return xy[0] + xy[1] + (a.z * b.z);
#else
    return a.x * b.x + a.y * b.y + a.z * b.z;
#endif
}

Vector3D vector3D_cross(const Vector3D a, const Vector3D b) {
    Vector3D result;
    
    // Cross product can't be easily vectorized with SSE
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    
    return result;
}

double vector3D_length(const Vector3D v) {
    return sqrt(vector3D_dot(v, v));
}

Vector3D vector3D_normalize(const Vector3D v) {
    double length = vector3D_length(v);
    if (length < BH_EPSILON) {
        Vector3D zero = {0.0, 0.0, 0.0};
        return zero;
    }
    return vector3D_scale(v, 1.0 / length);
}

// Leapfrog integrator for N-body simulations and geodesic tracing
void leapfrog_integrate(
    ODEFunctionSecondOrder f,
    double* x,    // Position
    double* v,    // Velocity
    int n,        // System dimension
    double t,     // Current time
    double dt,    // Time step
    void* params  // Additional parameters
)
{
    double* a = (double*)malloc(n * sizeof(double));
    
    // Half-step velocity: v(t+dt/2) = v(t) + a(t)*dt/2
    f(t, x, v, a, params);
    for (int i = 0; i < n; i++) {
        v[i] += 0.5 * dt * a[i];
    }
    
    // Full-step position: x(t+dt) = x(t) + v(t+dt/2)*dt
    for (int i = 0; i < n; i++) {
        x[i] += dt * v[i];
    }
    
    // Update acceleration: a(t+dt)
    f(t + dt, x, v, a, params);
    
    // Half-step velocity: v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
    for (int i = 0; i < n; i++) {
        v[i] += 0.5 * dt * a[i];
    }
    
    free(a);
}

/**
 * Runge-Kutta 4th order integrator for a system of ODEs
 */
void rk4_integrate(
    ODEFunction f,
    double* y,
    int n,
    double t,
    double h,
    void* params)
{
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* k4 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    
    // k1 = f(t, y)
    f(t, y, k1, params);
    
    // k2 = f(t + h/2, y + h*k1/2)
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + 0.5 * h * k1[i];
    }
    f(t + 0.5 * h, y_temp, k2, params);
    
    // k3 = f(t + h/2, y + h*k2/2)
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + 0.5 * h * k2[i];
    }
    f(t + 0.5 * h, y_temp, k3, params);
    
    // k4 = f(t + h, y + h*k3)
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + h * k3[i];
    }
    f(t + h, y_temp, k4, params);
    
    // y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    for (int i = 0; i < n; i++) {
        y[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(y_temp);
}

/**
 * Adaptive step size Runge-Kutta-Fehlberg integrator (RKF45)
 */
int rkf45_integrate(
    ODEFunction f,
    double y[],
    int n,
    double* t,
    double h_try,
    double* h_next,
    double eps_rel,
    void* params)
{
    // RKF45 coefficients
    const double a2 = 1.0/4.0;
    const double a3 = 3.0/8.0;
    const double a4 = 12.0/13.0;
    const double a5 = 1.0;
    const double a6 = 1.0/2.0;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    (void)a6;
    const double b21 = 1.0/4.0;
    
    const double b31 = 3.0/32.0;
    const double b32 = 9.0/32.0;
    
    const double b41 = 1932.0/2197.0;
    const double b42 = -7200.0/2197.0;
    const double b43 = 7296.0/2197.0;
    
    const double b51 = 439.0/216.0;
    const double b52 = -8.0;
    const double b53 = 3680.0/513.0;
    const double b54 = -845.0/4104.0;
    
    const double b61 = -8.0/27.0;
    const double b62 = 2.0;
    const double b63 = -3544.0/2565.0;
    const double b64 = 1859.0/4104.0;
    const double b65 = -11.0/40.0;
    
    // 4th order coefficients
    const double c1 = 25.0/216.0;
    const double c3 = 1408.0/2565.0;
    const double c4 = 2197.0/4104.0;
    const double c5 = -1.0/5.0;
    
    // 5th order coefficients
    const double d1 = 16.0/135.0;
    const double d3 = 6656.0/12825.0;
    const double d4 = 28561.0/56430.0;
    const double d5 = -9.0/50.0;
    const double d6 = 2.0/55.0;
    
    // Safety factor for step size adjustment
    const double SAFETY = 0.9;
    
    // Min and max step size scaling factors
    const double MIN_SCALE = 0.2;
    const double MAX_SCALE = 10.0;
    
    // Allocate working arrays
    double* y_temp = (double*)malloc(n * sizeof(double));
    double* y4 = (double*)malloc(n * sizeof(double));
    double* y5 = (double*)malloc(n * sizeof(double));
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* k4 = (double*)malloc(n * sizeof(double));
    double* k5 = (double*)malloc(n * sizeof(double));
    double* k6 = (double*)malloc(n * sizeof(double));
    
    double h = h_try;
    double t_current = *t;
    
    // Calculate k1
    f(t_current, y, k1, params);
    
    // Calculate k2
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + h * b21 * k1[i];
    }
    f(t_current + h * b21, y_temp, k2, params);
    
    // Calculate k3
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + h * (b31 * k1[i] + b32 * k2[i]);
    }
    f(t_current + h * b31, y_temp, k3, params);
    
    // Calculate k4
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + h * (b41 * k1[i] + b42 * k2[i] + b43 * k3[i]);
    }
    f(t_current + h * b41, y_temp, k4, params);
    
    // Calculate k5
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + h * (b51 * k1[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i]);
    }
    f(t_current + h * b51, y_temp, k5, params);
    
    // Calculate k6
    for (int i = 0; i < n; i++) {
        y_temp[i] = y[i] + h * (b61 * k1[i] + b62 * k2[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i]);
    }
    f(t_current + h * b61, y_temp, k6, params);
    
    // Calculate 4th and 5th order solutions
    for (int i = 0; i < n; i++) {
        y4[i] = y[i] + h * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i]);
        y5[i] = y[i] + h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i]);
    }
    
    // Calculate max error and RMS error
    double max_error = 0.0;
    double rms_error = 0.0;
    
    for (int i = 0; i < n; i++) {
        double scale = fmax(fabs(y[i]), fabs(y5[i]));
        if (scale < BH_EPSILON) {
            scale = BH_EPSILON;
        }
        
        double error = fabs(y5[i] - y4[i]) / scale;
        max_error = fmax(max_error, error);
        rms_error += error * error;
    }
    
    rms_error = sqrt(rms_error / n);
    
    // Calculate new step size based on error
    double error_ratio = max_error / eps_rel;
    double step_scale;
    
    if (error_ratio <= 1.0) {
        // Step succeeded, compute next step size
        if (error_ratio == 0.0) {
            step_scale = MAX_SCALE;
        } else {
            step_scale = SAFETY * pow(error_ratio, -0.2);
            step_scale = fmax(MIN_SCALE, fmin(MAX_SCALE, step_scale));
        }
        
        // Update solution with 5th order result
        for (int i = 0; i < n; i++) {
            y[i] = y5[i];
        }
        
        *t = t_current + h;
        *h_next = h * step_scale;
        
        free(y_temp);
        free(y4);
        free(y5);
        free(k1);
        free(k2);
        free(k3);
        free(k4);
        free(k5);
        free(k6);
        
        return 0; // Success
    } else {
        // Step failed, reduce step size and retry
        step_scale = SAFETY * pow(error_ratio, -0.25);
        step_scale = fmax(MIN_SCALE, fmin(MAX_SCALE, step_scale));
        
        *h_next = h * step_scale;
        
        free(y_temp);
        free(y4);
        free(y5);
        free(k1);
        free(k2);
        free(k3);
        free(k4);
        free(k5);
        free(k6);
        
        return 1; // Retry with smaller step
    }
}

/**
 * Convert RGB to temperature-based color (blackbody radiation)
 * Uses approximation based on Planck's law
 */
void temperature_to_rgb(double temperature, double rgb[3]) {
    // Temperature bounds (in K)
    const double MIN_TEMP = 1000.0;
    const double MAX_TEMP = 40000.0;
    
    // Clamp temperature to valid range
    temperature = clamp(temperature, MIN_TEMP, MAX_TEMP);
    
    // Normalize temperature (0-1) for the range
    double t = (temperature - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
    
    // Red component
    if (t < 0.5) {
        rgb[0] = t * 2.0;
    } else {
        rgb[0] = 1.0;
    }
    
    // Green component
    if (t < 0.25) {
        rgb[1] = 0.0;
    } else if (t < 0.75) {
        rgb[1] = (t - 0.25) * 2.0;
    } else {
        rgb[1] = 1.0;
    }
    
    // Blue component
    if (t < 0.5) {
        rgb[2] = 0.0;
    } else {
        rgb[2] = (t - 0.5) * 2.0;
    }
    
    // Brightness adjustment
    double brightness = 0.2 + 0.8 * (t * t); // Quadratic brightness increase
    
    rgb[0] *= brightness;
    rgb[1] *= brightness;
    rgb[2] *= brightness;
}

double clamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
} 