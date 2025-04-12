#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "include/blackhole_types.h"
#include "include/math_util.h"
#include "include/spacetime.h"

// Forward declarations
double simple_time_dilation(double r, double rs);
void print_vector(const char* name, Vector3D v);

/**
 * Test program for the core functionality of the black hole physics engine,
 * focusing on mathematical utilities and coordinate transformations.
 */
int main() {
    printf("Black Hole Physics Engine - Core Functionality Test\n");
    printf("=================================================\n\n");
    
    // Test vector operations
    printf("Testing Vector Operations:\n");
    printf("-----------------------\n");
    
    Vector3D v1 = {1.0, 2.0, 3.0};
    Vector3D v2 = {4.0, 5.0, 6.0};
    
    printf("v1 = (%.2f, %.2f, %.2f)\n", v1.x, v1.y, v1.z);
    printf("v2 = (%.2f, %.2f, %.2f)\n", v2.x, v2.y, v2.z);
    
    Vector3D v_add = vector3D_add(v1, v2);
    printf("v1 + v2 = (%.2f, %.2f, %.2f)\n", v_add.x, v_add.y, v_add.z);
    
    Vector3D v_sub = vector3D_sub(v1, v2);
    printf("v1 - v2 = (%.2f, %.2f, %.2f)\n", v_sub.x, v_sub.y, v_sub.z);
    
    double dot = vector3D_dot(v1, v2);
    printf("v1 · v2 = %.2f\n", dot);
    
    Vector3D cross = vector3D_cross(v1, v2);
    printf("v1 × v2 = (%.2f, %.2f, %.2f)\n", cross.x, cross.y, cross.z);
    
    double length = vector3D_length(v1);
    printf("Length of v1 = %.2f\n", length);
    
    Vector3D v_norm = vector3D_normalize(v1);
    printf("Normalized v1 = (%.2f, %.2f, %.2f)\n", v_norm.x, v_norm.y, v_norm.z);
    printf("Length of normalized v1 = %.2f\n", vector3D_length(v_norm));
    
    // Test coordinate transformations
    printf("\nTesting Coordinate Transformations:\n");
    printf("--------------------------------\n");
    
    // Create a vector in Cartesian coordinates
    Vector3D cart = {10.0, 10.0, 10.0};
    printf("Cartesian: (%.2f, %.2f, %.2f)\n", cart.x, cart.y, cart.z);
    
    // Convert to spherical
    Vector3D sph;
    cartesian_to_spherical(&cart, &sph);
    printf("Spherical: r=%.2f, theta=%.2f, phi=%.2f\n", sph.x, sph.y, sph.z);
    
    // Convert back to Cartesian
    Vector3D cart2;
    spherical_to_cartesian(&sph, &cart2);
    printf("Back to Cartesian: (%.2f, %.2f, %.2f)\n", cart2.x, cart2.y, cart2.z);
    
    // Calculate error
    double error = sqrt(pow(cart.x - cart2.x, 2) + 
                        pow(cart.y - cart2.y, 2) + 
                        pow(cart.z - cart2.z, 2));
    printf("Transformation error: %.10f\n", error);
    
    // Test black hole parameters
    printf("\nTesting Black Hole Parameter Calculation:\n");
    printf("--------------------------------------\n");
    
    BlackHoleParams bh;
    initialize_black_hole_params(&bh, 1.0, 0.0, 0.0);
    printf("Schwarzschild black hole (M=1, a=0):\n");
    printf("Schwarzschild radius: %.2f\n", bh.schwarzschild_radius);
    printf("ISCO radius: %.2f\n", bh.isco_radius);
    
    BlackHoleParams bh_kerr;
    initialize_black_hole_params(&bh_kerr, 1.0, 0.5, 0.0);
    printf("\nKerr black hole (M=1, a=0.5):\n");
    printf("Event horizon radius (r+): %.4f\n", bh_kerr.r_plus);
    printf("Inner horizon radius (r-): %.4f\n", bh_kerr.r_minus);
    printf("ISCO radius: %.4f\n", bh_kerr.isco_radius);
    printf("Ergosphere radius (equator): %.4f\n", bh_kerr.ergosphere_radius);
    
    // Test leapfrog integrator with a simple harmonic oscillator
    printf("\nTesting Leapfrog Integrator with Simple Harmonic Oscillator:\n");
    printf("--------------------------------------------------------\n");
    
    // Simple harmonic oscillator parameters
    double k = 1.0;  // Spring constant
    double m = 1.0;  // Mass
    struct {
        double k;
        double m;
    } params = {k, m};
    
    // Function for calculating acceleration in SHO (F = -kx)
    void sho_acceleration(double t, double* x, double* v, double* a, void* p) {
        struct { double k; double m; } *params = (struct { double k; double m; }*)p;
        a[0] = -params->k * x[0] / params->m;
    }
    
    // Initial conditions
    double x[1] = {1.0};   // Initial position
    double v[1] = {0.0};   // Initial velocity
    double t = 0.0;        // Initial time
    double dt = 0.1;       // Time step
    
    printf("Initial state: x=%.4f, v=%.4f\n", x[0], v[0]);
    
    // Run a few integration steps
    for (int i = 0; i < 10; i++) {
        t += dt;
        leapfrog_integrate(sho_acceleration, x, v, 1, t, dt, &params);
        
        // Analytical solution for SHO: x(t) = A*cos(ω*t), where ω = sqrt(k/m)
        double omega = sqrt(k / m);
        double x_analytical = 1.0 * cos(omega * t);
        
        printf("t=%.1f: x=%.4f, v=%.4f (analytical x=%.4f, error=%.6f)\n", 
               t, x[0], v[0], x_analytical, fabs(x[0] - x_analytical));
    }
    
    // Test clamp function
    printf("\nTesting Utility Functions:\n");
    printf("------------------------\n");
    
    printf("clamp(5.0, 0.0, 10.0) = %.2f\n", clamp(5.0, 0.0, 10.0));
    printf("clamp(-5.0, 0.0, 10.0) = %.2f\n", clamp(-5.0, 0.0, 10.0));
    printf("clamp(15.0, 0.0, 10.0) = %.2f\n", clamp(15.0, 0.0, 10.0));
    
    // Test temperature to RGB conversion
    printf("\nTesting Temperature to RGB Conversion:\n");
    printf("---------------------------------\n");
    
    double temps[] = {1500.0, 3000.0, 5000.0, 10000.0, 20000.0, 30000.0};
    int num_temps = sizeof(temps) / sizeof(temps[0]);
    
    for (int i = 0; i < num_temps; i++) {
        double rgb[3];
        temperature_to_rgb(temps[i], rgb);
        printf("%.0f K: RGB = (%.2f, %.2f, %.2f)\n", 
               temps[i], rgb[0], rgb[1], rgb[2]);
    }
    
    printf("\nCore functionality test completed successfully!\n");
    return 0;
}

// Simple time dilation for Schwarzschild metric
double simple_time_dilation(double r, double rs) {
    return 1.0 / sqrt(1.0 - rs / r);
}

// Utility function to print vectors
void print_vector(const char* name, Vector3D v) {
    printf("%s = (%.2f, %.2f, %.2f)\n", name, v.x, v.y, v.z);
} 