#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "include/blackhole_types.h"
#include "include/math_util.h"

// Demo-specific settings
#define DEMO_POINTS 20
#define DEMO_RADIUS 5.0
#define DEMO_STEPS  10

// Forward declarations of functions we would use from the physics engine
// These are just placeholders for the demo
double calculate_time_dilation(double r, double rs);
void print_orbit_demo(double mass, double radius, int points);
void print_ray_bending_demo(double mass, double impact_parameter, int steps);

int main() {
    printf("Black Hole Physics Engine - Demo\n");
    printf("================================\n\n");
    
    // Create a black hole with 10 solar masses
    BlackHoleParams blackhole;
    blackhole.mass = 10.0;
    blackhole.spin = 0.0;
    blackhole.charge = 0.0;
    blackhole.schwarzschild_radius = 2.0 * blackhole.mass;
    
    printf("Black Hole Parameters:\n");
    printf("---------------------\n");
    printf("Mass: %.2f units\n", blackhole.mass);
    printf("Schwarzschild radius: %.2f units\n", blackhole.schwarzschild_radius);
    
    // Time dilation demo
    printf("\nTime Dilation at Different Radii:\n");
    printf("--------------------------------\n");
    for (int i = 1; i <= 10; i++) {
        double r = i * blackhole.schwarzschild_radius;
        double dilation = calculate_time_dilation(r, blackhole.schwarzschild_radius);
        printf("At r = %.2f units: Time runs %.4f times slower than at infinity\n", 
               r, dilation);
    }
    
    // Circular orbit demo
    printf("\nCircular Orbit Points:\n");
    printf("---------------------\n");
    print_orbit_demo(blackhole.mass, DEMO_RADIUS, DEMO_POINTS);
    
    // Ray bending demo
    printf("\nRay Bending Near Black Hole:\n");
    printf("--------------------------\n");
    print_ray_bending_demo(blackhole.mass, 3.0 * blackhole.schwarzschild_radius, DEMO_STEPS);
    
    printf("\nNote: This is a demo showing how the physics engine would work.\n");
    printf("A complete implementation would calculate these effects accurately using\n");
    printf("relativistic equations from spacetime.c, raytracer.c, and other modules.\n");
    
    return 0;
}

// Simple time dilation calculation (Schwarzschild metric)
double calculate_time_dilation(double r, double rs) {
    return 1.0 / sqrt(1.0 - rs / r);
}

// Simulate points on a circular orbit
void print_orbit_demo(double mass, double radius, int points) {
    printf("Orbit radius: %.2f units\n", radius);
    
    // In a real implementation, we would calculate the orbital velocity
    // based on general relativity for close orbits
    double orbital_velocity = sqrt(mass / radius);
    printf("Orbital velocity: %.4f c\n", orbital_velocity);
    
    for (int i = 0; i < points; i++) {
        double angle = 2.0 * PI * i / points;
        double x = radius * cos(angle);
        double y = radius * sin(angle);
        printf("Point %2d: (%.2f, %.2f, 0.00)\n", i+1, x, y);
    }
}

// Simulate ray bending
void print_ray_bending_demo(double mass, double impact_parameter, int steps) {
    printf("Impact parameter: %.2f units\n", impact_parameter);
    
    // In a real implementation, we would integrate the geodesic equation
    // Here we just approximate the bending with a simple model
    const double rs = 2.0 * mass;
    double deflection_angle = 2.0 * rs / impact_parameter;
    printf("Approximate deflection angle: %.4f radians\n", deflection_angle);
    
    // Show a simplified path
    printf("Ray path (simplified approximation):\n");
    for (int i = 0; i <= steps; i++) {
        double z = -10.0 + 20.0 * i / steps;
        // Simple model of the deflection
        double bend = 0.0;
        if (z > -3.0 && z < 3.0) {
            // Apply more bending near the black hole
            double dist = sqrt(z*z);
            bend = 0.5 * deflection_angle * (1.0 - dist/3.0);
        }
        double x = impact_parameter - bend;
        printf("Step %2d: (%.2f, 0.00, %.2f)\n", i, x, z);
    }
} 