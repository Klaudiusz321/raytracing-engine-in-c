#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "include/blackhole_types.h"

// Simple constants
#define PI 3.14159265358979323846
#define G 6.67430e-11
#define C 299792458.0

typedef struct {
    double mass;
    double schwarzschild_radius;
    double spin;
} SimpleBlackHole;

// Simple functions that don't rely on the full physics engine
double simple_time_dilation(double r, double rs) {
    return 1.0 / sqrt(1.0 - rs / r);
}

double simple_orbital_velocity(double r, double M) {
    return sqrt(M / r);
}

double simple_ray_deflection(double b, double rs) {
    return 2.0 * rs / b;
}

void print_disk_visualization(double inner_radius, double outer_radius, int resolution) {
    printf("\nAccretion Disk Visualization:\n");
    
    // Calculate grid size
    int grid_size = resolution * 2 + 1;
    char grid[grid_size][grid_size];
    
    // Clear grid
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            grid[i][j] = ' ';
        }
    }
    
    // Draw disk
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            // Convert grid coordinates to physical coordinates
            double x = (i - resolution) * outer_radius / resolution;
            double y = (j - resolution) * outer_radius / resolution;
            double r = sqrt(x*x + y*y);
            
            // Check if point is in the disk
            if (r >= inner_radius && r <= outer_radius) {
                // Temperature decreases with radius
                double temperature = 1.0 - (r - inner_radius) / (outer_radius - inner_radius);
                
                // Choose character based on temperature
                if (temperature > 0.8) {
                    grid[i][j] = '#';  // Hottest
                } else if (temperature > 0.6) {
                    grid[i][j] = '*';
                } else if (temperature > 0.4) {
                    grid[i][j] = '+';
                } else if (temperature > 0.2) {
                    grid[i][j] = '.';
                } else {
                    grid[i][j] = '-';  // Coolest, using ASCII dash instead of UTF-8 dot
                }
            } else if (r < inner_radius) {
                // Inside inner radius is empty (or event horizon)
                grid[i][j] = ' ';
            }
        }
    }
    
    // Print grid
    for (int j = 0; j < grid_size; j++) {
        printf("    ");
        for (int i = 0; i < grid_size; i++) {
            printf("%c ", grid[i][j]);
        }
        printf("\n");
    }
}

int main() {
    printf("Black Hole Physics Engine - Simple Test\n");
    printf("=======================================\n\n");
    
    // Create a black hole
    SimpleBlackHole blackhole;
    blackhole.mass = 10.0;                             // 10 solar masses
    blackhole.schwarzschild_radius = 2.0 * blackhole.mass;  // 2M in geometric units
    blackhole.spin = 0.0;                             // Non-rotating (Schwarzschild)
    
    printf("Black Hole Parameters:\n");
    printf("---------------------\n");
    printf("Mass: %.2f M\n", blackhole.mass);
    printf("Schwarzschild radius: %.2f units\n", blackhole.schwarzschild_radius);
    
    // Time dilation
    printf("\nTime Dilation at Different Radii:\n");
    printf("--------------------------------\n");
    for (int i = 1; i <= 5; i++) {
        double r = blackhole.schwarzschild_radius * i;
        double dilation = simple_time_dilation(r, blackhole.schwarzschild_radius);
        printf("At r = %.2f units: Time runs %.4f times slower than at infinity\n", 
               r, dilation);
    }
    
    // Orbital velocities
    printf("\nOrbital Velocities:\n");
    printf("-----------------\n");
    printf("Radius (R_s)   |   Orbital Velocity (c)   |   Period (M)\n");
    printf("--------------------------------------------------------\n");
    
    for (int i = 3; i <= 10; i += 1) {
        double r = blackhole.schwarzschild_radius * i / 2.0;
        double v = simple_orbital_velocity(r, blackhole.mass);
        double period = 2.0 * PI * r / v;
        
        printf("%8.2f      |       %8.6f          |   %8.2f\n", 
               r / blackhole.schwarzschild_radius, v, period);
    }
    
    // Ray bending
    printf("\nGravitational Lensing (Ray Deflection):\n");
    printf("------------------------------------\n");
    printf("Impact Parameter (R_s)   |   Deflection Angle (rad)\n");
    printf("------------------------------------------------\n");
    
    for (int i = 3; i <= 10; i += 1) {
        double b = blackhole.schwarzschild_radius * i / 2.0;
        double deflection = simple_ray_deflection(b, blackhole.schwarzschild_radius);
        
        printf("%16.2f        |       %10.6f\n", 
               b / blackhole.schwarzschild_radius, deflection);
    }
    
    // Visualize accretion disk
    print_disk_visualization(blackhole.schwarzschild_radius * 1.5, 
                             blackhole.schwarzschild_radius * 5.0, 
                             10);
    
    printf("\nNote: This is a simplified test using basic approximations.\n");
    printf("The full physics engine would provide more accurate calculations\n");
    printf("based on general relativity.\n");
    
    return 0;
} 