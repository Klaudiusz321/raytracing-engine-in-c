/**
 * main.c
 * 
 * Simple test program for black hole physics engine.
 */

#include "../include/blackhole_api.h"
#include <stdio.h>
#include <stdlib.h>
#define NUM_POSITIONS 100

#include <math.h>
#ifndef PI
#define PI 3.14159265358979323846
#endif

void print_ray_result(RayTraceHit* hit) {
    printf("Ray result: ");
    
    switch (hit->result) {
        case RAY_HORIZON:
            printf("Hit event horizon\n");
            break;
        case RAY_DISK:
            printf("Hit accretion disk\n");
            break;
        case RAY_BACKGROUND:
            printf("Reached background\n");
            break;
        case RAY_MAX_DISTANCE:
            printf("Reached maximum distance\n");
            break;
        case RAY_MAX_STEPS:
            printf("Reached maximum steps\n");
            break;
        case RAY_ERROR:
            printf("Error during ray tracing\n");
            break;
        default:
            printf("Unknown result\n");
            break;
    }
    
    printf("  Hit position: (%.3f, %.3f, %.3f)\n", 
           hit->hit_position.x, hit->hit_position.y, hit->hit_position.z);
    printf("  Distance traveled: %.3f\n", hit->distance);
    printf("  Steps: %d\n", hit->steps);
    printf("  Time dilation: %.3f\n", hit->time_dilation);
    
    if (hit->result == RAY_BACKGROUND || hit->result == RAY_MAX_DISTANCE) {
        printf("  Sky direction: (%.3f, %.3f, %.3f)\n", 
               hit->sky_direction.x, hit->sky_direction.y, hit->sky_direction.z);
    }
    
    printf("\n");
}

/**
 * Test ray tracing
 */
void test_ray_tracing(BHContextHandle context) {
    printf("Testing ray tracing...\n");
    
    // Define a set of test rays
    const int NUM_RAYS = 5;
    Ray rays[NUM_RAYS];
    RayTraceHit hits[NUM_RAYS];
    
    // Ray 1: Directly toward the black hole
    rays[0].origin.x = 0.0;
    rays[0].origin.y = 0.0;
    rays[0].origin.z = 30.0;
    rays[0].direction.x = 0.0;
    rays[0].direction.y = 0.0;
    rays[0].direction.z = -1.0;
    
    // Ray 2: Near miss (grazing)
    rays[1].origin.x = 0.0;
    rays[1].origin.y = 0.0;
    rays[1].origin.z = 30.0;
    rays[1].direction.x = 0.2;
    rays[1].direction.y = 0.0;
    rays[1].direction.z = -1.0;
    
    // Ray 3: Far miss
    rays[2].origin.x = 0.0;
    rays[2].origin.y = 0.0;
    rays[2].origin.z = 30.0;
    rays[2].direction.x = 0.5;
    rays[2].direction.y = 0.0;
    rays[2].direction.z = -1.0;
    
    // Ray 4: Toward disk
    rays[3].origin.x = 0.0;
    rays[3].origin.y = 0.0;
    rays[3].origin.z = 30.0;
    rays[3].direction.x = 0.3;
    rays[3].direction.y = 0.0;
    rays[3].direction.z = -1.0;
    
    // Ray 5: From side
    rays[4].origin.x = 30.0;
    rays[4].origin.y = 0.0;
    rays[4].origin.z = 0.0;
    rays[4].direction.x = -1.0;
    rays[4].direction.y = 0.0;
    rays[4].direction.z = 0.1;
    
    // Trace rays
    BHErrorCode error = bh_trace_rays_batch(context, rays, hits, NUM_RAYS);
    
    if (error != BH_SUCCESS) {
        printf("Error tracing rays: %d\n", error);
        return;
    }
    
    // Print results
    for (int i = 0; i < NUM_RAYS; i++) {
        printf("Ray %d:\n", i + 1);
        printf("  Origin: (%.3f, %.3f, %.3f)\n", 
               rays[i].origin.x, rays[i].origin.y, rays[i].origin.z);
        printf("  Direction: (%.3f, %.3f, %.3f)\n", 
               rays[i].direction.x, rays[i].direction.y, rays[i].direction.z);
        print_ray_result(&hits[i]);
    }
}

/**
 * Test particle orbits
 */
void test_particle_orbits(BHContextHandle context) {
    printf("Testing particle orbit calculation...\n");
    
    // Test parameters
    // double mass = 10.0;
    int num_positions = 5;
    
    // Allocate space for positions dynamically
    double (*positions)[3] = (double(*)[3])malloc(num_positions * sizeof(double[3]));
    
    // Initialize positions without using initializers
    positions[0][0] = 20.0; positions[0][1] = 0.0; positions[0][2] = 0.0;
    positions[1][0] = 30.0; positions[1][1] = 0.0; positions[1][2] = 0.0;
    positions[2][0] = 40.0; positions[2][1] = 0.0; positions[2][2] = 0.0;
    positions[3][0] = 50.0; positions[3][1] = 0.0; positions[3][2] = 0.0;
    positions[4][0] = 60.0; positions[4][1] = 0.0; positions[4][2] = 0.0;
    
    printf("\nCalculating velocity for circular orbits at various radii:\n");
    printf("------------------------------------------------------\n");
    printf("Radius (M)   |   Orbital Velocity (c)   |   Period (M)\n");
    printf("------------------------------------------------------\n");
    
    for (int i = 0; i < num_positions; i++) {
        double r = positions[i][0];
        double v_phi;
        
        // Calculate orbital velocity for a circular orbit at this radius
        bh_calculate_orbital_velocity(context, r, &v_phi);
        
        // Calculate orbital period
        double period = 2.0 * PI * r / v_phi;
        
        printf("%10.2f   |   %20.6f   |   %10.2f\n", r, v_phi, period);
    }
    
    // Free the allocated memory
    free(positions);
}

/**
 * Test time dilation
 */
void test_time_dilation(int num_positions) {
    double (*positions)[3] = malloc(num_positions * sizeof(*positions));
    if (positions == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < num_positions; i++) {
        positions[i][0] = 0.0;
        positions[i][1] = 0.0;
        positions[i][2] = 0.0;
    }
    
    // Use positions for the test...
    
    free(positions);
}

int main(int argc, char* argv[]) {
    // Print welcome message
    printf("Black Hole Physics Engine - Test Program\n");
    printf("----------------------------------------\n\n");
    
    // Get API version
    int major, minor, patch;
    bh_get_version(&major, &minor, &patch);
    printf("API Version: %d.%d.%d\n\n", major, minor, patch);
    
    // Initialize the engine
    BHContextHandle context = bh_initialize();
    if (context == NULL) {
        printf("Error initializing physics engine\n");
        return 1;
    }
    
    // Configure black hole (10 solar masses)
    BHErrorCode error = bh_configure_black_hole(context, 1.0, 0.0, 0.0);
    if (error != BH_SUCCESS) {
        printf("Error configuring black hole: %d\n", error);
        bh_shutdown(context);
        return 1;
    }
    
    // Configure accretion disk
    error = bh_configure_accretion_disk(context, 6.0, 20.0, 1.0, 1.0);
    if (error != BH_SUCCESS) {
        printf("Error configuring accretion disk: %d\n", error);
        bh_shutdown(context);
        return 1;
    }
    
    // Configure simulation parameters
    error = bh_configure_simulation(context, 0.1, 100.0, 1000, 1.0e-6);
    if (error != BH_SUCCESS) {
        printf("Error configuring simulation: %d\n", error);
        bh_shutdown(context);
        return 1;
    }
    
    // Run tests
    test_ray_tracing(context);
    printf("\n");
    
    test_particle_orbits(context);
    printf("\n");
    
    test_time_dilation(NUM_POSITIONS);
    printf("\n");
    
    // Clean up
    bh_shutdown(context);
    
    printf("Tests completed.\n");
    
    return 0;
} 