#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/blackhole_api.h"

/**
 * Simple test for the fixed black hole physics engine
 */
int main(int argc, char* argv[]) {
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
    
    // Configure black hole (10 solar masses, with spin)
    double mass = 1.0;  // Mass in geometric units
    double spin = 0.6;  // Moderate spin (a/M)
    double charge = 0.0; // No charge
    
    BHErrorCode error = bh_configure_black_hole(context, mass, spin, charge);
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
    
    // Test 1: Time dilation at different radii
    printf("\nTest 1: Time Dilation at Different Radii\n");
    printf("---------------------------------------\n");
    
    // Define reference position (far from black hole)
    double ref_pos[3] = {100.0, 0.0, 0.0};
    
    // Test positions at different radii
    double test_radii[] = {3.0, 6.0, 10.0, 20.0, 50.0};
    int num_tests = sizeof(test_radii) / sizeof(test_radii[0]);
    
    for (int i = 0; i < num_tests; i++) {
        double test_pos[3] = {test_radii[i], 0.0, 0.0};
        double time_ratio;
        
        error = bh_calculate_time_dilation(context, test_pos, ref_pos, &time_ratio);
        
        if (error == BH_SUCCESS) {
            printf("At r = %.2f M: 1 second here = %.4f seconds at reference\n", 
                   test_radii[i], time_ratio);
        } else {
            printf("Error calculating time dilation at r = %.2f M\n", test_radii[i]);
        }
    }
    
    // Test 2: Gravitational lensing
    printf("\nTest 2: Gravitational Lensing\n");
    printf("--------------------------\n");
    
    // Trace rays passing at different impact parameters
    const int num_rays = 5;
    Ray rays[num_rays];
    RayTraceHit hits[num_rays];
    
    // Origin far from black hole
    double origin[3] = {0.0, 0.0, 100.0};
    
    // Different impact parameters
    double impacts[] = {0.0, 2.0, 5.0, 10.0, 20.0};
    
    for (int i = 0; i < num_rays; i++) {
        rays[i].origin.x = origin[0];
        rays[i].origin.y = origin[1];
        rays[i].origin.z = origin[2];
        
        // Direction toward black hole with offset
        rays[i].direction.x = impacts[i];
        rays[i].direction.y = 0.0;
        rays[i].direction.z = -1.0;
        
        // Normalize direction (will be done internally, but let's be explicit)
        double len = sqrt(rays[i].direction.x * rays[i].direction.x + 
                         rays[i].direction.y * rays[i].direction.y + 
                         rays[i].direction.z * rays[i].direction.z);
        
        rays[i].direction.x /= len;
        rays[i].direction.y /= len;
        rays[i].direction.z /= len;
    }
    
    error = bh_trace_rays_batch(context, rays, hits, num_rays);
    
    if (error == BH_SUCCESS) {
        for (int i = 0; i < num_rays; i++) {
            printf("Ray with impact parameter b = %.1f:\n", impacts[i]);
            printf("  Result: ");
            
            switch (hits[i].result) {
                case RAY_HORIZON:
                    printf("Captured by black hole\n");
                    break;
                case RAY_DISK:
                    printf("Hit accretion disk at (%.2f, %.2f, %.2f)\n",
                           hits[i].hit_position.x, 
                           hits[i].hit_position.y, 
                           hits[i].hit_position.z);
                    break;
                case RAY_BACKGROUND:
                    printf("Escaped to background\n");
                    break;
                default:
                    printf("Other result (%d)\n", hits[i].result);
                    break;
            }
            
            printf("  Deflection: %.2f degrees\n\n", 
                   acos(hits[i].sky_direction.z) * 180.0 / 3.14159265358979323846);
        }
    } else {
        printf("Error tracing rays: %d\n", error);
    }
    
    // Clean up
    bh_shutdown(context);
    
    printf("\nTests completed.\n");
    return 0;
} 