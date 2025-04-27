/**
 * blackhole_api.h
 * 
 * Public API for the black hole physics engine. This interface is designed
 * to be used by different frontend platforms (web via WASM or native desktop).
 */

#ifndef BLACKHOLE_API_H
#define BLACKHOLE_API_H

#include <stdint.h>
#include <stddef.h>

#include "blackhole_types.h"
#include "math_util.h"
#include "spacetime.h"
#include "raytracer.h"
#include "particle_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Version information */
#define BLACKHOLE_API_VERSION_MAJOR 0
#define BLACKHOLE_API_VERSION_MINOR 1
#define BLACKHOLE_API_VERSION_PATCH 0

/* Error codes */
typedef enum {
    BH_SUCCESS = 0,
    BH_ERROR_INVALID_PARAMETER = -1,
    BH_ERROR_MEMORY_ALLOCATION = -2,
    BH_ERROR_INITIALIZATION = -3,
    BH_ERROR_SIMULATION = -4
} BHErrorCode;



typedef struct BHContext_t* BHContextHandle;

/**
 * Initialize the black hole physics engine
 * 
 * @return A handle to the engine context or NULL on failure
 */
BHContextHandle bh_initialize(void);

/**
 * Shut down the black hole physics engine and free resources
 * 
 * @param context Engine context handle
 */
void bh_calculate_orbital_velocity(BHContextHandle context, double r, double* v_phi);
double blackhole_get_mass(BHContextHandle context);
void bh_shutdown(BHContextHandle context);



/**
 * Configure the black hole parameters
 * 
 * @param context Engine context handle
 * @param mass Black hole mass (geometric units)
 * @param spin Black hole spin (a/M, dimensionless)
 * @param charge Black hole charge (Q, optional)
 * @return Error code
 */
BHErrorCode bh_configure_black_hole(
    BHContextHandle context,
    double mass,
    double spin,
    double charge);

/**
 * Configure the accretion disk
 * 
 * @param context Engine context handle
 * @param inner_radius Inner radius of the disk
 * @param outer_radius Outer radius of the disk
 * @param temperature_scale Temperature scaling factor
 * @param density_scale Density scaling factor
 * @return Error code
 */
BHErrorCode bh_configure_accretion_disk(
    BHContextHandle context,
    double inner_radius,
    double outer_radius,
    double temperature_scale,
    double density_scale);

/**
 * Configure simulation parameters
 * 
 * @param context Engine context handle
 * @param time_step Time step for integration
 * @param max_ray_distance Maximum ray tracing distance
 * @param max_integration_steps Maximum integration steps
 * @param tolerance Integration tolerance
 * @return Error code
 */
BHErrorCode bh_configure_simulation(
    BHContextHandle context,
    double time_step,
    double max_ray_distance,
    int max_integration_steps,
    double tolerance);

/**
 * Trace a ray and get hit information
 * 
 * @param context Engine context handle
 * @param origin Ray origin (3D vector)
 * @param direction Ray direction (3D vector, will be normalized)
 * @param hit Output hit information
 * @return Error code
 */
BHErrorCode bh_trace_ray(
    BHContextHandle context,
    const double origin[3],
    const double direction[3],
    RayTraceHit* hit);

/**
 * Batch ray tracing for efficient rendering
 * 
 * @param context Engine context handle
 * @param rays Array of rays to trace
 * @param hits Output array for hit information
 * @param count Number of rays to trace
 * @return Error code
 */
BHErrorCode bh_trace_rays_batch(
    BHContextHandle context,
    const Ray* rays,
    RayTraceHit* hits,
    int count);

/**
 * Create a particle system
 * 
 * @param context Engine context handle
 * @param capacity Initial capacity
 * @return Handle to the particle system or NULL on failure
 */
void* bh_create_particle_system(
    BHContextHandle context,
    int capacity);

/**
 * Destroy a particle system
 * 
 * @param context Engine context handle
 * @param system Particle system handle
 */
void bh_destroy_particle_system(
    BHContextHandle context,
    void* system);

/**
 * Add a test particle to the system
 * 
 * @param context Engine context handle
 * @param system Particle system handle
 * @param position Initial position
 * @param velocity Initial velocity
 * @param mass Particle mass
 * @return Index of the new particle or negative value on error
 */
int bh_add_test_particle(
    BHContextHandle context,
    void* system,
    const double position[3],
    const double velocity[3],
    double mass);

/**
 * Create an accretion disk in the particle system
 * 
 * @param context Engine context handle
 * @param system Particle system handle
 * @param num_particles Number of particles to create
 * @return Number of particles created or negative value on error
 */
int bh_create_accretion_disk_particles(
    BHContextHandle context,
    void* system,
    int num_particles);

/**
 * Generate Hawking radiation particles
 * 
 * @param context Engine context handle
 * @param system Particle system handle
 * @param num_particles Number of particles to generate
 * @return Number of particles created or negative value on error
 */
int bh_generate_hawking_radiation(
    BHContextHandle context,
    void* system,
    int num_particles);

/**
 * Update all particles in the system for one time step
 * 
 * @param context Engine context handle
 * @param system Particle system handle
 * @return Error code
 */
BHErrorCode bh_update_particles(
    BHContextHandle context,
    void* system);

/**
 * Get particle data for rendering
 * 
 * @param context Engine context handle
 * @param system Particle system handle
 * @param positions Output array for positions (3 doubles per particle)
 * @param velocities Output array for velocities (3 doubles per particle)
 * @param types Output array for particle types
 * @param count Input: maximum number of particles to get; Output: actual number
 * @return Error code
 */
BHErrorCode bh_get_particle_data(
    BHContextHandle context,
    void* system,
    double* positions,
    double* velocities,
    int* types,
    int* count);

/**
 * Calculate time dilation between two points
 * 
 * @param context Engine context handle
 * @param position1 First position (3D vector)
 * @param position2 Second position (3D vector)
 * @param time_ratio Output - relative time passage ratio
 * @return Error code
 */
BHErrorCode bh_calculate_time_dilation(
    BHContextHandle context,
    const double position1[3],
    const double position2[3],
    double* time_ratio);

/**
 * Get the version of the API
 * 
 * @param major Output parameter for major version
 * @param minor Output parameter for minor version
 * @param patch Output parameter for patch version
 */
void bh_get_version(int* major, int* minor, int* patch);

/**
 * Generate ray tracing data for visualization
 * 
 * @param context The black hole physics context
 * @param observer_pos Observer position (x,y,z)
 * @param observer_dir Observer direction (normalized)
 * @param up_vector Camera up vector (normalized)
 * @param width Width of output buffer
 * @param height Height of output buffer
 * @param fov Field of view in radians
 * @param enable_doppler Enable Doppler effect (1=on, 0=off)
 * @param enable_redshift Enable gravitational redshift (1=on, 0=off)
 * @param show_disk Show accretion disk (1=on, 0=off)
 * @param output_buffer Output RGBA buffer (size = width*height*4)
 * 
 * @return BH_SUCCESS on success, error code otherwise
 */
BHErrorCode bh_generate_shader_data(
    void* context,
    const float observer_pos[3],
    const float observer_dir[3],
    const float up_vector[3],
    int width, 
    int height,
    float fov,
    int enable_doppler,
    int enable_redshift,
    int show_disk,
    float* output_buffer
);

#ifdef __cplusplus
}
#endif

#endif /* BLACKHOLE_API_H */ 