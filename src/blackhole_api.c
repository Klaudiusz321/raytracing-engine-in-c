/**
 * blackhole_api.c
 * 
 * Implementation of the public API for the black hole physics engine.
 */

#include "../include/blackhole_api.h"
#include "../include/blackhole_types.h"
#include "../include/spacetime.h"
#include "../include/raytracer.h"
#include "../include/particle_sim.h"
#include "../include/math_util.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef struct BHContext_t {
    BlackHoleParams blackhole;
    AccretionDiskParams disk;
    SimulationConfig config;
    int disk_enabled;
} BHContext;
// Implementation of blackhole_get_mass:
double blackhole_get_mass(BHContextHandle context) {
    if (context == NULL)
        return 0.0;
    // Assuming BHContext_t has a member 'blackhole' of type BlackHoleParams,
    // and that BlackHoleParams has a member 'mass'.
    return context->blackhole.mass;
}

// Implementation of bh_calculate_orbital_velocity:
void bh_calculate_orbital_velocity(BHContextHandle context, double r, double* v_phi) {
    if (context == NULL || v_phi == NULL || r <= 0)
        return;
    double mass = blackhole_get_mass(context);
    // Example: Using a Newtonian circular orbit formula, v = sqrt(mass/r)
    *v_phi = sqrt(mass / r);
}
/**
 * Initialize the black hole physics engine
 */
BHContextHandle bh_initialize(void) {
    BHContext* context = (BHContext*)malloc(sizeof(BHContext));
    if (context == NULL) {
        return NULL;
    }
    
    // Initialize with default values
    
    // Black hole parameters (default: Schwarzschild black hole with mass 1.0)
    context->blackhole.mass = 1.0;
    context->blackhole.schwarzschild_radius = 2.0 * context->blackhole.mass; // 2M in geometric units
    context->blackhole.spin = 0.0;
    context->blackhole.charge = 0.0;
    
    // Accretion disk parameters
    context->disk.inner_radius = 6.0; // Default is at ISCO (6M for Schwarzschild)
    context->disk.outer_radius = 20.0;
    context->disk.temperature_scale = 1.0;
    context->disk.density_scale = 1.0;
    context->disk_enabled = 0;  // Disabled by default
    
    // Simulation parameters
    context->config.time_step = 0.1;
    context->config.max_ray_distance = 100.0;
    context->config.max_integration_steps = 1000;
    context->config.tolerance = 1.0e-6;
    
    return context;
}

/**
 * Shut down the black hole physics engine and free resources
 */
void bh_shutdown(BHContextHandle context) {
    if (context != NULL) {
        free(context);
    }
}

/**
 * Configure the black hole parameters
 */
BHErrorCode bh_configure_black_hole(
    BHContextHandle context,
    double mass,
    double spin,
    double charge)
{
    if (context == NULL) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Validate parameters
    if (mass <= 0.0) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Spin should be in range [0, 1] for Kerr black hole
    if (spin < 0.0 || spin > 1.0) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Initialize all derived parameters
    initialize_black_hole_params(&context->blackhole, mass, spin, charge);
    
    return BH_SUCCESS;
}

/**
 * Configure the accretion disk
 */
BHErrorCode bh_configure_accretion_disk(
    BHContextHandle context,
    double inner_radius,
    double outer_radius,
    double temperature_scale,
    double density_scale)
{
    if (context == NULL) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Validate parameters
    if (inner_radius <= 0.0 || outer_radius <= inner_radius || 
        temperature_scale <= 0.0 || density_scale <= 0.0) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Set parameters
    context->disk.inner_radius = inner_radius;
    context->disk.outer_radius = outer_radius;
    context->disk.temperature_scale = temperature_scale;
    context->disk.density_scale = density_scale;
    context->disk_enabled = 1;  // Enable disk
    
    return BH_SUCCESS;
}

/**
 * Configure simulation parameters
 */
BHErrorCode bh_configure_simulation(
    BHContextHandle context,
    double time_step,
    double max_ray_distance,
    int max_integration_steps,
    double tolerance)
{
    if (context == NULL) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Validate parameters
    if (time_step <= 0.0 || max_ray_distance <= 0.0 || 
        max_integration_steps <= 0 || tolerance <= 0.0) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Set parameters
    context->config.time_step = time_step;
    context->config.max_ray_distance = max_ray_distance;
    context->config.max_integration_steps = max_integration_steps;
    context->config.tolerance = tolerance;
    
    return BH_SUCCESS;
}

/**
 * Trace a ray and get hit information
 */
BHErrorCode bh_trace_ray(
    BHContextHandle context,
    const double origin[3],
    const double direction[3],
    RayTraceHit* hit)
{
    if (context == NULL || origin == NULL || direction == NULL || hit == NULL) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Set up ray
    Ray ray;
    ray.origin.x = origin[0];
    ray.origin.y = origin[1];
    ray.origin.z = origin[2];
    
    ray.direction.x = direction[0];
    ray.direction.y = direction[1];
    ray.direction.z = direction[2];
    
    // Normalize the direction vector
    Vector3D vector3D_normalize(const Vector3D v);
    ray.direction = vector3D_normalize(ray.direction);

    
    // Trace ray
    RayTraceResult result = trace_ray(
        &ray,
        &context->blackhole,
        context->disk_enabled ? &context->disk : NULL,
        &context->config,
        hit);
    
    if (result == RAY_ERROR) {
        return BH_ERROR_SIMULATION;
    }
    
    return BH_SUCCESS;
}

/**
 * Batch ray tracing for efficient rendering
 */
BHErrorCode bh_trace_rays_batch(
    BHContextHandle context,
    const Ray* rays,
    RayTraceHit* hits,
    int count)
{
    if (context == NULL || rays == NULL || hits == NULL || count <= 0) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Trace rays in batch
    for (int i = 0; i < count; i++) {
        RayTraceResult result = trace_ray(
            &rays[i],
            &context->blackhole,
            context->disk_enabled ? &context->disk : NULL,
            &context->config,
            &hits[i]);
        
        if (result == RAY_ERROR) {
            return BH_ERROR_SIMULATION;
        }
    }
    
    return BH_SUCCESS;
}


/**
 * Create a particle system
 */
void* bh_create_particle_system(
    BHContextHandle context,
    int capacity)
{
    if (context == NULL || capacity <= 0) {
        return NULL;
    }
    
    ParticleSystem* system = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    if (system == NULL) {
        return NULL;
    }
    
    if (particle_system_init(system, capacity) != 0) {
        free(system);
        return NULL;
    }
    
    return system;
}

/**
 * Destroy a particle system
 */
void bh_destroy_particle_system(
    BHContextHandle context,
    void* system)
{
    if (context == NULL || system == NULL) {
        return;
    }
    
    ParticleSystem* particle_system = (ParticleSystem*)system;
    particle_system_cleanup(particle_system);
    free(particle_system);
}

/**
 * Add a test particle to the system
 */
int bh_add_test_particle(
    BHContextHandle context,
    void* system,
    const double position[3],
    const double velocity[3],
    double mass)
{
    if (context == NULL || system == NULL || position == NULL || velocity == NULL || mass < 0.0) {
        return -1;
    }
    
    ParticleSystem* particle_system = (ParticleSystem*)system;
    
    Vector3D pos = {position[0], position[1], position[2]};
    Vector3D vel = {velocity[0], velocity[1], velocity[2]};
    
    return add_particle(particle_system, &pos, &vel, mass, PARTICLE_TEST);
}

/**
 * Create an accretion disk in the particle system
 */
int bh_create_accretion_disk_particles(
    BHContextHandle context,
    void* system,
    int num_particles)
{
    if (context == NULL || system == NULL || num_particles <= 0) {
        return -1;
    }
    
    if (!context->disk_enabled) {
        return 0;  // Disk is not enabled
    }
    
    ParticleSystem* particle_system = (ParticleSystem*)system;
    
    return create_accretion_disk(
        particle_system,
        &context->blackhole,
        &context->disk,
        num_particles);
}

/**
 * Generate Hawking radiation particles
 */
int bh_generate_hawking_radiation(
    BHContextHandle context,
    void* system,
    int num_particles)
{
    if (context == NULL || system == NULL || num_particles <= 0) {
        return -1;
    }
    
    ParticleSystem* particle_system = (ParticleSystem*)system;
    
    return generate_hawking_radiation(
        particle_system,
        &context->blackhole,
        num_particles,
        &context->config);
}

/**
 * Update all particles in the system for one time step
 */
BHErrorCode bh_update_particles(
    BHContextHandle context,
    void* system)
{
    if (context == NULL || system == NULL) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    ParticleSystem* particle_system = (ParticleSystem*)system;
    
    if (update_particles(particle_system, &context->blackhole, &context->config) != 0) {
        return BH_ERROR_SIMULATION;
    }
    
    return BH_SUCCESS;
}

/**
 * Get particle data for rendering
 */
BHErrorCode bh_get_particle_data(
    BHContextHandle context,
    void* system,
    double* positions,
    double* velocities,
    int* types,
    int* count)
{
    if (context == NULL || system == NULL || positions == NULL || 
        velocities == NULL || types == NULL || count == NULL || *count <= 0) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    ParticleSystem* particle_system = (ParticleSystem*)system;
    
    int max_count = *count;
    int active_count = 0;
    
    // Copy data for active particles
    for (int i = 0; i < particle_system->count && active_count < max_count; i++) {
        Particle* p = &particle_system->particles[i];
        
        if (p->active) {
            // Copy position
            positions[active_count * 3 + 0] = p->position.x;
            positions[active_count * 3 + 1] = p->position.y;
            positions[active_count * 3 + 2] = p->position.z;
            
            // Copy velocity
            velocities[active_count * 3 + 0] = p->velocity.x;
            velocities[active_count * 3 + 1] = p->velocity.y;
            velocities[active_count * 3 + 2] = p->velocity.z;
            
            // Copy type
            types[active_count] = p->type;
            
            active_count++;
        }
    }
    
    *count = active_count;
    
    return BH_SUCCESS;
}

/**
 * Calculate time dilation between two points
 */
BHErrorCode bh_calculate_time_dilation(
    BHContextHandle context,
    const double position1[3],
    const double position2[3],
    double* time_ratio)
{
    if (context == NULL || position1 == NULL || position2 == NULL || time_ratio == NULL) {
        return BH_ERROR_INVALID_PARAMETER;
    }
    
    // Calculate distance from black hole for each position
    double r1 = sqrt(position1[0] * position1[0] + 
                    position1[1] * position1[1] + 
                    position1[2] * position1[2]);
                    
    double r2 = sqrt(position2[0] * position2[0] + 
                    position2[1] * position2[1] + 
                    position2[2] * position2[2]);
    
    // Calculate time dilation factors
    double dilation1 = calculate_time_dilation(r1, &context->blackhole);
    double dilation2 = calculate_time_dilation(r2, &context->blackhole);
    
    // Compute the ratio of time passage
    *time_ratio = dilation1 / dilation2;
    
    return BH_SUCCESS;
}

/**
 * Get the version of the API
 */
void bh_get_version(int* major, int* minor, int* patch) {
    if (major != NULL) {
        *major = BLACKHOLE_API_VERSION_MAJOR;
    }
    
    if (minor != NULL) {
        *minor = BLACKHOLE_API_VERSION_MINOR;
    }
    
    if (patch != NULL) {
        *patch = BLACKHOLE_API_VERSION_PATCH;
    }
} 

// In blackhole_api.c



