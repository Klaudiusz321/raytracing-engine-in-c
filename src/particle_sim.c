/**
 * particle_sim.c
 * 
 * Implementation of particle simulation for orbits and Hawking radiation.
 */

#include "../include/particle_sim.h"
#include "../include/spacetime.h"
#include "../include/math_util.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

/**
 * Structure to hold particle integration state
 */
typedef struct {
    const BlackHoleParams* blackhole;
    double mass;
} ParticleIntegrationParams;

/**
 * Compute derivative function for particle integration
 * 
 * @param state Current state (position and velocity)
 * @param derivatives Output derivatives
 * @param params Integration parameters
 */

static void particle_derivatives(const double state[], double derivatives[], void* params)
{
    (void)state; (void)derivatives; (void)params;
    ParticleIntegrationParams* particle_params = (ParticleIntegrationParams*)params;
    // Extract position and velocity components
    double position[4] = {
        state[0],  // t
        state[1],  // r
        state[2],  // theta
        state[3]   // phi
    };
    
    double velocity[4] = {
        state[4],  // dt/ds
        state[5],  // dr/ds
        state[6],  // dθ/ds
        state[7]   // dφ/ds
    };
    
    // Calculate acceleration using geodesic equation
    double acceleration[4] = {0};
    geodesic_equation(position, velocity, particle_params->blackhole, acceleration);
    
    // Fill derivatives array
    // First 4 entries are velocities
    derivatives[0] = velocity[0];
    derivatives[1] = velocity[1];
    derivatives[2] = velocity[2];
    derivatives[3] = velocity[3];
    
    // Next 4 entries are accelerations
    derivatives[4] = acceleration[0];
    derivatives[5] = acceleration[1];
    derivatives[6] = acceleration[2];
    derivatives[7] = acceleration[3];
}

/**
 * Initialize a particle system
 */
int particle_system_init(ParticleSystem* system, int capacity) {
    if (system == NULL || capacity <= 0) {
        return -1;
    }
    
    system->particles = (Particle*)malloc(capacity * sizeof(Particle));
    if (system->particles == NULL) {
        return -1;
    }
    
    system->capacity = capacity;
    system->count = 0;
    system->next_id = 1;
    
    // Initialize random number generator
    srand((unsigned int)time(NULL));
    
    return 0;
}

/**
 * Clean up a particle system
 */
void particle_system_cleanup(ParticleSystem* system) {
    if (system != NULL && system->particles != NULL) {
        free(system->particles);
        system->particles = NULL;
        system->capacity = 0;
        system->count = 0;
    }
}

/**
 * Add a particle to the system
 */
int add_particle(
    ParticleSystem* system,
    const Vector3D* position,
    const Vector3D* velocity,
    double mass,
    ParticleType type)
{
    if (system->count >= system->capacity) {
        return -1; // System full
    }
    
    // Get next available particle
    Particle* p = &system->particles[system->count++];
    
    // Initialize particle
    p->id = system->next_id++;
    p->position = *position; // Direct assignment for Vector3D
    p->velocity = *velocity; // Direct assignment for Vector3D
    p->mass = mass;
    p->type = type;
    p->active = 1;
    p->age = 0.0;
    p->temperature = 0.0;
    
    return p->id;
}

/**
 * Find a particle by ID
 */
Particle* find_particle(ParticleSystem* system, int particle_id) {
    if (system == NULL || particle_id <= 0) {
        return NULL;
    }
    
    for (int i = 0; i < system->count; i++) {
        if (system->particles[i].id == particle_id && system->particles[i].active) {
            return &system->particles[i];
        }
    }
    
    return NULL;
}

/**
 * Remove a particle from the system
 */
int remove_particle(ParticleSystem* system, int particle_id) {
    if (system == NULL || particle_id <= 0) {
        return -1;
    }
    
    for (int i = 0; i < system->count; i++) {
        if (system->particles[i].id == particle_id) {
            system->particles[i].active = 0;  // Mark as inactive
            return 0;
        }
    }
    
    return -1;  // Particle not found
}

/**
 * Calculate Keplerian orbital parameters for a particle
 */
static void calculate_orbit_parameters(
    const Vector3D* position,
    const Vector3D* velocity,
    const BlackHoleParams* blackhole,
    OrbitalParams* params)
{
    double r = vector3D_length(*position);
    double v = vector3D_length(*velocity);
    
    // Calculate specific angular momentum
    Vector3D l_vec = vector3D_cross(*position, *velocity);
    double L = vector3D_length(l_vec);
    
    // Calculate specific energy
    double E = 0.5 * v * v - blackhole->mass / r;
    
    // Calculate eccentricity
    Vector3D e_vec;
    Vector3D r_hat = vector3D_normalize(*position);
    
    // e = ((v^2 - μ/r)r - (r·v)v) / μ
    Vector3D term1 = vector3D_scale(r_hat, v * v - blackhole->mass / r);
    double r_dot_v = vector3D_dot(*position, *velocity);
    Vector3D term2 = vector3D_scale(*velocity, r_dot_v);
    e_vec = vector3D_scale(vector3D_sub(term1, term2), 1.0 / blackhole->mass);
    
    double e = vector3D_length(e_vec);
    
    // Calculate semi-major axis
    double a;
    if (E < 0) {
        // Bound orbit (elliptical)
        a = -blackhole->mass / (2.0 * E);
    } else if (E > 0) {
        // Unbound orbit (hyperbolic)
        a = blackhole->mass / (2.0 * E);
    } else {
        // Parabolic orbit (e = 1)
        a = INFINITY;
    }
    
    // Calculate inclination
    // Inclination is the angle between angular momentum vector and z-axis
    Vector3D z_axis = {0, 0, 1};
    (void)z_axis;
    double cos_i = l_vec.z / L;
    double inclination = acos(cos_i);
    
    // Fill in parameters
    params->semi_major_axis = a;
    params->eccentricity = e;
    params->inclination = inclination;
    params->specific_angular_momentum = L;
    params->specific_energy = E;
}

/**
 * Update particle position using geodesic equation
 */
static void update_particle_geodesic(
    Particle* particle,
    const BlackHoleParams* blackhole,
    const SimulationConfig* config)
{
    // Convert Cartesian to spherical coordinates
    Vector3D spherical_pos;
    cartesian_to_spherical(&particle->position, &spherical_pos);
    
    double r = spherical_pos.x;
    double theta = spherical_pos.y;
    double phi = spherical_pos.z;
    
    // Extract velocity components
    Vector3D vel = particle->velocity;
    
    // Calculate new position and velocity
    // Simple Euler integration for demonstration
    
    // For Schwarzschild metric, calculate derivatives
    double rs = blackhole->schwarzschild_radius;
    double factor = 1.0 - rs / r;
    (void)factor;
    
    // Update position
    r += vel.x * config->time_step;
    theta += vel.y * config->time_step;
    phi += vel.z * config->time_step;
    
    // Create new position in spherical coordinates
    Vector3D new_spherical = {r, theta, phi};
    
    // Convert back to Cartesian
    Vector3D new_position;
    spherical_to_cartesian(&new_spherical, &new_position);
    
    // Update particle
    particle->position = new_position;
    
    // Calculate time dilation at the new position
    particle->time_dilation = calculate_time_dilation(r, blackhole);
}

/**
 * Update particle position using Newtonian gravity
 */
static void update_particle_newtonian(
    Particle* particle,
    const BlackHoleParams* blackhole,
    const SimulationConfig* config)
{
    // Calculate distance from black hole
    double r = vector3D_length(particle->position);
    
    // Calculate acceleration due to gravity
    double accel_mag = blackhole->mass / (r * r);
    
    // Direction of acceleration (towards black hole)
    Vector3D accel_dir = vector3D_scale(particle->position, -1.0 / r);
    
    // Calculate acceleration vector
    Vector3D acceleration = vector3D_scale(accel_dir, accel_mag);
    
    // Update velocity (Euler integration)
    particle->velocity = vector3D_add(
        particle->velocity,
        vector3D_scale(acceleration, config->time_step)
    );
    
    // Update position
    particle->position = vector3D_add(
        particle->position,
        vector3D_scale(particle->velocity, config->time_step)
    );
}

/**
 * Create a disk of particles around the black hole
 */
int create_accretion_disk(
    ParticleSystem* system,
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    int num_particles)
{
    if (system == NULL || blackhole == NULL || disk == NULL || num_particles <= 0) {
        return -1;
    }
    
    // Check if there's enough space in the particle system
    if (system->count + num_particles > system->capacity) {
        return -1;
    }
    
    // Calculate inner and outer disk radius
    double inner_radius = disk->inner_radius;
    double outer_radius = disk->outer_radius;
    
    // Make sure inner radius is at least the ISCO
    if (inner_radius < blackhole->isco_radius) {
        inner_radius = blackhole->isco_radius;
    }
    
    // Ensure inner radius is not inside the event horizon
    if (inner_radius < blackhole->schwarzschild_radius) {
        inner_radius = blackhole->schwarzschild_radius * 1.1;
    }
    
    // Create particles distributed in the disk
    int particles_created = 0;
    
    for (int i = 0; i < num_particles; i++) {
        // Calculate radius using square root distribution for uniform density
        double t = (double)i / (num_particles - 1);
        double r = inner_radius + (outer_radius - inner_radius) * sqrt(t);
        
        // Generate random angle
        double phi = ((double)rand() / RAND_MAX) * 2.0 * PI;
        
        // Calculate position in the disk plane
        Vector3D position = {
            r * cos(phi),
            r * sin(phi),
            ((double)rand() / RAND_MAX - 0.5) * disk->thickness_factor * r  // Small z-variation for thickness
        };
        
        // Calculate Keplerian orbital velocity
        double v_orbit = sqrt(blackhole->mass / r);
        
        // Tangential velocity vector (perpendicular to radius)
        Vector3D velocity = {
            -position.y * v_orbit / r,
            position.x * v_orbit / r,
            0.0
        };
        
        // Add some small random variation to velocity (for disk turbulence)
        double v_random = v_orbit * 0.05;  // 5% random component
        velocity.x += ((double)rand() / RAND_MAX - 0.5) * v_random;
        velocity.y += ((double)rand() / RAND_MAX - 0.5) * v_random;
        velocity.z += ((double)rand() / RAND_MAX - 0.5) * v_random;
        
        // Calculate temperature based on distance
        double temp_factor = pow(inner_radius / r, 0.75);  // T ~ r^(-3/4) for thin disk
        double temperature = disk->temperature_scale * 10000.0 * temp_factor;
        
        // Create particle
        int id = add_particle(system, &position, &velocity, 0.0, PARTICLE_DISK);
        if (id < 0) {
            break;
        }
        
        // Set particle temperature
        Particle* p = find_particle(system, id);
        if (p != NULL) {
            p->temperature = temperature;
        }
        
        particles_created++;
    }
    
    return particles_created;
}

/**
 * Generate Hawking radiation particles
 */
int generate_hawking_radiation(
    ParticleSystem* system,
    const BlackHoleParams* blackhole,
    int num_particles,
    const SimulationConfig* config)
{
    if (system == NULL || blackhole == NULL || num_particles <= 0) {
        return -1;
    }
    
    // Check if there's enough space in the particle system
    if (system->count + num_particles > system->capacity) {
        return -1;
    }
    
    // Calculate Hawking temperature
    // T = ℏc³/(8πGMk) = 1/(8πM) in geometric units
    double hawking_temp = 1.0 / (8.0 * PI * blackhole->mass);
    
    // Scale temperature for visualization
    hawking_temp *= config->hawking_temp_factor;
    
    // Generate particles
    int particles_created = 0;
    
    for (int i = 0; i < num_particles; i++) {
        // Generate random position on event horizon
        double theta = ((double)rand() / RAND_MAX) * PI;
        double phi = ((double)rand() / RAND_MAX) * 2.0 * PI;
        
        double r = blackhole->schwarzschild_radius * 1.01;  // Just outside horizon
        
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        double sin_phi = sin(phi);
        double cos_phi = cos(phi);
        
        Vector3D position = {
            r * sin_theta * cos_phi,
            r * sin_theta * sin_phi,
            r * cos_theta
        };
        
        // Outward radial velocity
        double speed_of_light = 1.0;  // Geometric units
        Vector3D radial_dir = vector3D_normalize(position);
        
        Vector3D velocity = vector3D_scale(radial_dir, speed_of_light * 0.9);
        
        // Add some random perturbation to velocity
        velocity.x += ((double)rand() / RAND_MAX - 0.5) * 0.2;
        velocity.y += ((double)rand() / RAND_MAX - 0.5) * 0.2;
        velocity.z += ((double)rand() / RAND_MAX - 0.5) * 0.2;
        
        // Normalize to maintain near light speed
        velocity = vector3D_scale(vector3D_normalize(velocity), speed_of_light * 0.9);
        
        // Create particle
        int id = add_particle(system, &position, &velocity, 0.0, PARTICLE_HAWKING);
        if (id < 0) {
            break;
        }
        
        // Set particle temperature
        Particle* p = find_particle(system, id);
        if (p != NULL) {
            p->temperature = hawking_temp;
        }
        
        particles_created++;
    }
    
    return particles_created;
}

/**
 * Update all particles in the system
 */
int update_particles(
    ParticleSystem* system,
    const BlackHoleParams* blackhole,
    const SimulationConfig* config)
{
    if (system == NULL || blackhole == NULL || config == NULL) {
        return -1;
    }
    
    double event_horizon_radius = blackhole->schwarzschild_radius;
    
    // Update each active particle
    for (int i = 0; i < system->count; i++) {
        Particle* p = &system->particles[i];
        
        if (!p->active) {
            continue;
        }
        
        // Increase particle age
        p->age += config->time_step;
        
        // Choose integration method based on particle type and distance
        double r = vector3D_length(p->position);
        
        if (p->type == PARTICLE_TEST && r < 20.0 * event_horizon_radius) {
            // Use relativistic geodesic equation for test particles near the black hole
            update_particle_geodesic(p, blackhole, config);
        } else {
            // Use Newtonian gravity for particles far from the black hole
            update_particle_newtonian(p, blackhole, config);
        }
        
        // Check if particle has crossed the event horizon
        r = vector3D_length(p->position);
        
        if (r <= event_horizon_radius) {
            // Particle is captured by black hole
            p->active = 0;
            continue;
        }
        
        // Additional updates based on particle type
        switch (p->type) {
            case PARTICLE_DISK:
                // Handle disk particles
                break;
            case PARTICLE_HAWKING:
                // Handle Hawking radiation particles
                break;
            case PARTICLE_JET:
                // Handle jet particles: add your handling code here (or leave a comment if not needed)
                break;
            default:
                // Optionally handle unexpected values
                break;
        }

    }
    
    return 0;
}

/**
 * Calculate orbital parameters for a test particle
 */
int calculate_particle_orbit(
    const ParticleSystem* system,
    int particle_id,
    const BlackHoleParams* blackhole,
    OrbitalParams* params)
{
    if (system == NULL || blackhole == NULL || params == NULL) {
        return -1;
    }
    
    // Find the particle
    Particle* p = NULL;
    
    for (int i = 0; i < system->count; i++) {
        if (system->particles[i].id == particle_id && system->particles[i].active) {
            p = &system->particles[i];
            break;
        }
    }
    
    if (p == NULL) {
        return -1;  // Particle not found
    }
    
    // Calculate orbital parameters
    calculate_orbit_parameters(&p->position, &p->velocity, blackhole, params);
    
    return 0;
}

/**
 * Calculate stable circular orbit parameters at a given radius
 */
int calculate_circular_orbit(
    double r,
    const BlackHoleParams* blackhole,
    Vector3D* velocity)
{
    // Check if radius is outside the innermost stable circular orbit
    double isco_radius = get_isco_radius(blackhole);
    if (r <= isco_radius) {
        return -1; // No stable orbit exists
    }
    
    // For a Schwarzschild black hole, the orbital velocity is given by
    // v = sqrt(M/r)
    double v = sqrt(blackhole->mass / r);
    
    // Create velocity vector perpendicular to radial direction
    // For simplicity, we'll put the orbit in the x-y plane
    velocity->x = -v * sin(0.0); // phi = 0.0 for position on x-axis
    velocity->y = v * cos(0.0);
    velocity->z = 0.0;
    
    return 0;
}

/**
 * Get the innermost stable circular orbit (ISCO) radius
 */
