/**
 * particle_sim.h
 * 
 * Particle simulation for orbits and Hawking radiation around a black hole.
 */

#ifndef PARTICLE_SIM_H
#define PARTICLE_SIM_H

#include "blackhole_types.h"
#include "math_util.h"

/**
 * Type of particle in the simulation
 */
typedef enum {
    PARTICLE_TEST,       /* Test particle for trajectory visualization */
    PARTICLE_DISK,       /* Particle in the accretion disk */
    PARTICLE_HAWKING,    /* Particle representing Hawking radiation */
    PARTICLE_JET         /* Particle in relativistic jet */
} ParticleType;

/**
 * Particle state structure
 */
typedef struct {
    Vector3D position;    /* Current position */
    Vector3D velocity;    /* Current velocity */
    Vector3D acceleration; /* Current acceleration */
    double mass;          /* Particle mass */
    double energy;        /* Particle energy */
    double angular_momentum; /* Particle angular momentum */
    double proper_time;   /* Accumulated proper time */
    double coordinate_time; /* Coordinate time */
    ParticleType type;    /* Type of particle */
    int active;           /* Whether particle is active (1) or not (0) */
    int id;              /* Unique particle identifier */
    double age;          /* Time since particle creation */
    double temperature;  /* Temperature (for thermal particles) */
    double time_dilation; /* Time dilation factor at current position */
} Particle;

/**
 * Parameters for orbital calculations
 */
typedef struct {
    double semi_major_axis;  /* Semi-major axis of the orbit */
    double eccentricity;     /* Eccentricity of the orbit */
    double inclination;      /* Inclination to the equatorial plane (radians) */
    double longitude_of_ascending_node; /* Longitude of ascending node (radians) */
    double argument_of_periapsis;      /* Argument of periapsis (radians) */
    double mean_anomaly;  
    double specific_angular_momentum;  // <-- Add this field
    double specific_energy;        /* Mean anomaly at epoch (radians) */
} OrbitalParams;

/**
 * System of particles for simulation
 */
typedef struct {
    Particle* particles;        /* Array of particles */
    int capacity;               /* Maximum number of particles */
    int count;                  /* Current number of particles */
    int next_id;                /* Next available particle ID */
    BlackHoleParams* blackhole; /* Reference to black hole parameters */
} ParticleSystem;

/**
 * Initialize a particle system
 * 
 * @param system Particle system to initialize
 * @param initial_capacity Initial capacity for the particle array
 * @return 0 on success, non-zero on failure
 */
int particle_system_init(ParticleSystem* system, int initial_capacity);

/**
 * Free resources used by a particle system
 * 
 * @param system Particle system to clean up
 */
void particle_system_cleanup(ParticleSystem* system);

/**
 * Add a particle to the system
 * 
 * @param system Particle system
 * @param position Initial position
 * @param velocity Initial velocity
 * @param mass Particle mass
 * @param type Particle type
 * @return Index of added particle or -1 on failure
 */
int add_particle(
    ParticleSystem* system,
    const Vector3D* position,
    const Vector3D* velocity,
    double mass,
    ParticleType type);

/**
 * Update all particles in the system for one time step
 * 
 * @param system Particle system to update
 * @param blackhole Black hole parameters
 * @param config Simulation configuration
 * @return 0 on success, non-zero on failure
 */
int update_particles(
    ParticleSystem* system,
    const BlackHoleParams* blackhole,
    const SimulationConfig* config);

/**
 * Calculate stable circular orbit parameters at a given radius
 * 
 * @param r Orbit radius
 * @param blackhole Black hole parameters
 * @param velocity Output orbital velocity
 * @return 0 if stable orbit exists at this radius, non-zero otherwise
 */
int calculate_circular_orbit(
    double r,
    const BlackHoleParams* blackhole,
    Vector3D* velocity);

/**
 * Create particles representing the accretion disk
 * 
 * @param system Particle system to add particles to
 * @param blackhole Black hole parameters
 * @param disk Accretion disk parameters
 * @param num_particles Number of particles to create
 * @return Number of particles successfully added
 */
int create_accretion_disk(
    ParticleSystem* system,
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    int num_particles);

/**
 * Generate Hawking radiation particles near the event horizon
 * This is a simplified visual representation, not physically accurate
 * 
 * @param system Particle system
 * @param blackhole Black hole parameters
 * @param num_particles Number of particles to generate
 * @param config Simulation configuration
 * @return Number of particles successfully added
 */
int generate_hawking_radiation(
    ParticleSystem* system,
    const BlackHoleParams* blackhole,
    int num_particles,
    const SimulationConfig* config);

/**
 * Get the innermost stable circular orbit (ISCO) radius
 * 
 * @param blackhole Black hole parameters
 * @return ISCO radius
 */


#endif /* PARTICLE_SIM_H */ 