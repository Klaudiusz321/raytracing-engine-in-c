/**
 * raytracer.h
 * 
 * Core ray tracing functions to simulate light paths in curved spacetime.
 */

#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "blackhole_types.h"
#include "math_util.h"

/**
 * Result of a ray trace operation
 */
typedef enum {
    RAY_HORIZON,     /* Ray reached the event horizon */
    RAY_DISK,        /* Ray hit the accretion disk */
    RAY_BACKGROUND,  /* Ray reached background (stars/sky) */
    RAY_MAX_DISTANCE, /* Ray reached max distance without hitting anything */
    RAY_MAX_STEPS,    /* Maximum integration steps reached */
    RAY_ERROR         /* Numerical or other error */
} RayTraceResult;

/**
 * Integration method to use for ray tracing
 */
typedef enum {
    INTEGRATOR_RK4,       /* 4th order Runge-Kutta (fixed step) */
    INTEGRATOR_RKF45,     /* Runge-Kutta-Fehlberg 4-5 (adaptive step) */
    INTEGRATOR_LEAPFROG,  /* Leapfrog/Verlet (fixed step, symplectic) */
    INTEGRATOR_YOSHIDA    /* 4th order Yoshida (fixed step, symplectic) */
} IntegrationMethod;

/**
 * Jittering methods for supersampling
 */
typedef enum {
    JITTER_NONE,          /* No jittering */
    JITTER_REGULAR_GRID,  /* Regular grid pattern */
    JITTER_RANDOM,        /* Random jittering */
    JITTER_HALTON,        /* Halton sequence (quasi-random) */
    JITTER_BLUE_NOISE     /* Blue noise (high quality but expensive) */
} JitterMethod;

/**
 * Supersampling parameters 
 */
typedef struct {
    int samples_per_pixel;      /* Number of samples per pixel (e.g., 1, 4, 16) */
    JitterMethod jitter_method; /* Method to distribute samples within pixel */
    double jitter_strength;     /* Strength of jitter (0.0-1.0) */
} SupersamplingParams;

/**
 * Adaptive sampling parameters
 */
typedef struct {
    int enable_adaptive;        /* Whether to use adaptive sampling */
    int min_samples;            /* Minimum samples per pixel */
    int max_samples;            /* Maximum samples per pixel */
    double convergence_threshold; /* Threshold for adaptive convergence */
    double edge_threshold;      /* Threshold for edge detection */
} AdaptiveSamplingParams;

/**
 * Temporal accumulation parameters for reducing flicker
 */
typedef struct {
    int enable_accumulation;    /* Whether to use temporal accumulation */
    double blend_factor;        /* How much to blend with previous frame (0.0-1.0) */
    int max_accumulation_frames; /* Maximum number of frames to accumulate */
    int subpixel_jitter_index;  /* Current jitter index for progressive sampling */
} TemporalAccumulationParams;

/**
 * Information about ray intersection with an object
 */
typedef struct {
    RayTraceResult result;     /* Result of the ray trace */
    Vector3D hit_position;     /* Position where ray hit an object */
    Vector3D hit_normal;       /* Surface normal at hit point (for disks/objects) */
    double distance;           /* Distance traveled by the ray */
    int steps;                 /* Number of integration steps performed */
    double time_dilation;      /* Time dilation factor at hit position */
    Vector3D sky_direction;    /* Direction to look up in skybox (for background hits) */
    double doppler_factor;     /* Doppler shift factor at hit position */
    double temperature;        /* Temperature at hit position (for disk hits) */
    double color[3];           /* RGB color at hit point (for disk hits) */
    double redshift;           /* Gravitational redshift factor */
    double optical_depth;      /* Optical depth along ray path (for scattering) */
} RayTraceHit;

/**
 * GPU shader parameters for ray tracing
 */
typedef struct {
    double mass;               /* Black hole mass */
    double spin;               /* Black hole spin */
    double schwarzschild_radius; /* Event horizon radius */
    double disk_inner_radius;  /* Inner radius of the accretion disk */
    double disk_outer_radius;  /* Outer radius of the accretion disk */
    double disk_temp_scale;    /* Temperature scaling of the disk */
    double observer_distance;  /* Distance of the observer from the black hole */
    double fov;                /* Field of view in radians */
} GPUShaderParams;

/**
 * Adaptive step control parameters
 */
typedef struct {
    double abs_tolerance;      /* Absolute error tolerance (e.g., 1e-10) */
    double rel_tolerance;      /* Relative error tolerance (e.g., 1e-10) */
    double min_step;           /* Minimum allowed step size */
    double max_step;           /* Maximum allowed step size */
    double horizon_safety_factor; /* Step size limiter near horizon (e.g., 0.01) */
} AdaptiveStepParams;

/**
 * Trace a ray through curved spacetime around a black hole
 * 
 * @param ray The ray to trace (origin and direction)
 * @param blackhole Black hole parameters
 * @param disk Accretion disk parameters (can be NULL if no disk)
 * @param config Simulation configuration
 * @param hit Output hit information
 * @return Result code
 */
RayTraceResult trace_ray(const Ray* ray, 
                         const BlackHoleParams* blackhole,
                         const AccretionDiskParams* disk,
                         const SimulationConfig* config,
                         RayTraceHit* hit);

/**
 * Trace a pixel with supersampling and/or adaptive sampling
 * 
 * @param pixel_x X coordinate of the pixel
 * @param pixel_y Y coordinate of the pixel
 * @param width Width of the image
 * @param height Height of the image
 * @param camera Camera parameters
 * @param blackhole Black hole parameters
 * @param disk Accretion disk parameters
 * @param config Simulation configuration
 * @param ss_params Supersampling parameters
 * @param as_params Adaptive sampling parameters
 * @param color_out Output color (RGB)
 * @return Result code
 */
RayTraceResult trace_pixel(
    int pixel_x,
    int pixel_y,
    int width,
    int height,
    const Vector3D* camera_position,
    const Vector3D* camera_direction,
    const Vector3D* camera_up,
    double fov,
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    const SimulationConfig* config,
    const SupersamplingParams* ss_params,
    const AdaptiveSamplingParams* as_params,
    double color_out[3]);

/**
 * Integrates a photon path through curved spacetime using numerical methods
 * 
 * @param position Initial 4-position
 * @param direction Initial 3-direction
 * @param blackhole Black hole parameters
 * @param config Simulation configuration
 * @param method Integration method to use
 * @param path_positions Output buffer for path positions (can be NULL)
 * @param max_positions Maximum number of positions to store
 * @param num_positions Actual number of positions stored
 * @param hit Output hit information
 * @return Result code
 */
RayTraceResult integrate_photon_path(
    const Vector4D* position,
    const Vector3D* direction,
    const BlackHoleParams* blackhole,
    const SimulationConfig* config,
    IntegrationMethod method,
    Vector3D* path_positions,
    int max_positions,
    int* num_positions,
    RayTraceHit* hit);

/**
 * Batch ray tracing for multiple rays
 * Can use multi-threading for improved performance
 * 
 * @param rays Array of rays to trace
 * @param num_rays Number of rays to trace
 * @param blackhole Black hole parameters
 * @param disk Accretion disk parameters (can be NULL if no disk)
 * @param config Simulation configuration
 * @param hits Output array for hit information (must be pre-allocated)
 * @param num_threads Number of threads to use (0 for auto-detect)
 * @return 0 on success, non-zero on error
 */
int trace_rays_batch(
    const Ray* rays,
    int num_rays,
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    const SimulationConfig* config,
    RayTraceHit* hits,
    int num_threads);

/**
 * Generate shader parameters for GPU-based ray tracing
 * 
 * @param blackhole Black hole parameters
 * @param disk Accretion disk parameters
 * @param observer_distance Distance of observer from black hole
 * @param fov Field of view in radians
 * @param params Output shader parameters
 */
void generate_gpu_shader_params(
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    double observer_distance,
    double fov,
    GPUShaderParams* params);

/**
 * Check if a ray intersects with the accretion disk
 * 
 * @param position Current position
 * @param velocity Current velocity
 * @param prev_position Previous position
 * @param disk Accretion disk parameters
 * @param hit_position Output hit position if intersection found
 * @return 1 if intersection found, 0 otherwise
 */
int check_disk_intersection(
    const Vector3D* position,
    const Vector3D* velocity,
    const Vector3D* prev_position,
    const AccretionDiskParams* disk,
    Vector3D* hit_position);

/**
 * Calculate the temperature and color of the accretion disk at a given position
 * 
 * @param position Position on the disk
 * @param blackhole Black hole parameters
 * @param disk Accretion disk parameters
 * @param temperature Output temperature
 * @param color Output RGB color (each component 0.0-1.0)
 */
void calculate_disk_temperature(
    const Vector3D* position,
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    double* temperature,
    double color[3]);

/**
 * Apply relativistic effects to disk color (doppler shift, gravitational redshift)
 * 
 * @param position Position on the disk
 * @param velocity Velocity at that position
 * @param blackhole Black hole parameters
 * @param color Input/output RGB color values
 * @param doppler_factor Output doppler factor if pointer is not NULL
 */
void apply_relativistic_effects(
    const Vector3D* position,
    const Vector3D* velocity,
    const BlackHoleParams* blackhole,
    double color[3],
    double* doppler_factor);

/**
 * Calculate the size of the black hole shadow for a distant observer
 * 
 * @param blackhole Black hole parameters
 * @param inclination Observer inclination angle (0 = pole-on, PI/2 = equatorial)
 * @return Apparent radius of the black hole shadow
 */
double calculate_shadow_radius(
    const BlackHoleParams* blackhole,
    double inclination);

/**
 * Halton sequence for generating well-distributed quasi-random numbers
 * Used for jittering in supersampling
 * 
 * @param index The index in the sequence
 * @param base The base of the sequence (prime numbers work best)
 * @return A value in range [0,1)
 */
double halton_sequence(int index, int base);

#endif /* RAYTRACER_H */ 