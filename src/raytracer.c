/**
 * raytracer.c
 * 
 * Implementation of raytracing for curved spacetime around a black hole.
 */

#include "../include/raytracer.h"
#include "../include/spacetime.h"
#include "../include/math_util.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

/**
 * Structure to hold ray integration state
 */
typedef struct {
    const BlackHoleParams* blackhole;
    const SimulationConfig* config;
    Vector3D direction;
} RayIntegrationParams;

/**
 * Compute derivative function for ray integration
 * This function computes the right-hand side of the geodesic equations
 * 
 * @param t Current time parameter
 * @param state Current state vector (position and velocity)
 * @param derivatives Output derivatives
 * @param params Integration parameters
 */
static void ray_derivatives(double t, const double state[], double derivatives[], void *params) {
    RayIntegrationParams* ray_params = (RayIntegrationParams*)params;
    const BlackHoleParams* blackhole = ray_params->blackhole;
    
    // Extract position and velocity components
    double r = state[0];
    double theta = state[1];
    double phi = state[2];
    (void)phi;
    
    double v_r = state[3];
    double v_theta = state[4];
    double v_phi = state[5];
    
    // DEBUG: Check for invalid inputs
    printf("[DEBUG] ray_derivatives: t=%.6e, r=%.6e, theta=%.6e, phi=%.6e\n", t, r, theta, phi);
    printf("[DEBUG] ray_derivatives: v_r=%.6e, v_theta=%.6e, v_phi=%.6e\n", v_r, v_theta, v_phi);
    
    // Position derivatives are just velocities
    derivatives[0] = v_r;
    derivatives[1] = v_theta;
    derivatives[2] = v_phi;
    
    // For Schwarzschild metric, compute acceleration (velocity derivatives)
    double rs = blackhole->schwarzschild_radius;
    double M = blackhole->mass;
    
    if (blackhole->spin == 0.0) {
        // Schwarzschild geodesic equations (simplified version)
        double r_sq = r * r;
        double sin_theta = sin(theta);
        double sin_theta_sq = sin_theta * sin_theta;
        
        // Check for potential division by zero or extreme values
        if (r <= rs + BH_EPSILON) {
            printf("[ERROR] ray_derivatives: r (%.6e) too close to or inside event horizon (%.6e)\n", r, rs);
            // Clamp r to avoid numerical issues
            r = rs + BH_EPSILON;
            r_sq = r * r;
        }
        
        if (fabs(sin_theta) < BH_EPSILON) {
            printf("[WARNING] ray_derivatives: sin_theta (%.6e) near zero, at poles\n", sin_theta);
            // Avoid division by zero in dphi calculation
            sin_theta = (sin_theta >= 0.0) ? BH_EPSILON : -BH_EPSILON;
            sin_theta_sq = sin_theta * sin_theta;
        }
        
        // Calculate acceleration terms with safety checks
        double term1 = -M / (r_sq * (1.0 - rs / r)) * (1.0 - rs / r);
        double term2 = r * v_theta * v_theta;
        double term3 = r * sin_theta_sq * v_phi * v_phi;
        
        printf("[DEBUG] acceleration terms: term1=%.6e, term2=%.6e, term3=%.6e\n", 
               term1, term2, term3);
               
        // dr/dt equation
        derivatives[3] = term1 + term2 + term3;
        
        // dtheta/dt equation
        derivatives[4] = -2.0 * v_r * v_theta / r +
                          sin_theta * cos(theta) * v_phi * v_phi;
        
        // dphi/dt equation
        derivatives[5] = -2.0 * v_r * v_phi / r -
                          2.0 * v_theta * v_phi * cos(theta) / sin_theta;
    } else {
        // Kerr geodesic equations would go here (more complex)
        // For simplicity, we'll use Schwarzschild for now
        derivatives[3] = derivatives[4] = derivatives[5] = 0.0;
    }
    
    // DEBUG: Check for NaN or Inf in derivatives
    for (int i = 0; i < 6; i++) {
        if (isnan(derivatives[i]) || isinf(derivatives[i])) {
            printf("[ERROR] ray_derivatives: derivatives[%d] = %.6e is NaN or Inf\n", i, derivatives[i]);
            // Set to zero to prevent propagation of invalid values
            derivatives[i] = 0.0;
        }
    }
    
    printf("[DEBUG] derivatives: dr/dt=%.6e, dtheta/dt=%.6e, dphi/dt=%.6e\n", 
           derivatives[3], derivatives[4], derivatives[5]);
    
    return;
}

/**
 * Check if a ray intersects with the accretion disk
 */
int check_disk_intersection(
    const Vector3D* origin,
    const Vector3D* velocity,
    const Vector3D* disk_normal,
    const AccretionDiskParams* disk,
    Vector3D* hit_point) 
{
    // Disk assumed to be in x-y plane (normal = z-axis) for simplicity
    
    // Calculate how long until the ray intersects the plane
    double denom = vector3D_dot(*velocity, *disk_normal);
    
    if (fabs(denom) < BH_EPSILON) {
        // Ray is parallel to disk, no intersection
        return 0;
    }
    
    // Calculate distance to intersection
    double t = -(vector3D_dot(*origin, *disk_normal)) / denom;
    
    if (t < 0.0) {
        // Intersection is behind the ray
        return 0;
    }
    
    // Calculate intersection point
    Vector3D scaled_vel = vector3D_scale(*velocity, t);
    *hit_point = vector3D_add(*origin, scaled_vel);
    
    // Check if hit point is within disk
    double r = sqrt(hit_point->x * hit_point->x + hit_point->y * hit_point->y);
    
    if (r >= disk->inner_radius && r <= disk->outer_radius) {
        return 1; // Hit the disk
    }
    
    return 0; // Missed the disk
}

/**
 * Calculate the temperature and color of the accretion disk at a given position
 */
void calculate_disk_temperature(
    const Vector3D* position,
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    double* temperature,
    double color[3])
{
    // Distance from center
    double r = sqrt(position->x * position->x + position->y * position->y);
    
    // Disk temperature model: T ~ r^(-3/4) for thin disk
    // Normalized to range [0,1] based on inner and outer radius
    double normalized_radius = (r - disk->inner_radius) / 
                              (disk->outer_radius - disk->inner_radius);
    
    // Ensure radius is within bounds
    normalized_radius = clamp(normalized_radius, 0.0, 1.0);
    
    // Temperature peaks at inner edge and falls off with radius
    // T ~ r^(-3/4) is the profile for a standard thin accretion disk
    double temp_factor = pow(1.0 - normalized_radius, 0.75);
    
    // Calculate temperature (scaled by parameter)
    *temperature = disk->temperature_scale * (2000.0 + 18000.0 * temp_factor);
    
    // Convert temperature to RGB color
    temperature_to_rgb(*temperature, color);
}

/**
 * Apply relativistic effects to disk color (doppler shift, gravitational redshift)
 */
void apply_relativistic_effects(
    const Vector3D* position,
    const Vector3D* velocity,
    const BlackHoleParams* blackhole,
    double color[3],
    double* doppler_factor_out)
{
    // Distance from center
    double r = sqrt(position->x * position->x + position->y * position->y);
    
    // Calculate orbital velocity (Keplerian)
    double orbital_speed = sqrt(blackhole->mass / r);
    (void)orbital_speed;
    // Calculate the angle of the position vector
    double phi = atan2(position->y, position->x);
    
    // Tangential velocity direction
    Vector3D tangent;
    tangent.x = -sin(phi);
    tangent.y = cos(phi);
    tangent.z = 0.0;
    
    // Doppler shift factor (approaching = blueshift, receding = redshift)
    // Updated to use pass-by-value
    double doppler_factor = 1.0 + vector3D_dot(*velocity, tangent) * 0.1;
    
    // Gravitational redshift - fixed to pass the whole blackhole struct
    double grav_redshift = calculate_time_dilation(r, blackhole);
    
    // Redshift combined effect (doppler and gravitational)
    double redshift = doppler_factor / grav_redshift;
    
    // Apply redshift to RGB color
    // Simplified model: redshift < 1 shifts toward red, > 1 shifts toward blue
    if (redshift < 1.0) {
        // Shift toward red (decrease blue, increase red)
        color[2] *= redshift;
        color[0] = fmin(1.0, color[0] * (2.0 - redshift));
    } else {
        // Shift toward blue (decrease red, increase blue)
        color[0] *= 2.0 - redshift;
        color[2] = fmin(1.0, color[2] * redshift);
    }
    
    // Relativistic beaming (brightness enhancement in direction of motion)
    double beaming = pow(doppler_factor, 4);  // Intensity ~ doppler^4
    
    // Apply beaming effect (scale overall brightness)
    color[0] *= beaming;
    color[1] *= beaming;
    color[2] *= beaming;
    
    // Ensure colors remain in valid range
    color[0] = clamp(color[0], 0.0, 1.0);
    color[1] = clamp(color[1], 0.0, 1.0);
    color[2] = clamp(color[2], 0.0, 1.0);
    
    // Output Doppler factor if requested
    if (doppler_factor_out != NULL) {
        *doppler_factor_out = doppler_factor;
    }
}

/**
 * Fill hit information with current ray state
 */
static void fill_hit_info(
    RayTraceHit* hit,
    RayTraceResult result,
    const Vector3D* current_pos,
    double distance,
    int step_count,
    double r,
    double schwarzschild_radius,
    const double* velocity)
{
    if (hit != NULL) {
        hit->result = result;
        
        // Copy position component by component instead of direct assignment
        hit->hit_position.x = current_pos->x;
        hit->hit_position.y = current_pos->y;
        hit->hit_position.z = current_pos->z;
        
        hit->distance = distance;
        hit->steps = step_count;
        
        // Calculate time dilation at hit position - create a temporary BlackHoleParams
        BlackHoleParams temp_bh;
        temp_bh.schwarzschild_radius = schwarzschild_radius;
        hit->time_dilation = calculate_time_dilation(r, &temp_bh);
        
        // For background hits, set the sky direction
        if (result == RAY_BACKGROUND || result == RAY_MAX_DISTANCE) {
            // Use the final direction as the sky direction
            // In a more sophisticated implementation, this would account for light bending
            Vector3D current_vel = {velocity[1], velocity[2], velocity[3]};
            hit->sky_direction = vector3D_normalize(current_vel);
        }
    }
}

/**
 * Integrates a photon path through curved spacetime using numerical methods
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
    RayTraceHit* hit)
{
    printf("[DEBUG] Starting photon path integration\n");
    printf("[DEBUG] Initial position: (%.6e, %.6e, %.6e, %.6e)\n", 
           position->t, position->x, position->y, position->z);
    printf("[DEBUG] Direction: (%.6e, %.6e, %.6e)\n", direction->x, direction->y, direction->z);
    printf("[DEBUG] Black hole mass: %.6e, spin: %.6e, schwarzschild_radius: %.6e\n", 
           blackhole->mass, blackhole->spin, blackhole->schwarzschild_radius);
    
    // Normalize the direction vector
    Vector3D norm_direction = vector3D_normalize(*direction);
    
    // Set up state vector for integration
    // 8 components: 4 for position (t,r,θ,φ) and 4 for velocity
    double state[8];
    
    // Initialize position using spherical coordinates
    Vector3D cart_pos = {position->x, position->y, position->z};
    
    // Fixed: properly use cartesian_to_spherical with input and output parameters
    Vector3D spherical_pos;
    cartesian_to_spherical(&cart_pos, &spherical_pos);
    
    printf("[DEBUG] Initial cart_pos: (%.6e, %.6e, %.6e)\n", cart_pos.x, cart_pos.y, cart_pos.z);
    printf("[DEBUG] Initial spherical_pos: r=%.6e, theta=%.6e, phi=%.6e\n", 
           spherical_pos.x, spherical_pos.y, spherical_pos.z);
    
    state[0] = position->t;     // Time
    state[1] = spherical_pos.x; // Radial coordinate r
    state[2] = spherical_pos.y; // Polar angle θ
    state[3] = spherical_pos.z; // Azimuthal angle φ
    
    // Initialize velocity using initial direction
    // Convert Cartesian direction to spherical
    Vector3D cart_vel = norm_direction;
    
    // Here we need to compute the velocity components in spherical coordinates
    // This is a simplification - in reality, we would need to compute the proper
    // initial velocity for a null geodesic (light ray)
    
    // For a ray moving in flat space in Cartesian coordinates with direction (dx,dy,dz),
    // the spherical components (dr,dθ,dφ) can be computed using:
    
    double r = spherical_pos.x;
    double theta = spherical_pos.y;
    double phi = spherical_pos.z;
    
    // dr/dλ = sin(θ)cos(φ)dx + sin(θ)sin(φ)dy + cos(θ)dz
    double dr = sin(theta)*cos(phi)*cart_vel.x + 
                sin(theta)*sin(phi)*cart_vel.y + 
                cos(theta)*cart_vel.z;
                
    // dθ/dλ = (cos(θ)cos(φ)dx + cos(θ)sin(φ)dy - sin(θ)dz)/r
    double dtheta = (cos(theta)*cos(phi)*cart_vel.x + 
                    cos(theta)*sin(phi)*cart_vel.y - 
                    sin(theta)*cart_vel.z) / r;
                    
    // dφ/dλ = (-sin(φ)dx + cos(φ)dy)/(r*sin(θ))
    double dphi = (-sin(phi)*cart_vel.x + cos(phi)*cart_vel.y) / (r * sin(theta));
    
    // Check for potential numerical issues in initial velocity calculation
    if (fabs(sin(theta)) < BH_EPSILON) {
        printf("[WARNING] integrate_photon_path: sin(theta) near zero at poles, clamping to avoid division by zero\n");
        // Fix near-pole numerical issues
        dphi = 0.0; // At poles, azimuthal component is indeterminate
    }
    
    printf("[DEBUG] Initial velocities: dr=%.6e, dtheta=%.6e, dphi=%.6e\n", dr, dtheta, dphi);
    
    // Set velocity components
    // dt/dλ calculated from null geodesic condition (ds²=0)
    SchwarzschildMetric metric_components;
    
    // Updated to use the new function signature
    metric_components = calculate_schwarzschild_metric(r, blackhole);
    
    // For null geodesic: g_tt(dt/dλ)² + g_rr(dr/dλ)² + g_θθ(dθ/dλ)² + g_φφ(dφ/dλ)² = 0
    // Solving for dt/dλ:
    double dt_squared = -(metric_components.g_rr * dr * dr + 
                         metric_components.g_thth * dtheta * dtheta + 
                         metric_components.g_phph * dphi * dphi) / metric_components.g_tt;
    
    printf("[DEBUG] Metric components: g_tt=%.6e, g_rr=%.6e, g_thth=%.6e, g_phph=%.6e\n",
           metric_components.g_tt, metric_components.g_rr, metric_components.g_thth, metric_components.g_phph);
    printf("[DEBUG] dt_squared = %.6e\n", dt_squared);
    
    // Check for negative dt_squared which can occur due to numerical errors
    if (dt_squared < 0.0) {
        printf("[WARNING] integrate_photon_path: dt_squared (%.6e) is negative, setting to zero\n", dt_squared);
        dt_squared = 0.0;
    }
    
    // Ensure dt/dλ is positive (forward in time)
    double dt = sqrt(dt_squared);
    
    state[4] = dt;      // dt/dλ
    state[5] = dr;      // dr/dλ
    state[6] = dtheta;  // dθ/dλ
    state[7] = dphi;    // dφ/dλ
    
    printf("[DEBUG] Initial state vector:\n");
    printf("[DEBUG] (t,r,theta,phi) = (%.6e, %.6e, %.6e, %.6e)\n", 
           state[0], state[1], state[2], state[3]);
    printf("[DEBUG] (dt,dr,dtheta,dphi) = (%.6e, %.6e, %.6e, %.6e)\n", 
           state[4], state[5], state[6], state[7]);
    
    // Integration parameters
    RayIntegrationParams ray_params;
    ray_params.blackhole = blackhole;
    ray_params.config = config;
    ray_params.direction = norm_direction;
    
    // Adaptive step size parameters
    double t = 0.0;      // Integration parameter (not time)
    double h = config->time_step;
    double h_next = h;
    
    printf("[DEBUG] Starting integration with step size h=%.6e\n", h);
    printf("[DEBUG] Max steps: %d, tolerance: %.6e\n", config->max_integration_steps, config->tolerance);
    
    // Path tracing variables
    int step_count = 0;
    double distance_traveled = 0.0;
    
    Vector3D current_pos, prev_pos;
    
    // Convert initial spherical position to Cartesian
    Vector3D sph_pos = {state[1], state[2], state[3]};
    
    // Fixed: properly use spherical_to_cartesian with input and output parameters
    spherical_to_cartesian(&sph_pos, &current_pos);
    
    // Store the initial position if requested
    if (path_positions != NULL && max_positions > 0) {
        path_positions[0] = current_pos;
        *num_positions = 1;
    }
    
    RayTraceResult result = RAY_MAX_STEPS;
    
    // Integration loop
    while (step_count < config->max_integration_steps) {
        // Store previous position
        prev_pos = current_pos;
        
        // Perform integration step based on selected method
        int retry = 0;
        
        // Before integration, check all state values for issues
        int invalid_state = 0;
        for (int i = 0; i < 8; i++) {
            if (isnan(state[i]) || isinf(state[i])) {
                printf("[ERROR] Step %d: state[%d] = %.6e is NaN or Inf before integration\n", 
                       step_count, i, state[i]);
                invalid_state = 1;
            }
        }
        
        if (invalid_state) {
            printf("[ERROR] Aborting integration due to invalid state values\n");
            result = RAY_ERROR;
            break;
        }
        
        // Periodically log the integration progress
        if (step_count % 100 == 0) {
            printf("[DEBUG] Step %d: r=%.6e, distance=%.6e, h=%.6e\n", 
                   step_count, state[1], distance_traveled, h);
        }
        
        switch (method) {
            case INTEGRATOR_RK4:
                {
                    double temp_y[8];
                    memcpy(temp_y, state, 8 * sizeof(double));
                    rk4_integrate(ray_derivatives, temp_y, 8, t, h, &ray_params);
                    memcpy(state, temp_y, 8 * sizeof(double));
                    t += h;
                }
                break;
                
            case INTEGRATOR_RKF45:
                // Fixed to match the correct function signature
                {
                    double t_current = t;
                    double h_current = h;
                    
                    retry = rkf45_integrate(
                        ray_derivatives, 
                        state,
                        8,
                        &t_current,
                        h_current,
                        &h_next,
                        config->tolerance,
                        &ray_params
                    );
                    
                    // If step failed, retry with smaller step size
                    if (retry) {
                        printf("[DEBUG] Step %d: RKF45 step failed, retrying with h=%.6e\n", 
                               step_count, h_current * 0.5);
                        h = h_current * 0.5;
                        continue;
                    }
                    
                    // Update time and step size for next iteration
                    t = t_current;
                    h = h_next;
                }
                break;
                
            case INTEGRATOR_LEAPFROG:
                {
                    // Use the new leapfrog integrator function
                    // Need to create a second-order ODE function adapter
                    double pos[4] = {state[0], state[1], state[2], state[3]};
                    double vel[4] = {state[4], state[5], state[6], state[7]};
                    (void)pos;
                    (void)vel;
                    // Function to calculate acceleration for leapfrog
                    ODEFunctionSecondOrder accel_func = NULL;
                    (void)accel_func;
                    // This is a simplified version - in a real implementation we would
                    // need to adapt our ray_derivatives function to the leapfrog format
                    
                    // Just use RK4 for now
                    double temp_y[8];
                    memcpy(temp_y, state, 8 * sizeof(double));
                    rk4_integrate(ray_derivatives, temp_y, 8, t, h, &ray_params);
                    memcpy(state, temp_y, 8 * sizeof(double));
                    t += h;
                }
                break;
                
            case INTEGRATOR_YOSHIDA:
                // Yoshida 4th order symplectic integrator
                // Just use RK4 for now
                {
                    double temp_y[8];
                    memcpy(temp_y, state, 8 * sizeof(double));
                    rk4_integrate(ray_derivatives, temp_y, 8, t, h, &ray_params);
                    memcpy(state, temp_y, 8 * sizeof(double));
                    t += h;
                }
                break;
                
            default:
                // Default to RK4
                {
                    double temp_y[8];
                    memcpy(temp_y, state, 8 * sizeof(double));
                    rk4_integrate(ray_derivatives, temp_y, 8, t, h, &ray_params);
                    memcpy(state, temp_y, 8 * sizeof(double));
                    t += h;
                }
                break;
        }
        
        // After integration, check for invalid values
        invalid_state = 0;
        for (int i = 0; i < 8; i++) {
            if (isnan(state[i]) || isinf(state[i])) {
                printf("[ERROR] Step %d: state[%d] = %.6e is NaN or Inf after integration\n", 
                       step_count, i, state[i]);
                state[i] = (i < 4) ? prev_pos.x : 0.0; // Attempt recovery
                invalid_state = 1;
            }
        }
        
        if (invalid_state) {
            printf("[WARNING] Integration produced invalid values, attempting to continue with corrections\n");
        }
        
        // Convert current spherical position to Cartesian
        Vector3D new_sph_pos = {state[1], state[2], state[3]};
        
        // Fixed: properly use spherical_to_cartesian with input and output parameters
        spherical_to_cartesian(&new_sph_pos, &current_pos);
        
        // Calculate distance traveled in this step
        Vector3D step_vec = vector3D_sub(current_pos, prev_pos);
        double step_distance = vector3D_length(step_vec);
        
        // Check for unusually large step distance which might indicate numerical issues
        if (step_distance > 10.0 * h) {
            printf("[WARNING] Step %d: Large step distance %.6e detected, may indicate numerical issues\n", 
                   step_count, step_distance);
        }
        
        distance_traveled += step_distance;
        
        // Store the current position if requested
        if (path_positions != NULL && *num_positions < max_positions) {
            path_positions[*num_positions] = current_pos;
            (*num_positions)++;
        }
        
        // Check termination conditions
        
        // Check for hitting the event horizon
        if (state[1] <= blackhole->schwarzschild_radius + BH_EPSILON) {
            printf("[DEBUG] Ray reached event horizon at step %d, r=%.6e\n", 
                   step_count, state[1]);
            result = RAY_HORIZON;
            break;
        }
        
        // Check for maximum distance
        if (distance_traveled >= config->max_ray_distance) {
            printf("[DEBUG] Ray reached maximum distance (%.6e) at step %d\n", 
                   config->max_ray_distance, step_count);
            result = RAY_MAX_DISTANCE;
            break;
        }
        
        step_count++;
    }
    
    if (result == RAY_MAX_STEPS) {
        printf("[DEBUG] Ray reached maximum steps (%d) without termination\n", 
               config->max_integration_steps);
    }
    
    // Fill hit information
    fill_hit_info(hit, result, &current_pos, distance_traveled, step_count, 
                state[1], blackhole->schwarzschild_radius, &state[4]);
    
    printf("[DEBUG] Integration complete: result=%d, steps=%d, distance=%.6e\n", 
           result, step_count, distance_traveled);
    
    return result;
}

/**
 * Trace a ray through curved spacetime around a black hole
 */
RayTraceResult trace_ray(const Ray* ray, 
                         const BlackHoleParams* blackhole,
                         const AccretionDiskParams* disk,
                         const SimulationConfig* config,
                         RayTraceHit* hit)
{
    // Set up initial position (t=0)
    Vector4D position = {0.0, ray->origin.x, ray->origin.y, ray->origin.z};
    
    // Allocate space for path positions if we need to check disk intersections
    int max_positions = 0;
    Vector3D* path_positions = NULL;
    int num_positions = 0;
    
    if (disk != NULL) {
        // Need to track positions to check for disk intersections
        max_positions = config->max_integration_steps;
        path_positions = (Vector3D*)malloc(max_positions * sizeof(Vector3D));
    }
    
    // Trace the ray
    RayTraceResult result = integrate_photon_path(
        &position, 
        &ray->direction,
        blackhole,
        config,
        INTEGRATOR_RK4,
        path_positions,
        max_positions,
        &num_positions,
        hit);
    
    // If disk is present, check for intersections with stored path
    if (disk != NULL && path_positions != NULL && num_positions > 1) {
        for (int i = 1; i < num_positions; i++) {
            Vector3D hit_pos;
            if (check_disk_intersection(
                    &path_positions[i],
                    &ray->direction,   // Approximation - should use actual ray direction at this point
                    &path_positions[i-1],
                    disk,
                    &hit_pos)) {
                
                // Update hit information
                if (hit != NULL) {
                    hit->result = RAY_DISK;
                    
                    // Copy component by component instead of direct assignment
                    hit->hit_position.x = hit_pos.x;
                    hit->hit_position.y = hit_pos.y;
                    hit->hit_position.z = hit_pos.z;
                    
                    // Calculate distance to the hit point
                    double dist = 0.0;
                    for (int j = 1; j <= i; j++) {
                        Vector3D step = vector3D_sub(path_positions[j], path_positions[j-1]);
                        dist += vector3D_length(step);
                    }
                    
                    // Add partial distance to the actual intersection point
                    Vector3D final_step = vector3D_sub(hit_pos, path_positions[i-1]);
                    dist += vector3D_length(final_step);
                    
                    hit->distance = dist;
                    hit->steps = i;
                    
                    // Calculate time dilation at hit position
                    double r = vector3D_length(hit_pos);
                    hit->time_dilation = calculate_time_dilation(r, blackhole);
                }
                
                result = RAY_DISK;
                break;
            }
        }
    }
    
    // Free path positions
    if (path_positions != NULL) {
        free(path_positions);
    }
    
    return result;
}

/**
 * Batch ray tracing for efficient rendering
 * This implementation uses multi-threading for improved performance when available
 */
int trace_rays_batch(
    const Ray* rays,
    int num_rays,
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    const SimulationConfig* config,
    RayTraceHit* hits,
    int num_threads)
{
    // Basic error checking
    if (rays == NULL || num_rays <= 0 || blackhole == NULL || 
        config == NULL || hits == NULL) {
        return -1;  // Invalid parameters
    }

    // Check if multi-threading is requested and available
    if (num_threads > 1) {
        #ifdef _OPENMP
        // Set the number of threads to use
        int max_threads = omp_get_max_threads();
        int threads_to_use = (num_threads < max_threads) ? num_threads : max_threads;
        
        omp_set_num_threads(threads_to_use);
        
        // Process rays in parallel using OpenMP
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < num_rays; i++) {
            trace_ray(&rays[i], blackhole, disk, config, &hits[i]);
        }
        
        return 0;  // Success
        #else
        // OpenMP not available, fall back to sequential processing
        // Note: we still proceed with sequential processing below
        #endif
    }
    
    // Sequential processing (either because num_threads <= 1 or OpenMP is not available)
    for (int i = 0; i < num_rays; i++) {
        trace_ray(&rays[i], blackhole, disk, config, &hits[i]);
    }
    
    return 0;  // Success
}

void generate_gpu_shader_params(
    const BlackHoleParams* blackhole,
    const AccretionDiskParams* disk,
    double observer_distance,
    double fov,
    GPUShaderParams* params)
{
    params->mass = blackhole->mass;
    params->spin = blackhole->spin;
    params->schwarzschild_radius = blackhole->schwarzschild_radius;
    params->disk_inner_radius = disk->inner_radius;
    params->disk_outer_radius = disk->outer_radius;
    params->disk_temp_scale = disk->temperature_scale;
    params->observer_distance = observer_distance;
    params->fov = fov;
} 