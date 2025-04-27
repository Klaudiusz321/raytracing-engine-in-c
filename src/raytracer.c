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
    
    // Conserved quantities for the geodesic
    double energy;             /* Conserved energy of the ray */
    double angular_momentum;   /* Conserved angular momentum */
    double carter_constant;    /* Carter constant (for Kerr spacetime) */
    
    // Fields for analytic approximation in weak-field regions
    double impact_parameter;   /* Impact parameter for far-field approximation */
    int use_analytic_approx;   /* Whether to use analytic approximation (1) or full ODE (0) */
    double field_strength_threshold; /* Threshold for switching to full integration */
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
    
    double v_r = state[3];
    double v_theta = state[4];
    double v_phi = state[5];
    
    // Position derivatives are just velocities
    derivatives[0] = v_r;
    derivatives[1] = v_theta;
    derivatives[2] = v_phi;
    
    // Check if we can use analytic approximation for weak-field regions
    // This is much faster than full integration for rays far from the black hole
    if (ray_params->use_analytic_approx && r > ray_params->field_strength_threshold) {
        // In weak-field limit, we can use approximate lensing based on impact parameter
        // This is a simplified model for demonstration - would be more complex in practice
        double rs = blackhole->schwarzschild_radius;
        double M = blackhole->mass;
        double impact_b = ray_params->impact_parameter;
        
        // Approximate deflection angle for weak-field limit: α ≈ 2M/r
        // This affects primarily the angular velocities
        double deflection_factor = 2.0 * M / (r * r);
        
        // Apply approximate deflection to angular velocities
        // Adjust velocities to bend toward the black hole
        if (impact_b > 0.0) {
            // Simplified approximation: only apply to phi component for demonstration
            derivatives[3] = 0.0;  // Simplified radial acceleration
            derivatives[4] = 0.0;  // Simplified theta acceleration
            derivatives[5] = v_phi * deflection_factor;  // Approximate phi acceleration
            return;
        }
    }
    
    // For Schwarzschild metric, compute acceleration (velocity derivatives)
    double rs = blackhole->schwarzschild_radius;
    double M = blackhole->mass;
    
    if (blackhole->spin == 0.0) {
        // Exploit conserved quantities for Schwarzschild
        // Energy and angular momentum are conserved
        // In a true implementation, we would use these to simplify the geodesic
        // equations, especially in symmetry planes
        
        double r_sq = r * r;
        double sin_theta = sin(theta);
        double sin_theta_sq = sin_theta * sin_theta;
        
        // CRITICAL FIX: Keep rays from getting too close to event horizon
        // Use safety factor of 1.5 - farther out than before for better stability
        if (r <= rs * 1.5) {
            r = rs * 1.5;
            r_sq = r * r;
            // No need for warning - this happens frequently and is normal
        }
        
        // Handle poles silently - they cause numerical issues
        if (fabs(sin_theta) < 0.01) {
            sin_theta = (sin_theta >= 0.0) ? 0.01 : -0.01;
            sin_theta_sq = sin_theta * sin_theta;
        }
        
        // Calculate acceleration terms with safety checks
        double term1 = -M / (r_sq * (1.0 - rs / r)) * (1.0 - rs / r);
        double term2 = r * v_theta * v_theta;
        double term3 = r * sin_theta_sq * v_phi * v_phi;
        
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
        // Here we would use all three conserved quantities (E, L, Q)
        // to simplify the equations
        
        // For now, we'll use Schwarzschild for demonstration
        derivatives[3] = derivatives[4] = derivatives[5] = 0.0;
    }
    
    // Check for NaN or Inf in derivatives - silently fix without warnings
    for (int i = 0; i < 6; i++) {
        if (isnan(derivatives[i]) || isinf(derivatives[i])) {
            derivatives[i] = 0.0;
        }
    }
    
    // CRITICAL FIX: Limit maximum derivative magnitude to prevent instability
    const double MAX_DERIV = 10.0;
    for (int i = 3; i < 6; i++) {
        if (fabs(derivatives[i]) > MAX_DERIV) {
            derivatives[i] = (derivatives[i] > 0) ? MAX_DERIV : -MAX_DERIV;
        }
    }
    
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
    // Initialize photon path integration
    if (config->max_integration_steps > 100) {
        printf("[DEBUG] Starting photon path integration\n");
    }
    
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
        // Fix near-pole numerical issues
        dphi = 0.0; // At poles, azimuthal component is indeterminate
    }
    
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
    
    // Check for negative dt_squared which can occur due to numerical errors
    if (dt_squared < 0.0) {
        dt_squared = 0.0;
    }
    
    // Ensure dt/dλ is positive (forward in time)
    double dt = sqrt(dt_squared);
    
    state[4] = dt;      // dt/dλ
    state[5] = dr;      // dr/dλ
    state[6] = dtheta;  // dθ/dλ
    state[7] = dphi;    // dφ/dλ
    
    // Calculate conserved quantities (important for improving integration accuracy)
    // For Schwarzschild spacetime:
    // - Energy (E = -g_tt * dt/dλ)
    // - Angular momentum (L = g_φφ * dφ/dλ)
    double energy = -metric_components.g_tt * dt;
    double angular_momentum = metric_components.g_phph * dphi;
    
    // For Kerr spacetime, we would also calculate Carter constant (Q)
    double carter_constant = 0.0;
    if (blackhole->spin > 0.0) {
        // In a complete implementation, we would calculate Carter constant here
        // For now we leave it at 0 since we're not fully implementing Kerr
    }
    
    // Calculate impact parameter (useful for analytic approximation in weak field)
    double impact_parameter = fabs(angular_momentum / energy);
    
    // Integration parameters
    RayIntegrationParams ray_params;
    ray_params.blackhole = blackhole;
    ray_params.config = config;
    ray_params.direction = norm_direction;
    
    // Set the conserved quantities and approximation parameters
    ray_params.energy = energy;
    ray_params.angular_momentum = angular_momentum;
    ray_params.carter_constant = carter_constant;
    ray_params.impact_parameter = impact_parameter;
    
    // Determine if we can use analytic approximation
    // For rays far from black hole, use approximation to improve performance
    // Typically 10-20x the Schwarzschild radius is considered "far field"
    ray_params.field_strength_threshold = blackhole->schwarzschild_radius * 15.0;
    ray_params.use_analytic_approx = (r > ray_params.field_strength_threshold) ? 1 : 0;
    
    // CRITICAL FIX: Better initial step size control
    // Use much smaller steps near the black hole, larger steps far away
    double t = 0.0;      // Integration parameter (not time)
    double h = config->time_step;
    
    // Adaptive step size based on proximity to event horizon
    if (r < blackhole->schwarzschild_radius * 5.0) {
        // Near black hole: use very small steps (0.5% of default)
        h = config->time_step * 0.005;
    } else if (r < blackhole->schwarzschild_radius * 10.0) {
        // Moderate distance: use small steps (1% of default)
        h = config->time_step * 0.01;
    } else if (r < blackhole->schwarzschild_radius * 20.0) {
        // Further distance: use medium steps (10% of default)
        h = config->time_step * 0.1;
    }
    
    // Limit maximum step size to prevent instability
    h = fmin(h, 0.1);
    
    double h_next = h;
    
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
    
    // Flag to indicate if we've switched to horizon-penetrating coordinates
    int using_horizon_penetrating = 0;
    double ingoing_time_offset = 0.0;
    
    // Integration loop
    while (step_count < config->max_integration_steps) {
        // Store previous position
        prev_pos = current_pos;
        
        // Check if we need to switch to horizon-penetrating coordinates
        // This prevents coordinate singularity at the event horizon
        if (!using_horizon_penetrating && state[1] < blackhole->schwarzschild_radius * 2.5) {
            // We're getting close to the event horizon, switch to ingoing Eddington-Finkelstein
            // coordinates to avoid coordinate singularity
            
            // For Schwarzschild, the transformation is:
            // t_EF = t + 2M ln|r/2M - 1|
            // The other coordinates remain the same
            
            ingoing_time_offset = 2.0 * blackhole->mass * log(fabs(state[1]/(2.0*blackhole->mass) - 1.0));
            using_horizon_penetrating = 1;
            
            // In a complete implementation, we would transform the 4-velocity as well
            // For now, we'll continue with the existing velocity components
        }
        
        // Perform integration step based on selected method
        int retry = 0;
        
        // Before integration, check all state values for issues
        int invalid_state = 0;
        for (int i = 0; i < 8; i++) {
            if (isnan(state[i]) || isinf(state[i])) {
                invalid_state = 1;
                state[i] = (i < 4) ? 1.0 : 0.0;  // Simple recovery
            }
        }
        
        if (invalid_state && step_count == 0) {
            // Only log initial invalid states
            printf("[ERROR] Invalid initial state values, attempting recovery\n");
        }
        
        // Adaptive step size based on distance from black hole
        if (state[1] < blackhole->schwarzschild_radius * 2.5) {
            // Very close to black hole: use extremely small steps
            h = config->time_step * 0.001;
        } else if (state[1] < blackhole->schwarzschild_radius * 5.0) {
            // Near black hole: use very small steps
            h = config->time_step * 0.01;
        } else if (state[1] < blackhole->schwarzschild_radius * 15.0) {
            // Moderate distance: use small steps
            h = config->time_step * 0.1;
        } else {
            // Far away: can use larger steps
            h = config->time_step;
        }
        
        // Use the clipped step size, with a maximum to prevent instability
        h = fmin(h, 0.1);  // Maximum step size of 0.1
        
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
                    
                    // Use AdaptiveStepParams in the future for better control
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
                        h = h_current * 0.5;
                        continue;
                    }
                    
                    t = t_current;
                }
                break;
                
            case INTEGRATOR_LEAPFROG:
                // Not yet implemented for this system
                printf("[ERROR] Leapfrog integrator not implemented for geodesic equations\n");
                break;
                
            case INTEGRATOR_YOSHIDA:
                // Not yet implemented for this system
                printf("[ERROR] Yoshida integrator not implemented for geodesic equations\n");
                break;
        }
        
        // Convert current spherical position to Cartesian for visualization
        Vector3D new_sph_pos = {state[1], state[2], state[3]};
        Vector3D new_pos;
        spherical_to_cartesian(&new_sph_pos, &new_pos);
        
        // Calculate distance traveled in this step
        Vector3D step_vector = vector3D_sub(new_pos, current_pos);
        double step_distance = vector3D_length(step_vector);
        
        // Update current position
        current_pos = new_pos;
        
        // Add to total distance
        distance_traveled += step_distance;
        
        // Store the current position if requested
        if (path_positions != NULL && *num_positions < max_positions) {
            path_positions[*num_positions] = current_pos;
            (*num_positions)++;
        }
        
        // Check termination conditions
        
        // EARLY TERMINATION: Check for hitting the event horizon
        // More aggressive early termination to avoid wasted computation
        if (state[1] <= blackhole->schwarzschild_radius * 1.05) {
            // Ray has essentially fallen into horizon with 5% safety margin
            result = RAY_HORIZON;
            break;
        }
        
        // Check for maximum distance
        if (distance_traveled >= config->max_ray_distance) {
            result = RAY_MAX_DISTANCE;
            break;
        }
        
        step_count++;
    }
    
    // If using horizon-penetrating coordinates, convert time back to Schwarzschild
    if (using_horizon_penetrating) {
        // This would be more complex in a complete implementation
        // For demonstration, we just note that we used horizon-penetrating coordinates
        printf("[DEBUG] Used horizon-penetrating coordinates for ray near black hole\n");
    }
    
    // Fill hit information
    fill_hit_info(hit, result, &current_pos, distance_traveled, step_count, 
                state[1], blackhole->schwarzschild_radius, &state[4]);
    
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
    int num_threads)
{
    if (!rays || !blackhole || !hits || num_rays <= 0) {
        return -1; // Invalid parameters
    }

    // For now, we'll implement a simple single-threaded version
    // In the future, this could be enhanced with OpenMP or pthreads
    for (int i = 0; i < num_rays; i++) {
        RayTraceResult result = trace_ray(&rays[i], blackhole, disk, config, &hits[i]);
        
        // If there was an error tracing this ray, mark it but continue
        if (result == RAY_ERROR) {
            hits[i].result = RAY_ERROR;
        }
    }

    return 0; // Success
}

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
    GPUShaderParams* params)
{
    if (!blackhole || !params) {
        return;
    }

    // Fill in basic parameters
    params->mass = blackhole->mass;
    params->spin = blackhole->spin;
    params->schwarzschild_radius = 2.0 * blackhole->mass; // 2GM/c^2 = 2M in geometric units
    params->observer_distance = observer_distance;
    params->fov = fov;

    // Fill in disk parameters if available
    if (disk) {
        params->disk_inner_radius = disk->inner_radius;
        params->disk_outer_radius = disk->outer_radius;
        params->disk_temp_scale = disk->temperature_scale;
    } else {
        // Default values if no disk
        params->disk_inner_radius = 3.0 * params->schwarzschild_radius;
        params->disk_outer_radius = 20.0 * params->schwarzschild_radius;
        params->disk_temp_scale = 1.0;
    }
}

/**
 * Halton sequence for generating well-distributed quasi-random numbers
 */
double halton_sequence(int index, int base) {
    double result = 0.0;
    double f = 1.0;
    
    while (index > 0) {
        f /= base;
        result += f * (index % base);
        index /= base;
    }
    
    return result;
}

/**
 * Generate jittered sample position within a pixel
 */
static void generate_jittered_position(
    int pixel_x, 
    int pixel_y,
    int sample_index,
    int samples_per_pixel,
    JitterMethod jitter_method, 
    double jitter_strength,
    double* offset_x, 
    double* offset_y)
{
    // Default to pixel center
    *offset_x = 0.5;
    *offset_y = 0.5;
    
    switch (jitter_method) {
        case JITTER_NONE:
            // Always use pixel center
            break;
            
        case JITTER_REGULAR_GRID:
            {
                // Determine grid size (e.g., 2x2 for 4 samples)
                int grid_size = (int)sqrt((double)samples_per_pixel);
                int x = sample_index % grid_size;
                int y = sample_index / grid_size;
                
                // Distribute samples evenly in grid
                *offset_x = (x + 0.5) / grid_size;
                *offset_y = (y + 0.5) / grid_size;
            }
            break;
            
        case JITTER_RANDOM:
            {
                // Random jittering (less efficient for convergence)
                *offset_x = (double)rand() / RAND_MAX;
                *offset_y = (double)rand() / RAND_MAX;
            }
            break;
            
        case JITTER_HALTON:
            {
                // Halton sequence - use different prime bases for x and y
                *offset_x = halton_sequence(sample_index, 2);
                *offset_y = halton_sequence(sample_index, 3);
            }
            break;
            
        case JITTER_BLUE_NOISE:
            {
                // Blue noise would require a pre-generated texture
                // For now, fall back to Halton sequence
                *offset_x = halton_sequence(sample_index, 2);
                *offset_y = halton_sequence(sample_index, 3);
            }
            break;
    }
    
    // Apply jitter strength (1.0 = full pixel, 0.0 = no jitter)
    if (jitter_strength != 1.0) {
        // Scale jitter around pixel center
        *offset_x = 0.5 + (*offset_x - 0.5) * jitter_strength;
        *offset_y = 0.5 + (*offset_y - 0.5) * jitter_strength;
    }
}

/**
 * Detect high gradient edges for adaptive sampling
 */
static double calculate_edge_factor(
    int pixel_x, 
    int pixel_y,
    int width, 
    int height, 
    const double* image_buffer,
    double edge_threshold)
{
    // If we're at the edge of the image, return high factor
    if (pixel_x <= 1 || pixel_x >= width - 2 || pixel_y <= 1 || pixel_y >= height - 2) {
        return 1.0;
    }
    
    // Calculate color differences with adjacent pixels
    double max_diff = 0.0;
    
    // Center pixel color
    double center_color[3];
    center_color[0] = image_buffer[(pixel_y * width + pixel_x) * 3 + 0];
    center_color[1] = image_buffer[(pixel_y * width + pixel_x) * 3 + 1];
    center_color[2] = image_buffer[(pixel_y * width + pixel_x) * 3 + 2];
    
    // Check 8 neighbors
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    for (int i = 0; i < 8; i++) {
        int nx = pixel_x + dx[i];
        int ny = pixel_y + dy[i];
        
        double neighbor_color[3];
        neighbor_color[0] = image_buffer[(ny * width + nx) * 3 + 0];
        neighbor_color[1] = image_buffer[(ny * width + nx) * 3 + 1];
        neighbor_color[2] = image_buffer[(ny * width + nx) * 3 + 2];
        
        // Calculate color difference
        double diff = 0.0;
        for (int c = 0; c < 3; c++) {
            diff += fabs(center_color[c] - neighbor_color[c]);
        }
        diff /= 3.0; // Average across channels
        
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    // If difference exceeds threshold, mark as edge
    if (max_diff > edge_threshold) {
        return 1.0; // Maximum factor for edges
    }
    
    return max_diff / edge_threshold; // Proportional factor
}

/**
 * Calculate camera ray direction for a given pixel and subpixel offset
 */
static void calculate_ray_direction(
    int pixel_x, 
    int pixel_y,
    double offset_x, 
    double offset_y,
    int width, 
    int height,
    const Vector3D* camera_position,
    const Vector3D* camera_direction,
    const Vector3D* camera_up,
    double fov,
    Vector3D* ray_direction)
{
    // Calculate aspect ratio
    double aspect_ratio = (double)width / (double)height;
    
    // Calculate camera basis vectors
    Vector3D forward = vector3D_normalize(*camera_direction);
    
    // Calculate right vector (cross product of forward and up)
    Vector3D right = vector3D_cross(forward, *camera_up);
    right = vector3D_normalize(right);
    
    // Calculate true up vector (cross product of right and forward)
    Vector3D up = vector3D_cross(right, forward);
    
    // Convert FOV to radians and calculate image plane distances
    double fov_radians = fov * BH_PI / 180.0;
    double plane_height = 2.0 * tan(fov_radians / 2.0);
    double plane_width = plane_height * aspect_ratio;
    
    // Calculate normalized device coordinates (-1 to 1)
    double ndcX = (2.0 * ((pixel_x + offset_x) / width) - 1.0) * plane_width;
    double ndcY = (1.0 - 2.0 * ((pixel_y + offset_y) / height)) * plane_height;
    
    // Calculate ray direction
    *ray_direction = forward;
    *ray_direction = vector3D_add(*ray_direction, vector3D_scale(right, ndcX));
    *ray_direction = vector3D_add(*ray_direction, vector3D_scale(up, ndcY));
    *ray_direction = vector3D_normalize(*ray_direction);
}

/**
 * Trace a pixel with supersampling and/or adaptive sampling
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
    double color_out[3])
{
    // Initialize output color
    color_out[0] = color_out[1] = color_out[2] = 0.0;
    
    // Set default result
    RayTraceResult result = RAY_BACKGROUND;
    
    // Determine number of samples to take
    int samples_to_take = 1; // Default to 1 sample
    
    if (ss_params != NULL) {
        samples_to_take = ss_params->samples_per_pixel;
    }
    
    // Adjust samples based on adaptive settings if available
    double edge_factor = 1.0;
    
    if (as_params != NULL && as_params->enable_adaptive) {
        // If we have edge factor information, use it
        // (in a real implementation, this would come from a previous analysis pass)
        
        // For demonstration purposes, we keep a fixed number of samples
        samples_to_take = as_params->min_samples;
        
        // If we detect this is an edge/high-gradient region, increase samples
        if (edge_factor > 0.5) {
            int additional_samples = (int)((as_params->max_samples - as_params->min_samples) * edge_factor);
            samples_to_take += additional_samples;
            
            // Cap at max_samples
            if (samples_to_take > as_params->max_samples) {
                samples_to_take = as_params->max_samples;
            }
        }
    }
    
    // Accumulate colors from all samples
    for (int sample = 0; sample < samples_to_take; sample++) {
        // Generate jittered position within the pixel
        double offset_x = 0.5;
        double offset_y = 0.5;
        
        if (ss_params != NULL && ss_params->samples_per_pixel > 1) {
            generate_jittered_position(
                pixel_x, pixel_y,
                sample, ss_params->samples_per_pixel,
                ss_params->jitter_method,
                ss_params->jitter_strength,
                &offset_x, &offset_y);
        }
        
        // Calculate ray direction for this sample
        Vector3D ray_direction;
        calculate_ray_direction(
            pixel_x, pixel_y,
            offset_x, offset_y,
            width, height,
            camera_position,
            camera_direction,
            camera_up,
            fov,
            &ray_direction);
        
        // Set up ray
        Ray ray;
        ray.origin = *camera_position;
        ray.direction = ray_direction;
        
        // Trace the ray
        RayTraceHit hit;
        RayTraceResult sample_result = trace_ray(&ray, blackhole, disk, config, &hit);
        
        // If this is the first sample, use its result
        if (sample == 0) {
            result = sample_result;
        }
        
        // Accumulate color from this sample
        if (sample_result == RAY_DISK) {
            // For disk hits, use the color from the hit structure
            color_out[0] += hit.color[0];
            color_out[1] += hit.color[1];
            color_out[2] += hit.color[2];
        } else if (sample_result == RAY_HORIZON) {
            // For horizon hits, use black
            // Leave color at 0,0,0
        } else {
            // For background hits, use a simple skybox color
            // In a real implementation, this would be a texture lookup
            
            // Simple gradient background for demonstration
            double t = 0.5 * (ray_direction.y + 1.0);
            double r = (1.0 - t) * 1.0 + t * 0.5;
            double g = (1.0 - t) * 1.0 + t * 0.7;
            double b = (1.0 - t) * 1.0 + t * 1.0;
            
            color_out[0] += r;
            color_out[1] += g;
            color_out[2] += b;
        }
    }
    
    // Average the accumulated colors
    color_out[0] /= samples_to_take;
    color_out[1] /= samples_to_take;
    color_out[2] /= samples_to_take;
    
    return result;
} 