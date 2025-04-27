/**
 * spacetime.c
 * 
 * Implementation of spacetime geometry and general relativity calculations.
 */

#include "../include/spacetime.h"
#include "../include/math_util.h"
#include <math.h>
#include <string.h>

/**
 * Calculate the Schwarzschild metric at a given point in spherical coordinates
 */
SchwarzschildMetric calculate_schwarzschild_metric(double r, const BlackHoleParams* blackhole) {
    SchwarzschildMetric metric;
    
    // Schwarzschild radius (event horizon)
    double rs = blackhole->schwarzschild_radius;
    
    // Avoid singularity
    if (r <= rs + BH_EPSILON) {
        r = rs + BH_EPSILON;
    }
    
    // Metric components in spherical coordinates (t, r, θ, φ)
    metric.g_tt = -(1.0 - rs / r);
    metric.g_rr = 1.0 / (1.0 - rs / r);
    metric.g_thth = r * r;
    metric.g_phph = r * r * sin(BH_PI/2) * sin(BH_PI/2); // At equator for simplicity
    
    return metric;
}

/**
 * Calculate the Kerr metric at a given point in Boyer-Lindquist coordinates
 */
KerrMetric calculate_kerr_metric(double r, double theta, const BlackHoleParams* blackhole) {
    KerrMetric metric;
    
    // Black hole parameters
    double M = blackhole->mass;
    double a = blackhole->spin * M; // a = J/M, J = angular momentum
    
    // Avoid singularity
    if (r <= blackhole->r_plus + BH_EPSILON) {
        r = blackhole->r_plus + BH_EPSILON;
    }
    
    // Intermediate calculations for Kerr metric
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    (void)cos_theta;
    double sin_theta_sq = sin_theta * sin_theta;
    double Sigma = r*r + a*a * sin_theta_sq;
    double Delta = r*r - 2.0*M*r + a*a;
    double A = (r*r + a*a) * (r*r + a*a) - Delta * a*a * sin_theta_sq;
    (void)A;
    
    // Metric components in Boyer-Lindquist coordinates
    metric.g_tt = -(1.0 - 2.0*M*r/Sigma);
    metric.g_tphi = -2.0*M*a*r*sin_theta_sq/Sigma;
    metric.g_rr = Sigma/Delta;
    metric.g_thth = Sigma;
    metric.g_phiphi = (r*r + a*a + 2.0*M*r*a*a*sin_theta_sq/Sigma) * sin_theta_sq;
    metric.g_phit = metric.g_tphi; // Symmetric component
    
    return metric;
}

/**
 * Calculate the black hole metric based on type (Schwarzschild or Kerr)
 */
BlackHoleMetric calculate_metric(double r, double theta, const BlackHoleParams* blackhole) {
    BlackHoleMetric metric;
    
    if (blackhole->spin == 0.0) {
        // Schwarzschild metric
        metric.is_kerr = 0;
        metric.metric.schwarzschild = calculate_schwarzschild_metric(r, blackhole);
    } else {
        // Kerr metric
        metric.is_kerr = 1;
        metric.metric.kerr = calculate_kerr_metric(r, theta, blackhole);
    }
    
    return metric;
}

/**
 * Calculate the Christoffel symbols (connection coefficients) for Schwarzschild metric
 */
void calculate_christoffel_symbols(double r, double theta, const BlackHoleParams* blackhole, double christoffel[4][4][4]) {
    // Initialize all symbols to zero
    memset(christoffel, 0, 4 * 4 * 4 * sizeof(double));
    
    if (blackhole->spin == 0.0) {
        // Schwarzschild metric
        double rs = blackhole->schwarzschild_radius;
        
        // Avoid singularity
        if (r <= rs + BH_EPSILON) {
            r = rs + BH_EPSILON;
        }
        
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        
        // Non-zero Christoffel symbols for Schwarzschild metric
        // Time-radial components
        christoffel[0][0][1] = christoffel[0][1][0] = rs / (2.0 * r * (r - rs));
        
        // Radial components
        christoffel[1][0][0] = rs * (r - rs) / (2.0 * r * r * r);
        christoffel[1][1][1] = -rs / (2.0 * r * (r - rs));
        christoffel[1][2][2] = -(r - rs);
        christoffel[1][3][3] = -(r - rs) * sin_theta * sin_theta;
        
        // Theta components
        christoffel[2][1][2] = christoffel[2][2][1] = 1.0 / r;
        christoffel[2][3][3] = -sin_theta * cos_theta;
        
        // Phi components
        christoffel[3][1][3] = christoffel[3][3][1] = 1.0 / r;
        christoffel[3][2][3] = christoffel[3][3][2] = cos_theta / sin_theta;
    } else {
        // Kerr metric - much more complex Christoffel symbols
        double M = blackhole->mass;
        double a = blackhole->spin * M;
        
        // Avoid singularity
        if (r <= blackhole->r_plus + BH_EPSILON) {
            r = blackhole->r_plus + BH_EPSILON;
        }
        
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        double sin_theta_sq = sin_theta * sin_theta;
        double cos_theta_sq = cos_theta * cos_theta;
        
        // Intermediate calculations for Kerr metric
        double Sigma = r*r + a*a * cos_theta_sq;
        double Delta = r*r - 2.0*M*r + a*a;
        (void)Delta;
        double Sigma_sq = Sigma * Sigma;
        double Sigma_cube = Sigma_sq * Sigma;
        (void)Sigma_cube;
        
        // These are just a few of the Christoffel symbols for Kerr metric
        // The full set is quite extensive and complex
        
        // Some of the radial components
        christoffel[0][0][1] = M * (r*r - a*a*cos_theta_sq) / Sigma_sq;
        christoffel[0][1][0] = christoffel[0][0][1];
        christoffel[0][1][3] = -a*M*sin_theta_sq * (r*r - a*a*cos_theta_sq) / Sigma_sq;
        christoffel[0][3][1] = christoffel[0][1][3];
        
        // This is just a small subset of the Kerr Christoffel symbols
        // In a full implementation, all 40 non-zero components would be computed
    }
}

/**
 * Compute geodesic equation right-hand side for numerical integration
 */
void geodesic_equation(const double position[4], const double velocity[4], 
                       const BlackHoleParams* blackhole, double acceleration[4]) {
    double r = position[1];
    double theta = position[2];
    
    // Initialize accelerations
    memset(acceleration, 0, 4 * sizeof(double));
    
    // Calculate Christoffel symbols at current position
    double christoffel[4][4][4];
    calculate_christoffel_symbols(r, theta, blackhole, christoffel);
    
    // Calculate acceleration using the geodesic equation
    // d²x^μ/dλ² + Γ^μ_αβ (dx^α/dλ)(dx^β/dλ) = 0
    for (int mu = 0; mu < 4; mu++) {
        for (int alpha = 0; alpha < 4; alpha++) {
            for (int beta = 0; beta < 4; beta++) {
                acceleration[mu] -= christoffel[mu][alpha][beta] * velocity[alpha] * velocity[beta];
            }
        }
    }
}

/**
 * Calculate the gravitational time dilation factor at a given radius
 */
double calculate_time_dilation(double r, const BlackHoleParams* blackhole) {
    double rs = blackhole->schwarzschild_radius;
    // Schwarzschild time dilation formula
    return 1.0 / sqrt(1.0 - rs / r);
}

/**
 * Convert Cartesian coordinates to spherical coordinates
 */
void cartesian_to_spherical(const Vector3D* cartesian, Vector3D* spherical) {
    double x = cartesian->x;
    double y = cartesian->y;
    double z = cartesian->z;
    
    // Calculate r (radius)
    double r = sqrt(x*x + y*y + z*z);
    
    // Calculate theta (polar angle)
    double theta = 0.0;
    if (r > BH_EPSILON) {
        theta = acos(z / r);
    }
    
    // Calculate phi (azimuthal angle)
    double phi = atan2(y, x);
    if (phi < 0.0) {
        phi += BH_TWO_PI;
    }
    
    spherical->x = r;
    spherical->y = theta;
    spherical->z = phi;
}

/**
 * Convert spherical coordinates to Cartesian coordinates
 */
void spherical_to_cartesian(const Vector3D* spherical, Vector3D* cartesian) {
    double r = spherical->x;
    double theta = spherical->y;
    double phi = spherical->z;
    
    cartesian->x = r * sin(theta) * cos(phi);
    cartesian->y = r * sin(theta) * sin(phi);
    cartesian->z = r * cos(theta);
}

/**
 * Calculate the effective potential for a particle in orbit around the black hole
 */
double calculate_effective_potential(double r, double l, const BlackHoleParams* blackhole) {
    if (blackhole->spin == 0.0) {
        // Schwarzschild effective potential
        double rs = blackhole->schwarzschild_radius;
        
        // Avoid singularity
        if (r <= rs + BH_EPSILON) {
            r = rs + BH_EPSILON;
        }
        
        // Effective potential for test particle (simplified)
        // V_eff = (1 - rs/r) * (1 + l²/(r²))
        return (1.0 - rs / r) * (1.0 + (l * l) / (r * r));
    } else {
        // Kerr effective potential (simplified, equatorial)
        double M = blackhole->mass;
        double a = blackhole->spin * M;
        double E = 1.0; // Assuming unit energy
        
        // Avoid singularity
        if (r <= blackhole->r_plus + BH_EPSILON) {
            r = blackhole->r_plus + BH_EPSILON;
        }
        
        // Delta parameter
        double Delta = r*r - 2.0*M*r + a*a;
        (void)Delta;
        // Effective potential for equatorial orbits in Kerr
        // This is a simplified version
        double term1 = E*E - 1.0;
        double term2 = 2.0*M / r;
        double term3 = l*l / (r*r);
        double term4 = -2.0*M*a*l / (r*r*r);
        
        return term1 + term2 * (term3 + term4);
    }
}

/**
 * Get the innermost stable circular orbit (ISCO) radius
 * For Schwarzschild: 6M
 * For Kerr: depends on spin and orbit direction
 */
double get_isco_radius(const BlackHoleParams* blackhole) {
    double M = blackhole->mass;
    double a = blackhole->spin * M;
    
    if (blackhole->spin == 0.0) {
        // Schwarzschild black hole, ISCO is at 6M
        return 6.0 * M;
    } else {
        // Kerr black hole - prograde orbit (co-rotating with BH)
        // Formula from Bardeen, Press, Teukolsky (1972)
        double Z1 = 1.0 + pow(1.0 - a*a/(M*M), 1.0/3.0) * 
                   (pow(1.0 + a/(M), 1.0/3.0) + pow(1.0 - a/(M), 1.0/3.0));
        double Z2 = sqrt(3.0*a*a/(M*M) + Z1*Z1);
        
        // For prograde orbits
        double r_isco_prograde = M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2)));
        
        // For retrograde orbits
        double r_isco_retrograde = M * (3.0 + Z2 + sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2)));
        (void)r_isco_retrograde;
        // Return prograde ISCO (smaller, closer to black hole)
        return r_isco_prograde;
    }
}

/**
 * Calculate the ergosphere radius at a given angle theta
 * Only exists for Kerr black holes (a > 0)
 */
double calculate_ergosphere_radius(double theta, const BlackHoleParams* blackhole) {
    double M = blackhole->mass;
    double a = blackhole->spin * M;
    
    if (a <= BH_EPSILON) {
        // For Schwarzschild, ergosphere coincides with event horizon
        return 2.0 * M;
    }
    
    // For Kerr: r_ergo = M + sqrt(M² - a²cos²θ)
    double cos_theta = cos(theta);
    return M + sqrt(M*M - a*a*cos_theta*cos_theta);
}

/**
 * Initialize black hole parameters, calculating derived values
 */
void initialize_black_hole_params(BlackHoleParams* blackhole, double mass, double spin, double charge) {
    blackhole->mass = mass;
    blackhole->spin = spin;
    blackhole->charge = charge;
    
    if (spin == 0.0 && charge == 0.0) {
        // Schwarzschild black hole
        blackhole->schwarzschild_radius = 2.0 * mass;
        blackhole->r_plus = 2.0 * mass;
        blackhole->r_minus = 0.0;  // No inner horizon
        blackhole->ergosphere_radius = 2.0 * mass;  // Coincides with horizon
    } else if (spin > 0.0 && charge == 0.0) {
        // Kerr black hole
        double a = spin * mass;
        double r_plus = mass + sqrt(mass*mass - a*a);
        double r_minus = mass - sqrt(mass*mass - a*a);
        
        blackhole->schwarzschild_radius = 2.0 * mass;  // Reference value
        blackhole->r_plus = r_plus;
        blackhole->r_minus = r_minus;
        blackhole->ergosphere_radius = 2.0 * mass;  // At equator
    } else {
        // Charged and/or rotating black hole
        double a = spin * mass;
        double r_plus = mass + sqrt(mass*mass - a*a - charge*charge);
        double r_minus = mass - sqrt(mass*mass - a*a - charge*charge);
        
        blackhole->schwarzschild_radius = 2.0 * mass;  // Reference value
        blackhole->r_plus = r_plus;
        blackhole->r_minus = r_minus;
        blackhole->ergosphere_radius = 2.0 * mass;  // Approximation at equator
    }
    
    // Calculate ISCO radius
    blackhole->isco_radius = get_isco_radius(blackhole);
}

/**
 * Calculate the Kerr metric components at a given position in Boyer-Lindquist coordinates
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param metric Output metric tensor components
 * @return 0 if successful, error code otherwise
 */
int calculate_kerr_metric_bl(const double position[4], double a, double M, KerrMetric* metric) {
    if (position == NULL || metric == NULL || a < 0 || a >= 1) {
        return -1;  // Invalid parameters
    }
    
    // Extract coordinates
    double r = position[1];
    double theta = position[2];
    
    // Common terms in Kerr metric
    double sin_theta = sin(theta);
    double sin_theta_sq = sin_theta * sin_theta;
    double cos_theta = cos(theta);
    double cos_theta_sq = cos_theta * cos_theta;
    
    double a_sq = a * a;
    double r_sq = r * r;
    double two_mr = 2.0 * M * r;
    
    double Sigma = r_sq + a_sq * cos_theta_sq;
    double Delta = r_sq - two_mr + a_sq;
    double A = (r_sq + a_sq) * (r_sq + a_sq) - Delta * a_sq * sin_theta_sq;
    (void)A;  // Silence unused variable warning
    
    // Calculate metric components
    // Time-time component
    metric->g_tt = -(1.0 - two_mr / Sigma);
    
    // Time-phi component
    metric->g_tphi = -two_mr * a * sin_theta_sq / Sigma;
    
    // r-r component
    metric->g_rr = Sigma / Delta;
    
    // theta-theta component
    metric->g_thetatheta = Sigma;
    
    // phi-phi component
    metric->g_phiphi = (r_sq + a_sq + two_mr * a_sq * sin_theta_sq / Sigma) * sin_theta_sq;
    
    return 0;
}

/**
 * Calculate the inverse Kerr metric components
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param inv_metric Output inverse metric tensor components
 * @return 0 if successful, error code otherwise
 */
int calculate_inverse_kerr_metric(const double position[4], double a, double M, KerrMetric* inv_metric) {
    if (position == NULL || inv_metric == NULL || a < 0 || a >= 1) {
        return -1;  // Invalid parameters
    }
    
    // Extract coordinates
    double r = position[1];
    double theta = position[2];
    
    // Common terms
    double sin_theta = sin(theta);
    double sin_theta_sq = sin_theta * sin_theta;
    double cos_theta = cos(theta);
    double cos_theta_sq = cos_theta * cos_theta;
    
    double a_sq = a * a;
    double r_sq = r * r;
    double two_mr = 2.0 * M * r;
    
    double Sigma = r_sq + a_sq * cos_theta_sq;
    double Delta = r_sq - two_mr + a_sq;
    
    // Calculate inverse metric components
    // Time-time component
    inv_metric->g_tt = -((r_sq + a_sq) * (r_sq + a_sq) - Delta * a_sq * sin_theta_sq) / (Sigma * Delta);
    
    // Time-phi component
    inv_metric->g_tphi = -two_mr * a / (Sigma * Delta);
    
    // r-r component
    inv_metric->g_rr = Delta / Sigma;
    
    // theta-theta component
    inv_metric->g_thetatheta = 1.0 / Sigma;
    
    // phi-phi component
    inv_metric->g_phiphi = (Delta - a_sq * sin_theta_sq) / (Sigma * Delta * sin_theta_sq);
    
    return 0;
}

/**
 * Calculate Kerr Christoffel symbols at a given position
 * 
 * Note: This is a complex calculation with many components.
 * For implementation simplicity, we only calculate the most important
 * components for geodesic calculations.
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param christoffel Output array for Christoffel symbols (4x4x4 array)
 * @return 0 if successful, error code otherwise
 */
int calculate_kerr_christoffel(const double position[4], double a, double M, double christoffel[4][4][4]) {
    if (position == NULL || christoffel == NULL || a < 0 || a >= 1) {
        return -1;  // Invalid parameters
    }
    
    // Zero out the array first
    memset(christoffel, 0, 4 * 4 * 4 * sizeof(double));
    
    // Extract coordinates
    double r = position[1];
    double theta = position[2];
    (void)theta;
    // Common terms
    double sin_theta = sin(theta);
    double sin_theta_sq = sin_theta * sin_theta;
    (void)sin_theta_sq;
    double cos_theta = cos(theta);
    
    double a_sq = a * a;
    double r_sq = r * r;
    double two_mr = 2.0 * M * r;
    
    double Sigma = r_sq + a_sq * cos_theta * cos_theta;
    double Sigma_sq = Sigma * Sigma;
    double Delta = r_sq - two_mr + a_sq;
    
    // Some components of the Christoffel symbols for Kerr metric
    // Note: This is not a complete implementation - implementing all components
    // would require significant space
    
    // Γ^t_tr = M * (r_sq - a_sq * cos_theta * cos_theta) / (Sigma_sq * Delta)
    christoffel[0][0][1] = M * (r_sq - a_sq * cos_theta * cos_theta) / (Sigma_sq * Delta);
    
    // Γ^t_tθ = -2 * M * r * a_sq * sin_theta * cos_theta / Sigma_sq
    christoffel[0][0][2] = -2.0 * M * r * a_sq * sin_theta * cos_theta / Sigma_sq;
    
    // Γ^r_tt = Delta * M * (r_sq - a_sq * cos_theta * cos_theta) / Sigma_sq
    christoffel[1][0][0] = Delta * M * (r_sq - a_sq * cos_theta * cos_theta) / Sigma_sq;
    
    // Γ^r_rr = (M * (r_sq - a_sq * cos_theta * cos_theta) - r * Delta) / (Sigma * Delta)
    christoffel[1][1][1] = (M * (r_sq - a_sq * cos_theta * cos_theta) - r * Delta) / (Sigma * Delta);
    
    // Γ^θ_θr = r / Sigma
    christoffel[2][2][1] = r / Sigma;
    
    // Γ^θ_θθ = -a_sq * sin_theta * cos_theta / Sigma
    christoffel[2][2][2] = -a_sq * sin_theta * cos_theta / Sigma;
    
    // Γ^φ_rφ = (r * Delta - M * (r_sq - a_sq * cos_theta * cos_theta)) / (Sigma * Delta)
    christoffel[3][1][3] = (r * Delta - M * (r_sq - a_sq * cos_theta * cos_theta)) / (Sigma * Delta);
    
    // Γ^φ_θφ = cot(θ) - a_sq * sin_theta * cos_theta / Sigma
    christoffel[3][2][3] = 1.0 / tan(theta) - a_sq * sin_theta * cos_theta / Sigma;
    
    return 0;
}

/**
 * Calculate the ISCO (Innermost Stable Circular Orbit) radius for a Kerr black hole
 * 
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param prograde Whether to calculate for prograde (true) or retrograde (false) orbits
 * @return The ISCO radius in geometrized units (GM/c²)
 */
double calculate_kerr_isco(double a, double M, bool prograde) {
    double a_sign = prograde ? a : -a;
    double Z1 = 1.0 + pow(1.0 - a_sign * a_sign, 1.0/3.0) * 
                (pow(1.0 + a_sign, 1.0/3.0) + pow(1.0 - a_sign, 1.0/3.0));
    double Z2 = sqrt(3.0 * a_sign * a_sign + Z1 * Z1);
    
    double r_isco = M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
    return r_isco;
}

/**
 * Calculate the event horizon radius for a Kerr black hole
 * 
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @return The event horizon radius in geometrized units
 */
double calculate_kerr_event_horizon(double a, double M) {
    return M * (1.0 + sqrt(1.0 - a * a));
}

/**
 * Calculate the ergosphere radius at a given polar angle for a Kerr black hole
 * 
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param theta Polar angle
 * @return The ergosphere radius at the given angle
 */
double calculate_kerr_ergosphere(double a, double M, double theta) {
    return M * (1.0 + sqrt(1.0 - a * a * cos(theta) * cos(theta)));
}

/**
 * Calculate the relativistic velocity components for frame dragging
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param velocity Output array for velocity components (v^r, v^theta, v^phi)
 * @return 0 if successful, error code otherwise
 */
int calculate_frame_dragging(const double position[4], double a, double M, double velocity[3]) {
    if (position == NULL || velocity == NULL || a < 0 || a >= 1) {
        return -1;  // Invalid parameters
    }
    
    // Extract coordinates
    double r = position[1];
    double theta = position[2];
    
    // Calculate frame-dragging angular velocity
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double Sigma = r * r + a * a * cos_theta * cos_theta;
    double omega = 2.0 * M * r * a / (Sigma * (r * r + a * a) + 2.0 * M * r * a * a * sin_theta * sin_theta);
    
    // Set velocity components
    velocity[0] = 0.0;  // v^r
    velocity[1] = 0.0;  // v^theta
    velocity[2] = omega; // v^phi - angular velocity from frame dragging
    
    return 0;
}

/**
 * Calculate the Kerr geodesic equations for a given position, velocity, and black hole parameters
 * This computes the rate of change of velocity components (accelerations)
 * 
 * @param position The position in Boyer-Lindquist coordinates (t, r, theta, phi)
 * @param velocity The velocity components (v^t, v^r, v^theta, v^phi)
 * @param a Dimensionless spin parameter of the black hole (0 <= a < 1)
 * @param M Mass of the black hole
 * @param acceleration Output array for accelerations (a^t, a^r, a^theta, a^phi)
 * @return 0 if successful, error code otherwise
 */
int calculate_kerr_geodesic(
    const double position[4], 
    const double velocity[4], 
    double a, 
    double M, 
    double acceleration[4]) 
{
    if (position == NULL || velocity == NULL || acceleration == NULL || a < 0 || a >= 1) {
        return -1;  // Invalid parameters
    }
    
    // Calculate Christoffel symbols
    double christoffel[4][4][4] = {0};
    int result = calculate_kerr_christoffel(position, a, M, christoffel);
    if (result != 0) {
        return result;
    }
    
    // Initialize acceleration array to zeros
    memset(acceleration, 0, 4 * sizeof(double));
    
    // Compute the geodesic equation: a^μ = -Γ^μ_αβ v^α v^β
    for (int mu = 0; mu < 4; mu++) {
        for (int alpha = 0; alpha < 4; alpha++) {
            for (int beta = 0; beta < 4; beta++) {
                acceleration[mu] -= christoffel[mu][alpha][beta] * velocity[alpha] * velocity[beta];
            }
        }
    }
    
    return 0;
} 