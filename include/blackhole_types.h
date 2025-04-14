/**
 * blackhole_types.h
 * 
 * Common type definitions for the black hole physics engine.
 */

#ifndef BLACKHOLE_TYPES_H
#define BLACKHOLE_TYPES_H

#include <stdint.h>

/**
 * Vector in 3D space
 */
typedef struct {
    double x;
    double y;
    double z;
} Vector3D;

/**
 * Vector in 4D spacetime (t, x, y, z)
 */
typedef struct {
    double t;  /* Time component */
    double x;
    double y;
    double z;
} Vector4D;

/**
 * Represents a ray in 3D space with origin and direction
 */
typedef struct {
    Vector3D origin;
    Vector3D direction;  /* Should be normalized */
} Ray;

/**
 * Metric tensor components for Schwarzschild metric
 */
typedef struct {
    double g_tt;  /* Time-time component */
    double g_rr;  /* Radial-radial component */
    double g_thth; /* Theta-theta component */
    double g_phph; /* Phi-phi component */
} SchwarzschildMetric;

/**
 * Metric tensor components for Kerr metric (rotating black hole)
 * Only storing the non-zero components
 */
typedef struct {
    double g_tt;       /* Time-time component */
    double g_tphi;     /* Time-phi cross component */
    double g_rr;       /* Radial-radial component */
    double g_thth;     /* Theta-theta component */
    double g_thetatheta; /* Same as g_thth for implementation compatibility */
    double g_phiphi;   /* Phi-phi component */
    double g_phit;     /* Phi-time cross component (same as g_tphi) */
} KerrMetric;

/**
 * Black hole metrics - union to store either Schwarzschild or Kerr parameters
 */
typedef struct {
    int is_kerr;  /* 0 for Schwarzschild, 1 for Kerr */
    union {
        SchwarzschildMetric schwarzschild;
        KerrMetric kerr;
    } metric;
} BlackHoleMetric;

/**
 * Parameters for the black hole
 */
typedef struct {
    double mass;               /* Mass in geometric units (M) */
    double schwarzschild_radius; /* Event horizon radius (2M for Schwarzschild) */
    double spin;               /* Dimensionless spin parameter (a/M) for Kerr metric */
    double charge;             /* Electric charge (Q) for Reissner-Nordström metric */
    double r_plus;             /* Outer horizon radius (depends on spin) */
    double r_minus;            /* Inner horizon radius (depends on spin) */
    double isco_radius;        /* Innermost stable circular orbit */
    double ergosphere_radius;  /* Radius of the ergosphere at equator */
} BlackHoleParams;

/**
 * Accretion disk parameters
 */
typedef struct {
    double inner_radius;       /* Inner edge of the disk */
    double outer_radius;       /* Outer edge of the disk */
    double temperature_scale;  /* Temperature scaling factor */
    double density_scale;      /* Density scaling factor */
    double thickness_factor;   /* Thickness of the disk relative to radius (h/r) */
    double alpha_viscosity;    /* Shakura-Sunyaev alpha viscosity parameter */
} AccretionDiskParams;

/**
 * Simulation configuration parameters
 */
typedef struct {
    double time_step;          /* Time step for integration */
    double max_ray_distance;   /* Maximum ray travel distance */
    int max_integration_steps; /* Maximum integration steps for trajectories */
    double tolerance;          /* Numerical integration tolerance */
    int use_adaptive_step;     /* Whether to use adaptive step size (1) or fixed (0) */
    int use_gpu_raytracing;    /* Whether to use GPU (1) or CPU (0) for ray tracing */
    double doppler_factor;     /* Scaling for Doppler effect visualization */
    double hawking_temp_factor; /* Scaling for Hawking radiation temperature */
} SimulationConfig;

/**
 * Frame data for rendering
 * Used to transfer all necessary data from physics engine to renderer
 */
typedef struct {
    double observer_time;       /* Current observer time */
    double shadow_radius;       /* Black hole shadow apparent radius */
    double time_dilation;       /* Time dilation at observer position */
    Vector3D observer_position; /* Observer position */
    double event_horizon_size;  /* Visual size of event horizon */
    double ergosphere_size;     /* Visual size of ergosphere */
    double disk_inner_edge;     /* Visual inner edge of accretion disk */
} FrameData;




#endif /* BLACKHOLE_TYPES_H */ 