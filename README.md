# Black Hole Physics Simulation Engine

![Black Hole Simulation](https://via.placeholder.com/800x400?text=Black+Hole+Simulation)

## Overview

This project implements a physics-accurate black hole simulation engine with real-time ray tracing capabilities. It models how light and matter behave in the curved spacetime around black holes, visualizing phenomena like gravitational lensing, accretion disks, relativistic effects, and particle orbits.

## Features

- **Relativistic Ray Tracing**: Accurate simulation of photon paths in curved spacetime using geodesic equations
- **Accretion Disk Simulation**: Realistic visualization of matter orbiting black holes
- **Particle Orbit Calculator**: Computation of stable and unstable orbits around black holes
- **Relativistic Effects**: Visualization of:
  - Gravitational lensing
  - Gravitational redshift
  - Doppler shift and relativistic beaming
  - Time dilation
- **Multiple Integration Methods**:
  - 4th-order Runge-Kutta (RK4)
  - Adaptive Runge-Kutta-Fehlberg (RKF45)
  - Leapfrog integration
  - Symplectic integrators

## Technical Architecture

The engine is implemented in C for performance, with a modular architecture:

- **Core Modules**:
  - `spacetime.c`: Implements the mathematical models for curved spacetime (Schwarzschild and Kerr metrics)
  - `raytracer.c`: Handles light ray propagation through curved spacetime
  - `particle_sim.c`: Simulates massive particles orbiting black holes
  - `math_util.c`: Mathematical utilities, including vector operations and numerical integrators
  - `blackhole_api.c`: High-level API for application integration

- **Numerical Methods**:
  The engine provides multiple integration methods for solving the geodesic equations, balancing accuracy and performance.

## Building the Project

### Prerequisites

- C compiler (GCC or Clang recommended)
- OpenMP support (optional, for multi-threading)

### Compilation

```bash
gcc -Wall -Iinclude -o blackhole_sim.exe src/main.c src/math_util.c src/spacetime.c src/raytracer.c src/particle_sim.c src/blackhole_api.c
```

For optimized build with OpenMP support:

```bash
gcc -O3 -fopenmp -Wall -Iinclude -o blackhole_sim.exe src/main.c src/math_util.c src/spacetime.c src/raytracer.c src/particle_sim.c src/blackhole_api.c
```

## Usage Examples

### Basic Simulation

```c
// Initialize black hole parameters
BlackHoleParams blackhole;
blackhole.mass = 1.0;  // Mass in geometric units (M)
blackhole.spin = 0.0;  // Non-rotating Schwarzschild black hole
blackhole.schwarzschild_radius = 2.0 * blackhole.mass;

// Configure simulation
SimulationConfig config;
config.max_integration_steps = 1000;
config.time_step = 0.1;
config.tolerance = 1e-6;
config.max_ray_distance = 100.0;

// Create a ray
Ray ray;
ray.origin = (Vector3D){0.0, 0.0, 30.0};  // Starting 30M away from black hole
ray.direction = (Vector3D){0.0, 0.0, -1.0};  // Pointing toward black hole

// Trace the ray
RayTraceHit hit;
RayTraceResult result = trace_ray(&ray, &blackhole, NULL, &config, &hit);

// Print results
printf("Ray result: %d\n", result);
printf("Hit position: (%.2f, %.2f, %.2f)\n", 
       hit.hit_position.x, hit.hit_position.y, hit.hit_position.z);
printf("Distance traveled: %.3f\n", hit.distance);
```

### Visualizing an Accretion Disk

```c
// Create accretion disk parameters
AccretionDiskParams disk;
disk.inner_radius = 3.0 * blackhole.mass;  // Inner edge at ISCO
disk.outer_radius = 20.0 * blackhole.mass;
disk.temperature_scale = 1.0;

// Trace rays with disk intersection check
RayTraceResult result = trace_ray(&ray, &blackhole, &disk, &config, &hit);

if (result == RAY_DISK) {
    // Calculate disk temperature and color at hit position
    double temperature;
    double color[3];
    calculate_disk_temperature(&hit.hit_position, &blackhole, &disk, &temperature, color);
    
    // Apply relativistic effects to color
    apply_relativistic_effects(&hit.hit_position, &ray.direction, &blackhole, color, NULL);
    
    // Use color for rendering
    printf("Disk hit color: RGB(%.2f, %.2f, %.2f)\n", color[0], color[1], color[2]);
}
```

## Performance Optimization

The engine incorporates several performance optimizations:

- **SIMD Vector Operations**: Accelerated vector math using CPU SIMD instructions when available
- **Multi-threading**: Parallel ray tracing using OpenMP
- **Adaptive Step Size**: RKF45 integrator adjusts step size based on local error
- **Efficient Memory Management**: Minimizes allocations during critical processing

## Development Status

This project is currently in active development. Key areas of ongoing work:

- Adding support for rotating (Kerr) black holes
- Implementing more sophisticated accretion disk models
- Improving numerical stability in extreme gravity regions
- Adding GPU acceleration for real-time visualization

## License

[MIT License](LICENSE) - Feel free to use, modify, and distribute this code.

## Acknowledgments

This project draws inspiration from:
- The work of Kip Thorne and the visual effects team for the movie "Interstellar"
- Olli Seiskari's real-time black hole visualization techniques
- Academic papers on numerical relativity and black hole physics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Directory Structure

```
├── include/              # Header files
│   ├── blackhole_api.h   # Public API
│   ├── blackhole_types.h # Common type definitions
│   ├── math_util.h       # Mathematical utilities
│   ├── particle_sim.h    # Particle simulation
│   ├── raytracer.h       # Ray tracing
│   └── spacetime.h       # Spacetime geometry
├── src/                  # Source files
├── bin/                  # Compiled binaries
├── lib/                  # Libraries
├── tests/                # Test files
├── Makefile              # Build system
└── README.md             # This file
```

## Building

### Requirements

- C compiler (GCC or Clang)
- Make

### Build Commands

```bash
# Build the default executable
make

# Create a static library
make lib

# Clean the build
make clean

# Rebuild everything
make rebuild

# Run the test program
make run

# Build WebAssembly version (requires Emscripten)
make wasm
```

## Using the Library

The library provides a simple API for integrating into your applications. Here's a quick example:

```c
#include "blackhole_api.h"

// Initialize the engine
BHContextHandle context = bh_initialize();

// Configure black hole parameters
bh_configure_black_hole(context, 1.0, 0.0, 0.0);  // Mass, spin, charge

// Configure accretion disk
bh_configure_accretion_disk(context, 6.0, 20.0, 1.0, 1.0);

// Trace a ray
Ray ray = {
    .origin = {0.0, 0.0, 30.0},
    .direction = {0.1, 0.0, -1.0}
};
RayTraceHit hit;
bh_trace_ray(context, ray.origin, ray.direction, &hit);

// Clean up
bh_shutdown(context);
```

## Science Behind the Simulation

The simulation implements various aspects of general relativity, including:

- **Geodesic Equation** for tracing light paths in curved spacetime
- **Schwarzschild Metric** for modeling the spacetime around a non-rotating black hole
- **Kerr Metric** for rotating black holes (partial implementation)
- **Relativistic Doppler and Gravitational Redshift** for accretion disk visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project draws inspiration from various scientific visualizations and simulations of black holes, including work by NASA's Goddard Space Flight Center and Kip Thorne's work for the movie "Interstellar".