# Black Hole Physics Engine

A high-performance simulation engine for visualizing black hole physics phenomena, including gravitational lensing, time dilation, accretion disk dynamics, particle trajectories, and Hawking radiation.

## Features

- **Gravitational Lensing**: Simulates the bending of light around a black hole, including Einstein rings and multiple imaging.
- **Time Dilation**: Calculates the differing passage of time for observers at various gravitational potentials.
- **Accretion Disk Dynamics**: Models a physically accurate accretion disk with relativistic effects.
- **Particle Trajectories**: Simulates paths of test particles moving under the black hole's gravity.
- **Hawking Radiation**: Provides a conceptual visualization of particle/antiparticle pairs at the event horizon.

## Architecture

The engine is designed with a modular architecture allowing it to work across multiple platforms:

- **Core Physics Engine**: Written in C for maximum performance and portability
- **Platform Abstraction**: Can be compiled as a native library or to WebAssembly for web use
- **Full API**: Well-documented interface for easy integration

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

This project draws inspiration from various scientific visualizations and simulations of black holes, including work by NASA's Goddard Space Flight Center and Kip Thorne's work for the movie "Interstellar". #   r a y t r a c i n g - e n g i n e - i n - c  
 