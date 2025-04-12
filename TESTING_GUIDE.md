# Black Hole Physics Engine - Testing Guide

This guide provides instructions on how to test the Black Hole Physics Engine on Windows.

## Prerequisites

- MinGW-W64 GCC compiler installed and in your PATH
- Windows PowerShell or Command Prompt

## Quick Test: Simplified Version

The easiest way to test the core concepts of the physics engine is to use the simplified test:

1. Run the simple build script:
   ```
   .\simple_build.bat
   ```

   This will:
   - Compile the simplified test program
   - Run the test to demonstrate key black hole physics concepts
   - Show results in text format, including a simple ASCII visualization

2. The simplified test demonstrates:
   - Time dilation at different distances from a black hole
   - Orbital velocities and periods for circular orbits
   - Gravitational lensing (ray deflection angles)
   - A basic ASCII visualization of an accretion disk with temperature gradient

This test uses simplified equations and approximations rather than the full general relativistic calculations.

## Core Functionality Test

If you want to test the core mathematical utilities with vector operations:

1. Run the core test build script:
   ```
   .\test_build.bat
   ```

   This will test:
   - Basic vector operations
   - Coordinate transformations
   - Simple time dilation calculations

## Full Project Status

The complete project is currently being fixed to address several issues:

1. **Working Components**:
   - Basic data structures (Vector3D, BlackHoleParams, etc.)
   - Mathematical utilities (vector operations, constants)
   - Simplified physics calculations

2. **Needs Additional Fixes**:
   - Function signature mismatches between headers and implementations
   - ODE integrator function compatibility
   - Coordinate transformation function calls
   - Various parameter type issues

## Next Steps for Fixing the Project

To complete the fixes for the full project:

1. Update all function calls to cartesian_to_spherical and spherical_to_cartesian to use the pointer-based API:
   ```c
   // Instead of:
   position = spherical_to_cartesian(spherical);
   
   // Use:
   spherical_to_cartesian(&spherical, &position);
   ```

2. Update all calls to calculate_time_dilation to use BlackHoleParams pointers:
   ```c
   // Instead of:
   dilation = calculate_time_dilation(r, rs);
   
   // Use:
   dilation = calculate_time_dilation(r, blackhole);
   ```

3. Fix the RKF45 integrator function signature and calls.

4. Update the test_main.c and main.c implementations to avoid array initializers with variable length arrays.

## Running the Simplified Test 

The simplified test is the most reliable way to verify the basic physics concepts:

```
.\simple_build.bat
```

This test will show:
- Correct calculation of time dilation factors
- Proper orbital velocities at various radii
- Accurate gravitational lensing calculations
- Visual representation of an accretion disk

The output provides a good demonstration of the physics principles while the full implementation is being fixed.

## Testing Demonstrations

Several demo programs are available to test different aspects of the engine:

1. Basic structure test:
   ```
   .\build.bat
   ```

   This runs demonstrations of:
   - Vector and parameter handling
   - Mathematical operations
   - Simple physics calculations
   - A visual representation of orbits and ray bending

## Current Project Status

The project is a work in progress with the following status:

1. **Working Correctly**:
   - Core mathematical utilities
   - Vector operations and transformations
   - Basic black hole parameter handling
   - Time dilation calculations

2. **Partially Working**:
   - Simplified ray tracing with basic gravitational lensing
   - Basic orbit calculations
   - Coordinate transformations

3. **Needs Additional Fixes**:
   - Consistency between header declarations and implementations
   - Function signature mismatches in numerical integrators
   - Parameter type issues in various physics functions
   - Linking all components together for the full simulation

## Full Project Testing

To test the full project once all fixes are applied:

1. Build the complete project:
   ```
   gcc -Wall -Iinclude -o blackhole_sim.exe src/main.c src/math_util.c src/spacetime.c src/raytracer.c src/particle_sim.c src/blackhole_api.c
   ```

2. Run the simulation:
   ```
   .\blackhole_sim.exe
   ```

## Troubleshooting

If you encounter compilation errors:

1. Check for type definition conflicts:
   - Vector3D and Vector4D should only be defined in blackhole_types.h
   - Function signatures should match between .h and .c files

2. Parameter mismatch errors:
   - Ensure all function calls use the correct parameter types
   - Check that all pointers are properly dereferenced

3. Missing header includes:
   - Make sure all necessary headers are included in the correct order
   - blackhole_types.h should be included before math_util.h

## Testing Individual Components

To test specific components:

1. Spacetime calculations:
   ```
   gcc -Wall -Iinclude -o test_spacetime.exe test_spacetime.c test_math_util.c
   ```

2. Ray tracing:
   ```
   gcc -Wall -Iinclude -o test_raytracer.exe test_raytracer.c test_math_util.c
   ```

3. Particle simulation:
   ```
   gcc -Wall -Iinclude -o test_particles.exe test_particles.c test_math_util.c
   ```

Note: The test files mentioned above are placeholders - you would need to create them to test specific components.

## Progress Tracking

As you fix issues:

1. Start with the basic tests using test_build.bat
2. Move on to more complex demos with build.bat
3. Fix specific component issues using individual test programs
4. Finally, build and test the complete project 