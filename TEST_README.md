# Black Hole Physics Engine - Testing Guide

This document provides instructions on how to test the Black Hole Physics Engine on Windows.

## Prerequisites

- MinGW-W64 GCC compiler installed and in your PATH
- Windows PowerShell or Command Prompt

## Quick Test

To quickly test if the project's basic components work:

1. Run the build script from PowerShell or Command Prompt:
   ```
   .\build.bat
   ```

   This will:
   - Compile the basic structure test
   - Compile the math utilities test
   - Compile the demonstration program
   - Run all three programs to show their output

2. Verify the output shows:
   - Basic vector and black hole parameter tests
   - Mathematical operations like dot products and cross products
   - Time dilation calculations
   - Simplified circular orbit visualization
   - Ray bending demonstration near a black hole

## Manual Testing

If you want to test components individually:

1. Test basic structure:
   ```
   gcc -Wall -o test.exe test.c
   .\test.exe
   ```

2. Test math utilities:
   ```
   gcc -Wall -o test3.exe test3.c test_math_util.c
   .\test3.exe
   ```

3. Run the demonstration:
   ```
   gcc -Wall -o demo.exe demo.c test_math_util.c
   .\demo.exe
   ```

## Current Project Status

The project is in development with the following components:

1. **Working Correctly**:
   - Basic data structures for black hole parameters
   - Vector operations (3D and 4D)
   - Mathematical constants and utility functions
   - Simple time dilation calculations

2. **Needs Correction**:
   - Incompatible function signatures between headers and implementation
   - Duplicate type definitions
   - Include order issues between header files

3. **Demo Functionality**:
   - The demo shows a simplified version of what the full engine will do
   - Ray bending, time dilation, and orbital mechanics are shown with basic approximations
   - A full implementation would use proper relativistic calculations

## Full Engine Testing (Once Fixed)

When the full engine is operational, you would be able to:

1. Compile the complete project:
   ```
   gcc -Wall -Iinclude -o blackhole_sim.exe src/*.c
   ```

2. Run the simulation:
   ```
   .\blackhole_sim.exe
   ```

3. Observe accurate physics simulations including:
   - Proper gravitational lensing of light rays
   - Accurate time dilation calculations
   - Particle trajectories following geodesics
   - Accretion disk dynamics
   - Hawking radiation effects 