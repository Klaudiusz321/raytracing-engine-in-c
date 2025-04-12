# Black Hole Physics Engine - Testing Guide

This document provides step-by-step instructions on how to test the Black Hole Physics Engine.

## Quick Testing with Simplified Demo

The fastest way to see the physics engine working is to run the simplified test program:

```
.\simple_build.bat
```

This program demonstrates the core physics concepts:

1. **Time Dilation**: Shows how time slows down near the black hole's event horizon
2. **Orbital Mechanics**: Displays orbital velocities and periods at different radii
3. **Gravitational Lensing**: Calculates deflection angles for light passing near the black hole
4. **Accretion Disk Visualization**: Provides a simple ASCII representation of an accretion disk with temperature gradient

This test does not require the full physics engine implementation and uses simplified equations that capture the essential physics.

## Understanding the Test Results

The test output shows several important aspects of black hole physics:

### Time Dilation

```
At r = 20.00 units: Time runs inf times slower than at infinity
At r = 40.00 units: Time runs 1.4142 times slower than at infinity
At r = 60.00 units: Time runs 1.2247 times slower than at infinity
```

This shows the gravitational time dilation effect: time passes more slowly in stronger gravitational fields. At the event horizon (r = 20 units), time effectively stops from an outside observer's perspective.

### Orbital Velocities

```
Radius (R_s)   |   Orbital Velocity (c)   |   Period (M)
--------------------------------------------------------
    1.50      |       0.577350          |     326.48
    2.00      |       0.500000          |     502.65
```

This table shows how orbital velocity decreases with distance from the black hole. Close to the black hole, objects must orbit at a significant fraction of the speed of light to maintain stable orbits.

### Gravitational Lensing

```
Impact Parameter (R_s)   |   Deflection Angle (rad)
------------------------------------------------
            1.50        |         1.333333
            2.00        |         1.000000
```

This demonstrates how light rays are bent when passing near a black hole. Rays that pass within 1.5 Schwarzschild radii are deflected by more than 1 radian (about 57 degrees).

### Accretion Disk

```
                        -
                - - - - - - - - -
            - - - . . . . . . . - - -
          - - - . . . + + + . . . - - -
        - - . . . + + + + + + + . . . - -
        - - . + + + * * * * * + + + . - -
      - - . . + * * * # # # * * * + . . - -
```

This ASCII visualization shows a simulated accretion disk around the black hole. Different characters represent temperature gradients, with the inner edge being hotter (# symbols) and outer regions cooler (- symbols).

## Testing Individual Components

For more detailed testing of specific components:

### 1. Vector Math Operations

Run the vector math test:

```
.\test_build.bat
```

This tests the core mathematical utilities including:
- Vector addition, subtraction, and scaling
- Dot and cross products
- Vector normalization
- Coordinate transformations

### 2. Fixing the Full Engine

The complete physics engine has several issues that need to be fixed:

1. Update function signatures in header files to match implementations
2. Fix coordinate transformation function calls
3. Correct the time dilation function calls
4. Update the numerical integrators

## What You Should See When the Engine Works

A properly functioning black hole physics engine should demonstrate these phenomena:

1. **Gravitational Lensing**: Light bending around the black hole, creating an "Einstein Ring"
2. **Time Dilation**: Clock rates changing based on proximity to the black hole
3. **Orbital Dynamics**: Particles following correct relativistic orbits, with those closer to the ISCO (Innermost Stable Circular Orbit) showing more pronounced relativistic effects
4. **Accretion Disk**: Material orbiting the black hole with correct temperatures and relativistic effects, including redshift and blueshift based on motion relative to the observer

## Next Steps

1. Run the simplified test to understand the key concepts
2. Run the vector math test to verify the mathematical foundation
3. Continue fixing the issues in the full engine implementation
4. Once fixed, build the complete engine for more accurate simulations

## Conclusion

The simplified test provides a good demonstration of the core physics even while the full engine is being fixed. Run `.\simple_build.bat` to see these concepts in action. 