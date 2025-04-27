# Black Hole Physics Visualization

This directory contains the desktop visualization frontend for the Black Hole Physics Engine. It provides a GPU-accelerated, real-time visualization of black hole physics using OpenGL and the underlying C physics engine.

## Features

- Real-time rendering of black hole gravitational lensing
- Visualization of accretion disk with relativistic effects
- Interactive UI controls using Dear ImGui
- Multi-threaded architecture with physics simulation running in a separate thread
- Camera controls to explore the black hole from different angles

## Requirements

- CMake 3.10 or higher
- C++11 compatible compiler (GCC, MSVC, Clang)
- OpenGL 3.3 or higher
- GLFW 3.3 or higher
- ImGui
- GLAD (for OpenGL function loading)

## Building

### Getting Dependencies

Before building, you need to download the required dependencies:

1. **GLFW**: Download from https://github.com/glfw/glfw/releases and extract to `external/glfw`
2. **GLAD**: Generate at https://glad.dav1d.de/ (select OpenGL 3.3 Core) and extract to `external/glad`
3. **ImGui**: Download from https://github.com/ocornut/imgui/releases and extract to `external/imgui`

### Windows (MinGW)

Use the provided build script:

```
.\build_visualization.bat
```

Or manually:

```
mkdir -p build/visualization
cd build/visualization
cmake ../../src/visualization -G "MinGW Makefiles"
cmake --build .
```

### Linux

```
mkdir -p build/visualization
cd build/visualization
cmake ../../src/visualization
make
```

### macOS

```
mkdir -p build/visualization
cd build/visualization
cmake ../../src/visualization
make
```

## Running

After building, run the executable:

```
.\build\visualization\blackhole_visualizer
```

Command-line options:
- `--width <pixels>`: Set window width (default: 1280)
- `--height <pixels>`: Set window height (default: 720)
- `--title <string>`: Set window title
- `--help`: Show help message

## Controls

### UI Controls
- Mass slider: Adjust the black hole mass
- Spin slider: Adjust the black hole spin (a/M)
- Accretion Rate slider: Adjust the accretion disk density
- Show/Hide toggles for accretion disk, grid, stars
- Enable/Disable Doppler effect and gravitational redshift
- Integration Steps slider: Adjust ray tracing accuracy
- Time Scale slider: Adjust simulation speed

### Camera Controls
- The camera automatically orbits the black hole
- In a full implementation, keyboard and mouse controls would be added

## Architecture

The visualization is built on a multi-threaded architecture:

1. **Main Thread**: Handles rendering, UI, and user input
2. **Physics Thread**: Runs the black hole physics simulation

Communication between threads is done using double-buffered data structures with thread synchronization.

## Shaders

The visualization uses several GLSL shaders:

1. **Black Hole Shader**: Implements ray tracing in curved spacetime
2. **Accretion Disk Shader**: Renders the accretion disk with relativistic effects
3. **Skybox Shader**: Renders the background stars

## Future Improvements

- User-controlled camera
- Advanced visualization effects (lens flares, bloom)
- VR support
- Vulkan renderer for improved performance
- More detailed accretion disk model
- Support for different black hole metrics

## License

This code is licensed under the same license as the main project. 