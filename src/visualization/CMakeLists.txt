cmake_minimum_required(VERSION 3.10)
project(BlackHoleVisualizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenGL REQUIRED)

# Define GLFW paths
set(GLFW_DIR ${CMAKE_SOURCE_DIR}/../../lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64)
set(GLFW_INCLUDE_DIR ${GLFW_DIR}/include)
set(GLFW_LIBRARY ${GLFW_DIR}/lib-mingw-w64/libglfw3.a)

# Add GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Updated path to GLFW - using the lib directory instead of looking for it in src/visualization/external
# We're actually using a pre-built GLFW from lib directory
# add_subdirectory(${CMAKE_SOURCE_DIR}/external/glfw EXCLUDE_FROM_ALL)

# Add GLAD
# Updated path to GLAD - we need to create a custom target
add_library(glad STATIC ${CMAKE_SOURCE_DIR}/../../src/gl.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/../../include)

# Add ImGui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/../../external/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC 
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLFW_INCLUDE_DIR}  # Add GLFW include directory for ImGui GLFW backend
)
target_link_libraries(imgui PUBLIC ${GLFW_LIBRARY})

# Add black hole physics library
add_library(blackhole_physics STATIC
    ${CMAKE_SOURCE_DIR}/../../src/math_util.c
    ${CMAKE_SOURCE_DIR}/../../src/spacetime.c
    ${CMAKE_SOURCE_DIR}/../../src/raytracer.c
    ${CMAKE_SOURCE_DIR}/../../src/particle_sim.c
    ${CMAKE_SOURCE_DIR}/../../src/blackhole_api.c
)
target_include_directories(blackhole_physics PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
)

# Add executable
add_executable(blackhole_visualizer
    main.cpp
    renderer.cpp
)

# Link libraries
target_link_libraries(blackhole_visualizer
    blackhole_physics
    OpenGL::GL
    glad
    ${GLFW_LIBRARY}
    imgui
    -lgdi32  # Required for GLFW on Windows
)

# Set include directories
target_include_directories(blackhole_visualizer PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
    ${GLFW_INCLUDE_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_SOURCE_DIR}/../../external/glm
)

# Create shader directory in build
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

# Installation
install(TARGETS blackhole_visualizer DESTINATION bin)
# Skip shader copy since we don't have shaders directory yet
# install(DIRECTORY ${CMAKE_SOURCE_DIR}/shaders/ DESTINATION bin/shaders) 