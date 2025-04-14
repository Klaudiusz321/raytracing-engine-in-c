cmake_minimum_required(VERSION 3.10)
project(BlackHoleVisualizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenGL REQUIRED)

# Add GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/glfw EXCLUDE_FROM_ALL)

# Add GLAD
add_library(glad STATIC ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/external/glad/include)

# Add ImGui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
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
)
target_link_libraries(imgui PUBLIC glfw)

# Add black hole physics library
add_library(blackhole_physics STATIC
    ${CMAKE_SOURCE_DIR}/src/math_util.c
    ${CMAKE_SOURCE_DIR}/src/spacetime.c
    ${CMAKE_SOURCE_DIR}/src/raytracer.c
    ${CMAKE_SOURCE_DIR}/src/particle_sim.c
    ${CMAKE_SOURCE_DIR}/src/blackhole_api.c
)
target_include_directories(blackhole_physics PUBLIC
    ${CMAKE_SOURCE_DIR}/include
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
    glfw
    imgui
)

# Set include directories
target_include_directories(blackhole_visualizer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/glfw/include
    ${CMAKE_SOURCE_DIR}/external/glad/include
    ${CMAKE_SOURCE_DIR}/external/imgui
)

# Copy shaders to build directory
file(GLOB SHADER_FILES ${CMAKE_SOURCE_DIR}/src/visualization/shaders/*)
file(COPY ${SHADER_FILES} DESTINATION ${CMAKE_BINARY_DIR}/shaders)

# Installation
install(TARGETS blackhole_visualizer DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/visualization/shaders/ DESTINATION bin/shaders) 