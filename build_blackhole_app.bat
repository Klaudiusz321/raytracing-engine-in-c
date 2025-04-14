@echo off
setlocal enabledelayedexpansion

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Set include and library paths - using the actual project structure
set INCLUDE_DIRS=-I include -I lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64/include -I external/imgui

REM GLFW library path - using the actual location
set GLFW_LIB=lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64/lib-mingw-w64/libglfw3.a

REM Compile GLAD (which is actually gl.c in your project)
echo Compiling GLAD/GL...
g++ -c src/gl.c -o build/gl.o -I include

REM Compile ImGui (assuming you downloaded it to external/imgui)
echo Compiling ImGui...
g++ -c external/imgui/imgui.cpp -o build/imgui.o %INCLUDE_DIRS%
g++ -c external/imgui/imgui_demo.cpp -o build/imgui_demo.o %INCLUDE_DIRS%
g++ -c external/imgui/imgui_draw.cpp -o build/imgui_draw.o %INCLUDE_DIRS%
g++ -c external/imgui/imgui_tables.cpp -o build/imgui_tables.o %INCLUDE_DIRS%
g++ -c external/imgui/imgui_widgets.cpp -o build/imgui_widgets.o %INCLUDE_DIRS%
g++ -c external/imgui/backends/imgui_impl_glfw.cpp -o build/imgui_impl_glfw.o %INCLUDE_DIRS%
g++ -c external/imgui/backends/imgui_impl_opengl3.cpp -o build/imgui_impl_opengl3.o %INCLUDE_DIRS%

REM Compile renderer
echo Compiling renderer...
g++ -c src/visualization/renderer.cpp -o build/renderer.o %INCLUDE_DIRS%

REM Compile main
echo Compiling main...
g++ -c src/visualization/main.cpp -o build/main.o %INCLUDE_DIRS%

REM Compile blackhole physics engine files
echo Compiling physics engine...
g++ -c src/math_util.c -o build/math_util.o -I include
g++ -c src/spacetime.c -o build/spacetime.o -I include 
g++ -c src/raytracer.c -o build/raytracer.o -I include
g++ -c src/particle_sim.c -o build/particle_sim.o -I include
g++ -c src/blackhole_api.c -o build/blackhole_api.o -I include

REM Link everything
echo Linking...
g++ build/main.o build/renderer.o build/gl.o ^
    build/imgui.o build/imgui_demo.o build/imgui_draw.o ^
    build/imgui_tables.o build/imgui_widgets.o ^
    build/imgui_impl_glfw.o build/imgui_impl_opengl3.o ^
    build/math_util.o build/spacetime.o build/raytracer.o ^
    build/particle_sim.o build/blackhole_api.o ^
    %GLFW_LIB% -o build/blackhole_visualizer.exe ^
    -lgdi32 -lopengl32 -static -static-libgcc -static-libstdc++

if %ERRORLEVEL% equ 0 (
    echo Build successful!
    echo Run build/blackhole_visualizer.exe to start the application
) else (
    echo Build failed with error code %ERRORLEVEL%
)

endlocal 