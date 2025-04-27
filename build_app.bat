@echo off
REM Build script for black hole visualization application

REM Clean previous build
if exist blackhole_visualizer.exe del /Q blackhole_visualizer.exe

REM Compile the black hole physics backend
echo Compiling physics backend...
g++ -c src/math_util.c -o math_util.o -I./include
if %errorlevel% neq 0 goto error

g++ -c src/spacetime.c -o spacetime.o -I./include
if %errorlevel% neq 0 goto error

g++ -c src/raytracer.c -o raytracer.o -I./include
if %errorlevel% neq 0 goto error

g++ -c src/particle_sim.c -o particle_sim.o -I./include
if %errorlevel% neq 0 goto error

g++ -c src/blackhole_api.c -o blackhole_api.o -I./include
if %errorlevel% neq 0 goto error

g++ -c src/gl.c -o gl.o -I./include
if %errorlevel% neq 0 goto error

REM Compile ImGui source files
echo Compiling ImGui...
g++ -c external/imgui/imgui.cpp -o imgui.o -I./external/imgui
if %errorlevel% neq 0 goto error

g++ -c external/imgui/imgui_demo.cpp -o imgui_demo.o -I./external/imgui
if %errorlevel% neq 0 goto error

g++ -c external/imgui/imgui_draw.cpp -o imgui_draw.o -I./external/imgui
if %errorlevel% neq 0 goto error

g++ -c external/imgui/imgui_tables.cpp -o imgui_tables.o -I./external/imgui
if %errorlevel% neq 0 goto error

g++ -c external/imgui/imgui_widgets.cpp -o imgui_widgets.o -I./external/imgui
if %errorlevel% neq 0 goto error

g++ -c external/imgui/backends/imgui_impl_glfw.cpp -o imgui_impl_glfw.o -I./external/imgui -I./lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64/include
if %errorlevel% neq 0 goto error

g++ -c external/imgui/backends/imgui_impl_opengl3.cpp -o imgui_impl_opengl3.o -I./external/imgui
if %errorlevel% neq 0 goto error

REM Compile the visualization frontend
echo Compiling visualization frontend...
g++ -c src/visualization/main.cpp -o main.o -I. -I./include -I./external -I./external/imgui -I./external/imgui/backends -I./external/glm -I./lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64/include
if %errorlevel% neq 0 goto error

g++ -c src/visualization/renderer.cpp -o renderer.o -I. -I./include -I./external -I./external/imgui -I./external/imgui/backends -I./external/glm -I./lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64/include
if %errorlevel% neq 0 goto error

REM Link everything together
echo Linking...
g++ -o blackhole_visualizer.exe main.o renderer.o math_util.o spacetime.o raytracer.o particle_sim.o blackhole_api.o gl.o imgui.o imgui_demo.o imgui_draw.o imgui_tables.o imgui_widgets.o imgui_impl_glfw.o imgui_impl_opengl3.o -L./lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64/lib-mingw-w64 -lglfw3 -lopengl32 -lgdi32
if %errorlevel% neq 0 goto error

REM Clean up temporary files
echo Cleaning up...
del *.o

echo Build complete. Run blackhole_visualizer.exe to start the application.
goto end

:error
echo Build failed with error code %errorlevel%
exit /b %errorlevel%

:end 