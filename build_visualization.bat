@echo off
setlocal enabledelayedexpansion

REM Create the build directory if it doesn't exist
if not exist build\visualization mkdir build\visualization

REM Check for CMake
where cmake >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo CMake not found. Please install CMake and add it to your PATH.
    exit /b 1
)

REM Check for external dependencies
if not exist external (
    mkdir external
    echo External dependencies directory created. Please download and extract the following:
    echo 1. GLFW to external\glfw
    echo 2. GLAD to external\glad
    echo 3. ImGui to external\imgui
    echo.
    echo You can download these from:
    echo GLFW: https://github.com/glfw/glfw/releases
    echo GLAD: https://glad.dav1d.de/ (select OpenGL 3.3 Core)
    echo ImGui: https://github.com/ocornut/imgui/releases
    exit /b 1
)

REM Configure and build using CMake
cd build\visualization
cmake ..\..\src\visualization -G "MinGW Makefiles" || (
    echo CMake configuration failed
    exit /b 1
)

cmake --build . || (
    echo Build failed
    exit /b 1
)

echo.
echo Build completed successfully!
echo You can find the executable in build\visualization\blackhole_visualizer.exe
echo.

endlocal 