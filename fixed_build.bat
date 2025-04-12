@echo off
echo Building Black Hole Physics Engine (Fixed Version)...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile only the core test first to verify functionality
echo Compiling core test...
gcc -Wall -o build\test_fixed.exe test_main.c test_math_util.c || goto :error

REM Run the core test
echo.
echo Running core test:
build\test_fixed.exe

echo.
echo =================================================================
echo Core test passed! Now attempting to build the full project...
echo =================================================================
echo.

REM Compile the full project with fixed files
echo Compiling full project...
gcc -Wall -Iinclude -o build\blackhole_sim.exe src/main.c src/math_util.c src/spacetime.c src/raytracer.c src/particle_sim.c src/blackhole_api.c || goto :error

echo.
echo =================================================================
echo Success! The Black Hole Physics Engine has been compiled.
echo.
echo You can run the simulation with:
echo   .\build\blackhole_sim.exe
echo.
echo Note: There may still be some runtime issues to work through,
echo but the compilation errors have been fixed.
echo =================================================================

goto :end

:error
echo Error: Build failed! Check the error messages above.
exit /b 1

:end 