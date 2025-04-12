@echo off
echo Building Black Hole Physics Engine - Simple Test...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile the simple test program
echo Compiling simple test...
gcc -Wall -o build\simple_test.exe simple_test.c || goto :error

REM Run the test
echo.
echo Running simple test:
build\simple_test.exe

echo.
echo =================================================================
echo Simple test complete!
echo.
echo This simplified test demonstrates the core concepts of the
echo black hole physics engine without requiring all the modules:
echo.
echo 1. Time dilation near a black hole
echo 2. Orbital velocities at different radii
echo 3. Gravitational lensing (ray deflection)
echo 4. Visualization of an accretion disk
echo.
echo These calculations use simplified approximations rather than
echo the full general relativistic equations.
echo =================================================================

goto :end

:error
echo Error: Build failed!
exit /b 1

:end 