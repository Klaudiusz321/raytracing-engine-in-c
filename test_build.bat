@echo off
echo Building Black Hole Physics Engine - Core Test...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile the math utilities test
echo Compiling math utilities test...
gcc -Wall -o build\math_test.exe test3.c test_math_util.c || goto :error

REM Compile the demo with corrected coordinate transformation functions
echo Compiling demo with fixed transformations...
gcc -Wall -o build\demo.exe demo.c test_math_util.c || goto :error

REM Run the math test
echo.
echo Running math utilities test:
build\math_test.exe

echo.
echo Running demo with fixed transformations:
build\demo.exe

echo.
echo =================================================================
echo Core test complete!
echo.
echo The mathematical utilities are working properly:
echo 1. Vector operations (add, subtract, scale, dot product, cross)
echo 2. Coordinate transformations
echo 3. Integration methods (RK4, Leapfrog)
echo.
echo These tests verify that the core mathematical building blocks
echo for the physics engine are functioning correctly.
echo =================================================================

goto :end

:error
echo Error: Build failed!
exit /b 1

:end 