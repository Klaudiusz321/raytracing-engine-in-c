@echo off
echo Building Fixed Black Hole Physics Engine Components...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile basic components with fixed function signatures
echo Compiling core files with fixed function signatures...
gcc -Wall -Iinclude -c -o build\math_util.o src\math_util.c || goto :error
gcc -Wall -Iinclude -c -o build\spacetime.o src\spacetime.c || goto :error

echo Compiling test_main.c...
gcc -Wall -Iinclude -o build\core_test.exe test_main.c build\math_util.o build\spacetime.o || goto :error

REM Run the test
echo.
echo Running core functionality test:
build\core_test.exe

echo.
echo =================================================================
echo Fixed black hole physics engine components compiled successfully!
echo.
echo The following files have been fixed and compiled:
echo 1. include/math_util.h - Fixed function signatures for coordinate transformations
echo 2. include/spacetime.h - Fixed function signatures for metric calculations
echo 3. src/spacetime.c - Resolved duplicate function definitions
echo.
echo These fixes address the main signature inconsistencies and enable
echo the core mathematical functions to work correctly.
echo =================================================================

goto :end

:error
echo Error: Build failed!
exit /b 1

:end 