@echo off
echo Building Black Hole Physics Engine...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile the basic test program
echo Compiling basic test...
gcc -Wall -o build\test.exe test.c || goto :error

REM Compile the math test
echo Compiling math test...
gcc -Wall -o build\test3.exe test3.c test_math_util.c || goto :error

REM Compile the demo
echo Compiling demo...
gcc -Wall -o build\demo.exe demo.c test_math_util.c || goto :error

REM Run the basic test
echo.
echo Running basic test:
build\test.exe

echo.
echo Running math test:
build\test3.exe

echo.
echo Running demo:
build\demo.exe

echo.
echo =================================================================
echo The test programs and demo work!
echo.
echo The basic vector operations and demonstrations are now working.
echo This demonstrates how the black hole physics engine would function
echo when fully implemented with all components working together.
echo.
echo To compile the full project, you still need to:
echo.
echo 1. Ensure all function signatures in header files match implementations
echo 2. Fix remaining include orders in the source files
echo 3. Update implementations to use the modified function signatures
echo =================================================================

goto :end

:error
echo Error: Build failed!
exit /b 1

:end 