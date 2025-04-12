@echo off
echo Black Hole Physics Engine - Testing Options
echo ===========================================
echo.

:menu
echo Available Tests:
echo.
echo 1. Run Simplified Demo (recommended)
echo 2. Run Core Math Test
echo 3. Run Full Project Test (incomplete/has errors)
echo 4. View Project Status
echo 5. Exit
echo.
set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" (
    echo.
    echo Running Simplified Demo...
    echo.
    call simple_build.bat
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Running Core Math Test...
    echo.
    call test_build.bat
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo Attempting to build full project (this will show errors)...
    echo.
    call fixed_build.bat
    goto menu
)

if "%choice%"=="4" (
    echo.
    type PROJECT_STATUS.md
    echo.
    pause
    goto menu
)

if "%choice%"=="5" (
    echo.
    echo Thank you for testing the Black Hole Physics Engine!
    exit /b 0
)

echo.
echo Invalid choice. Please try again.
goto menu 