@echo off
REM Compile the program
g++ main.cpp

REM Check if the compilation was successful
if %errorlevel% neq 0 (
    echo Compilation failed
    exit /b %errorlevel%
)

REM Run the program
a.exe
