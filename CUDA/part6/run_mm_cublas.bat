@echo off
setlocal enabledelayedexpansion

set SRC=matrix_cublas.cu
set EXE=matrix_cublas.exe
set CSV=matrix_cublas_results.csv
set TRIALS=5

echo Compiling...
nvcc %SRC% -o %EXE% -lcublas
if errorlevel 1 (
  echo Compile failed.
  exit /b 1
)

echo timestamp,N,trial,exit_code,stdout > %CSV%

for %%N in (256 512 1024 2048 4096) do (
  echo Running N=%%N...
  for /L %%T in (1,1,%TRIALS%) do (
    for /f "usebackq delims=" %%O in (`powershell -NoProfile -Command "Get-Date -Format s"`) do set TS=%%O

    REM Capture program output (works best if program prints ONE line)
    set OUT=
    for /f "usebackq delims=" %%R in (`.\%EXE% %%N 2^>^&1`) do set OUT=%%R
    set EXITCODE=!errorlevel!

    echo "!TS!",%%N,%%T,!EXITCODE!,"!OUT!" >> %CSV%
  )
)

echo Done. Results saved to %CSV%
endlocal