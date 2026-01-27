@echo off
setlocal enabledelayedexpansion

set SRC=matrix_cpu.c
set EXE=matrix_cpu.exe
set CSV=matrix_cpu_results.csv
set TRIALS=5

echo Compiling...
gcc %SRC% -O2 -o %EXE%
if errorlevel 1 (
  echo Compile failed.
  exit /b 1
)

echo timestamp,N,trial,exit_code,stdout > %CSV%

for %%N in (256 512 1024 2048 4096) do (
  echo Running N=%%N...
  for /L %%T in (1,1,%TRIALS%) do (
    for /f "usebackq delims=" %%O in (`powershell -NoProfile -Command "Get-Date -Format s"`) do set TS=%%O
    for /f "usebackq delims=" %%R in (`%EXE% %%N 2^>^&1`) do (
      set OUT=%%R
      echo "!TS!",%%N,%%T,0,"!OUT!" >> %CSV%
    )
  )
)

echo Done. Results saved to %CSV%
endlocal