@echo off
setlocal enabledelayedexpansion
title Push Gear Model Project to GitHub
color 0b

echo ================================================================
echo       GEAR TOOTH UNDERCUT DETECTOR - GITHUB REPO PUBLISHER       
echo ================================================================
echo.

:: 1. Navigate directly to the project folder
cd /d "C:\Users\Administrator\Desktop\gear model"
echo Project Directory: %CD%
echo.

:: 2. Ensure Git is in PATH
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if exist "C:\Program Files\Git\cmd\git.exe" (
        set "PATH=C:\Program Files\Git\cmd;!PATH!"
    ) else (
        echo [ERROR] Git is not found! Please ensure Git is installed.
        pause
        exit /b 1
    )
)

:: 3. Verify / Set Remote URL
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Configuring remote origin...
    git remote add origin https://github.com/ashvinthnalan-2806/Gear_Tooth_Undercutting.git
)

echo Target GitHub Repository:
git remote -v
echo.

echo [1/3] Staging latest files...
git add .
echo.

echo [2/3] Verifying commit...
git commit -m "Update Gear Tooth Undercutting Detection System" >nul 2>&1
git branch -M main
echo.

echo [3/3] Pushing to GitHub (main branch)...
echo ------------------------------------------------------------
echo Target: https://github.com/ashvinthnalan-2806/Gear_Tooth_Undercutting
echo NOTE: A GitHub sign-in window may appear in your browser.
echo Simply click 'Sign in' / 'Authorize' to complete the upload.
echo ------------------------------------------------------------
echo.
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================================
    echo      SUCCESS! Your project is now safely stored on GitHub!      
    echo ================================================================
) else (
    echo.
    echo [ERROR] Push did not complete. If you see an authentication error,
    echo make sure you sign in to GitHub when prompted.
)

echo.
pause
