@echo off
setlocal enabledelayedexpansion
title Push Gear Model Project to GitHub
color 0b

echo ================================================================
echo       GEAR TOOTH UNDERCUT DETECTOR - GITHUB REPO PUBLISHER       
echo ================================================================
echo.

:: Ensure Git is in PATH
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

echo [1/5] Checking Git Status...
git status
echo.

echo [2/5] Staging files (respecting .gitignore)...
git add .
echo Staging complete.
echo.

set /p COMMIT_MSG="Enter commit message [Default: Initial commit - Gear Undercutting Detector]: "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial commit - Gear Undercutting Detector

echo [3/5] Committing changes...
git commit -m "%COMMIT_MSG%"
echo.

echo [4/5] Setting main branch...
git branch -M main
echo.

:: Check if remote 'origin' already exists
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [5/5] GitHub Remote URL Setup
    echo ------------------------------------------------------------
    echo Before proceeding, create an empty repository on GitHub:
    echo 1. Go to https://github.com/new
    echo 2. Name your repo (e.g. gear-tooth-undercut-detector)
    echo 3. Leave "Add README", "Add .gitignore", and "Choose a license" UNCHECKED!
    echo 4. Copy the repository URL (e.g. https://github.com/<username>/gear-tooth-undercut-detector.git)
    echo ------------------------------------------------------------
    echo.
    set /p REPO_URL="Enter your GitHub Repository URL: "
    if "!REPO_URL!"=="" (
        echo [ERROR] No URL provided. Cannot push without remote URL.
        pause
        exit /b 1
    )
    git remote add origin !REPO_URL!
) else (
    echo Existing remote origin found:
    git remote -v
)

echo.
echo Pushing project to GitHub (main branch)...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================================
    echo      SUCCESS! Your project is now safely stored on GitHub!      
    echo ================================================================
) else (
    echo.
    echo [NOTE] If you were prompted for credentials:
    echo - Use your GitHub Personal Access Token (PAT) as password,
    echo   OR sign in via the GitHub browser prompt.
)

pause
