@echo off
echo ========================================
echo Adding Remote and Pushing to GitHub
echo ========================================
echo.
echo Please enter your GitHub repository URL:
echo Example: https://github.com/username/repo-name.git
echo.
set /p REPO_URL="Repository URL: "

if "%REO_URL%"=="" (
    echo Error: No URL provided
    pause
    exit /b 1
)

echo.
echo Step 1: Adding remote repository...
git remote add origin %REPO_URL% 2>nul
if errorlevel 1 (
    echo Remote might already exist, trying to set URL...
    git remote set-url origin %REPO_URL%
)
echo.

echo Step 2: Verifying remote...
git remote -v
echo.

echo Step 3: Adding files...
git add app.py rmvpe.py requirements.txt
echo.

echo Step 4: Committing...
git commit -m "001"
echo.

echo Step 5: Pushing to GitHub (main branch)...
git push -u origin main
echo.

if errorlevel 1 (
    echo.
    echo If push failed, try:
    echo   git push -u origin master
    echo   or
    echo   git push -u origin main --force
)

echo.
echo Done!
pause

