@echo off
echo ========================================
echo Quick Push to GitHub
echo ========================================
echo.

echo Checking for remote repository...
git remote -v
echo.

set REMOTE_EXISTS=0
git remote get-url origin >nul 2>&1
if %errorlevel%==0 set REMOTE_EXISTS=1

if %REMOTE_EXISTS%==0 (
    echo ERROR: No remote repository configured!
    echo.
    echo Please run: add_remote_and_push.bat
    echo Or manually add remote:
    echo   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
    pause
    exit /b 1
)

echo Remote found! Proceeding with push...
echo.

echo Step 1: Adding files...
git add app.py rmvpe.py requirements.txt
echo.

echo Step 2: Committing...
git commit -m "002"
echo.

echo Step 3: Pushing to GitHub...
git push -u origin main
if errorlevel 1 (
    echo Trying master branch...
    git push -u origin master
)
echo.

echo Done!
pause

