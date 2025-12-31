@echo off
echo ========================================
echo Pulling changes and pushing commit 002
echo ========================================
echo.

echo Step 1: Pulling latest changes from GitHub...
git pull origin main --no-rebase
if errorlevel 1 (
    echo Trying with rebase...
    git pull origin main --rebase
)
echo.

echo Step 2: Adding all changed files...
git add app.py rmvpe.py requirements.txt
git add .github/workflows/*.yml 2>nul
git add README*.md 2>nul
echo.

echo Step 3: Committing changes (002)...
git commit -m "002"
echo.

echo Step 4: Pushing to GitHub (main branch)...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push failed. Trying with force (be careful!)...
    echo Press Ctrl+C to cancel, or wait 5 seconds...
    timeout /t 5 /nobreak >nul
    git push -u origin main --force
)
echo.

echo ========================================
echo Done! Check your repository at:
echo https://github.com/shaharprod/rvc-tts-webui
echo ========================================
pause

