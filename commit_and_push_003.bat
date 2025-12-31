@echo off
echo ========================================
echo Commit and Push 003
echo Repository: https://github.com/shaharprod/rvc-tts-webui
echo ========================================
echo.

echo Step 1: Pulling latest changes from GitHub...
git pull origin main --no-rebase
if errorlevel 1 (
    echo Trying with rebase...
    git pull origin main --rebase
)
echo.

echo Step 2: Checking status...
git status
echo.

echo Step 3: Adding all changed files...
git add -A
echo.

echo Step 4: Committing changes (003)...
git commit -m "003"
echo.

echo Step 5: Pushing to GitHub (main branch)...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push failed. Trying master branch...
    git push -u origin master
)
echo.

echo ========================================
echo Done! Check your repository at:
echo https://github.com/shaharprod/rvc-tts-webui
echo ========================================
pause

