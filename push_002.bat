@echo off
echo ========================================
echo Pushing to GitHub - Commit 002
echo Repository: https://github.com/shaharprod/rvc-tts-webui
echo ========================================
echo.

echo Step 1: Checking for remote repository...
git remote -v
echo.

echo Step 2: Adding all changed files...
git add app.py rmvpe.py requirements.txt
git add .github/workflows/*.yml
git add README*.md
git add *.bat
echo.

echo Step 3: Checking status...
git status
echo.

echo Step 4: Committing changes (002)...
git commit -m "002"
echo.

echo Step 5: Pushing to GitHub (main branch)...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push to main failed, trying master branch...
    git push -u origin master
)
echo.

echo ========================================
echo Done! Check your repository at:
echo https://github.com/shaharprod/rvc-tts-webui
echo ========================================
pause

