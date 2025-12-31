@echo off
echo ========================================
echo Pushing to GitHub Repository
echo Repository: https://github.com/shaharprod/rvc-tts-webui
echo ========================================
echo.

echo Step 1: Checking for existing remote...
git remote -v
echo.

echo Step 2: Adding/Updating remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/shaharprod/rvc-tts-webui.git
echo Remote set to: https://github.com/shaharprod/rvc-tts-webui.git
echo.

echo Step 3: Verifying remote...
git remote -v
echo.

echo Step 4: Adding changed files...
git add app.py rmvpe.py requirements.txt
echo.

echo Step 5: Committing changes...
git commit -m "001"
echo.

echo Step 6: Pushing to GitHub (main branch)...
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

