@echo off
echo ========================================
echo Setting up and pushing to GitHub
echo ========================================
echo.

echo Step 1: Checking current remotes...
git remote -v
echo.

echo Step 2: If no remote found, you'll need to add it:
echo   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
echo.

echo Step 3: Adding files...
git add app.py rmvpe.py requirements.txt
echo.

echo Step 4: Committing...
git commit -m "001"
echo.

echo Step 5: Checking current branch...
git branch
echo.

echo Step 6: Pushing to GitHub...
echo   (If first time, use: git push -u origin main)
echo   (Or if branch is master: git push -u origin master)
git push -u origin main 2>nul || git push -u origin master 2>nul || echo Please run: git push -u origin YOUR_BRANCH_NAME
echo.

echo Done!
pause

