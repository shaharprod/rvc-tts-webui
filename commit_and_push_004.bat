@echo off
chcp 65001 >nul
echo ============================================================
echo יצירת קומיט 004 ופוש ל-GitHub
echo Repository: https://github.com/shaharprod/rvc-tts-webui
echo ============================================================
echo.

echo Step 1: בודק remote repository...
git remote -v
echo.

echo Step 2: מושך שינויים אחרונים מ-GitHub...
git pull origin main --no-rebase
if errorlevel 1 (
    echo מנסה עם rebase...
    git pull origin main --rebase
)
echo.

echo Step 3: מוסיף קבצים...
git add app.py requirements.txt commit_and_push_004.bat
git add .vscode/*.json
git add -A
echo.

echo Step 4: בודק סטטוס...
git status
echo.

echo Step 5: יוצר קומיט 004...
git commit -m "004: הוספת תמיכה במנועי TTS נוספים (OpenAI, Google Cloud, ElevenLabs) עם API keys"
echo.

echo Step 6: מבצע push ל-GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push ל-main נכשל, מנסה master branch...
    git push -u origin master
)
echo.

echo ============================================================
echo הושלם! בדוק את ה-repository ב:
echo https://github.com/shaharprod/rvc-tts-webui
echo ============================================================
pause

