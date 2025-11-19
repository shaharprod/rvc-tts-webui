@echo off
chcp 65001 >nul
echo ============================================================
echo בדיקת הגדרות Python
echo ============================================================
echo.

echo בדיקת גרסת Python:
python --version
if %errorlevel% neq 0 (
    echo ✗ Python לא נמצא! התקן Python מ-https://www.python.org/
    pause
    exit /b 1
)
echo ✓ Python מותקן
echo.

echo בדיקת נתיב Python:
where python
echo.

echo בדיקת קובץ app.py:
if exist app.py (
    echo ✓ קובץ app.py נמצא
) else (
    echo ✗ קובץ app.py לא נמצא!
)
echo.

echo בדיקת תיקיית .vscode:
if exist .vscode (
    echo ✓ תיקיית .vscode קיימת
    if exist .vscode\launch.json (
        echo   ✓ launch.json קיים
    ) else (
        echo   ✗ launch.json חסר!
    )
    if exist .vscode\settings.json (
        echo   ✓ settings.json קיים
    ) else (
        echo   ✗ settings.json חסר!
    )
) else (
    echo ✗ תיקיית .vscode לא קיימת!
)
echo.

echo ============================================================
echo המלצות:
echo ============================================================
echo 1. ב-Cursor/VS Code: לחץ Ctrl+Shift+P
echo 2. חפש: Python: Select Interpreter
echo 3. בחר את המתורגמן Python
echo 4. ודא שהרחבת Python מותקנת (Extensions ^> Python)
echo 5. נסה להריץ את app.py עם F5 או כפתור הפליי
echo ============================================================
echo.
pause

