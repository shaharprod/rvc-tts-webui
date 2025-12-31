# מדריך להגדרת Python ב-Cursor/VS Code

## שלב 1: התקנת הרחבת Python

1. פתח את Cursor/VS Code
2. לחץ על `Ctrl+Shift+X` (או `View > Extensions`)
3. חפש: **"Python"** (מ-Microsoft)
4. לחץ **Install**

## שלב 2: בחירת Python Interpreter

1. לחץ `Ctrl+Shift+P` (או `View > Command Palette`)
2. חפש: **"Python: Select Interpreter"**
3. בחר את גרסת Python שלך (למשל: `Python 3.10.x`)

## שלב 3: בדיקת ההגדרות

הרץ את הסקריפט:
```powershell
.\check_python.bat
```

או:
```powershell
python check_python_setup.py
```

## שלב 4: הפעלת הקובץ

### שיטה 1: כפתור הפליי (▶️)
- לחץ על כפתור הפליי בפינה הימנית העליונה של `app.py`
- או לחץ `F5`

### שיטה 2: דרך התפריט
- לחץ `F5` או `Run > Start Debugging`
- בחר **"Python: Run app.py"**

### שיטה 3: Code Runner (אם מותקן)
- לחץ `Ctrl+Alt+N`
- או לחץ על כפתור Run בפינה הימנית העליונה

### שיטה 4: מהטרמינל
```powershell
python app.py
```

## פתרון בעיות

### בעיה: כפתור הפליי לא עובד
**פתרון:**
1. ודא שהרחבת Python מותקנת
2. ודא ש-Python Interpreter נבחר (`Ctrl+Shift+P` > `Python: Select Interpreter`)
3. נסה להריץ מהטרמינל: `python app.py`

### בעיה: "Python not found"
**פתרון:**
1. התקן Python מ-https://www.python.org/
2. ודא ש-Python ב-PATH (בחר "Add Python to PATH" בהתקנה)
3. הפעל מחדש את Cursor/VS Code

### בעיה: "Module not found"
**פתרון:**
```powershell
pip install -r requirements.txt
```

### בעיה: "Port already in use"
**פתרון:**
```powershell
python app.py --port 7866
```

## קבצי הגדרות שנוצרו

- `.vscode/launch.json` - הגדרות להפעלה עם F5
- `.vscode/tasks.json` - הגדרות לביצוע משימות
- `.vscode/settings.json` - הגדרות כלליות

## בדיקת מצב

הרץ:
```powershell
.\check_python.bat
```

זה יבדוק:
- ✓ גרסת Python
- ✓ נתיב Python
- ✓ קובץ app.py
- ✓ קבצי הגדרות .vscode
- ✓ חבילות נדרשות

