# תיקון מהיר - הפעלת app.py עם כפתור הפליי

## מה תוקן:

1. ✅ הוספתי shebang line (`#!/usr/bin/env python`) ל-`app.py`
2. ✅ שיפרתי את `.vscode/launch.json` עם הגדרות נכונות
3. ✅ שיפרתי את `.vscode/settings.json` עם הגדרות Python
4. ✅ תיקנתי את `input()` ב-`app.py` - עכשיו לא חוסם בעת הפעלה מ-IDE
5. ✅ יצרתי `.vscode/extensions.json` עם הרחבות מומלצות

## איך להפעיל עכשיו:

### שיטה 1: כפתור הפליי (▶️)
1. פתח את `app.py`
2. לחץ על כפתור הפליי (▶️) בפינה הימנית העליונה
3. או לחץ `F5`

### שיטה 2: דרך התפריט
1. לחץ `F5` או `Run > Start Debugging`
2. בחר **"Python: Run app.py"**

### שיטה 3: Code Runner
1. לחץ `Ctrl+Alt+N`
2. או לחץ על כפתור Run בפינה הימנית העליונה

## אם זה עדיין לא עובד:

1. **ודא שהרחבת Python מותקנת:**
   - לחץ `Ctrl+Shift+X`
   - חפש: "Python" (מ-Microsoft)
   - לחץ Install

2. **בחר Python Interpreter:**
   - לחץ `Ctrl+Shift+P`
   - חפש: "Python: Select Interpreter"
   - בחר את גרסת Python שלך

3. **הפעל מחדש את Cursor/VS Code**

4. **נסה מהטרמינל:**
   ```powershell
   python app.py
   ```

5. **אם יש שגיאות - שלח אותן**

## קבצים שנוצרו/עודכנו:

- ✅ `app.py` - הוספתי shebang ותיקנתי `input()`
- ✅ `.vscode/launch.json` - שיפרתי הגדרות
- ✅ `.vscode/settings.json` - הוספתי הגדרות Python
- ✅ `.vscode/extensions.json` - יצרתי רשימת הרחבות מומלצות

