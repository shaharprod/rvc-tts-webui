# הפעלה דרך GitHub Actions

GitHub Actions יכול לעזור לך להפעיל את התוכנה, אבל יש כמה אפשרויות:

## אפשרויות להפעלה דרך GitHub

### 1. GitHub Codespaces (מומלץ - הכי קל)

GitHub Codespaces מאפשר להריץ את התוכנה בסביבת cloud:

1. לך ל-repository שלך ב-GitHub
2. לחץ על כפתור "Code" > "Codespaces" > "Create codespace"
3. ב-Codespace, הרץ:
   ```bash
   python app.py
   ```
4. Codespace יפתח את התוכנה אוטומטית

**יתרונות:**
- חינם (עד 60 שעות בחודש)
- סביבה מוכנה
- נגיש מכל מקום

### 2. GitHub Actions + Self-Hosted Runner

אם יש לך שרת משלך, אפשר להפעיל runner:

1. התקן GitHub Actions Runner על השרת
2. הוסף workflow (יש קובץ `.github/workflows/deploy.yml`)
3. כל push יריץ את התוכנה

### 3. GitHub Actions + Render/Railway (Cloud Deployment)

**Render (חינם):**

1. צור חשבון ב-https://render.com
2. צור Web Service חדש
3. חבר ל-GitHub repository
4. הגדר:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Port: `7865`

**Railway (חינם עם מגבלות):**

1. צור חשבון ב-https://railway.app
2. New Project > Deploy from GitHub repo
3. הגדר את ה-command: `python app.py`

### 4. GitHub Actions + Hugging Face Spaces

הכי מומלץ! Hugging Face Spaces תומך ב-GitHub integration:

1. לך ל-https://huggingface.co/spaces
2. צור Space חדש
3. בחר "Import from GitHub"
4. בחר את ה-repository שלך
5. ה-Space ירוץ אוטומטית!

## השוואה

| שיטה | קושי | עלות | זמן התקנה | נגישות |
|------|------|------|-----------|--------|
| Codespaces | קל | חינם (מוגבל) | 2 דקות | זמני |
| Render | בינוני | חינם | 5 דקות | קבוע |
| Railway | בינוני | חינם (מוגבל) | 5 דקות | קבוע |
| Hugging Face Spaces | קל | חינם | 3 דקות | קבוע |

## המלצה

**לשימוש קבוע:** Hugging Face Spaces - הכי קל ותומך ב-GitHub integration
**לפיתוח/בדיקה:** GitHub Codespaces - סביבה מוכנה מיד

## הגדרת Hugging Face Spaces (מומלץ)

1. לך ל-https://huggingface.co/spaces
2. לחץ "Create new Space"
3. בחר:
   - SDK: Gradio
   - Python version: 3.10
   - Visibility: Public
4. בחר "Import from GitHub"
5. בחר את ה-repository: `shaharprod/rvc-tts-webui`
6. לחץ "Create Space"

ה-Space יתקין הכל אוטומטית ויהיה נגיש ב:
`https://huggingface.co/spaces/shaharprod/rvc-tts-webui`

## קבצי הגדרה

יצרתי קבצי workflow ב-`.github/workflows/`:
- `deploy.yml` - לבדיקות אוטומטיות
- `deploy-to-render.yml` - לדפלוי ל-Render

