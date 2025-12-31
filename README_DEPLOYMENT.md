# דפלוי והפעלה ציבורית

## אפשרויות להפעלה ציבורית (כמו GitHub Pages)

### 1. Gradio Share (הכי קל - כמו GitHub Pages)

הפעל את התוכנה עם שיתוף ציבורי:

```bash
# Windows PowerShell
$env:GRADIO_SHARE="true"
python app.py

# או ישירות בקוד - שנה share=False ל-share=True ב-app.py
```

זה ייצור קישור ציבורי כמו: `https://xxxxx.gradio.live` שדומה ל-GitHub Pages.

### 2. Hugging Face Spaces (מומלץ - חינם)

1. לך ל-https://huggingface.co/spaces
2. צור Space חדש
3. העלה את הקבצים:
   - `app.py`
   - `requirements.txt`
   - `rmvpe.py`
   - `config.py`
   - `vc_infer_pipeline.py`
   - כל התיקייה `lib/`
4. הוסף קובץ `README.md` עם:
   ```yaml
   ---
   title: RVC TTS WebUI
   emoji: 🎙️
   colorFrom: blue
   colorTo: purple
   sdk: gradio
   sdk_version: 3.38.0
   app_file: app.py
   pinned: false
   ---
   ```
5. ה-Space ירוץ אוטומטית ויהיה נגיש ציבורית!

### 3. הרצה מקומית עם גישה ציבורית

התוכנה כבר מוגדרת לרוץ על `0.0.0.0` - זה אומר שהיא נגישה ברשת המקומית.

להפוך לציבורית:
1. פתח פורט ב-router (port forwarding)
2. או השתמש ב-ngrok:
   ```bash
   ngrok http 7865
   ```

### 4. VPS/Cloud (AWS, Google Cloud, etc.)

העלה את הקוד ל-VPS והפעל:
```bash
python app.py
```

השרת יהיה נגיש דרך כתובת ה-IP של ה-VPS.

## השוואה

| שיטה | קושי | עלות | זמן התקנה |
|------|------|------|-----------|
| Gradio Share | קל מאוד | חינם | 1 דקה |
| Hugging Face Spaces | קל | חינם | 5 דקות |
| ngrok | בינוני | חינם (מוגבל) | 2 דקות |
| VPS | קשה | בתשלום | 30 דקות |

## המלצה

**לשימוש ציבורי:** השתמש ב-Hugging Face Spaces - זה הכי דומה ל-GitHub Pages אבל תומך ב-Python apps.

**לשימוש מהיר:** השתמש ב-Gradio Share (share=True) - זה הכי מהיר וקל.

