# מדריך התקנה - RVC TTS WebUI

## ✅ התקנה מהירה (מומלצת)

המערכת שלך הותקנה בהצלחה! כדי להריץ בעתיד:

```bash
python app.py
```

---

## 🔧 מה תוקן

### 1. בעיית PyTorch DLL ❌→✅
**בעיה:** PyTorch 2.9.1 (גרסה לא קיימת!) גרם לשגיאת DLL
**פתרון:** הותקן PyTorch 2.0.1+cpu תקין

### 2. בעיית NumPy API ❌→✅
**בעיה:** NumPy 1.22.0 לא תואם ל-fairseq
**פתרון:** שודרג ל-NumPy 1.26.4

### 3. בעיית Gradio KeyError ❌→✅
**בעיה:** Gradio 3.34.0 עם שגיאת 'dataset'
**פתרון:** שודרג ל-Gradio 3.50.2

### 4. אזהרות TensorBoard/Protobuf ⚠️
**בעיה:** קונפליקט בגרסאות protobuf
**השפעה:** אזהרות בלבד, לא משפיע על RVC TTS

---

## 📦 גרסאות מותקנות

| חבילה | גרסה | מצב |
|------|------|-----|
| PyTorch | 2.0.1+cpu | ✅ |
| NumPy | 1.26.4 | ✅ |
| Gradio | 3.50.2 | ✅ |
| Edge TTS | 7.2.7 | ✅ |
| librosa | 0.10.0 | ✅ |
| fairseq | 0.12.2 | ✅ |

---

## 🚀 הפעלה

```bash
python app.py
```

האפליקציה תיפתח אוטומטית בדפדפן ב-http://localhost:7865

---

## ⚙️ אופציות נוספות

### הפעלה על פורט אחר
```bash
python app.py --port 8080
```

### הפעלה ללא פתיחת דפדפן
```bash
python app.py --noautoopen
```

---

## 🔄 אם יש בעיות

### בעיית DLL חוזרת
אם אתה מקבל שוב שגיאת DLL, הרץ:
```bash
fix_pytorch.bat
```

### בעיות עם Gradio
```bash
pip install gradio==3.50.2 --force-reinstall
```

### אתחול מלא
```bash
pip uninstall torch torchaudio torchvision -y
pip cache purge
pip install -r requirements.txt
```

---

## 📝 תכונות

- ✅ **Edge TTS** - המרת טקסט לדיבור חינם (ללא API key)
- ✅ **OpenAI TTS** - קולות איכותיים (דורש API key)
- ✅ **Google Cloud TTS** - תמיכה במגוון שפות (דורש API key)
- ✅ **ElevenLabs** - קולות מציאותיים (דורש API key)
- ✅ **RVC Voice Conversion** - המרת כל הקולות למודל שלך
- ✅ **תמיכה בעברית** - קולות עבריים מובנים

---

## 🎯 שימוש בסיסי

1. בחר **מודל RVC** מהרשימה (בתיקיית `weights/`)
2. בחר **מנוע TTS** (Edge TTS מומלץ להתחלה)
3. בחר **קול** מהרשימה
4. הזן **טקסט** להמרה
5. התאם **מהירות** ו-**גובה צליל**
6. לחץ **Convert**

---

## 📂 מבנה תיקיות

```
rvc-tts-webui-main/
├── app.py                 # קובץ ראשי
├── config.py             # הגדרות
├── weights/              # מודלים של RVC (הוסף כאן)
├── requirements.txt      # תלויות
├── fix_pytorch.bat       # תיקון אוטומטי
├── BUGFIXES.md          # תיעוד תיקונים
└── PYTORCH_FIX.md       # מדריך תיקון PyTorch
```

---

## 🆘 תמיכה

אם יש בעיות:
1. בדוק שכל התלויות מותקנות: `pip install -r requirements.txt`
2. קרא את [BUGFIXES.md](BUGFIXES.md) לתיקונים נפוצים
3. הרץ את [fix_pytorch.bat](fix_pytorch.bat) לתיקון PyTorch
4. קרא את [PYTORCH_FIX.md](PYTORCH_FIX.md) לפתרונות מתקדמים

---

## 📌 הערות חשובות

⚠️ **אזהרות על תלויות:** הן נורמליות! האפליקציה פותחה עם גרסאות ספציפיות שעובדות ביחד.

⚠️ **שגיאות TensorBoard:** לא משפיעות על פעולת RVC TTS.

⚠️ **GPU:** אם יש לך GPU של NVIDIA, הרץ `fix_pytorch_cuda.bat` במקום `fix_pytorch.bat`.

✅ **המערכת עובדת!** כל האזהרות הן קוסמטיות בלבד.

---

נוצר: 2025-12-13
מעודכן: לאחר תיקון כל הבעיות
