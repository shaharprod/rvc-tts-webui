# הגדרת מודלים ב-Railway

המודלים שלך גדולים מדי ל-GitHub (מעל 100MB), אז צריך להוריד אותם בזמן ה-build ב-Railway.

## אפשרות 1: Google Drive

1. העלה את תיקיית `weights/shmoolik` ל-Google Drive כ-ZIP
2. שתף את הקובץ עם "כל מי שיש לו את הקישור"
3. קבל את ה-File ID מהקישור:
   - קישור נראה כך: `https://drive.google.com/file/d/FILE_ID/view`
   - העתק את ה-`FILE_ID`
4. ב-Railway, הוסף environment variable:
   - שם: `GOOGLE_DRIVE_MODEL_ID`
   - ערך: `FILE_ID` שלך

## אפשרות 2: קישור ישיר (ZIP)

1. העלה את תיקיית `weights/shmoolik` כ-ZIP לאחסון כלשהו (Dropbox, OneDrive, וכו')
2. קבל קישור ישיר להורדה
3. ב-Railway, הוסף environment variable:
   - שם: `MODEL_DOWNLOAD_URL`
   - ערך: הקישור הישיר ל-ZIP

## מבנה ה-ZIP

ה-ZIP צריך להכיל את המבנה הבא:
```
shmoolik/
├── shmoolik.pth
├── config.json
└── added_IVF894_Flat_nprobe_1_shmoolik_v2.index
```

או:
```
weights/
└── shmoolik/
    ├── shmoolik.pth
    ├── config.json
    └── added_IVF894_Flat_nprobe_1_shmoolik_v2.index
```

## בדיקה

לאחר ה-build, המודל `shmoolik` אמור להיות זמין ב-dropdown של המודלים באפליקציה.
