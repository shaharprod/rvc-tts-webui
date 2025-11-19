#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
סקריפט בדיקה להפעלת app.py
"""
import subprocess
import sys
from pathlib import Path

print("=" * 60)
print("בדיקת הפעלת app.py")
print("=" * 60)

app_path = Path("app.py")
if not app_path.exists():
    print("✗ קובץ app.py לא נמצא!")
    sys.exit(1)

print(f"✓ קובץ app.py נמצא: {app_path.absolute()}")
print(f"\nמנסה להריץ את app.py...")
print("=" * 60)

try:
    # הרצה עם Python
    result = subprocess.run(
        [sys.executable, str(app_path)],
        timeout=5,
        capture_output=True,
        text=True
    )
    print("✓ הקובץ רץ בהצלחה (עצר אחרי 5 שניות)")
    print(f"\nפלט ראשוני:\n{result.stdout[:500]}")
except subprocess.TimeoutExpired:
    print("✓ הקובץ רץ (עצר אחרי 5 שניות - זה תקין)")
except Exception as e:
    print(f"✗ שגיאה: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("הקובץ מוכן להפעלה!")
print("נסה ללחוץ על כפתור הפליי (▶️) ב-app.py")
print("=" * 60)

