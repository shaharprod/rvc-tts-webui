#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
סקריפט לבדיקת הגדרות Python ב-Cursor/VS Code
"""
import sys
import os
from pathlib import Path

print("=" * 60)
print("בדיקת הגדרות Python")
print("=" * 60)

# בדיקת גרסת Python
print(f"\n✓ גרסת Python: {sys.version}")
print(f"✓ נתיב Python: {sys.executable}")
print(f"✓ תיקיית עבודה: {os.getcwd()}")

# בדיקת קובץ app.py
app_path = Path("app.py")
if app_path.exists():
    print(f"\n✓ קובץ app.py נמצא: {app_path.absolute()}")
else:
    print(f"\n✗ קובץ app.py לא נמצא!")

# בדיקת קבצי הגדרות
vscode_dir = Path(".vscode")
if vscode_dir.exists():
    print(f"\n✓ תיקיית .vscode קיימת")
    
    launch_json = vscode_dir / "launch.json"
    if launch_json.exists():
        print(f"  ✓ launch.json קיים")
    else:
        print(f"  ✗ launch.json חסר!")
    
    settings_json = vscode_dir / "settings.json"
    if settings_json.exists():
        print(f"  ✓ settings.json קיים")
    else:
        print(f"  ✗ settings.json חסר!")
else:
    print(f"\n✗ תיקיית .vscode לא קיימת!")

# בדיקת חבילות נדרשות
print(f"\nבדיקת חבילות נדרשות:")
required_packages = [
    "gradio",
    "edge_tts",
    "torch",
    "librosa",
    "fairseq",
    "numpy"
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        print(f"  ✓ {package} מותקן")
    except ImportError:
        print(f"  ✗ {package} חסר!")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠ חבילות חסרות: {', '.join(missing_packages)}")
    print(f"הרץ: pip install {' '.join(missing_packages)}")
else:
    print(f"\n✓ כל החבילות הנדרשות מותקנות")

# בדיקת קבצי מודלים
print(f"\nבדיקת קבצי מודלים:")
model_files = ["hubert_base.pt", "rmvpe.pt"]
for model_file in model_files:
    model_path = Path(model_file)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {model_file} קיים ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {model_file} חסר!")

print("\n" + "=" * 60)
print("המלצות:")
print("=" * 60)
print("1. ב-Cursor/VS Code: לחץ Ctrl+Shift+P")
print("2. חפש: 'Python: Select Interpreter'")
print("3. בחר את המתורגמן: " + sys.executable)
print("4. ודא שהרחבת Python מותקנת")
print("5. נסה להריץ את app.py עם F5 או כפתור הפליי")
print("=" * 60)

