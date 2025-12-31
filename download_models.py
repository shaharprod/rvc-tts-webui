#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download RVC models from Google Drive for Railway deployment
"""
import os
import urllib.request
from pathlib import Path
import zipfile
import shutil

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using file ID"""
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"Downloading {destination} from Google Drive...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Successfully downloaded {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {destination}: {e}")
        return False

def download_from_url(url, destination):
    """Download a file from a direct URL"""
    print(f"Downloading {destination} from URL...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Successfully downloaded {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {destination}: {e}")
        return False

def download_models():
    """Download models from Google Drive or direct URL"""
    model_id = os.environ.get("GOOGLE_DRIVE_MODEL_ID")
    model_url = os.environ.get("MODEL_DOWNLOAD_URL")
    
    if not model_id and not model_url:
        print("GOOGLE_DRIVE_MODEL_ID or MODEL_DOWNLOAD_URL not set - skipping model download")
        print("To download models, set one of these environment variables in Railway:")
        print("  - GOOGLE_DRIVE_MODEL_ID: Google Drive file ID")
        print("  - MODEL_DOWNLOAD_URL: Direct download URL (ZIP file)")
        return
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Download as ZIP and extract
    zip_path = "models.zip"
    success = False
    
    if model_url:
        success = download_from_url(model_url, zip_path)
    elif model_id:
        success = download_file_from_google_drive(model_id, zip_path)
    
    if success:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(weights_dir)
            print("Models extracted successfully!")
            os.remove(zip_path)
        except Exception as e:
            print(f"Error extracting models: {e}")

if __name__ == "__main__":
    download_models()
