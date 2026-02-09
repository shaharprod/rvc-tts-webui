#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RVC TTS WebUI - Voice Conversion Text-to-Speech Application
"""
import asyncio
import datetime
import logging
import os
import time
import traceback
import urllib.request
from pathlib import Path
import sys
import json
import base64

import edge_tts
import gradio as gr
import librosa
import torch
from fairseq import checkpoint_utils

# Optional imports for TTS engines
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to set edge-tts user agent to avoid 401 errors
try:
    import edge_tts.constants
    # Update user agent if possible
    if hasattr(edge_tts.constants, 'EDGE_TTS_DOMAIN'):
        pass  # Keep default
except:
    pass

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = False  # Disabled - no limitations on text length or audio duration

# Download required model files if they don't exist
def ensure_model_files():
    try:
        root_dir = Path(__file__).parent.absolute()
    except NameError:
        root_dir = Path.cwd()
    
    hubert_path = root_dir / "hubert_base.pt"
    rmvpe_path = root_dir / "rmvpe.pt"
    
    if not hubert_path.exists():
        print("Downloading hubert_base.pt...")
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
                str(hubert_path)
            )
            print("hubert_base.pt downloaded successfully!")
        except Exception as e:
            print(f"Error downloading hubert_base.pt: {e}")
    else:
        print("hubert_base.pt already exists")
    
    if not rmvpe_path.exists():
        print("Downloading rmvpe.pt...")
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
                str(rmvpe_path)
            )
            print("rmvpe.pt downloaded successfully!")
        except Exception as e:
            print(f"Error downloading rmvpe.pt: {e}")
    else:
        print("rmvpe.pt already exists")

# Ensure we're in the right directory
try:
    ensure_model_files()
except Exception as e:
    print(f"Warning: Could not ensure model files: {e}")

try:
    config = Config()
except Exception as e:
    print(f"Error loading config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

edge_output_filename = "edge_output.mp3"

# Try to load TTS voices, fallback to default if fails
try:
    tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
    tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
    # Ensure Hebrew voices are included even if API works
    hebrew_voices = ["he-IL-AvriNeural-Male", "he-IL-HilaNeural-Female"]
    for voice in hebrew_voices:
        if voice not in tts_voices:
            tts_voices.append(voice)
    print(f"Loaded {len(tts_voices)} TTS voices (including Hebrew)")
except Exception as e:
    print(f"Warning: Could not load TTS voices list: {e}")
    print("Using default voice list...")
    # Default voices as fallback - sorted alphabetically by language
    tts_voices = [
        # Arabic
        "ar-EG-SalmaNeural-Female",
        "ar-EG-ShakirNeural-Male",
        # Chinese
        "zh-CN-XiaoxiaoNeural-Female",
        "zh-CN-YunxiNeural-Male",
        # Danish
        "da-DK-ChristelNeural-Female",
        "da-DK-JeppeNeural-Male",
        # Dutch
        "nl-NL-ColetteNeural-Female",
        "nl-NL-MaartenNeural-Male",
        # English
        "en-US-AriaNeural-Female",
        "en-US-DavisNeural-Male",
        # Finnish
        "fi-FI-NooraNeural-Female",
        # French
        "fr-FR-DeniseNeural-Female",
        "fr-FR-HenriNeural-Male",
        # German
        "de-DE-AmalaNeural-Female",
        "de-DE-ConradNeural-Male",
        # Greek
        "el-GR-AthinaNeural-Female",
        # Hebrew
        "he-IL-AvriNeural-Male",
        "he-IL-HilaNeural-Female",
        # Hindi
        "hi-IN-SwaraNeural-Female",
        "hi-IN-MadhurNeural-Male",
        # Indonesian
        "id-ID-GadisNeural-Female",
        "id-ID-ArdiNeural-Male",
        # Italian
        "it-IT-ElsaNeural-Female",
        "it-IT-DiegoNeural-Male",
        # Japanese
        "ja-JP-NanamiNeural-Female",
        "ja-JP-KeitaNeural-Male",
        # Korean
        "ko-KR-SunHiNeural-Female",
        "ko-KR-InJoonNeural-Male",
        # Malay
        "ms-MY-YasminNeural-Female",
        "ms-MY-OsmanNeural-Male",
        # Norwegian
        "nb-NO-PernilleNeural-Female",
        "nb-NO-FinnNeural-Male",
        # Polish
        "pl-PL-AgnieszkaNeural-Female",
        "pl-PL-MarekNeural-Male",
        # Portuguese
        "pt-PT-RaquelNeural-Female",
        "pt-PT-DuarteNeural-Male",
        # Russian
        "ru-RU-SvetlanaNeural-Female",
        "ru-RU-DmitryNeural-Male",
        # Spanish
        "es-ES-ElviraNeural-Female",
        "es-ES-AlvaroNeural-Male",
        # Swedish
        "sv-SE-SofieNeural-Female",
        "sv-SE-MattiasNeural-Male",
        # Tamil
        "ta-IN-PallaviNeural-Female",
        # Thai
        "th-TH-PremwadeeNeural-Female",
        "th-TH-NiwatNeural-Male",
        # Turkish
        "tr-TR-EmelNeural-Female",
        "tr-TR-AhmetNeural-Male",
        # Ukrainian
        "uk-UA-PolinaNeural-Female",
        # Vietnamese
        "vi-VN-HoaiMyNeural-Female",
        "vi-VN-NamMinhNeural-Male",
    ]

# Country code mappings for display
country_codes = {
    "AD": "ANDORRA",
    "AE": "UNITED ARAB EMIRATES",
    "AF": "AFGHANISTAN",
    "AG": "ANTIGUA AND BARBUDA",
    "AI": "ANGUILLA",
    "AL": "ALBANIA",
    "AM": "ARMENIA",
    "AO": "ANGOLA",
    "AQ": "ANTARCTICA",
    "AR": "ARGENTINA",
    "AS": "AMERICAN SAMOA",
    "AT": "AUSTRIA",
    "AU": "AUSTRALIA",
    "AW": "ARUBA",
    "AX": "ALAND ISLANDS",
    "AZ": "AZERBAIJAN",
    "BA": "BOSNIA AND HERZEGOVINA",
    "BB": "BARBADOS",
    "BD": "BANGLADESH",
    "BE": "BELGIUM",
    "BF": "BURKINA FASO",
    "BG": "BULGARIA",
    "BH": "BAHRAIN",
    "BI": "BURUNDI",
    "BJ": "BENIN",
    "BL": "SAINT BARTHELEMY",
    "BM": "BERMUDA",
    "BN": "BRUNEI",
    "BO": "BOLIVIA",
    "BQ": "BONAIRE",
    "BR": "BRAZIL",
    "BS": "BAHAMAS",
    "BT": "BHUTAN",
    "BV": "BOUVET ISLAND",
    "BW": "BOTSWANA",
    "BY": "BELARUS",
    "BZ": "BELIZE",
    "CA": "CANADA",
    "CC": "COCOS ISLANDS",
    "CD": "DEMOCRATIC REPUBLIC OF THE CONGO",
    "CF": "CENTRAL AFRICAN REPUBLIC",
    "CG": "CONGO",
    "CH": "SWITZERLAND",
    "CI": "IVORY COAST",
    "CK": "COOK ISLANDS",
    "CL": "CHILE",
    "CM": "CAMEROON",
    "CN": "CHINA",
    "CO": "COLOMBIA",
    "CR": "COSTA RICA",
    "CU": "CUBA",
    "CV": "CAPE VERDE",
    "CW": "CURACAO",
    "CX": "CHRISTMAS ISLAND",
    "CY": "CYPRUS",
    "CZ": "CZECH REPUBLIC",
    "DE": "GERMANY",
    "DJ": "DJIBOUTI",
    "DK": "DENMARK",
    "DM": "DOMINICA",
    "DO": "DOMINICAN REPUBLIC",
    "DZ": "ALGERIA",
    "EC": "ECUADOR",
    "EE": "ESTONIA",
    "EG": "EGYPT",
    "EH": "WESTERN SAHARA",
    "ER": "ERITREA",
    "ES": "SPAIN",
    "ET": "ETHIOPIA",
    "FI": "FINLAND",
    "FJ": "FIJI",
    "FK": "FALKLAND ISLANDS",
    "FM": "MICRONESIA",
    "FO": "FAROE ISLANDS",
    "FR": "FRANCE",
    "GA": "GABON",
    "GB": "UNITED KINGDOM",
    "GD": "GRENADA",
    "GE": "GEORGIA",
    "GF": "FRENCH GUIANA",
    "GG": "GUERNSEY",
    "GH": "GHANA",
    "GI": "GIBRALTAR",
    "GL": "GREENLAND",
    "GM": "GAMBIA",
    "GN": "GUINEA",
    "GP": "GUADELOUPE",
    "GQ": "EQUATORIAL GUINEA",
    "GR": "GREECE",
    "GS": "SOUTH GEORGIA",
    "GT": "GUATEMALA",
    "GU": "GUAM",
    "GW": "GUINEA-BISSAU",
    "GY": "GUYANA",
    "HK": "HONG KONG",
    "HM": "HEARD ISLAND",
    "HN": "HONDURAS",
    "HR": "CROATIA",
    "HT": "HAITI",
    "HU": "HUNGARY",
    "ID": "INDONESIA",
    "IE": "IRELAND",
    "IL": "ISRAEL",
    "IM": "ISLE OF MAN",
    "IN": "INDIA",
    "IO": "BRITISH INDIAN OCEAN TERRITORY",
    "IQ": "IRAQ",
    "IR": "IRAN",
    "IS": "ICELAND",
    "IT": "ITALY",
    "JE": "JERSEY",
    "JM": "JAMAICA",
    "JO": "JORDAN",
    "JP": "JAPAN",
    "KE": "KENYA",
    "KG": "KYRGYZSTAN",
    "KH": "CAMBODIA",
    "KI": "KIRIBATI",
    "KM": "COMOROS",
    "KN": "SAINT KITTS AND NEVIS",
    "KP": "NORTH KOREA",
    "KR": "SOUTH KOREA",
    "KW": "KUWAIT",
    "KY": "CAYMAN ISLANDS",
    "KZ": "KAZAKHSTAN",
    "LA": "LAOS",
    "LB": "LEBANON",
    "LC": "SAINT LUCIA",
    "LI": "LIECHTENSTEIN",
    "LK": "SRI LANKA",
    "LR": "LIBERIA",
    "LS": "LESOTHO",
    "LT": "LITHUANIA",
    "LU": "LUXEMBOURG",
    "LV": "LATVIA",
    "LY": "LIBYA",
    "MA": "MOROCCO",
    "MC": "MONACO",
    "MD": "MOLDOVA",
    "ME": "MONTENEGRO",
    "MF": "SAINT MARTIN",
    "MG": "MADAGASCAR",
    "MH": "MARSHALL ISLANDS",
    "MK": "NORTH MACEDONIA",
    "ML": "MALI",
    "MM": "MYANMAR",
    "MN": "MONGOLIA",
    "MO": "MACAO",
    "MP": "NORTHERN MARIANA ISLANDS",
    "MQ": "MARTINIQUE",
    "MR": "MAURITANIA",
    "MS": "MONTSERRAT",
    "MT": "MALTA",
    "MU": "MAURITIUS",
    "MV": "MALDIVES",
    "MW": "MALAWI",
    "MX": "MEXICO",
    "MY": "MALAYSIA",
    "MZ": "MOZAMBIQUE",
    "NA": "NAMIBIA",
    "NC": "NEW CALEDONIA",
    "NE": "NIGER",
    "NF": "NORFOLK ISLAND",
    "NG": "NIGERIA",
    "NI": "NICARAGUA",
    "NL": "NETHERLANDS",
    "NO": "NORWAY",
    "NP": "NEPAL",
    "NR": "NAURU",
    "NU": "NIUE",
    "NZ": "NEW ZEALAND",
    "OM": "OMAN",
    "PA": "PANAMA",
    "PE": "PERU",
    "PF": "FRENCH POLYNESIA",
    "PG": "PAPUA NEW GUINEA",
    "PH": "PHILIPPINES",
    "PK": "PAKISTAN",
    "PL": "POLAND",
    "PM": "SAINT PIERRE AND MIQUELON",
    "PN": "PITCAIRN",
    "PR": "PUERTO RICO",
    "PS": "PALESTINE",
    "PT": "PORTUGAL",
    "PW": "PALAU",
    "PY": "PARAGUAY",
    "QA": "QATAR",
    "RE": "REUNION",
    "RO": "ROMANIA",
    "RS": "SERBIA",
    "RU": "RUSSIA",
    "RW": "RWANDA",
    "SA": "SAUDI ARABIA",
    "SB": "SOLOMON ISLANDS",
    "SC": "SEYCHELLES",
    "SD": "SUDAN",
    "SE": "SWEDEN",
    "SG": "SINGAPORE",
    "SH": "SAINT HELENA",
    "SI": "SLOVENIA",
    "SJ": "SVALBARD AND JAN MAYEN",
    "SK": "SLOVAKIA",
    "SL": "SIERRA LEONE",
    "SM": "SAN MARINO",
    "SN": "SENEGAL",
    "SO": "SOMALIA",
    "SR": "SURINAME",
    "SS": "SOUTH SUDAN",
    "ST": "SAO TOME AND PRINCIPE",
    "SV": "EL SALVADOR",
    "SX": "SINT MAARTEN",
    "SY": "SYRIA",
    "SZ": "SWAZILAND",
    "TC": "TURKS AND CAICOS ISLANDS",
    "TD": "CHAD",
    "TF": "FRENCH SOUTHERN TERRITORIES",
    "TG": "TOGO",
    "TH": "THAILAND",
    "TJ": "TAJIKISTAN",
    "TK": "TOKELAU",
    "TL": "TIMOR-LESTE",
    "TM": "TURKMENISTAN",
    "TN": "TUNISIA",
    "TO": "TONGA",
    "TR": "TURKEY",
    "TT": "TRINIDAD AND TOBAGO",
    "TV": "TUVALU",
    "TW": "TAIWAN",
    "TZ": "TANZANIA",
    "UA": "UKRAINE",
    "UG": "UGANDA",
    "UM": "UNITED STATES MINOR OUTLYING ISLANDS",
    "US": "UNITED STATES",
    "UY": "URUGUAY",
    "UZ": "UZBEKISTAN",
    "VA": "VATICAN CITY",
    "VC": "SAINT VINCENT AND THE GRENADINES",
    "VE": "VENEZUELA",
    "VG": "BRITISH VIRGIN ISLANDS",
    "VI": "U.S. VIRGIN ISLANDS",
    "VN": "VIETNAM",
    "VU": "VANUATU",
    "WF": "WALLIS AND FUTUNA",
    "WS": "SAMOA",
    "YE": "YEMEN",
    "YT": "MAYOTTE",
    "ZA": "SOUTH AFRICA",
    "ZM": "ZAMBIA",
    "ZW": "ZIMBABWE",
}

country_full = {v: k for k, v in country_codes.items()}

def expand_country_codes(voice):
    parts = voice.split('-')
    if len(parts) >= 2:
        lang, code = parts[0], parts[1]
        full_code = country_codes.get(code.upper(), code)
        parts[1] = full_code
        return '-'.join(parts)
    return voice

# Build display→original mapping and expand for display
# This is the ONLY reliable way to convert back - no string parsing needed
voice_display_to_original = {}
for v in tts_voices:
    display = expand_country_codes(v)
    voice_display_to_original[display] = v

# Sort expanded names alphabetically for the dropdown
tts_voices = sorted(voice_display_to_original.keys())

def get_original_voice(display_voice):
    """Reliably convert expanded display name back to original Edge TTS voice name."""
    return voice_display_to_original.get(display_voice, display_voice)

# Language name to preferred country prefix (expanded) for auto-selecting speaker
# Uses the "standard" country for each language to avoid picking regional variants
language_to_preferred = {
    "Arabic": "ar-EGYPT-",
    "Chinese": "zh-CHINA-",
    "Danish": "da-DENMARK-",
    "Dutch": "nl-NETHERLANDS-",
    "English": "en-UNITED STATES-",
    "Finnish": "fi-FINLAND-",
    "French": "fr-FRANCE-",
    "German": "de-GERMANY-",
    "Greek": "el-GREECE-",
    "Hebrew": "he-ISRAEL-",
    "Hindi": "hi-INDIA-",
    "Indonesian": "id-INDONESIA-",
    "Italian": "it-ITALY-",
    "Japanese": "ja-JAPAN-",
    "Korean": "ko-SOUTH KOREA-",
    "Malay": "ms-MALAYSIA-",
    "Norwegian": "nb-NORWAY-",
    "Polish": "pl-POLAND-",
    "Portuguese": "pt-PORTUGAL-",
    "Russian": "ru-RUSSIA-",
    "Spanish": "es-SPAIN-",
    "Swedish": "sv-SWEDEN-",
    "Tamil": "ta-INDIA-",
    "Thai": "th-THAILAND-",
    "Turkish": "tr-TURKEY-",
    "Ukrainian": "uk-UKRAINE-",
    "Vietnamese": "vi-VIETNAM-",
}

# Sample sentences for each language
sample_sentences = {
    "Arabic": "مرحباً، هذه جملة مثال باللغة العربية.",
    "Chinese": "你好，这是中文的示例句子。",
    "Danish": "Hej, dette er en eksempelsætning på dansk.",
    "Dutch": "Hallo, dit is een voorbeeldzin in het Nederlands.",
    "English": "Hello, this is a sample sentence in English.",
    "Finnish": "Hei, tämä on esimerkkilause suomeksi.",
    "French": "Bonjour, ceci est une phrase d'exemple en français.",
    "German": "Hallo, dies ist ein Beispielsatz auf Deutsch.",
    "Greek": "Γεια σας, αυτό είναι ένα δείγμα πρότασης στα ελληνικά.",
    "Hebrew": "שלום, זהו משפט דוגמה בעברית.",
    "Hindi": "नमस्ते, यह हिंदी में एक नमूना वाक्य है।",
    "Indonesian": "Halo, ini adalah kalimat contoh dalam bahasa Indonesia.",
    "Italian": "Ciao, questa è una frase di esempio in italiano.",
    "Japanese": "こんにちは、これは日本語のサンプル文です。",
    "Korean": "안녕하세요, 이것은 한국어 샘플 문장입니다.",
    "Malay": "Hai, ini adalah ayat contoh dalam bahasa Melayu.",
    "Norwegian": "Hei, dette er en eksempelsetning på norsk.",
    "Polish": "Cześć, to jest przykładowe zdanie po polsku.",
    "Portuguese": "Olá, esta é uma frase de exemplo em português.",
    "Russian": "Здравствуйте, это пример предложения на русском.",
    "Spanish": "Hola, esta es una oración de ejemplo en español.",
    "Swedish": "Hej, detta är en exempelmening på svenska.",
    "Tamil": "வணக்கம், இது தமிழில் ஒரு மாதிரி வாக்கியம்.",
    "Thai": "สวัสดี, นี่คือประโยคตัวอย่างในภาษาไทย.",
    "Turkish": "Merhaba, bu Türkçe bir örnek cümledir.",
    "Ukrainian": "Привіт, це зразок речення українською.",
    "Vietnamese": "Xin chào, đây là một câu mẫu bằng tiếng Việt.",
}

model_root = "weights"
try:
    models = [
        d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
    ]
    if len(models) == 0:
        print("Warning: No model found in `weights` folder")
        print("Please add RVC models to the weights/ directory")
        models = []  # Allow app to start even without models
    else:
        models.sort()
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
    models = []


def model_data(model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")
    pth_path = pth_files[0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0


def load_hubert():
    global hubert_model
    # Fix for PyTorch 2.6+ weights_only=True default
    # Patch fairseq's checkpoint_utils to use weights_only=False
    import fairseq.checkpoint_utils
    original_load_checkpoint = fairseq.checkpoint_utils.load_checkpoint_to_cpu
    
    def patched_load_checkpoint(filename, arg_overrides=None):
        # Temporarily patch torch.load to use weights_only=False
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load
        try:
            result = original_load_checkpoint(filename, arg_overrides)
        finally:
            torch.load = original_torch_load
        return result
    
    fairseq.checkpoint_utils.load_checkpoint_to_cpu = patched_load_checkpoint
    
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["hubert_base.pt"],
            suffix="",
        )
    finally:
        # Restore original function
        fairseq.checkpoint_utils.load_checkpoint_to_cpu = original_load_checkpoint
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


print("Loading hubert model...")
try:
    hubert_model = load_hubert()
    print("Hubert model loaded.")
except Exception as e:
    print(f"Error loading hubert model: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing anyway...")
    hubert_model = None

print("Loading rmvpe model...")
try:
    import numpy as np
    # Test numpy availability
    test_array = np.array([1, 2, 3])
    rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
    print("rmvpe model loaded.")
except RuntimeError as e:
    if "Numpy" in str(e) or "numpy" in str(e).lower():
        print(f"Error: NumPy compatibility issue: {e}")
        print("Trying to reinitialize NumPy...")
        import numpy as np
        import importlib
        importlib.reload(np)
        rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
        print("rmvpe model loaded after NumPy reinit.")
    else:
        raise
except Exception as e:
    print(f"Error loading rmvpe model: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing anyway...")
    rmvpe_model = None


# TTS Engine Functions
def get_edge_voice_name(voice):
    """Strip the -Male/-Female gender suffix to get the Edge TTS ShortName.
    Edge TTS expects ShortName only (e.g., 'he-IL-AvriNeural'), not 'he-IL-AvriNeural-Male'."""
    if voice.endswith("-Male"):
        return voice[:-5]
    elif voice.endswith("-Female"):
        return voice[:-7]
    return voice


def _run_edge_tts(communicate, output_filename):
    """Run edge-tts communicate.save() in a clean event loop.
    Handles Windows ProactorEventLoop cleanup issues gracefully."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(communicate.save(output_filename))
    finally:
        # Gracefully shutdown: allow pending callbacks to complete
        try:
            # Cancel remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def generate_tts_edge(tts_text, tts_voice, speed_str, output_filename):
    """Generate TTS using Edge TTS (free, no API key needed)"""
    # Strip gender suffix - Edge TTS expects ShortName only
    voice_name = get_edge_voice_name(tts_voice)
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(tts_text, voice_name, rate=speed_str)
            _run_edge_tts(communicate, output_filename)

            if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                return True, None
            else:
                last_error = f"Edge TTS produced empty output for voice '{voice_name}'"
                print(f"  Attempt {attempt+1}: {last_error}")
        except Exception as e:
            last_error = f"Edge TTS error (voice='{voice_name}'): {e}"
            print(f"  Attempt {attempt+1}: {last_error}")

        if attempt < max_retries - 1:
            time.sleep(1)

    return False, last_error or "Failed after all retries"


def generate_tts_openai(tts_text, tts_voice, speed, api_key, output_filename):
    """Generate TTS using OpenAI TTS API"""
    if not OPENAI_AVAILABLE:
        return False, "OpenAI library not installed. Install with: pip install openai"
    
    if not api_key:
        return False, "OpenAI API key is required"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Map voice format to OpenAI voice names
        voice_map = {
            "male": "onyx",
            "female": "nova",
            "alloy": "alloy",
            "echo": "echo",
            "fable": "fable",
            "onyx": "onyx",
            "nova": "nova",
            "shimmer": "shimmer"
        }
        
        voice_name = "nova"  # default
        if "-" in tts_voice:
            parts = tts_voice.split("-")
            if len(parts) >= 4:
                gender = parts[-1].lower()
                if gender in voice_map:
                    voice_name = voice_map[gender]
            for part in parts:
                if part.lower() in voice_map:
                    voice_name = voice_map[part.lower()]
        
        speed_value = 1.0 + (speed / 100.0)
        speed_value = max(0.25, min(4.0, speed_value))
        
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice_name,
            input=tts_text,
            speed=speed_value
        )
        
        with open(output_filename, "wb") as f:
            f.write(response.content)
        
        return True, None
    except Exception as e:
        return False, str(e)


def generate_tts_google(tts_text, tts_voice, speed, api_key_json, output_filename):
    """Generate TTS using Google Cloud TTS"""
    if not GOOGLE_TTS_AVAILABLE:
        return False, "Google Cloud TTS library not installed. Install with: pip install google-cloud-texttospeech"
    
    if not api_key_json:
        return False, "Google Cloud API key (JSON) is required"
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(api_key_json)
            temp_key_file = f.name
        
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_key_file
            
            client = texttospeech.TextToSpeechClient()
            
            voice_parts = tts_voice.split("-")
            if len(voice_parts) >= 3:
                language_code = f"{voice_parts[0]}-{voice_parts[1]}"
                voice_name = tts_voice.replace(f"{voice_parts[0]}-{voice_parts[1]}-", "")
            else:
                language_code = "he-IL"
                voice_name = tts_voice
            
            ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL
            if len(voice_parts) >= 4:
                gender = voice_parts[-1].lower()
                if "male" in gender or "male" in voice_name.lower():
                    ssml_gender = texttospeech.SsmlVoiceGender.MALE
                elif "female" in gender or "female" in voice_name.lower():
                    ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
            
            speed_value = 1.0 + (speed / 100.0)
            speed_value = max(0.25, min(4.0, speed_value))
            
            synthesis_input = texttospeech.SynthesisInput(text=tts_text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name if voice_name else None,
                ssml_gender=ssml_gender
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speed_value
            )
            
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            with open(output_filename, "wb") as f:
                f.write(response.audio_content)
            
            return True, None
        finally:
            try:
                os.unlink(temp_key_file)
                if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                    del os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            except:
                pass
    except Exception as e:
        return False, str(e)


def generate_tts_elevenlabs(tts_text, tts_voice, speed, api_key, voice_id, output_filename):
    """Generate TTS using ElevenLabs API"""
    if not REQUESTS_AVAILABLE:
        return False, "requests library not installed. Install with: pip install requests"
    
    if not api_key:
        return False, "ElevenLabs API key is required"
    
    if not voice_id:
        return False, "ElevenLabs Voice ID is required"
    
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        speed_value = 1.0 + (speed / 100.0)
        speed_value = max(0.25, min(4.0, speed_value))
        
        data = {
            "text": tts_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "speed": speed_value
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        with open(output_filename, "wb") as f:
            f.write(response.content)
        
        return True, None
    except Exception as e:
        return False, str(e)


def tts(
    model_name,
    speed,
    pitch,
    tts_text,
    tts_voice,
    tts_engine,
    openai_api_key,
    google_api_key_json,
    elevenlabs_api_key,
    elevenlabs_voice_id,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
):
    print("------------------")
    print(datetime.datetime.now())
    # Convert expanded display name back to original Edge TTS voice name
    tts_voice = get_original_voice(tts_voice)
    print("tts_text:")
    print(tts_text)
    print(f"tts_voice: {tts_voice}")
    print(f"Model name: {model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:
        # Validate text input
        if not tts_text or not tts_text.strip():
            return ("שגיאה: הטקסט ריק. אנא הכנס טקסט.", None, None)

        # Check if model is selected (before loading model)
        if not model_name or (len(models) > 0 and model_name not in models):
            return (
                "שגיאה: לא נבחר מודל או שהמודל לא קיים.\n\n"
                "אנא בחר מודל מהרשימה או הוסף מודל RVC לתיקיית weights/",
                None,
                None,
            )

        # Validate voice selection
        if not tts_voice or tts_voice.strip() == "":
            return ("שגיאה: לא נבחר קול/דובר. אנא בחר קול מהרשימה.", None, None)
        
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        t0 = time.time()

        # Use unique temp file per request to avoid conflicts
        import tempfile
        temp_output = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name

        # Format speed string for edge-tts
        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"

        print(f"TTS Engine: {tts_engine}")
        print(f"Text: {tts_text[:50]}...")
        print(f"Voice: {tts_voice}")
        print(f"Speed: {speed}%")
        print(f"Pitch adjustment: {pitch} semitones")

        # Generate TTS based on selected engine
        tts_success = False
        tts_error = None

        if tts_engine == "Edge TTS (Free)":
            tts_success, tts_error = generate_tts_edge(tts_text, tts_voice, speed_str, temp_output)
        elif tts_engine == "OpenAI TTS":
            tts_success, tts_error = generate_tts_openai(tts_text, tts_voice, speed, openai_api_key, temp_output)
        elif tts_engine == "Google Cloud TTS":
            tts_success, tts_error = generate_tts_google(tts_text, tts_voice, speed, google_api_key_json, temp_output)
        elif tts_engine == "ElevenLabs":
            tts_success, tts_error = generate_tts_elevenlabs(tts_text, tts_voice, speed, elevenlabs_api_key, elevenlabs_voice_id, temp_output)
        else:
            return (
                f"שגיאה: מנוע TTS לא נתמך: {tts_engine}",
                None,
                None,
            )
        
        if not tts_success:
            error_msg = tts_error or "Unknown error"
            print(f"TTS generation failed: {error_msg}")
            
            # Return error message based on engine
            engine_name = tts_engine or "TTS"
            return (
                f"שגיאה ביצירת אודיו {engine_name}:\n\n"
                f"{error_msg}\n\n"
                "פתרונות אפשריים:\n"
                "1. בדוק את חיבור האינטרנט\n"
                "2. ודא שה-API key תקין (אם נדרש)\n"
                "3. נסה קול אחר\n"
                "4. בדוק שהטקסט תקין",
                None,
                None,
            )
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(temp_output, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")

        # Combine user's f0_up_key with pitch slider value
        user_f0_up_key = int(f0_up_key)
        final_f0_up_key = user_f0_up_key + int(pitch)
        print(f"Final f0_up_key: {final_f0_up_key} (user: {user_f0_up_key}, pitch slider: {pitch})")

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            temp_output,
            times,
            final_f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        return (
            info,
            temp_output,
            (tgt_sr, audio_opt),
        )
    except EOFError:
        info = (
            "שגיאה: הפלט מ-Edge TTS לא תקין.\n"
            "ייתכן שהטקסט והדובר לא תואמים.\n"
            "נסה דובר אחר או שנה את הטקסט."
        )
        print(info)
        return info, None, None
    except Exception as e:
        info = f"שגיאה: {str(e)}"
        print(f"Exception in tts(): {traceback.format_exc()}")
        return info, None, None


initial_md = """
# RVC text-to-speech webui

This is a text-to-speech webui of RVC models.

Input text ➡[(edge-tts)](https://github.com/rany2/edge-tts)➡ Speech mp3 file ➡[(RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)➡ Final output
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink")) as app:
    gr.Markdown(initial_md)
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(
                label="Model", 
                choices=models, 
                value=models[0] if len(models) > 0 else None,
                interactive=len(models) > 0
            )
            f0_key_up = gr.Number(
                label="Transpose (the best value depends on the models and speakers)",
                value=0,
            )
        with gr.Column():
            f0_method = gr.Radio(
                label="Pitch extraction method (pm: very fast, low quality, rmvpe: a little slow, high quality)",
                choices=["pm", "rmvpe"],  # harvest and crepe is too slow
                value="rmvpe",
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label="Index rate",
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label="Protect",
                value=0.33,
                step=0.01,
                interactive=True,
            )
    with gr.Row():
        with gr.Column():
            tts_engine = gr.Dropdown(
                label="TTS Engine",
                choices=["Edge TTS (Free)", "OpenAI TTS", "Google Cloud TTS", "ElevenLabs"],
                value="Edge TTS (Free)",
                interactive=True,
            )
            tts_voice = gr.Dropdown(
                label="Speaker/Voice",
                choices=tts_voices,
                allow_custom_value=False,
                value=expand_country_codes("he-IL-AvriNeural-Male") if expand_country_codes("he-IL-AvriNeural-Male") in tts_voices else (expand_country_codes("he-IL-HilaNeural-Female") if expand_country_codes("he-IL-HilaNeural-Female") in tts_voices else tts_voices[0] if tts_voices else "ja-JP-NanamiNeural-Female"),
            )
            openai_api_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                visible=False,
            )
            google_api_key_json = gr.Textbox(
                label="Google Cloud API Key (JSON)",
                type="password",
                placeholder='{"type": "service_account", ...}',
                visible=False,
                lines=5,
            )
            elevenlabs_api_key = gr.Textbox(
                label="ElevenLabs API Key",
                type="password",
                placeholder="...",
                visible=False,
            )
            elevenlabs_voice_id = gr.Textbox(
                label="ElevenLabs Voice ID",
                placeholder="21m00Tcm4TlvDq8ikWAM",
                visible=False,
            )
            speed = gr.Slider(
                minimum=-100,
                maximum=100,
                label="Speech speed (%)",
                value=0,
                step=10,
                interactive=True,
            )
            pitch = gr.Slider(
                minimum=-12,
                maximum=12,
                label="Pitch (semitones)",
                value=0,
                step=1,
                interactive=True,
            )
            sample_radio = gr.Radio(
                label="Sample Sentences",
                choices=list(sample_sentences.keys()),
                value=None,
                interactive=True,
            )
            tts_text = gr.Textbox(label="Input Text", value="זה משפט ברירת המחדל.")
        with gr.Column():
            but0 = gr.Button("Convert", variant="primary")
            info_text = gr.Textbox(label="Output info")
        with gr.Column():
            edge_tts_output = gr.Audio(label="Edge Voice", type="filepath")
            tts_output = gr.Audio(label="Result")
        
        # Function to show/hide API key fields based on selected engine
        def update_api_fields(engine):
            return [
                gr.update(visible=(engine == "OpenAI TTS")),
                gr.update(visible=(engine == "Google Cloud TTS")),
                gr.update(visible=(engine == "ElevenLabs")),
                gr.update(visible=(engine == "ElevenLabs")),
            ]
        
        tts_engine.change(
            update_api_fields,
            inputs=[tts_engine],
            outputs=[openai_api_key, google_api_key_json, elevenlabs_api_key, elevenlabs_voice_id],
        )
        
        but0.click(
            tts,
            [
                model_name,
                speed,
                pitch,
                tts_text,
                tts_voice,
                tts_engine,
                openai_api_key,
                google_api_key_json,
                elevenlabs_api_key,
                elevenlabs_voice_id,
                f0_key_up,
                f0_method,
                index_rate,
                protect0,
            ],
            [info_text, edge_tts_output, tts_output],
        )
        # Update text box and auto-select matching speaker when sample sentence is selected
        def on_sample_selected(language, current_voice):
            text = sample_sentences.get(language, "")
            # Try preferred country first (e.g., "de-GERMANY-" instead of "de-AUSTRIA-")
            preferred = language_to_preferred.get(language, "")
            if preferred:
                matching = [v for v in tts_voices if v.startswith(preferred)]
                if matching:
                    return text, matching[0]
                # Fallback: any voice with same language code (e.g., "de-")
                short_prefix = preferred.split("-")[0] + "-"
                matching = [v for v in tts_voices if v.startswith(short_prefix)]
                if matching:
                    return text, matching[0]
            return text, current_voice

        sample_radio.change(
            fn=on_sample_selected,
            inputs=[sample_radio, tts_voice],
            outputs=[tts_text, tts_voice],
        )
    
    # Examples component removed due to Gradio 3.34.0 compatibility issue
    # (causes KeyError: 'dataset' in API info endpoint)
    # with gr.Row():
    #     examples = gr.Examples(
    #         examples_per_page=100,
    #         examples=[
    #             ["זה משפט ברירת המחדל.", "he-IL-AvriNeural-Male"],
    #             ["זהו דמו של המרת טקסט עברי לדיבור. קול נשי.", "he-IL-HilaNeural-Female"],
    #             # ... more examples
    #         ],
    #         inputs=[tts_text, tts_voice],
    #     )


if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("Starting RVC TTS WebUI...")
        print("="*50)
        print(f"Working directory: {os.getcwd()}")
        print(f"Python version: {sys.version}")
        print("="*50)

        # Get port from config or use default
        port = config.listen_port if hasattr(config, 'listen_port') else 7865

        # Allow access from network (0.0.0.0 means accessible from any IP)
        # For localhost only, use "127.0.0.1"
        server_name = "0.0.0.0"  # Accessible from network

        # Option to create public share link (works like GitHub Pages but via Gradio)
        # Set to True to get a public URL like: https://xxxxx.gradio.live
        enable_share = os.getenv("GRADIO_SHARE", "True").lower() == "true"

        print(f"Server will be accessible at:")
        print(f"  - Local: http://localhost:{port}")
        print(f"  - Network: http://<your-ip>:{port}")
        if enable_share:
            print(f"  - Public share link will be generated (like GitHub Pages)")
        print(f"  - To find your IP, run: ipconfig (Windows) or ifconfig (Linux/Mac)")
        print("="*50)

        try:
            app.queue()
            app.launch(
                inbrowser=True,
                server_name=server_name,
                server_port=port,
                share=enable_share,  # Set to True for public Gradio share link (like GitHub Pages)
                debug=True  # Enable debug mode to see server-side errors
            )
        except Exception as e:
            print(f"\nError starting server: {e}")
            import traceback
            traceback.print_exc()
            print("\nTrying to launch without opening browser...")
            try:
                app.launch(
                    inbrowser=False,
                    server_name=server_name,
                    server_port=port,
                    share=enable_share
                )
            except Exception as e2:
                print(f"\nError: {e2}")
                import traceback
                traceback.print_exc()
                print("\n" + "="*50)
                print("Failed to start server. Please check the errors above.")
                print("Common issues:")
                print("1. Missing dependencies - run: pip install -r requirements.txt")
                print("2. Missing model files - check if hubert_base.pt and rmvpe.pt exist")
                print("3. Port already in use - try: python app.py --port 7866")
                print("="*50)
                # Don't use input() when running from IDE - it blocks execution
                if sys.stdin.isatty():
                    input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        # Don't use input() when running from IDE - it blocks execution
        if sys.stdin.isatty():
            input("Press Enter to exit...")
