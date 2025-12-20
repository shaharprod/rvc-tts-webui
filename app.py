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
    # Default voices as fallback (including Hebrew)
    tts_voices = [
        # Hebrew voices
        "he-IL-AvriNeural-Male",
        "he-IL-HilaNeural-Female",
        # Japanese
        "ja-JP-NanamiNeural-Female",
        "ja-JP-KeitaNeural-Male",
        # English
        "en-US-AriaNeural-Female",
        "en-US-DavisNeural-Male",
        # Chinese
        "zh-CN-XiaoxiaoNeural-Female",
        "zh-CN-YunxiNeural-Male",
        # Korean
        "ko-KR-SunHiNeural-Female",
        "ko-KR-InJoonNeural-Male",
        # French
        "fr-FR-DeniseNeural-Female",
        "fr-FR-HenriNeural-Male",
        # German
        "de-DE-AmalaNeural-Female",
        "de-DE-ConradNeural-Male",
        # Spanish
        "es-ES-ElviraNeural-Female",
        "es-ES-AlvaroNeural-Male",
        # Italian
        "it-IT-ElsaNeural-Female",
        "it-IT-DiegoNeural-Male",
        # Portuguese
        "pt-PT-RaquelNeural-Female",
        "pt-PT-DuarteNeural-Male",
        # Russian
        "ru-RU-SvetlanaNeural-Female",
        "ru-RU-DmitryNeural-Male",
        # Arabic
        "ar-EG-SalmaNeural-Female",
        "ar-EG-ShakirNeural-Male",
        # More languages
        "fi-FI-NooraNeural-Female",
        "uk-UA-PolinaNeural-Female",
        "el-GR-AthinaNeural-Female",
        "ta-IN-PallaviNeural-Female",
    ]

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
def generate_tts_edge(tts_text, tts_voice, speed_str, output_filename):
    """Generate TTS using Edge TTS (free, no API key needed)"""
    voice_parts = tts_voice.split("-")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                if len(voice_parts) >= 4:
                    voice_name = "-".join(voice_parts[:-1])
                else:
                    voice_name = tts_voice
            elif attempt == 1:
                voice_name = tts_voice
            else:
                if len(voice_parts) >= 3:
                    voice_name = "-".join(voice_parts[:3])
                else:
                    voice_name = tts_voice
            
            communicate = edge_tts.Communicate(tts_text, voice_name, rate=speed_str)
            asyncio.run(communicate.save(output_filename))
            
            if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                return True, None
        except Exception as e:
            if attempt == max_retries - 1:
                return False, str(e)
            time.sleep(2)
    
    return False, "Failed after all retries"


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
    print("tts_text:")
    print(tts_text)
    print(f"tts_voice: {tts_voice}")
    print(f"Model name: {model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:
        # No limitations - removed for better usability
        # if limitation and len(tts_text) > max_text_length:
        #     print("Error: Text too long")
        #     return (
        #         f"Text characters should be at most {max_text_length} in this huggingface space, but got {len(tts_text)} characters.",
        #         None,
        #         None,
        #     )
        
        # Check if model is selected (before loading model)
        if not model_name or (len(models) > 0 and model_name not in models):
            return (
                "שגיאה: לא נבחר מודל או שהמודל לא קיים.\n\n"
                "אנא בחר מודל מהרשימה או הוסף מודל RVC לתיקיית weights/",
                None,
                None,
            )
        
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        t0 = time.time()
        
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
            tts_success, tts_error = generate_tts_edge(tts_text, tts_voice, speed_str, edge_output_filename)
        elif tts_engine == "OpenAI TTS":
            tts_success, tts_error = generate_tts_openai(tts_text, tts_voice, speed, openai_api_key, edge_output_filename)
        elif tts_engine == "Google Cloud TTS":
            tts_success, tts_error = generate_tts_google(tts_text, tts_voice, speed, google_api_key_json, edge_output_filename)
        elif tts_engine == "ElevenLabs":
            tts_success, tts_error = generate_tts_elevenlabs(tts_text, tts_voice, speed, elevenlabs_api_key, elevenlabs_voice_id, edge_output_filename)
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
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        # No duration limitation - removed for better usability
        # if limitation and duration >= max_duration:
        #     print("Error: Audio too long")
        #     return (
        #         f"Audio should be less than {max_duration} seconds in this huggingface space, but got {duration:.2f}s.",
        #         edge_output_filename,
        #         None,
        #     )

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
            edge_output_filename,
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
            edge_output_filename,
            (tgt_sr, audio_opt),
        )
    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None


initial_md = """
# RVC text-to-speech webui

This is a text-to-speech webui of RVC models.

Input text ➡[(edge-tts)](https://github.com/rany2/edge-tts)➡ Speech mp3 file ➡[(RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)➡ Final output
"""

# Create Gradio interface
with gr.Blocks() as app:
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
                value="he-IL-AvriNeural-Male" if "he-IL-AvriNeural-Male" in tts_voices else ("he-IL-HilaNeural-Female" if "he-IL-HilaNeural-Female" in tts_voices else tts_voices[0] if tts_voices else "ja-JP-NanamiNeural-Female"),
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
        enable_share = os.getenv("GRADIO_SHARE", "False").lower() == "true"

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
