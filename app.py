import asyncio
import datetime
import logging
import os
import time
import traceback
import urllib.request
from pathlib import Path
import sys

import edge_tts
import gradio as gr
import librosa
import torch
from fairseq import checkpoint_utils

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

limitation = os.getenv("SYSTEM") == "spaces"

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

ensure_model_files()

config = Config()

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
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if len(models) == 0:
    raise ValueError("No model found in `weights` folder")
models.sort()


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
    cpt = torch.load(pth_path, map_location="cpu")
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
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


print("Loading hubert model...")
hubert_model = load_hubert()
print("Hubert model loaded.")

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


def tts(
    model_name,
    speed,
    tts_text,
    tts_voice,
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
        if limitation and len(tts_text) > 280:
            print("Error: Text too long")
            return (
                f"Text characters should be at most 280 in this huggingface space, but got {len(tts_text)} characters.",
                None,
                None,
            )
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        t0 = time.time()
        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"
        # Try to generate TTS with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        voice_parts = tts_voice.split("-")
        
        for attempt in range(max_retries):
            try:
                # Extract voice name - remove the last part (Gender) from format "lang-Country-Name-Gender"
                # On retry, try different formats
                if attempt == 0:
                    # First attempt: remove Gender suffix
                    if len(voice_parts) >= 4:
                        voice_name = "-".join(voice_parts[:-1])
                    else:
                        voice_name = tts_voice
                elif attempt == 1:
                    # Second attempt: use full name
                    voice_name = tts_voice
                else:
                    # Third attempt: try ShortName only (first 3 parts)
                    if len(voice_parts) >= 3:
                        voice_name = "-".join(voice_parts[:3])
                    else:
                        voice_name = tts_voice
                
                print(f"Using voice: {voice_name} (from {tts_voice}, attempt {attempt + 1})")
                print(f"Text: {tts_text[:50]}...")
                print(f"Rate: {speed_str}")
                
                # Try to create communicate object and save
                # Use positional arguments (text, voice) as edge-tts expects
                communicate = edge_tts.Communicate(tts_text, voice_name, rate=speed_str)
                
                # Save the audio file
                asyncio.run(communicate.save(edge_output_filename))
                
                # Verify file was created and has content
                if not os.path.exists(edge_output_filename):
                    raise Exception("Output file was not created")
                if os.path.getsize(edge_output_filename) == 0:
                    raise Exception("Output file is empty")
                
                break  # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)
                print(f"Edge TTS error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Handle specific error types
                if "401" in error_msg or "Unauthorized" in error_msg or "WSServerHandshakeError" in error_msg:
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        return (
                            "שגיאה: לא ניתן להתחבר לשירות Microsoft Edge TTS (401 Unauthorized).\n\n"
                            "זה יכול להיות בגלל:\n"
                            "1. בעיית רשת או חומת אש\n"
                            "2. הגבלת קצב (rate limiting) - נסה שוב בעוד כמה דקות\n"
                            "3. Microsoft שינתה את מדיניות האימות\n\n"
                            "פתרונות אפשריים:\n"
                            "1. בדוק את חיבור האינטרנט שלך\n"
                            "2. המתן 5-10 דקות ונסה שוב\n"
                            "3. עדכן edge-tts: pip install --upgrade edge-tts\n"
                            "4. נסה להשתמש ב-VPN או proxy\n"
                            "5. בדוק אם יש עדכונים ל-edge-tts בגיטהאב",
                            None,
                            None,
                        )
                elif "No audio was received" in error_msg or "empty" in error_msg.lower() or "not created" in error_msg.lower():
                    # Try with different voice format or parameters
                    if attempt < max_retries - 1:
                        print(f"Audio not received, trying different voice format in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return (
                            f"שגיאה: לא התקבל אודיו מ-edge-tts לאחר {max_retries} ניסיונות.\n\n"
                            f"פרטי השגיאה: {error_msg}\n\n"
                            "זה יכול להיות בגלל:\n"
                            "1. בעיית רשת או חיבור ל-Microsoft Edge TTS\n"
                            "2. שם הקול לא תקין או לא נתמך\n"
                            "3. הטקסט ריק או לא תקין\n"
                            "4. בעיה עם edge-tts - ייתכן שצריך לעדכן\n"
                            "5. הגבלת קצב (rate limiting) מ-Microsoft\n\n"
                            "פתרונות אפשריים:\n"
                            "1. בדוק את חיבור האינטרנט שלך\n"
                            "2. נסה קול אחר מהרשימה (למשל en-US-AriaNeural-Female)\n"
                            "3. ודא שהטקסט לא ריק\n"
                            "4. עדכן edge-tts: pip install --upgrade edge-tts\n"
                            "5. המתן 5-10 דקות ונסה שוב (rate limiting)\n"
                            "6. נסה להריץ edge-tts ישירות מהטרמינל לבדיקה:\n"
                            "   edge-tts --voice en-US-AriaNeural --text 'Hello' --write-media test.mp3",
                            None,
                            None,
                        )
                else:
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return (
                            f"שגיאה ביצירת אודיו TTS לאחר {max_retries} ניסיונות: {error_msg}\n\n"
                            "נסה:\n"
                            "1. לבדוק את חיבור האינטרנט\n"
                            "2. לנסות קול אחר\n"
                            "3. לבדוק שהטקסט תקין\n"
                            "4. לעדכן edge-tts: pip install --upgrade edge-tts",
                            None,
                            None,
                        )
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        if limitation and duration >= 20:
            print("Error: Audio too long")
            return (
                f"Audio should be less than 20 seconds in this huggingface space, but got {duration}s.",
                edge_output_filename,
                None,
            )

        f0_up_key = int(f0_up_key)

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
            f0_up_key,
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

app = gr.Blocks()
with app:
    gr.Markdown(initial_md)
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(label="Model", choices=models, value=models[0])
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
            tts_voice = gr.Dropdown(
                label="Edge-tts speaker (format: language-Country-Name-Gender)",
                choices=tts_voices,
                allow_custom_value=False,
                value="he-IL-HilaNeural-Female" if "he-IL-HilaNeural-Female" in tts_voices else tts_voices[0] if tts_voices else "ja-JP-NanamiNeural-Female",
            )
            speed = gr.Slider(
                minimum=-100,
                maximum=100,
                label="Speech speed (%)",
                value=0,
                step=10,
                interactive=True,
            )
            tts_text = gr.Textbox(label="Input Text", value="זהו דמו של המרת טקסט עברי לדיבור.")
        with gr.Column():
            but0 = gr.Button("Convert", variant="primary")
            info_text = gr.Textbox(label="Output info")
        with gr.Column():
            edge_tts_output = gr.Audio(label="Edge Voice", type="filepath")
            tts_output = gr.Audio(label="Result")
        but0.click(
            tts,
            [
                model_name,
                speed,
                tts_text,
                tts_voice,
                f0_key_up,
                f0_method,
                index_rate,
                protect0,
            ],
            [info_text, edge_tts_output, tts_output],
        )
    with gr.Row():
        examples = gr.Examples(
            examples_per_page=100,
            examples=[
                ["זהו דמו של המרת טקסט עברי לדיבור.", "he-IL-HilaNeural-Female"],
                ["זהו דמו של המרת טקסט עברי לדיבור.", "he-IL-AvriNeural-Male"],
                ["これは日本語テキストから音声への変換デモです。", "ja-JP-NanamiNeural-Female"],
                [
                    "This is an English text to speech conversation demo.",
                    "en-US-AriaNeural-Female",
                ],
                ["这是一个中文文本到语音的转换演示。", "zh-CN-XiaoxiaoNeural-Female"],
                ["한국어 텍스트에서 음성으로 변환하는 데모입니다.", "ko-KR-SunHiNeural-Female"],
                [
                    "Il s'agit d'une démo de conversion du texte français à la parole.",
                    "fr-FR-DeniseNeural-Female",
                ],
                [
                    "Dies ist eine Demo zur Umwandlung von Deutsch in Sprache.",
                    "de-DE-AmalaNeural-Female",
                ],
                [
                    "Tämä on suomenkielinen tekstistä puheeksi -esittely.",
                    "fi-FI-NooraNeural-Female",
                ],
                [
                    "Это демонстрационный пример преобразования русского текста в речь.",
                    "ru-RU-SvetlanaNeural-Female",
                ],
                [
                    "Αυτή είναι μια επίδειξη μετατροπής ελληνικού κειμένου σε ομιλία.",
                    "el-GR-AthinaNeural-Female",
                ],
                [
                    "Esta es una demostración de conversión de texto a voz en español.",
                    "es-ES-ElviraNeural-Female",
                ],
                [
                    "Questa è una dimostrazione di sintesi vocale in italiano.",
                    "it-IT-ElsaNeural-Female",
                ],
                [
                    "Esta é uma demonstração de conversão de texto em fala em português.",
                    "pt-PT-RaquelNeural-Female",
                ],
                [
                    "Це демонстрація тексту до мовлення українською мовою.",
                    "uk-UA-PolinaNeural-Female",
                ],
                [
                    "هذا عرض توضيحي عربي لتحويل النص إلى كلام.",
                    "ar-EG-SalmaNeural-Female",
                ],
                [
                    "இது தமிழ் உரையிலிருந்து பேச்சு மாற்ற டெமோ.",
                    "ta-IN-PallaviNeural-Female",
                ],
            ],
            inputs=[tts_text, tts_voice],
        )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting RVC TTS WebUI...")
    print("="*50)
    try:
        app.launch(
            inbrowser=True,
            server_name="127.0.0.1",
            server_port=7865,
            share=False
        )
    except Exception as e:
        print(f"\nError starting server: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying to launch without opening browser...")
        try:
            app.launch(
                inbrowser=False,
                server_name="127.0.0.1",
                server_port=7865,
                share=False
            )
        except Exception as e2:
            print(f"\nError: {e2}")
            import traceback
            traceback.print_exc()
