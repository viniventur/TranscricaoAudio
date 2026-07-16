# -*- coding: utf-8 -*-
"""
Transcrição de Áudio - App desktop 100% local e offline.

- Motor de transcrição: faster-whisper (roda localmente, sem API, sem custo por uso)
- Interface: tkinter (tema claro/escuro)
- Captura (3 modos): Microfone, Áudio do sistema (WASAPI loopback) ou os dois juntos
- GPU NVIDIA (CUDA) usada automaticamente quando disponível; caso contrário, CPU.

Otimizado para Português (language="pt").
"""

import os
import sys

# ---------------------------------------------------------------------------
# Blindagem para o modo --windowed do PyInstaller: sem console, sys.stdout /
# sys.stderr podem ser None. Bibliotecas (tqdm, logging) que escrevem nesses
# streams causariam exceção. Redirecionamos para "nada".
# ---------------------------------------------------------------------------
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import base64
import hashlib
import io
import json
import queue
import re
import threading
import time
import traceback
import wave
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pyaudiowpatch as pyaudio


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
APP_TITLE = "Transcrição de Áudio"
TARGET_RATE = 16000
READ_CHUNK = 1024
CAPTURE_SECONDS = 1.0          # a captura entrega blocos curtos de ~1s...
# ...e a transcrição os acumula e só corta em PAUSAS (silêncio) ou no tamanho
# máximo, o que melhora MUITO a precisão vs. cortes fixos "no seco".
MIN_FLUSH_SECONDS = 5.0
MAX_FLUSH_SECONDS = 15.0
SILENCE_RMS = 0.012            # abaixo disso o trecho é considerado "pausa"
CONTEXT_TAIL_CHARS = 220       # contexto passado ao modelo entre blocos
MAX_BACKLOG_SECONDS = 90       # teto da fila; acima disso descarta o mais antigo

# Modelos locais (faster-whisper). 'large-v3-turbo' = quase a qualidade do
# large-v3 com muito mais velocidade (ótimo com GPU).
MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
DEFAULT_MODEL = "small"
LANGUAGE = "pt"

# Motores de transcrição
ENGINE_LOCAL = "local"
ENGINE_OPENAI = "openai"
DIARIZE_MODEL = "gpt-4o-transcribe-diarize"   # separa falantes (só na OpenAI)
OPENAI_MODELS = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1", DIARIZE_MODEL]
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-transcribe"
OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"

MODE_MIC = "mic"
MODE_SYS = "sys"
MODE_BOTH = "both"
MODE_LABELS = {MODE_MIC: "🎤  Microfone",
               MODE_SYS: "🔊  Sistema",
               MODE_BOTH: "🎤+🔊  Ambos"}
# Rótulos das fontes no modo "Ambos" (separa você x os outros)
LABEL_MIC = "🎤 Você"
LABEL_SYS = "🔊 Outros"

# ---------------------------------------------------------------------------
# Identificação de falantes por voz (offline) — sherpa-onnx + modelo CAM++ ONNX
# ---------------------------------------------------------------------------
# Cada pessoa é "cadastrada" gravando alguns segundos de fala; extraímos uma
# "impressão vocal" (embedding de 192 dimensões) via sherpa-onnx. Durante a
# transcrição, cada trecho vira um embedding comparado por similaridade de
# cosseno com os perfis; o mais parecido (acima do limiar) define o NOME, e
# abaixo do limiar o rótulo vira "Desconhecido". Roda 100% local — nenhum áudio
# sai da máquina. Modelo: 3D-Speaker CAM++ (Apache-2.0, bilíngue ZH+EN, ~28 MB).
SPEAKER_MODEL_REL = os.path.join("assets", "speaker_campplus.onnx")
SPEAKER_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                     "speaker-recongition-models/"
                     "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
SPEAKER_MODEL_TAG = "3dspeaker_campplus_zh_en_advanced"  # invalida perfis se mudar
SPEAKER_DIM = 192
VOICES_DIR = os.path.join(os.path.expanduser("~"), ".transcricao_audio_voices")
SPKR_THRESHOLD = 0.5           # limiar de cosseno p/ aceitar a identidade (0..1)
SPKR_ENROLL_SECONDS = 12.0     # duração da gravação de cadastro por pessoa
SPKR_MIN_ID_SECONDS = 2.5      # trechos mais curtos que isso não são identificados
SPKR_MIN_ENROLL_SECONDS = 3.0  # mínimo de voz p/ aceitar um cadastro
SPKR_REF_MAX_SECONDS = 9.0     # clipe de referência p/ OpenAI (limite oficial: 2–10 s)
UNKNOWN_SPEAKER = "Desconhecido"
OPENAI_MAX_KNOWN_SPEAKERS = 4  # limite da API known_speaker_references

FONT = "Segoe UI"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".transcricao_audio.json")

# ---------------------------------------------------------------------------
# Paletas de cores (tema claro / escuro)
# ---------------------------------------------------------------------------
THEMES = {
    "light": {
        "bg":        "#eef1f6",
        "surface":   "#ffffff",
        "surface2":  "#f5f7fa",
        "fg":        "#1c2430",
        "muted":     "#5b6675",
        "border":    "#d3d9e2",
        "accent":    "#3b6fed",
        "accent_fg": "#ffffff",
        "start":     "#1f9d55",
        "start_hi":  "#188047",
        "stop":      "#d64545",
        "stop_hi":   "#b83636",
        "btn":       "#e7ebf2",
        "btn_hi":    "#dbe1ea",
        "sel":       "#cfe0ff",
    },
    "dark": {
        "bg":        "#181a1f",
        "surface":   "#22252b",
        "surface2":  "#2a2e35",
        "fg":        "#e6e9ef",
        "muted":     "#9aa4b2",
        "border":    "#363b44",
        "accent":    "#5b8bff",
        "accent_fg": "#0d1117",
        "start":     "#2ecc71",
        "start_hi":  "#27ae60",
        "stop":      "#e05555",
        "stop_hi":   "#c0392b",
        "btn":       "#2f333b",
        "btn_hi":    "#3a3f48",
        "sel":       "#33507f",
    },
}


# ---------------------------------------------------------------------------
# Configuração persistente (tema, modo, modelo)
# ---------------------------------------------------------------------------
def load_config():
    try:
        # utf-8-sig tolera BOM (alguns editores/ferramentas do Windows gravam BOM).
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        # Garante um dicionário: um JSON válido porém não-objeto (null, lista,
        # número) quebraria os .get() posteriores.
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(cfg):
    # Escrita atômica: grava num .tmp e troca; evita corromper o arquivo se o
    # processo for interrompido no meio da escrita.
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass


def detect_windows_dark():
    """True se o Windows estiver em modo escuro para aplicativos."""
    if os.name != "nt":
        return False
    try:
        import winreg
        k = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        val, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
        winreg.CloseKey(k)
        return val == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Utilidades de áudio
# ---------------------------------------------------------------------------
def resample_to_16k(mono_f32, orig_rate):
    """Reamostra mono float32 para 16 kHz (interpolação linear)."""
    if orig_rate == TARGET_RATE or mono_f32.size == 0:
        return mono_f32.astype(np.float32, copy=False)
    n_out = int(round(mono_f32.size * TARGET_RATE / float(orig_rate)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=mono_f32.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, mono_f32).astype(np.float32)


def int16_bytes_to_mono_f32(raw, channels):
    """int16 intercalado -> float32 mono em [-1, 1]."""
    if not raw:
        return np.zeros(0, dtype=np.float32)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False)


def rms(audio):
    """Raiz média quadrática (volume) de um sinal float32; 0 se vazio."""
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def float32_to_wav_bytes(audio16k):
    """Empacota um sinal float32 mono 16 kHz em bytes de um WAV PCM16."""
    pcm = np.clip(audio16k, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_file_to_f32_16k(path):
    """Lê um WAV do disco e devolve float32 mono 16 kHz em [-1, 1]."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = int16_bytes_to_mono_f32(raw, channels)
    if rate != TARGET_RATE:
        audio = resample_to_16k(audio, rate)
    return audio


def resource_path(rel):
    """Caminho de um recurso empacotado: funciona no dev e no exe do PyInstaller
    (que extrai os dados para sys._MEIPASS)."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def trim_to_voiced(audio16k, max_seconds=None, frame_ms=30):
    """Recorta silêncio do início/fim (por RMS de janelas curtas). Se max_seconds
    for dado, limita a duração à parte com voz. Nunca retorna vazio se houver
    algum áudio (cai para o sinal original quando não detecta voz)."""
    if audio16k is None or audio16k.size == 0:
        return np.zeros(0, dtype=np.float32)
    win = max(1, int(TARGET_RATE * frame_ms / 1000))
    voiced = [i for i in range(0, audio16k.size, win)
              if rms(audio16k[i:i + win]) >= SILENCE_RMS]
    if voiced:
        clip = audio16k[voiced[0]:min(audio16k.size, voiced[-1] + win)]
    else:
        clip = audio16k
    if max_seconds is not None:
        cap = int(TARGET_RATE * max_seconds)
        if clip.size > cap:
            clip = clip[:cap]
    return clip.astype(np.float32, copy=False)


class SpeakerBank:
    """Cadastro e identificação de falantes por voz, 100% local (sherpa-onnx).

    - Perfis ficam em ~/.transcricao_audio_voices/<chave>.json (embedding) e
      <chave>.wav (clipe de referência p/ o modo OpenAI diarize).
    - O modelo ONNX é resolvido em 3 níveis: embutido no exe (assets/) ->
      ao lado do script (dev) -> cache do usuário (baixado na 1ª vez).
    """

    def __init__(self, status_cb=None, threshold=SPKR_THRESHOLD):
        self.status_cb = status_cb or (lambda m: None)
        self.threshold = float(threshold)
        self.dim = SPEAKER_DIM
        self._ext = None            # SpeakerEmbeddingExtractor (sherpa-onnx)
        self._mgr = None            # SpeakerEmbeddingManager
        self._model_path = None
        # RLock: serializa acesso ao extrator/manager nativos (worker chama
        # identify; a gravação de cadastro chama enroll) e permite reentrância
        # (enroll -> ensure_loaded -> _embed, todos sob o mesmo lock).
        self._lock = threading.RLock()
        self._profiles = self._read_profiles_from_disk()   # nome -> meta

    # -------------------------------------------------------- disponibilidade
    @staticmethod
    def available():
        """True se o componente de voz (sherpa-onnx) puder ser importado."""
        try:
            import importlib.util
            return importlib.util.find_spec("sherpa_onnx") is not None
        except Exception:
            return False

    def num_profiles(self):
        return len(self._profiles)

    def list_profiles(self):
        return sorted(self._profiles.keys())

    # ------------------------------------------------------------- persistência
    def _read_profiles_from_disk(self):
        out = {}
        try:
            entries = os.listdir(VOICES_DIR)
        except FileNotFoundError:
            return out
        for fn in entries:
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(VOICES_DIR, fn), "r", encoding="utf-8-sig") as f:
                    d = json.load(f)
                name = d.get("name")
                vec = d.get("vector")
                if not name or not isinstance(vec, list):
                    continue
                # Ignora perfis de outro modelo/dimensão (vetores incompatíveis).
                if d.get("model") != SPEAKER_MODEL_TAG or d.get("dim") != SPEAKER_DIM:
                    continue
                d["_key"] = fn[:-5]
                out[str(name)] = d
            except Exception:
                continue
        return out

    @staticmethod
    def _key(name):
        slug = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_") or "perfil"
        h = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:6]
        return f"{slug}_{h}"

    # --------------------------------------------------------------- modelo
    def _resolve_model_path(self):
        # 1) embutido (PyInstaller) ou ao lado do script (dev)
        p = resource_path(SPEAKER_MODEL_REL)
        if os.path.isfile(p):
            return p
        # 2) cache do usuário (já baixado antes)
        cache = os.path.join(VOICES_DIR, "model", "speaker_campplus.onnx")
        if os.path.isfile(cache):
            return cache
        # 3) baixa uma única vez
        self.status_cb("Baixando modelo de voz (~28 MB, só na 1ª vez)...")
        import httpx
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp"
        with httpx.stream("GET", SPEAKER_MODEL_URL, timeout=120.0,
                          follow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(65536):
                    f.write(chunk)
        os.replace(tmp, cache)
        return cache

    def ensure_loaded(self):
        """Carrega o extrator e reconstrói o manager com os perfis salvos.
        Idempotente e protegido por lock (pode ser chamado de várias threads)."""
        with self._lock:
            if self._ext is not None:
                return
            import sherpa_onnx as so
            path = self._resolve_model_path()
            cfg = so.SpeakerEmbeddingExtractorConfig(
                model=path, num_threads=1, provider="cpu")
            if not cfg.validate():
                raise RuntimeError("Configuração do modelo de voz inválida.")
            ext = so.SpeakerEmbeddingExtractor(cfg)
            mgr = so.SpeakerEmbeddingManager(ext.dim)
            for name, meta in self._profiles.items():
                vec = meta.get("vector")
                if isinstance(vec, list) and len(vec) == ext.dim:
                    try:
                        mgr.add(name, vec)
                    except Exception:
                        pass
            self._ext, self._mgr, self._model_path, self.dim = ext, mgr, path, ext.dim

    def _embed(self, audio16k):
        with self._lock:
            stream = self._ext.create_stream()
            stream.accept_waveform(TARGET_RATE, audio16k.astype(np.float32, copy=False))
            stream.input_finished()
            if not self._ext.is_ready(stream):
                return None
            return np.asarray(self._ext.compute(stream), dtype=np.float32)

    # ------------------------------------------------------------- operações
    def enroll(self, name, audio16k, source_kind="mic"):
        """Cadastra (ou regrava) o perfil de voz de uma pessoa."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Informe um nome para a pessoa.")
        self.ensure_loaded()
        clip = trim_to_voiced(audio16k)
        if clip.size < int(TARGET_RATE * SPKR_MIN_ENROLL_SECONDS):
            raise ValueError("Áudio muito curto ou silencioso. Fale por alguns segundos.")
        emb = self._embed(clip)
        if emb is None or emb.size != self.dim:
            raise RuntimeError("Falha ao extrair o embedding de voz.")
        os.makedirs(VOICES_DIR, exist_ok=True)
        key = self._key(name)
        # Clipe de referência (2–10 s) usado no modo OpenAI diarize (atômico).
        ref = trim_to_voiced(audio16k, max_seconds=SPKR_REF_MAX_SECONDS)
        wav_tmp = os.path.join(VOICES_DIR, key + ".wav.tmp")
        with open(wav_tmp, "wb") as f:
            f.write(float32_to_wav_bytes(ref))
            f.flush()
            os.fsync(f.fileno())
        os.replace(wav_tmp, os.path.join(VOICES_DIR, key + ".wav"))
        meta = {"name": name, "vector": emb.tolist(), "model": SPEAKER_MODEL_TAG,
                "dim": int(emb.size), "source": source_kind,
                "created": datetime.now().isoformat(timespec="seconds")}
        tmp = os.path.join(VOICES_DIR, key + ".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(VOICES_DIR, key + ".json"))
        meta["_key"] = key
        self._profiles[name] = meta
        with self._lock:
            if self._mgr is not None:
                try:
                    self._mgr.remove(name)
                except Exception:
                    pass
                self._mgr.add(name, emb.tolist())

    def identify(self, audio16k):
        """Devolve o nome do falante mais parecido (>= limiar) ou "" (indeciso)."""
        if self._ext is None or self._mgr is None or self._mgr.num_speakers == 0:
            return ""
        if audio16k is None or audio16k.size < int(TARGET_RATE * SPKR_MIN_ID_SECONDS):
            return ""
        clip = trim_to_voiced(audio16k)
        if clip.size < int(TARGET_RATE * SPKR_MIN_ID_SECONDS):
            return ""
        emb = self._embed(clip)
        if emb is None:
            return ""
        try:
            with self._lock:
                return self._mgr.search(emb.tolist(), float(self.threshold)) or ""
        except Exception:
            return ""

    def remove(self, name):
        meta = self._profiles.pop(name, None)
        if meta and meta.get("_key"):
            for ext in (".json", ".wav"):
                try:
                    os.remove(os.path.join(VOICES_DIR, meta["_key"] + ext))
                except OSError:
                    pass
        with self._lock:
            if self._mgr is not None:
                try:
                    self._mgr.remove(name)
                except Exception:
                    pass
        return True

    def openai_references(self, limit=OPENAI_MAX_KNOWN_SPEAKERS):
        """Lista (nome, audio16k) dos clipes de referência p/ known_speaker_references."""
        out = []
        for name in self.list_profiles():
            key = self._profiles[name].get("_key", "")
            wav_path = os.path.join(VOICES_DIR, key + ".wav")
            if not os.path.isfile(wav_path):
                continue
            try:
                audio = wav_file_to_f32_16k(wav_path)
            except Exception:
                continue
            if audio.size:
                out.append((name, audio))
            if len(out) >= limit:
                break
        return out


def openai_transcribe(audio16k, api_key, model, prompt=""):
    """Transcreve um bloco pela API da OpenAI. Requer chave. Levanta em erro."""
    import httpx
    wav = float32_to_wav_bytes(audio16k)
    data = {"model": model, "language": LANGUAGE, "response_format": "text",
            "temperature": "0"}
    if prompt:
        data["prompt"] = prompt[-1000:]
    files = {"file": ("audio.wav", wav, "audio/wav")}
    r = httpx.post(OPENAI_URL, headers={"Authorization": f"Bearer {api_key}"},
                   data=data, files=files, timeout=120.0)
    r.raise_for_status()
    return r.text.strip()


def openai_diarize(audio16k, api_key, known=None):
    """Transcreve COM separação de falantes (gpt-4o-transcribe-diarize).
    `known` = lista de (nome, audio16k) para known_speaker_references (máx. 4,
    clipes de 2–10 s): quando informada, a API devolve os NOMES reais em vez de
    "A/B/C" e mantém a identidade estável entre blocos.
    Retorna lista de (falante, texto). Diarize não aceita prompt/idioma."""
    import httpx
    wav = float32_to_wav_bytes(audio16k)
    # data como lista de tuplas p/ permitir chaves repetidas (known_speaker_*[]).
    data = [("model", DIARIZE_MODEL), ("response_format", "diarized_json"),
            ("chunking_strategy", "auto")]
    for name, ref_audio in (known or [])[:OPENAI_MAX_KNOWN_SPEAKERS]:
        b64 = base64.b64encode(float32_to_wav_bytes(ref_audio)).decode("ascii")
        data.append(("known_speaker_names[]", name))
        data.append(("known_speaker_references[]", "data:audio/wav;base64," + b64))
    files = {"file": ("audio.wav", wav, "audio/wav")}
    r = httpx.post(OPENAI_URL, headers={"Authorization": f"Bearer {api_key}"},
                   data=data, files=files, timeout=180.0)
    r.raise_for_status()
    js = r.json()
    out = []
    for seg in js.get("segments", []):
        txt = (seg.get("text") or "").strip()
        if txt:
            out.append((str(seg.get("speaker", "?")), txt))
    if not out:
        full = (js.get("text") or "").strip()
        if full:
            out.append(("?", full))
    return out


def add_cuda_dll_dirs():
    if os.name != "nt":
        return
    if getattr(add_cuda_dll_dirs, "_done", False):
        return
    add_cuda_dll_dirs._done = True
    try:
        import nvidia  # namespace package (PEP 420): usa __path__
        for base in list(getattr(nvidia, "__path__", [])):
            if not os.path.isdir(base):
                continue
            for sub in os.listdir(base):
                bin_dir = os.path.join(base, sub, "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
                    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def detect_device():
    add_cuda_dll_dirs()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


# ---------------------------------------------------------------------------
# Dispositivos de áudio
# ---------------------------------------------------------------------------
class AudioDevices:
    def __init__(self, pa):
        self.pa = pa

    def list_microphones(self):
        result = []
        for i in range(self.pa.get_device_count()):
            try:
                info = self.pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0 and not info.get("isLoopbackDevice", False):
                    result.append(self._fmt(info))
            except Exception:
                continue
        return result

    def list_loopbacks(self):
        result = []
        try:
            for info in self.pa.get_loopback_device_info_generator():
                result.append(self._fmt(info))
        except Exception:
            pass
        return result

    def default_microphone_index(self):
        try:
            return int(self.pa.get_default_input_device_info()["index"])
        except Exception:
            mics = self.list_microphones()
            return mics[0]["index"] if mics else None

    def default_loopback_index(self):
        try:
            return int(self.pa.get_default_wasapi_loopback()["index"])
        except Exception:
            lb = self.list_loopbacks()
            return lb[0]["index"] if lb else None

    @staticmethod
    def _fmt(info):
        return {
            "index": int(info["index"]),
            "name": info["name"],
            "rate": int(info.get("defaultSampleRate", 44100)),
            "channels": max(1, int(info.get("maxInputChannels", 1))),
            "label": f'{info["name"]}  ·  {int(info.get("defaultSampleRate", 44100))} Hz',
        }


# ---------------------------------------------------------------------------
# Thread de captura (grava um dispositivo -> fila de blocos 16k mono)
# ---------------------------------------------------------------------------
class CaptureThread(threading.Thread):
    def __init__(self, pa, device_index, rate, channels, out_queue, stop_event, error_cb):
        super().__init__(daemon=True)
        self.pa = pa
        self.device_index = device_index
        self.rate = int(rate)
        self.channels = int(channels)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.error_cb = error_cb
        self.stream = None

    def run(self):
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16, channels=self.channels, rate=self.rate,
                input=True, frames_per_buffer=READ_CHUNK,
                input_device_index=self.device_index)
        except Exception as e:
            self.stop_event.set()
            self.error_cb(f"Não foi possível abrir o dispositivo de áudio:\n{e}")
            return

        frames_needed = int(self.rate * CAPTURE_SECONDS)
        parts, collected, errors = [], 0, 0
        try:
            while not self.stop_event.is_set():
                try:
                    data = self.stream.read(READ_CHUNK, exception_on_overflow=False)
                    errors = 0
                except Exception as e:
                    errors += 1
                    if errors >= 50:
                        self.stop_event.set()
                        self.error_cb(f"Dispositivo de áudio perdeu conexão:\n{e}")
                        break
                    self.stop_event.wait(0.05)
                    continue
                parts.append(data)
                collected += READ_CHUNK
                if collected >= frames_needed:
                    self._emit(parts)
                    parts, collected = [], 0
            if parts:
                self._emit(parts)
        except Exception as e:
            self.stop_event.set()
            self.error_cb(f"Erro na captura de áudio:\n{e}")
        finally:
            try:
                if self.stream is not None:
                    self.stream.stop_stream()
                    self.stream.close()
            except Exception:
                pass

    def _emit(self, parts):
        raw = b"".join(parts)
        audio16k = resample_to_16k(int16_bytes_to_mono_f32(raw, self.channels), self.rate)
        if audio16k.size:
            self.out_queue.put(audio16k)


# ---------------------------------------------------------------------------
# Título da janela em modo escuro (barra do Windows)
# ---------------------------------------------------------------------------
def set_titlebar_dark(root, dark):
    if os.name != "nt":
        return
    try:
        from ctypes import windll, byref, c_int, sizeof
        root.update_idletasks()
        hwnd = windll.user32.GetParent(root.winfo_id())
        val = c_int(1 if dark else 0)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (20 novo, 19 antigo)
            windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, byref(val), sizeof(val))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Aplicação principal
# ---------------------------------------------------------------------------
class TranscricaoApp:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.theme_name = self.cfg.get(
            "theme", "dark" if detect_windows_dark() else "light")

        self.root.title(APP_TITLE)
        self.root.geometry("900x680")
        self.root.minsize(760, 560)

        self.pa = pyaudio.PyAudio()
        self.devices = AudioDevices(self.pa)

        # Estado
        self.model = None
        self.model_size_loaded = None
        self._force_cpu = False
        self._error_shown = False   # coalescer diálogos de erro numa sessão
        self.audio_queue = queue.Queue()
        self.mic_q = queue.Queue()
        self.sys_q = queue.Queue()
        self.ui_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.all_threads = []
        self.feeders = []
        self.running = False
        self._mic_map = {}
        self._sys_map = {}
        self.show_timestamps = bool(self.cfg.get("timestamps", True))

        # Motor / OpenAI / vocabulário (carregados da config)
        self.engine = self.cfg.get("engine", ENGINE_LOCAL)
        if self.engine not in (ENGINE_LOCAL, ENGINE_OPENAI):
            self.engine = ENGINE_LOCAL
        # Coage para str: uma config editada à mão pode ter tipos errados
        # (número/None), o que quebraria .strip() e travaria o worker.
        self.openai_key = str(self.cfg.get("openai_key", "") or "")
        self.openai_model = str(self.cfg.get("openai_model", DEFAULT_OPENAI_MODEL))
        if self.openai_model not in OPENAI_MODELS:
            self.openai_model = DEFAULT_OPENAI_MODEL
        self.vocab = str(self.cfg.get("vocab", "") or "")

        # Identificação de falantes por voz (offline)
        self.identify_speakers = bool(self.cfg.get("identify_speakers", False))
        self.my_name = str(self.cfg.get("my_name", "") or "")
        try:
            self.spkr_threshold = float(self.cfg.get("spkr_threshold", SPKR_THRESHOLD))
        except (TypeError, ValueError):
            self.spkr_threshold = SPKR_THRESHOLD
        self.spk = SpeakerBank(status_cb=lambda m: self._post("status", m),
                               threshold=self.spkr_threshold)
        self._spk_ready = False
        self._identify_active = False
        self._known_refs = None
        self._enroll_stop = None     # sinaliza p/ abortar uma gravação de cadastro
        self._enroll_cap = None      # CaptureThread ativa do cadastro (p/ fechar limpo)

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._build_ui()
        self._apply_theme()
        self._on_mode_change()
        self._refresh_devices()
        self._update_subtitle()
        self._poll_ui_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        saved_mode = self.cfg.get("mode", MODE_MIC)
        saved_model = self.cfg.get("model", DEFAULT_MODEL)
        self.mode_var = tk.StringVar(value=saved_mode if saved_mode in MODE_LABELS else MODE_MIC)
        self.model_var = tk.StringVar(value=saved_model if saved_model in MODELS else DEFAULT_MODEL)
        self.mic_device_var = tk.StringVar()
        self.sys_device_var = tk.StringVar()

        # -------- Cabeçalho --------
        header = ttk.Frame(self.root, style="Header.TFrame")
        header.pack(fill="x", side="top")
        ttk.Label(header, text="🎙️", style="Logo.TLabel").pack(side="left", padx=(18, 6), pady=14)
        ttk.Label(header, text="Transcrição de Áudio", style="Title.TLabel").pack(side="left", pady=14)
        self.subtitle_lbl = ttk.Label(header, text="local · offline", style="Subtitle.TLabel")
        self.subtitle_lbl.pack(side="left", padx=10, pady=(20, 12))
        self.theme_btn = ttk.Button(header, text="", width=12, style="Ghost.TButton",
                                    command=self._toggle_theme)
        self.theme_btn.pack(side="right", padx=(6, 16))
        self.settings_btn = ttk.Button(header, text="⚙  Configurações", style="Ghost.TButton",
                                       command=self._open_settings)
        self.settings_btn.pack(side="right", padx=6)

        # -------- Card de controles --------
        outer = ttk.Frame(self.root, style="Bg.TFrame")
        outer.pack(fill="x", padx=16, pady=(14, 8))
        card = ttk.Frame(outer, style="Card.TFrame", padding=16)
        card.pack(fill="x")

        # Modo (segmentado)
        ttk.Label(card, text="Capturar de", style="Field.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        seg = ttk.Frame(card, style="Card.TFrame")
        seg.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6))
        for i, m in enumerate((MODE_MIC, MODE_SYS, MODE_BOTH)):
            ttk.Radiobutton(seg, text=MODE_LABELS[m], value=m, variable=self.mode_var,
                            style="Mode.Toolbutton", command=self._on_mode_change
                            ).grid(row=0, column=i, padx=(0 if i == 0 else 6, 0))

        self.refresh_btn = ttk.Button(card, text="⟳  Atualizar", style="Ghost.TButton",
                                      command=self._refresh_devices)
        self.refresh_btn.grid(row=0, column=2, sticky="e")

        # Microfone
        self.mic_row_lbl = ttk.Label(card, text="Microfone", style="Field.TLabel")
        self.mic_combo = ttk.Combobox(card, textvariable=self.mic_device_var,
                                      state="readonly", style="TCombobox")
        # Sistema
        self.sys_row_lbl = ttk.Label(card, text="Sistema", style="Field.TLabel")
        self.sys_combo = ttk.Combobox(card, textvariable=self.sys_device_var,
                                      state="readonly", style="TCombobox")
        # Modelo
        self.model_lbl = ttk.Label(card, text="Modelo", style="Field.TLabel")
        self.model_combo = ttk.Combobox(card, textvariable=self.model_var, state="readonly",
                                        width=12, values=MODELS, style="TCombobox")

        self.mic_row_lbl.grid(row=1, column=0, sticky="w", pady=4)
        self.mic_combo.grid(row=1, column=1, columnspan=2, sticky="we", padx=(12, 0), pady=4)
        self.sys_row_lbl.grid(row=2, column=0, sticky="w", pady=4)
        self.sys_combo.grid(row=2, column=1, columnspan=2, sticky="we", padx=(12, 0), pady=4)
        self.model_lbl.grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.model_combo.grid(row=3, column=1, sticky="w", padx=(12, 0), pady=(4, 0))
        card.columnconfigure(1, weight=1)

        # -------- Botões de ação --------
        actions = ttk.Frame(self.root, style="Bg.TFrame")
        actions.pack(fill="x", padx=16, pady=(4, 8))
        self.start_btn = ttk.Button(actions, text="▶  Iniciar", style="Start.TButton", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="■  Parar", style="Stop.TButton", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.clear_btn = ttk.Button(actions, text="🧹  Limpar", style="TButton", command=self.clear)
        self.clear_btn.pack(side="left")
        self.save_btn = ttk.Button(actions, text="💾  Salvar .txt", style="TButton", command=self.save)
        self.save_btn.pack(side="left", padx=8)
        self.enroll_btn = ttk.Button(actions, text="🧑‍🤝‍🧑  Identificar vozes",
                                     style="TButton", command=self._open_voice_enrollment)
        self.enroll_btn.pack(side="left")

        # -------- Status --------
        self.status_var = tk.StringVar(value="Pronto.")
        statusbar = ttk.Frame(self.root, style="Status.TFrame")
        statusbar.pack(fill="x", padx=16)
        self.status_dot = ttk.Label(statusbar, text="●", style="DotIdle.TLabel")
        self.status_dot.pack(side="left", padx=(10, 6), pady=6)
        self.status_lbl = ttk.Label(statusbar, textvariable=self.status_var, style="Status.TLabel", anchor="w")
        self.status_lbl.pack(side="left", fill="x", expand=True, pady=6)

        # -------- Área de texto --------
        textwrap = ttk.Frame(self.root, style="Bg.TFrame")
        textwrap.pack(fill="both", expand=True, padx=16, pady=(8, 16))
        self.text = tk.Text(textwrap, wrap="word", font=(FONT, 12), relief="flat",
                            borderwidth=0, padx=14, pady=12, highlightthickness=0)
        vsb = ttk.Scrollbar(textwrap, orient="vertical", style="Vertical.TScrollbar",
                            command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

    # ---------------------------------------------------------------- tema --
    def _apply_theme(self):
        c = THEMES[self.theme_name]
        s = self.style

        self.root.configure(bg=c["bg"])
        s.configure("Bg.TFrame", background=c["bg"])
        s.configure("Header.TFrame", background=c["surface"])
        s.configure("Card.TFrame", background=c["surface"])
        s.configure("Status.TFrame", background=c["surface2"])

        s.configure("Logo.TLabel", background=c["surface"], foreground=c["fg"], font=(FONT, 20))
        s.configure("Title.TLabel", background=c["surface"], foreground=c["fg"], font=(FONT, 17, "bold"))
        s.configure("Subtitle.TLabel", background=c["surface"], foreground=c["muted"], font=(FONT, 10))
        s.configure("Field.TLabel", background=c["surface"], foreground=c["muted"], font=(FONT, 10, "bold"))
        s.configure("Status.TLabel", background=c["surface2"], foreground=c["fg"], font=(FONT, 10))
        s.configure("Hint.TLabel", background=c["surface"], foreground=c["muted"], font=(FONT, 9))
        s.configure("TLabel", background=c["surface"], foreground=c["fg"])

        # Entry e Checkbutton (usados na janela de Configurações)
        s.configure("TEntry", fieldbackground=c["surface2"], foreground=c["fg"],
                    bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                    insertcolor=c["fg"], padding=6)
        s.configure("TCheckbutton", background=c["surface"], foreground=c["fg"],
                    font=(FONT, 10), indicatorcolor=c["surface2"], bordercolor=c["border"])
        s.map("TCheckbutton",
              background=[("active", c["surface"])],
              indicatorcolor=[("selected", c["accent"]), ("!selected", c["surface2"])])

        # status dot muda de cor conforme estado (idle/rec)
        s.configure("DotIdle.TLabel", background=c["surface2"], foreground=c["muted"], font=(FONT, 11))
        s.configure("DotRec.TLabel", background=c["surface2"], foreground=c["stop"], font=(FONT, 11))

        # Botões
        s.configure("TButton", background=c["btn"], foreground=c["fg"], bordercolor=c["border"],
                    focuscolor=c["surface"], relief="flat", padding=(14, 9), font=(FONT, 10))
        s.map("TButton", background=[("active", c["btn_hi"]), ("disabled", c["surface2"])],
              foreground=[("disabled", c["muted"])])

        s.configure("Ghost.TButton", background=c["surface"], foreground=c["muted"],
                    bordercolor=c["border"], relief="flat", padding=(10, 7), font=(FONT, 10))
        s.map("Ghost.TButton", background=[("active", c["btn_hi"])], foreground=[("active", c["fg"])])

        s.configure("Start.TButton", background=c["start"], foreground="#ffffff",
                    relief="flat", padding=(18, 9), font=(FONT, 10, "bold"))
        s.map("Start.TButton", background=[("active", c["start_hi"]), ("disabled", c["surface2"])],
              foreground=[("disabled", c["muted"])])

        s.configure("Stop.TButton", background=c["stop"], foreground="#ffffff",
                    relief="flat", padding=(18, 9), font=(FONT, 10, "bold"))
        s.map("Stop.TButton", background=[("active", c["stop_hi"]), ("disabled", c["surface2"])],
              foreground=[("disabled", c["muted"])])

        # Segmentado (radiobutton estilo toolbutton)
        s.configure("Mode.Toolbutton", background=c["btn"], foreground=c["fg"], bordercolor=c["border"],
                    relief="flat", padding=(16, 9), font=(FONT, 10), anchor="center")
        s.map("Mode.Toolbutton",
              background=[("selected", c["accent"]), ("active", c["btn_hi"])],
              foreground=[("selected", c["accent_fg"])])

        # Combobox
        s.configure("TCombobox", fieldbackground=c["surface2"], background=c["btn"],
                    foreground=c["fg"], arrowcolor=c["fg"], bordercolor=c["border"],
                    padding=(8, 6), relief="flat")
        s.map("TCombobox", fieldbackground=[("readonly", c["surface2"])],
              foreground=[("readonly", c["fg"])], arrowcolor=[("active", c["accent"])])
        self.root.option_add("*TCombobox*Listbox.background", c["surface"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", c["accent_fg"])
        self.root.option_add("*TCombobox*Listbox.font", (FONT, 10))
        # option_add só afeta popdowns criados DEPOIS. Reestiliza os já existentes
        # para o tema mudar na hora (ao alternar claro/escuro).
        for combo in (getattr(self, "mic_combo", None), getattr(self, "sys_combo", None),
                      getattr(self, "model_combo", None)):
            if combo is None:
                continue
            try:
                pop = combo.tk.call("ttk::combobox::PopdownWindow", combo)
                lb = pop + ".f.l"
                combo.tk.call(lb, "configure", "-background", c["surface"],
                              "-foreground", c["fg"], "-selectbackground", c["accent"],
                              "-selectforeground", c["accent_fg"])
            except Exception:
                pass

        # Scrollbar
        s.configure("Vertical.TScrollbar", background=c["btn"], troughcolor=c["surface2"],
                    bordercolor=c["surface2"], arrowcolor=c["muted"], relief="flat")
        s.map("Vertical.TScrollbar", background=[("active", c["btn_hi"])])

        # Área de texto
        self.text.configure(background=c["surface"], foreground=c["fg"],
                            insertbackground=c["fg"], selectbackground=c["accent"],
                            selectforeground=c["accent_fg"])

        self.theme_btn.configure(text=("☀  Claro" if self.theme_name == "dark" else "🌙  Escuro"))
        set_titlebar_dark(self.root, self.theme_name == "dark")

    def _toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self._apply_theme()
        self.cfg["theme"] = self.theme_name
        save_config(self.cfg)

    # ------------------------------------------------------- modo/dispos. --
    def _on_mode_change(self):
        mode = self.mode_var.get()
        show_mic = mode in (MODE_MIC, MODE_BOTH)
        show_sys = mode in (MODE_SYS, MODE_BOTH)
        for w, show in ((self.mic_row_lbl, show_mic), (self.mic_combo, show_mic),
                        (self.sys_row_lbl, show_sys), (self.sys_combo, show_sys)):
            if show:
                w.grid()
            else:
                w.grid_remove()

    def _refresh_devices(self):
        mics = self.devices.list_microphones()
        lbs = self.devices.list_loopbacks()
        self._mic_map = {it["label"]: it for it in mics}
        self._sys_map = {it["label"]: it for it in lbs}
        self.mic_combo["values"] = list(self._mic_map.keys())
        self.sys_combo["values"] = list(self._sys_map.keys())

        self._preselect(self.mic_combo, self.mic_device_var, mics,
                        self.devices.default_microphone_index())
        self._preselect(self.sys_combo, self.sys_device_var, lbs,
                        self.devices.default_loopback_index())
        self._set_status(f"{len(mics)} microfone(s), {len(lbs)} saída(s) de sistema detectada(s).")

    @staticmethod
    def _preselect(combo, var, items, default_idx):
        if var.get() in [it["label"] for it in items]:
            return
        chosen = None
        for it in items:
            if it["index"] == default_idx:
                chosen = it["label"]
                break
        if chosen is None and items:
            chosen = items[0]["label"]
        var.set(chosen or "")

    def _selected_mic(self):
        return self._mic_map.get(self.mic_device_var.get())

    def _selected_sys(self):
        return self._sys_map.get(self.sys_device_var.get())

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _update_subtitle(self):
        if self.engine == ENGINE_OPENAI:
            self.subtitle_lbl.config(text=f"OpenAI · {self.openai_model}")
        else:
            self.subtitle_lbl.config(text="local · offline")

    # ---------------------------------------------------------- configurações --
    def _open_settings(self):
        c = THEMES[self.theme_name]
        win = tk.Toplevel(self.root)
        win.title("Configurações")
        win.configure(bg=c["surface"])
        win.transient(self.root)
        win.resizable(False, False)
        try:
            set_titlebar_dark(win, self.theme_name == "dark")
        except Exception:
            pass

        pad = {"padx": 16}
        engine_var = tk.StringVar(value=self.engine)
        key_var = tk.StringVar(value=self.openai_key)
        omodel_var = tk.StringVar(value=self.openai_model)
        ts_var = tk.BooleanVar(value=self.show_timestamps)
        idv_var = tk.BooleanVar(value=self.identify_speakers)
        myname_var = tk.StringVar(value=self.my_name)

        ttk.Label(win, text="Motor de transcrição", style="Field.TLabel").pack(anchor="w", pady=(16, 4), **pad)
        eng = ttk.Frame(win, style="Card.TFrame")
        eng.pack(anchor="w", **pad)
        ttk.Radiobutton(eng, text="💻  Local (faster-whisper) — offline, grátis",
                        value=ENGINE_LOCAL, variable=engine_var, style="Mode.Toolbutton").pack(side="left")
        ttk.Radiobutton(eng, text="☁  OpenAI (nuvem) — mais preciso",
                        value=ENGINE_OPENAI, variable=engine_var, style="Mode.Toolbutton").pack(side="left", padx=6)

        ttk.Label(win, text="Chave da API OpenAI (sk-...)", style="Field.TLabel").pack(anchor="w", pady=(16, 4), **pad)
        key_entry = ttk.Entry(win, textvariable=key_var, show="•", width=54)
        key_entry.pack(anchor="w", **pad)
        show_var = tk.BooleanVar(value=False)
        def _toggle_show():
            key_entry.config(show="" if show_var.get() else "•")
        ttk.Checkbutton(win, text="Mostrar chave", variable=show_var, command=_toggle_show,
                        style="TCheckbutton").pack(anchor="w", pady=(4, 0), **pad)
        ttk.Label(win, text="A chave é salva localmente em ~/.transcricao_audio.json (texto puro).\n"
                            "Alternativa: definir a variável de ambiente OPENAI_API_KEY.",
                  style="Hint.TLabel", justify="left").pack(anchor="w", pady=(2, 0), **pad)

        ttk.Label(win, text="Modelo OpenAI", style="Field.TLabel").pack(anchor="w", pady=(16, 4), **pad)
        ttk.Combobox(win, textvariable=omodel_var, state="readonly", values=OPENAI_MODELS,
                     width=28, style="TCombobox").pack(anchor="w", **pad)
        ttk.Label(win, text="gpt-4o-transcribe-diarize = separa os falantes (Falante 1, 2...).",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 0), **pad)

        ttk.Checkbutton(win, text="Mostrar horário [hh:mm] em cada trecho", variable=ts_var,
                        style="TCheckbutton").pack(anchor="w", pady=(16, 0), **pad)

        ttk.Checkbutton(win, text="Identificar falantes por voz (perfis locais, offline)",
                        variable=idv_var, style="TCheckbutton").pack(anchor="w", pady=(16, 0), **pad)
        ttk.Label(win, text="Cadastre as vozes no botão “🧑‍🤝‍🧑 Identificar vozes”. Sem perfis,\n"
                            "os rótulos padrão (Você/Outros/Falante) são mantidos.",
                  style="Hint.TLabel", justify="left").pack(anchor="w", pady=(2, 0), **pad)

        ttk.Label(win, text="Seu nome (fixa o microfone no modo 🎤+🔊 Ambos)",
                  style="Field.TLabel").pack(anchor="w", pady=(16, 4), **pad)
        ttk.Entry(win, textvariable=myname_var, width=30).pack(anchor="w", **pad)

        ttk.Label(win, text="Vocabulário / contexto (nomes, siglas, jargão — melhora muito)",
                  style="Field.TLabel").pack(anchor="w", pady=(16, 4), **pad)
        vocab_txt = tk.Text(win, width=54, height=4, wrap="word", font=(FONT, 10),
                            relief="flat", padx=8, pady=6, highlightthickness=1,
                            bg=c["surface2"], fg=c["fg"], insertbackground=c["fg"],
                            highlightbackground=c["border"], highlightcolor=c["accent"])
        vocab_txt.pack(anchor="w", **pad)
        vocab_txt.insert("1.0", self.vocab)

        btns = ttk.Frame(win, style="Card.TFrame")
        btns.pack(anchor="e", pady=16, **pad)

        def _save():
            self.engine = engine_var.get()
            self.openai_key = key_var.get().strip()
            self.openai_model = omodel_var.get()
            self.vocab = vocab_txt.get("1.0", "end").strip()
            self.show_timestamps = ts_var.get()
            self.identify_speakers = idv_var.get()
            self.my_name = myname_var.get().strip()
            self.cfg.update({"engine": self.engine, "openai_key": self.openai_key,
                             "openai_model": self.openai_model, "vocab": self.vocab,
                             "timestamps": self.show_timestamps,
                             "identify_speakers": self.identify_speakers,
                             "my_name": self.my_name,
                             "spkr_threshold": self.spkr_threshold})
            save_config(self.cfg)
            self._update_subtitle()
            self._set_status("Configurações salvas.")
            win.destroy()

        ttk.Button(btns, text="Cancelar", style="TButton", command=win.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btns, text="💾  Salvar", style="Start.TButton", command=_save).pack(side="right")

        win.update_idletasks()
        win.grab_set()
        key_entry.focus_set()

    # ------------------------------------------------------ cadastro de vozes --
    def _open_voice_enrollment(self):
        if self.running:
            return
        if not SpeakerBank.available():
            messagebox.showerror("Identificar vozes",
                                 "O componente de identificação de voz (sherpa-onnx) não está "
                                 "instalado neste ambiente.")
            return

        c = THEMES[self.theme_name]
        win = tk.Toplevel(self.root)
        win.title("Identificar vozes")
        win.configure(bg=c["surface"])
        win.transient(self.root)
        win.resizable(False, False)
        try:
            set_titlebar_dark(win, self.theme_name == "dark")
        except Exception:
            pass
        pad = {"padx": 16}

        prog_q = queue.Queue()
        state = {"recording": False, "ready": False, "cancelled": False, "stop": None}

        ttk.Label(win, text="Cadastro de vozes (offline)", style="Title.TLabel").pack(
            anchor="w", pady=(14, 0), **pad)
        ttk.Label(win, text="Cada pessoa fala alguns segundos; o app aprende a “impressão vocal”\n"
                            "dela e passa a rotular os trechos com o nome certo durante a transcrição.",
                  style="Hint.TLabel", justify="left").pack(anchor="w", pady=(2, 8), **pad)

        # -------- Lista de perfis cadastrados --------
        ttk.Label(win, text="Pessoas cadastradas", style="Field.TLabel").pack(anchor="w", **pad)
        listwrap = ttk.Frame(win, style="Card.TFrame")
        listwrap.pack(fill="x", pady=(4, 0), **pad)
        listbox = tk.Listbox(listwrap, height=5, font=(FONT, 10), relief="flat",
                             activestyle="none", highlightthickness=1,
                             bg=c["surface2"], fg=c["fg"], selectbackground=c["accent"],
                             selectforeground=c["accent_fg"], highlightbackground=c["border"],
                             highlightcolor=c["accent"])
        listbox.pack(side="left", fill="x", expand=True)
        remove_btn = ttk.Button(listwrap, text="🗑  Remover", style="TButton")
        remove_btn.pack(side="left", padx=(8, 0))

        def _refresh_list():
            listbox.delete(0, "end")
            for name in self.spk.list_profiles():
                listbox.insert("end", name)

        # -------- Adicionar / regravar --------
        ttk.Label(win, text="Adicionar pessoa (ou regravar uma existente)",
                  style="Field.TLabel").pack(anchor="w", pady=(16, 4), **pad)

        row = ttk.Frame(win, style="Card.TFrame")
        row.pack(anchor="w", fill="x", **pad)
        ttk.Label(row, text="Nome", style="TLabel").pack(side="left")
        name_var = tk.StringVar()
        name_entry = ttk.Entry(row, textvariable=name_var, width=24)
        name_entry.pack(side="left", padx=(8, 0))

        src_row = ttk.Frame(win, style="Card.TFrame")
        src_row.pack(anchor="w", pady=(8, 0), **pad)
        ttk.Label(src_row, text="Fonte", style="TLabel").pack(side="left")
        src_var = tk.StringVar(value="sys" if self.mode_var.get() == MODE_SYS else "mic")
        ttk.Radiobutton(src_row, text="🎤 Microfone", value="mic", variable=src_var,
                        style="Mode.Toolbutton").pack(side="left", padx=(8, 0))
        ttk.Radiobutton(src_row, text="🔊 Sistema", value="sys", variable=src_var,
                        style="Mode.Toolbutton").pack(side="left", padx=(6, 0))

        record_btn = ttk.Button(win, text="●  Gravar 12 s", style="Start.TButton")
        record_btn.pack(anchor="w", pady=(12, 0), **pad)

        ttk.Label(win, text="Nível", style="Hint.TLabel").pack(anchor="w", pady=(10, 0), **pad)
        level_bar = ttk.Progressbar(win, orient="horizontal", mode="determinate",
                                    maximum=100, length=360)
        level_bar.pack(anchor="w", **pad)
        prog_bar = ttk.Progressbar(win, orient="horizontal", mode="determinate",
                                   maximum=100, length=360)
        prog_bar.pack(anchor="w", pady=(6, 0), **pad)

        status_var = tk.StringVar(value="Preparando modelo de voz...")
        ttk.Label(win, textvariable=status_var, style="Status.TLabel").pack(
            anchor="w", pady=(8, 0), **pad)

        ttk.Label(win, text="🔒 Os perfis de voz são dados biométricos e ficam SÓ no seu PC\n"
                            "(~/.transcricao_audio_voices). Nada é enviado à nuvem.",
                  style="Hint.TLabel", justify="left").pack(anchor="w", pady=(10, 0), **pad)

        btns = ttk.Frame(win, style="Card.TFrame")
        btns.pack(anchor="e", pady=14, **pad)
        close_btn = ttk.Button(btns, text="Fechar", style="TButton", command=win.destroy)
        close_btn.pack(side="right")

        # -------- Lógica --------
        def _set_controls(enabled):
            st = "normal" if (enabled and state["ready"]) else "disabled"
            record_btn.config(state=st)
            remove_btn.config(state="normal" if enabled else "disabled")

        def _source_info():
            return self._selected_sys() if src_var.get() == "sys" else self._selected_mic()

        def _start_record():
            if state["recording"]:
                return
            name = name_var.get().strip()
            if not name:
                status_var.set("Digite um nome antes de gravar.")
                return
            info = _source_info()
            if not info:
                status_var.set("Fonte de áudio indisponível. Escolha mic/sistema válido "
                               "na janela principal.")
                return
            state["recording"] = True
            state["cancelled"] = False
            stop = threading.Event()
            state["stop"] = stop
            self._enroll_stop = stop
            outcome = {"posted": False}   # já enviou saved/recerror?
            _set_controls(False)
            prog_bar["value"] = 0
            status_var.set("Gravando... fale naturalmente.")
            src_kind = src_var.get()

            def _post_term(item):
                outcome["posted"] = True
                prog_q.put(item)

            def _on_err(m):
                stop.set()
                _post_term(("recerror", m))

            def _worker_rec():
                try:
                    q = queue.Queue()
                    cap = CaptureThread(self.pa, info["index"], info["rate"],
                                        info["channels"], q, stop, _on_err)
                    cap.start()
                    self._enroll_cap = cap
                    buf, secs = [], 0.0
                    while secs < SPKR_ENROLL_SECONDS and not stop.is_set():
                        try:
                            chunk = q.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        buf.append(chunk)
                        secs += chunk.size / float(TARGET_RATE)
                        prog_q.put(("prog", secs, rms(chunk)))
                    stop.set()
                    cap.join(timeout=3.0)
                    if state.get("cancelled") or outcome["posted"]:
                        return                       # janela fechada ou erro já reportado
                    if not buf:
                        _post_term(("recerror", "Não capturou áudio."))
                        return
                    self.spk.enroll(name, np.concatenate(buf), src_kind)
                    _post_term(("saved", name))
                except Exception as e:
                    _post_term(("recerror", str(e)))
                finally:
                    self._enroll_cap = None
                    self._enroll_stop = None
                    # Rede de segurança: garante que a UI reabilite em qualquer saída.
                    if not outcome["posted"] and not state.get("cancelled"):
                        _post_term(("recerror", "Gravação interrompida."))

            threading.Thread(target=_worker_rec, daemon=True).start()

        def _remove_selected():
            sel = listbox.curselection()
            if not sel:
                return
            name = listbox.get(sel[0])
            if messagebox.askyesno("Remover", f"Remover o perfil de voz de “{name}”?", parent=win):
                self.spk.remove(name)
                _refresh_list()
                status_var.set(f"Perfil removido: {name}")

        record_btn.config(command=_start_record)
        remove_btn.config(command=_remove_selected)

        def _on_win_close():
            # Fechar durante a gravação: aborta e descarta (não salva perfil).
            state["cancelled"] = True
            if state.get("stop") is not None:
                state["stop"].set()
            win.destroy()

        close_btn.config(command=_on_win_close)
        win.protocol("WM_DELETE_WINDOW", _on_win_close)

        def _poll():
            try:
                while True:
                    item = prog_q.get_nowait()
                    kind = item[0]
                    if kind == "prog":
                        _, secs, level = item
                        prog_bar["value"] = min(100.0, secs / SPKR_ENROLL_SECONDS * 100.0)
                        level_bar["value"] = min(100.0, level * 500.0)
                        status_var.set("Gravando... fale naturalmente (%d s)" % int(secs))
                    elif kind == "saved":
                        state["recording"] = False
                        prog_bar["value"] = 0
                        level_bar["value"] = 0
                        name_var.set("")
                        _refresh_list()
                        _set_controls(True)
                        status_var.set(f"Perfil salvo: {item[1]}")
                    elif kind == "recerror":
                        state["recording"] = False
                        prog_bar["value"] = 0
                        level_bar["value"] = 0
                        _set_controls(True)
                        status_var.set("Erro na gravação: " + str(item[1]))
                    elif kind == "ready":
                        state["ready"] = True
                        _set_controls(True)
                        n = self.spk.num_profiles()
                        status_var.set("Pronto. %d voz(es) cadastrada(s)." % n if n
                                       else "Pronto. Digite um nome e grave a 1ª voz.")
                    elif kind == "preperror":
                        state["ready"] = False
                        _set_controls(False)
                        status_var.set("Falha ao preparar o modelo: " + str(item[1]))
            except queue.Empty:
                pass
            if win.winfo_exists():
                win.after(80, _poll)

        def _prep():
            try:
                self.spk.ensure_loaded()
                prog_q.put(("ready", None))
            except Exception as e:
                prog_q.put(("preperror", str(e)))

        _refresh_list()
        _set_controls(False)
        threading.Thread(target=_prep, daemon=True).start()
        win.after(80, _poll)
        win.update_idletasks()
        win.grab_set()
        name_entry.focus_set()

    # --------------------------------------------------------------- ações --
    def start(self):
        if self.running:
            return
        mode = self.mode_var.get()
        mic_info = self._selected_mic() if mode in (MODE_MIC, MODE_BOTH) else None
        sys_info = self._selected_sys() if mode in (MODE_SYS, MODE_BOTH) else None
        if mode in (MODE_MIC, MODE_BOTH) and not mic_info:
            messagebox.showwarning("Dispositivo", "Selecione um microfone válido.")
            return
        if mode in (MODE_SYS, MODE_BOTH) and not sys_info:
            messagebox.showwarning("Dispositivo", "Selecione uma saída de sistema (loopback) válida.")
            return

        self.stop_event.clear()
        self.running = True
        self._error_shown = False
        self._set_running_ui(True)
        for q in (self.audio_queue, self.mic_q, self.sys_q):
            self._drain_queue(q)

        self.worker_thread = threading.Thread(
            target=self._worker, args=(mode, self.model_var.get(), mic_info, sys_info), daemon=True)
        self.worker_thread.start()

        # salva preferências
        self.cfg.update({"mode": mode, "model": self.model_var.get()})
        save_config(self.cfg)

    def stop(self):
        if not self.running:
            return
        self._set_status("Parando... processando os últimos blocos.")
        self.stop_event.set()
        self.stop_btn.config(state="disabled")

    def clear(self):
        self.text.delete("1.0", "end")
        self._set_status("Texto limpo.")

    def save(self):
        content = self.text.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Salvar", "Não há texto para salvar.")
            return
        path = filedialog.asksaveasfilename(
            title="Salvar transcrição", defaultextension=".txt",
            initialfile="transcricao_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
            filetypes=[("Arquivo de texto", "*.txt"), ("Todos os arquivos", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            self._set_status(f"Salvo em: {path}")
        except Exception as e:
            messagebox.showerror("Salvar", f"Falha ao salvar:\n{e}")

    def _set_running_ui(self, running):
        state_ctrl = "disabled" if running else "normal"
        combo_state = "disabled" if running else "readonly"
        self.start_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")
        self.refresh_btn.config(state=state_ctrl)
        self.settings_btn.config(state=state_ctrl)
        self.enroll_btn.config(state=state_ctrl)
        self.model_combo.config(state=combo_state)
        self.mic_combo.config(state=combo_state)
        self.sys_combo.config(state=combo_state)
        self.status_dot.config(style="DotRec.TLabel" if running else "DotIdle.TLabel")

    # --------------------------------------------------------------- worker --
    def _worker(self, mode, model_size, mic_info, sys_info):
        self._context_tail = ""
        engine = self.engine
        if engine == ENGINE_OPENAI:
            if not self._openai_key():
                self._post("error", "Modo OpenAI selecionado, mas nenhuma chave foi informada.\n"
                                    "Abra ⚙ Configurações, cole sua chave da OpenAI (ou defina a "
                                    "variável de ambiente OPENAI_API_KEY) e tente novamente.")
                self._post("stopped", None)
                return
        else:
            try:
                self._load_model_if_needed(model_size)
            except Exception as e:
                self._post("error", f"Falha ao carregar o modelo:\n{e}")
                self._post("stopped", None)
                return

        err = lambda m: self._post("error", m)
        self.all_threads = []

        # ---- Identificação de falantes por voz (offline), se ligada ----------
        self._spk_ready = False
        self._identify_active = False
        self._known_refs = None
        if self.identify_speakers:
            if not SpeakerBank.available():
                self._post("warn", "Identificação de voz indisponível (componente ausente). "
                                   "Seguindo com rótulos padrão.")
            elif self.spk.num_profiles() == 0:
                self._post("warn", "Identificação ligada, mas nenhuma voz cadastrada — "
                                   "usando rótulos padrão (use “🧑‍🤝‍🧑 Identificar vozes”).")
            else:
                try:
                    self._post("status", "Preparando identificação de voz...")
                    self.spk.threshold = self.spkr_threshold
                    self.spk.ensure_loaded()
                    self._spk_ready = True
                    self._identify_active = True
                except Exception as e:
                    self._post("warn", f"Falha ao carregar identificação de voz ({e}). "
                                       "Seguindo com rótulos padrão.")
        # Referências (nome + clipe) p/ o diarize da OpenAI devolver nomes reais.
        if (engine == ENGINE_OPENAI and self.openai_model == DIARIZE_MODEL
                and self._identify_active):
            try:
                self._known_refs = self.spk.openai_references()
                if self.spk.num_profiles() > OPENAI_MAX_KNOWN_SPEAKERS:
                    self._post("warn", f"OpenAI diarize aceita só {OPENAI_MAX_KNOWN_SPEAKERS} "
                                       f"vozes conhecidas; usando as {OPENAI_MAX_KNOWN_SPEAKERS} "
                                       "primeiras (as demais podem sair como Falante A/B...).")
            except Exception:
                self._known_refs = None

        active = self._identify_active
        my_name = (self.my_name or "").strip()

        # Cada FONTE de áudio é um "stream" independente (buffer + contexto
        # próprios). No modo "Ambos" isso separa você (microfone) dos outros
        # (sistema) em vez de misturar — mais qualidade e já rotula quem falou.
        def make_stream(info, q, base_label, identify):
            self._drain_queue(q)
            cap = CaptureThread(self.pa, info["index"], info["rate"], info["channels"],
                                q, self.stop_event, err)
            cap.start()
            self.all_threads.append(cap)
            return {"q": q, "base_label": base_label, "identify": identify,
                    "buf": [], "secs": 0.0, "tail": ""}

        streams = []
        if mode == MODE_MIC:
            streams.append(make_stream(mic_info, self.mic_q, "", active))
        elif mode == MODE_SYS:
            streams.append(make_stream(sys_info, self.sys_q, "", active))
        else:  # MODE_BOTH -> mic fixado em você; sistema identificado por voz
            mic_label = my_name if (self.identify_speakers and my_name) else LABEL_MIC
            streams.append(make_stream(mic_info, self.mic_q, mic_label, False))
            streams.append(make_stream(sys_info, self.sys_q, LABEL_SYS, active))

        if engine == ENGINE_OPENAI:
            where = f"OpenAI · {self.openai_model}"
            min_flush, max_flush = 8.0, 24.0
        else:
            where = f"modelo '{model_size}' em {self.device_used}"
            min_flush, max_flush = MIN_FLUSH_SECONDS, MAX_FLUSH_SECONDS
        self._post("status", f"● Gravando... ({where}). Cortando em pausas p/ mais precisão.")

        stop_seen_at = None
        dropped_warned = False
        while True:
            got = False
            for st in streams:
                try:
                    chunk = st["q"].get_nowait()
                except queue.Empty:
                    continue
                got = True
                st["buf"].append(chunk)
                st["secs"] += chunk.size / float(TARGET_RATE)
                if st["secs"] >= max_flush or (st["secs"] >= min_flush
                                               and rms(chunk) < SILENCE_RMS):
                    self._flush_stream(st, engine)
                # Válvula de segurança: descarta excesso se não acompanha o tempo real.
                if st["q"].qsize() > MAX_BACKLOG_SECONDS:
                    while st["q"].qsize() > MAX_BACKLOG_SECONDS:
                        try:
                            st["q"].get_nowait()
                        except queue.Empty:
                            break
                    if not dropped_warned:
                        dropped_warned = True
                        self._post("warn", "A máquina não está acompanhando o tempo real — "
                                           "descartando áudio atrasado. Use um modelo menor ou a GPU.")
            if got:
                continue
            # Nada disponível agora -> checa término (espera captura morrer e fila
            # esvaziar p/ não perder a "cauda"; 'grace' escapa de travamento nativo).
            captures_dead = all(not t.is_alive() for t in self.all_threads)
            if self.stop_event.is_set() and stop_seen_at is None:
                stop_seen_at = time.monotonic()
            grace = stop_seen_at is not None and (time.monotonic() - stop_seen_at) > 4.0
            all_empty = all(st["q"].empty() for st in streams)
            if (captures_dead or grace) and all_empty:
                for st in streams:
                    self._flush_stream(st, engine)
                break
            self.stop_event.wait(0.05)

        # Espera as capturas morrerem antes de liberar novo Start (abrir streams
        # novos enquanto os antigos ainda fecham pode causar crash nativo).
        for th in list(self.all_threads):
            if th is not threading.current_thread():
                th.join(timeout=3.0)
        self._post("status", "Parado.")
        self._post("stopped", None)

    def _load_model_if_needed(self, model_size):
        if self.model is not None and self.model_size_loaded == model_size:
            return
        self.model = None
        self.model_size_loaded = None
        device, compute = ("cpu", "int8") if self._force_cpu else detect_device()
        self._post("status", f"Carregando modelo '{model_size}' em {device} "
                             f"(na primeira vez o modelo é BAIXADO; pode demorar)...")
        from faster_whisper import WhisperModel
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute)
            self._warmup(model)
            self.device_used, self.compute_used = device, compute
        except Exception as e:
            if device == "cuda":
                self._post("status", f"GPU indisponível ({e.__class__.__name__}). Usando CPU (int8)...")
                self._force_cpu = True
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                self._warmup(model)
                self.device_used, self.compute_used = "cpu", "int8"
            else:
                raise
        self.model = model
        self.model_size_loaded = model_size

    def _warmup(self, model):
        segments, _ = model.transcribe(np.zeros(TARGET_RATE, dtype=np.float32),
                                       language=LANGUAGE, beam_size=1)
        list(segments)

    def _build_prompt(self, tail=""):
        """Contexto passado ao modelo: vocabulário do usuário + fim do texto já
        transcrito daquele stream (ajuda em nomes/jargão e mantém continuidade)."""
        parts = []
        if self.vocab and self.vocab.strip():
            parts.append(self.vocab.strip())
        if tail and tail.strip():
            parts.append(tail.strip())
        return " ".join(parts).strip()

    def _openai_key(self):
        return (self.openai_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()

    def _emit_line(self, label, text):
        """Formata e envia um parágrafo: [hora] Rótulo: frase1 / frase2 / ...
        (uma frase por linha, para não ficar tudo embolado)."""
        text = (text or "").strip()
        if not text:
            return
        head = ""
        if self.show_timestamps:
            head += "[" + datetime.now().strftime("%H:%M") + "] "
        if label:
            head += label + ": "
        sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text]
        block = head + sentences[0]
        for s in sentences[1:]:
            block += "\n" + s
        self._post("line", block)

    def _flush_stream(self, st, engine):
        if not st["buf"]:
            return
        audio = np.concatenate(st["buf"])
        st["buf"] = []
        st["secs"] = 0.0
        self._transcribe_stream(audio, engine, st)

    def _transcribe_stream(self, audio16k, engine, st):
        if audio16k is None or audio16k.size == 0:
            return
        try:
            if engine == ENGINE_OPENAI and self.openai_model == DIARIZE_MODEL:
                # Separação de falantes: cada segmento vira uma linha rotulada.
                # Com referências cadastradas, o "speaker" já vem como o NOME real.
                known = self._known_refs if st.get("identify") else None
                for speaker, text in openai_diarize(audio16k, self._openai_key(), known):
                    if known:
                        label = speaker                      # já é o nome real
                    elif st.get("base_label"):
                        label = st["base_label"]             # fonte fixa (mic no modo Ambos)
                    else:
                        label = f"🗣 Falante {speaker}"
                    self._emit_line(label, text)
                return
            if engine == ENGINE_OPENAI:
                text = openai_transcribe(audio16k, self._openai_key(),
                                         self.openai_model, self._build_prompt(st["tail"]))
            else:
                prompt = self._build_prompt(st["tail"])
                segments, _ = self.model.transcribe(
                    audio16k, language=LANGUAGE, beam_size=5, best_of=5,
                    vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
                    condition_on_previous_text=True, initial_prompt=prompt or None)
                text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                st["tail"] = (st["tail"] + " " + text)[-CONTEXT_TAIL_CHARS:]
                self._emit_line(self._resolve_label(st, audio16k), text)
        except Exception as e:
            self._handle_transcribe_error(e, engine)

    def _resolve_label(self, st, audio16k):
        """Rótulo do trecho: nome identificado por voz (quando ligado e há perfis)
        ou o rótulo base da fonte (Você/Outros/vazio). Se identificar mas ninguém
        bater com o limiar, retorna "Desconhecido" (nunca chuta um nome errado)."""
        if st.get("identify") and self._spk_ready:
            try:
                name = self.spk.identify(audio16k)
            except Exception:
                name = ""
            return name or UNKNOWN_SPEAKER
        return st.get("base_label", "")

    def _handle_transcribe_error(self, e, engine):
        status = getattr(getattr(e, "response", None), "status_code", None)
        body = ""
        try:
            body = (e.response.text or "").lower()
        except Exception:
            pass
        quota = "insufficient_quota" in body or "exceeded your current quota" in body
        if engine == ENGINE_OPENAI and status in (401, 403):
            self.stop_event.set()
            self._post("error", f"Chave da OpenAI recusada (HTTP {status}). "
                                f"Ajuste a chave em ⚙ Configurações e tente novamente.")
        elif engine == ENGINE_OPENAI and status == 429 and quota:
            self.stop_event.set()
            self._post("error", "Sua conta OpenAI está SEM CRÉDITOS/COTA (HTTP 429).\n\n"
                                "Adicione créditos em:\n"
                                "https://platform.openai.com/settings/organization/billing\n\n"
                                "Ou use o motor 'Local' (grátis) — de preferência com a GPU "
                                "(Executar-GPU.bat + modelo large-v3-turbo).")
        elif engine == ENGINE_OPENAI and status == 429:
            self._post("warn", "OpenAI: limite de requisições (429). Reduzindo o ritmo...")
            time.sleep(5)
        elif engine == ENGINE_OPENAI:
            self._post("warn", f"Falha na API OpenAI (trecho ignorado): {e}")
        else:
            self._post("warn", f"Falha ao transcrever um trecho (ignorado): {e}")

    # ------------------------------------------------------- comunicação UI --
    def _post(self, kind, payload):
        self.ui_queue.put((kind, payload))

    def _poll_ui_queue(self):
        try:
            while True:
                try:
                    kind, payload = self.ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if kind == "line":
                        self._append_text(payload)
                    elif kind in ("status", "warn"):
                        self._set_status(payload)
                    elif kind == "error":
                        first_line = (payload or "Erro.").splitlines()[0]
                        self._set_status("Erro: " + first_line)
                        # Modal só uma vez por sessão (em "Ambos", as duas capturas
                        # podem falhar e empilhariam diálogos).
                        if not self._error_shown:
                            self._error_shown = True
                            messagebox.showerror("Erro", payload)
                    elif kind == "stopped":
                        self._on_stopped()
                except Exception:
                    pass  # um erro num handler não pode matar o polling da UI
        finally:
            # Reagenda SEMPRE (senão a UI pararia de atualizar para sempre).
            try:
                self.root.after(100, self._poll_ui_queue)
            except Exception:
                pass

    def _append_text(self, block):
        at_end = self.text.yview()[1] >= 0.999
        if self.text.index("end-1c") != "1.0":   # já há conteúdo -> separa parágrafos
            self.text.insert("end", "\n\n")
        self.text.insert("end", block)
        if at_end:
            self.text.see("end")

    def _on_stopped(self):
        self.running = False
        self.stop_event.clear()
        self._set_running_ui(False)

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def _drain_queue(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _on_close(self):
        self.cfg.update({"theme": self.theme_name, "mode": self.mode_var.get(),
                         "model": self.model_var.get(), "engine": self.engine,
                         "openai_key": self.openai_key, "openai_model": self.openai_model,
                         "vocab": self.vocab, "timestamps": self.show_timestamps,
                         "identify_speakers": self.identify_speakers,
                         "my_name": self.my_name, "spkr_threshold": self.spkr_threshold})
        save_config(self.cfg)
        self.stop_event.set()
        # Encerra uma gravação de cadastro em andamento (usa self.pa).
        enroll_cap = self._enroll_cap
        if self._enroll_stop is not None:
            self._enroll_stop.set()
        if enroll_cap is not None:
            try:
                enroll_cap.join(timeout=3.0)
            except Exception:
                pass
        threads_done = True
        try:
            for t in list(self.all_threads):
                if t.is_alive():
                    t.join(timeout=3.0)
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3.0)
            threads_done = (all(not t.is_alive() for t in self.all_threads)
                            and (enroll_cap is None or not enroll_cap.is_alive()))
        except Exception:
            threads_done = False
        if threads_done:
            try:
                self.pa.terminate()
            except Exception:
                pass
        self.root.destroy()


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    TranscricaoApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            import tkinter.messagebox as mb
            mb.showerror("Erro fatal", traceback.format_exc())
        except Exception:
            pass
        raise
