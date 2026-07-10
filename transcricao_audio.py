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

import json
import queue
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pyaudiowpatch as pyaudio


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
APP_TITLE = "Transcrição de Áudio"
TARGET_RATE = 16000
BLOCK_SECONDS = 5.0
READ_CHUNK = 1024
MODELS = ["tiny", "base", "small", "medium"]
DEFAULT_MODEL = "small"
LANGUAGE = "pt"

MODE_MIC = "mic"
MODE_SYS = "sys"
MODE_BOTH = "both"
MODE_LABELS = {MODE_MIC: "🎤  Microfone",
               MODE_SYS: "🔊  Sistema",
               MODE_BOTH: "🎤+🔊  Ambos"}

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
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
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

        frames_needed = int(self.rate * BLOCK_SECONDS)
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
# Thread mixadora (combina microfone + sistema num único bloco)
# ---------------------------------------------------------------------------
class MixerThread(threading.Thread):
    def __init__(self, mic_q, sys_q, out_q, stop_event, sources):
        super().__init__(daemon=True)
        self.mic_q = mic_q
        self.sys_q = sys_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.sources = sources  # threads de captura (mic, sys)

    def run(self):
        while True:
            sources_dead = all(not t.is_alive() for t in self.sources)
            if sources_dead and self.mic_q.empty() and self.sys_q.empty():
                break
            m = self._get(self.mic_q)
            s = self._get(self.sys_q)
            if m is None and s is None:
                continue
            block = self._mix(m, s)
            if block is not None and block.size:
                self.out_q.put(block)

    @staticmethod
    def _get(q, timeout=1.0):
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    @staticmethod
    def _mix(m, s):
        if m is None:
            return s
        if s is None:
            return m
        n = max(m.size, s.size)
        if m.size < n:
            m = np.pad(m, (0, n - m.size))
        if s.size < n:
            s = np.pad(s, (0, n - s.size))
        mixed = m + s
        np.clip(mixed, -1.0, 1.0, out=mixed)
        return mixed.astype(np.float32)


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

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._build_ui()
        self._apply_theme()
        self._on_mode_change()
        self._refresh_devices()
        self._poll_ui_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        self.mode_var = tk.StringVar(value=self.cfg.get("mode", MODE_MIC))
        self.model_var = tk.StringVar(value=self.cfg.get("model", DEFAULT_MODEL))
        self.mic_device_var = tk.StringVar()
        self.sys_device_var = tk.StringVar()

        # -------- Cabeçalho --------
        header = ttk.Frame(self.root, style="Header.TFrame")
        header.pack(fill="x", side="top")
        ttk.Label(header, text="🎙️", style="Logo.TLabel").pack(side="left", padx=(18, 6), pady=14)
        ttk.Label(header, text="Transcrição de Áudio", style="Title.TLabel").pack(side="left", pady=14)
        ttk.Label(header, text="local · offline", style="Subtitle.TLabel").pack(side="left", padx=10, pady=(20, 12))
        self.theme_btn = ttk.Button(header, text="", width=12, style="Ghost.TButton",
                                    command=self._toggle_theme)
        self.theme_btn.pack(side="right", padx=16)

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
        self.text = ScrolledText(textwrap, wrap="word", font=(FONT, 12), relief="flat",
                                 borderwidth=0, padx=14, pady=12)
        self.text.pack(fill="both", expand=True)

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
        s.configure("TLabel", background=c["surface"], foreground=c["fg"])

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

        # Scrollbar
        s.configure("Vertical.TScrollbar", background=c["btn"], troughcolor=c["surface2"],
                    bordercolor=c["surface2"], arrowcolor=c["muted"], relief="flat")

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
        self.model_combo.config(state=combo_state)
        self.mic_combo.config(state=combo_state)
        self.sys_combo.config(state=combo_state)
        self.status_dot.config(style="DotRec.TLabel" if running else "DotIdle.TLabel")

    # --------------------------------------------------------------- worker --
    def _worker(self, mode, model_size, mic_info, sys_info):
        try:
            self._load_model_if_needed(model_size)
        except Exception as e:
            self._post("error", f"Falha ao carregar o modelo:\n{e}")
            self._post("stopped", None)
            return

        err = lambda m: self._post("error", m)
        self.all_threads = []
        if mode == MODE_MIC:
            cap = CaptureThread(self.pa, mic_info["index"], mic_info["rate"],
                                mic_info["channels"], self.audio_queue, self.stop_event, err)
            cap.start()
            self.all_threads = [cap]
            self.feeders = [cap]
        elif mode == MODE_SYS:
            cap = CaptureThread(self.pa, sys_info["index"], sys_info["rate"],
                                sys_info["channels"], self.audio_queue, self.stop_event, err)
            cap.start()
            self.all_threads = [cap]
            self.feeders = [cap]
        else:  # MODE_BOTH
            miccap = CaptureThread(self.pa, mic_info["index"], mic_info["rate"],
                                   mic_info["channels"], self.mic_q, self.stop_event, err)
            syscap = CaptureThread(self.pa, sys_info["index"], sys_info["rate"],
                                   sys_info["channels"], self.sys_q, self.stop_event, err)
            mixer = MixerThread(self.mic_q, self.sys_q, self.audio_queue,
                                self.stop_event, [miccap, syscap])
            miccap.start(); syscap.start(); mixer.start()
            self.all_threads = [miccap, syscap, mixer]
            self.feeders = [mixer]

        self._post("status", f"● Gravando... (modelo '{model_size}' em {self.device_used}).")

        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                if all(not t.is_alive() for t in self.feeders) and self.audio_queue.empty():
                    break
                continue
            self._transcribe_chunk(chunk)
            backlog = self.audio_queue.qsize()
            if backlog > 1:
                self._post("status", f"Transcrevendo... ({backlog} bloco(s) na fila)")

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

    def _transcribe_chunk(self, audio16k):
        try:
            segments, _ = self.model.transcribe(
                audio16k, language=LANGUAGE, beam_size=5, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                self._post("text", text)
        except Exception as e:
            self._post("warn", f"Falha ao transcrever um bloco (ignorado): {e}")

    # ------------------------------------------------------- comunicação UI --
    def _post(self, kind, payload):
        self.ui_queue.put((kind, payload))

    def _poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "text":
                    self._append_text(payload)
                elif kind in ("status", "warn"):
                    self._set_status(payload)
                elif kind == "error":
                    first_line = (payload or "Erro.").splitlines()[0]
                    self._set_status("Erro: " + first_line)
                    # Mostra o pop-up modal só uma vez por sessão (em "Ambos", as
                    # duas capturas podem falhar e empilhariam diálogos).
                    if not self._error_shown:
                        self._error_shown = True
                        messagebox.showerror("Erro", payload)
                elif kind == "stopped":
                    self._on_stopped()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_ui_queue)

    def _append_text(self, text):
        at_end = self.text.yview()[1] >= 0.999
        self.text.insert("end", text + " ")
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
                         "model": self.model_var.get()})
        save_config(self.cfg)
        self.stop_event.set()
        threads_done = True
        try:
            for t in list(self.all_threads):
                if t.is_alive():
                    t.join(timeout=3.0)
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3.0)
            threads_done = all(not t.is_alive() for t in self.all_threads)
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
