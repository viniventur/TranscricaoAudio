# -*- mode: python ; coding: utf-8 -*-
"""
Spec do PyInstaller para gerar TranscricaoAudio.exe (--onefile --windowed).

Build:
    venv\\Scripts\\activate
    pyinstaller --noconfirm TranscricaoAudio.spec

Observações:
- Este build é CPU (portátil, funciona em qualquer PC Windows). O app detecta
  GPU em tempo de execução; sem as DLLs CUDA no pacote, ele usa a CPU.
- Os arquivos do MODELO Whisper NÃO são embutidos: são baixados na 1ª execução
  para %USERPROFILE%\\.cache\\huggingface (precisa de internet só nessa 1ª vez).
- Para embutir suporte a GPU, veja o README (seção "Build com GPU"): exige
  adicionar os pacotes nvidia-*-cu12 (pacote fica com vários GB).
"""
from PyInstaller.utils.hooks import collect_all

# Coleta código, dados (ex.: modelo VAD silero .onnx) e binários (DLLs) de cada
# dependência que carrega recursos/arquivos nativos em tempo de execução.
_packages = [
    "faster_whisper",   # inclui assets/silero_vad_v6.onnx (VAD)
    "ctranslate2",      # engine Whisper (DLLs nativas)
    "av",               # PyAV (decodificação de áudio)
    "onnxruntime",      # execução do modelo VAD
    "tokenizers",       # tokenizer do Whisper
    "huggingface_hub",  # download do modelo na 1ª execução
    "pyaudiowpatch",    # captura de áudio (mic + WASAPI loopback)
    "httpx",            # chamadas à API da OpenAI (motor opcional em nuvem)
    "certifi",          # bundle de certificados p/ HTTPS (OpenAI)
]

datas, binaries, hiddenimports = [], [], []
for _pkg in _packages:
    d, b, h = collect_all(_pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Extensão C de nível superior do PyAudioWPatch (importada como _portaudiowpatch).
hiddenimports += ["_portaudiowpatch"]


a = Analysis(
    ["transcricao_audio.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="TranscricaoAudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # --windowed (sem console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="icone.ico",     # opcional: descomente e forneça um .ico
)
