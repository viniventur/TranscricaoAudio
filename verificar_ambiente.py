# -*- coding: utf-8 -*-
"""Diagnóstico do ambiente: confirma libs, dispositivos de áudio e GPU.

Uso:
    venv\\Scripts\\python.exe verificar_ambiente.py
"""
import inspect
import sys

try:  # console em UTF-8 para acentos aparecerem corretamente
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    print("=" * 60)
    print("VERIFICAÇÃO DO AMBIENTE - Transcrição de Áudio")
    print("=" * 60)

    # --- Versões -----------------------------------------------------------
    import numpy, ctranslate2, faster_whisper
    import pyaudiowpatch as pyaudio
    print(f"numpy           : {numpy.__version__}")
    print(f"ctranslate2     : {ctranslate2.__version__}")
    print(f"faster_whisper  : {faster_whisper.__version__}")
    print(f"pyaudiowpatch   : {pyaudio.__version__}")

    # Identificação de falantes por voz (opcional)
    import os
    try:
        import sherpa_onnx
        print(f"sherpa_onnx     : {sherpa_onnx.__version__} (identificação de voz OK)")
    except Exception as e:
        print(f"sherpa_onnx     : AUSENTE ({e.__class__.__name__}) — sem identificação de voz")
    _model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "assets", "speaker_campplus.onnx")
    if os.path.isfile(_model):
        print(f"modelo de voz   : embutível ({os.path.getsize(_model) / 1e6:.1f} MB em assets/)")
    else:
        print("modelo de voz   : ausente em assets/ (será baixado na 1ª vez; "
              "para embutir no .exe rode baixar_modelo_voz.py)")

    # --- API do PyAudioWPatch (loopback) -----------------------------------
    print("-" * 60)
    print("API PyAudioWPatch:")
    print("  get_loopback_device_info_generator:",
          hasattr(pyaudio.PyAudio, "get_loopback_device_info_generator"))
    print("  get_default_wasapi_loopback       :",
          hasattr(pyaudio.PyAudio, "get_default_wasapi_loopback"))

    # --- Assinatura de WhisperModel.transcribe -----------------------------
    params = set(inspect.signature(faster_whisper.WhisperModel.transcribe).parameters)
    print("-" * 60)
    print("Parâmetros suportados por WhisperModel.transcribe:")
    for k in ["language", "beam_size", "vad_filter", "vad_parameters",
              "condition_on_previous_text"]:
        print(f"  {k:<28}: {k in params}")

    # --- GPU / CUDA --------------------------------------------------------
    print("-" * 60)
    try:
        import os
        try:
            import nvidia
            base = os.path.dirname(nvidia.__file__)
            for sub in ("cublas", "cudnn", "cuda_runtime"):
                b = os.path.join(base, sub, "bin")
                if os.path.isdir(b):
                    os.add_dll_directory(b)
        except Exception:
            pass
        n = ctranslate2.get_cuda_device_count()
        print(f"GPUs CUDA visíveis (ctranslate2): {n}")
        print(f"Dispositivo escolhido: {'cuda / float16' if n > 0 else 'cpu / int8'}")
    except Exception as e:
        print(f"Falha ao consultar CUDA: {e}")

    # --- Dispositivos de áudio ---------------------------------------------
    print("-" * 60)
    pa = pyaudio.PyAudio()
    try:
        print("MICROFONES (entradas):")
        mic_count = 0
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0 and not info.get("isLoopbackDevice", False):
                mic_count += 1
                print(f"  [{info['index']}] {info['name']} "
                      f"({int(info['defaultSampleRate'])} Hz, {int(info['maxInputChannels'])}ch)")
        if mic_count == 0:
            print("  (nenhum)")

        print("LOOPBACKS (áudio do sistema):")
        lb_count = 0
        for info in pa.get_loopback_device_info_generator():
            lb_count += 1
            print(f"  [{info['index']}] {info['name']} "
                  f"({int(info['defaultSampleRate'])} Hz, {int(info['maxInputChannels'])}ch)")
        if lb_count == 0:
            print("  (nenhum)")

        try:
            d = pa.get_default_input_device_info()
            print(f"Microfone padrão : [{d['index']}] {d['name']}")
        except Exception:
            print("Microfone padrão : (indisponível)")
        try:
            d = pa.get_default_wasapi_loopback()
            print(f"Loopback padrão  : [{d['index']}] {d['name']}")
        except Exception:
            print("Loopback padrão  : (indisponível)")
    finally:
        pa.terminate()

    print("=" * 60)
    print("OK - ambiente verificado.")


if __name__ == "__main__":
    main()
