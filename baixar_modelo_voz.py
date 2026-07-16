# -*- coding: utf-8 -*-
"""Baixa o modelo de identificação de voz (CAM++ ONNX) para assets/.

Necessário ANTES de empacotar com o PyInstaller (o .spec embute este arquivo
no .exe, para a identificação de falantes funcionar offline desde a 1ª
execução). Rodando o app a partir do código-fonte, o modelo também é baixado
automaticamente na 1ª vez que a identificação é usada.

Uso:
    venv\\Scripts\\python.exe baixar_modelo_voz.py
"""
import os
import sys

# 3D-Speaker CAM++ (Apache-2.0, bilíngue ZH+EN, embedding 192-d, ~28 MB).
URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
       "speaker-recongition-models/"
       "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "assets", "speaker_campplus.onnx")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if os.path.isfile(DEST):
        print(f"Modelo já existe: {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB)")
        return
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print("Baixando modelo de voz (~28 MB)...")
    import httpx
    tmp = DEST + ".tmp"
    got = 0
    with httpx.stream("GET", URL, timeout=120.0, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(65536):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1e6:5.1f} / {total / 1e6:.1f} MB", end="")
    print()
    os.replace(tmp, DEST)
    print(f"OK -> {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
