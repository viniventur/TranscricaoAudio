@echo off
REM ============================================================
REM  Executa o app pelo CODIGO-FONTE usando a GPU (venv c/ CUDA).
REM  Use este atalho para a MELHOR qualidade: rode com um modelo
REM  grande (large-v3-turbo / large-v3) na sua GPU NVIDIA.
REM  (O .exe empacotado roda so na CPU.)
REM ============================================================
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
  echo Ambiente virtual nao encontrado.
  echo Rode primeiro:  python -m venv venv  ^&^&  venv\Scripts\activate  ^&^&  pip install -r requirements.txt
  pause
  exit /b 1
)
start "" "venv\Scripts\pythonw.exe" "transcricao_audio.py"
