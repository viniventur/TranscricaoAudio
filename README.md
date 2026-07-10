# 🎙️ Transcrição de Áudio (local & offline)

App desktop para **Windows** que transcreve áudio **100% localmente** — sem API,
sem nuvem e sem custo por uso — usando [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
(Whisper via CTranslate2). Interface em `tkinter` com **tema claro/escuro**,
otimizado para **Português** (`pt`).

Captura em **3 modos** (selecionáveis na tela):

1. **🎤 Microfone** — a entrada padrão (ou qualquer microfone escolhido)
2. **🔊 Sistema** — o áudio que está tocando no PC (WASAPI *loopback* via
   `PyAudioWPatch`): reuniões, vídeos, chamadas, etc.
3. **🎤+🔊 Ambos** — mistura microfone **e** áudio do sistema num só texto
   (ideal para transcrever uma reunião inteira: você + os outros)

A transcrição roda em **tempo real**, em blocos de ~5 s, numa **thread separada**
(a interface não trava), e usa **GPU NVIDIA (CUDA)** automaticamente quando
disponível — caso contrário, cai para **CPU** sem você precisar fazer nada.

O **tema** (claro/escuro) segue o Windows na primeira vez e pode ser trocado no
botão ☀/🌙 do canto superior direito; sua preferência (tema, modo, modelo) fica
salva em `%USERPROFILE%\.transcricao_audio.json`.

---

## ✅ Requisitos

- **Windows 10/11**
- **Python 3.13 ou 3.14** (64-bit) — testado no 3.14
- (Opcional) **GPU NVIDIA** com driver recente para acelerar

> Verifique sua versão: `python --version`

---

## 📦 Instalação

Abra o **PowerShell** ou o **Prompt de Comando** na pasta do projeto:

```bat
:: 1) Criar o ambiente virtual
python -m venv venv

:: 2) Ativar o ambiente virtual
venv\Scripts\activate

:: 3) Instalar as dependências (CPU) - roda offline após isso
pip install -r requirements.txt
```

### (Opcional) Aceleração por GPU NVIDIA

Só se você tem uma GPU NVIDIA. Baixa ~1,3 GB de bibliotecas CUDA. **Sem isto o
app funciona igual, usando a CPU.**

```bat
pip install -r requirements-gpu.txt
```

---

## ▶️ Como usar

Com o ambiente ativado (`venv\Scripts\activate`):

```bat
python transcricao_audio.py
```

Na janela:

1. **Capturar de** — escolha o modo: **Microfone**, **Sistema** ou **Ambos**.
   - Em **Microfone** aparece o seletor de microfone.
   - Em **Sistema** aparece o seletor de saída (loopback).
   - Em **Ambos** aparecem os dois — o app mistura as duas fontes.
2. **Dispositivo(s)** — selecione a entrada/saída desejada (clique **⟳ Atualizar**
   se plugar um dispositivo novo).
3. **Modelo** — `tiny`, `base`, `small` (padrão) ou `medium`.
   - `tiny`/`base` = mais rápidos, menos precisos.
   - `small` = ótimo equilíbrio (padrão).
   - `medium` = mais preciso, mais pesado (recomendado com GPU).
4. **▶ Iniciar** — começa a transcrever. O texto aparece na caixa rolável.
5. **■ Parar** — encerra (processa os últimos blocos antes de parar).
6. **🧹 Limpar** — apaga o texto.
7. **💾 Salvar .txt** — grava a transcrição em um arquivo de texto (UTF-8).
8. **☀/🌙 (canto superior direito)** — alterna entre tema claro e escuro.

### ⚠️ Primeira execução: download do modelo

Na **primeira vez** que você usa um tamanho de modelo, ele é **baixado**
automaticamente (o `small` tem ~480 MB). A barra de status mostra
*"Carregando modelo… (na primeira vez o modelo é BAIXADO)"*. Precisa de internet
**só nessa primeira vez**; depois fica em cache e roda **offline**.

O modelo fica em: `%USERPROFILE%\.cache\huggingface\hub`

### 🔊 Dica para "Áudio do sistema (loopback)"

O loopback captura o que sai pelo dispositivo de **reprodução padrão**. Se não
aparecer texto, confirme que algo está tocando e que o dispositivo de loopback
selecionado é o mesmo que está com som (ex.: seus alto-falantes/fones ativos).

---

## 🩺 Diagnóstico do ambiente

Para conferir libs, GPU e listar todos os dispositivos de áudio detectados:

```bat
venv\Scripts\python.exe verificar_ambiente.py
```

---

## 🏗️ Gerar o executável (.exe) com PyInstaller

O projeto já vem com `TranscricaoAudio.spec` configurado.

```bat
:: com o venv ativado e o pyinstaller instalado:
pip install pyinstaller

:: (recomendado) build usando o .spec já pronto:
pyinstaller --noconfirm TranscricaoAudio.spec
```

O executável final será: **`dist\TranscricaoAudio.exe`**

### Comando equivalente em uma linha (sem o .spec)

```bat
pyinstaller --noconfirm --onefile --windowed --name TranscricaoAudio --collect-all faster_whisper --collect-all ctranslate2 --collect-all av --collect-all onnxruntime --collect-all tokenizers --collect-all huggingface_hub --collect-all pyaudiowpatch --hidden-import _portaudiowpatch transcricao_audio.py
```

### Observações importantes sobre o build

- **`--collect-all` é essencial.** `faster-whisper`, `ctranslate2`, `av`,
  `onnxruntime`, `tokenizers` e `pyaudiowpatch` trazem **DLLs nativas** e
  **arquivos de dados** (ex.: o modelo VAD `silero_vad_v6.onnx`). Sem coletá-los,
  o `.exe` quebra ao iniciar. O `--hidden-import _portaudiowpatch` garante o
  módulo de captura de áudio.
- **O modelo Whisper NÃO é embutido.** Ele é baixado na 1ª execução do `.exe`
  (para o cache do usuário) e reaproveitado depois. Assim o `.exe` fica muito
  menor. Para uso totalmente offline em outra máquina, rode o `.exe` uma vez com
  internet OU copie a pasta `%USERPROFILE%\.cache\huggingface\hub` junto.
- **Este build é para CPU** (portátil: roda em qualquer PC Windows). O app tenta
  GPU em tempo de execução, e se as DLLs CUDA não estiverem presentes, usa a CPU
  automaticamente.
- **Build com GPU (avançado):** embutir CUDA deixa o `.exe` com **vários GB** e
  exige empacotar as DLLs dos pacotes `nvidia-*-cu12`. Para uso com GPU, o
  recomendado é **rodar pelo código-fonte** com `requirements-gpu.txt` instalado.
- **Primeira abertura do `.exe` (--onefile) é lenta**, pois ele se descompacta em
  uma pasta temporária. Para abertura mais rápida, troque `--onefile` por
  `--onedir` (gera uma pasta em vez de um único arquivo).
- Alguns **antivírus** dão falso-positivo em `.exe` gerados com `--onefile`. Se
  isso incomodar, use `--onedir` ou adicione uma exceção.

---

## 🧠 Como funciona (resumo técnico)

| Componente        | Papel |
|-------------------|-------|
| `faster-whisper`  | Motor de transcrição (Whisper otimizado, roda local) |
| `PyAudioWPatch`   | Captura de microfone e de **loopback WASAPI** |
| `numpy`           | Conversão int16→float32, downmix p/ mono e reamostragem p/ 16 kHz |
| `tkinter`         | Interface gráfica |
| Threads + filas   | Captura, transcrição e UI desacopladas via `queue.Queue` |

- O áudio é capturado em `int16`, convertido para **mono float32** e
  **reamostrado para 16 kHz** (formato que o Whisper espera).
- Cada bloco de ~5 s é transcrito de forma **independente**, com `vad_filter`
  (ignora silêncio e reduz "alucinações"). Como os blocos são independentes,
  pode haver pequenos cortes de palavra na fronteira dos blocos — comportamento
  esperado para transcrição em tempo real por blocos.
- GPU: `device="cuda"`, `compute_type="float16"`. CPU: `device="cpu"`,
  `compute_type="int8"`. A escolha e o *fallback* são automáticos.
- Se a máquina **não acompanha o tempo real** (ex.: modelo `medium` na CPU), os
  blocos se acumulam numa fila e o texto aparece **com atraso** (a barra de
  status mostra quantos blocos estão na fila). Nenhum áudio é descartado — só
  atrasa. Para tempo real fluido em CPU, prefira `tiny`/`base`/`small`; use
  `medium` com GPU.

---

## 🔧 Solução de problemas

| Problema | Solução |
|---|---|
| *"Nenhum dispositivo encontrado"* | Clique **Atualizar**. Verifique se há microfone/saída ativos no Windows. |
| Loopback sem texto | Confirme que há áudio tocando e que o loopback escolhido é o dispositivo de reprodução ativo. |
| Não usa a GPU | Instale `requirements-gpu.txt`. Rode `verificar_ambiente.py` para ver "GPUs CUDA visíveis". Se falhar, o app usa CPU. |
| Download do modelo travado | Precisa de internet na 1ª vez. Verifique proxy/firewall. |
| `.exe` não abre | Rode pelo código-fonte para ver o erro, ou gere o build com console (`console=True` no `.spec`) para depurar. |

---

## 📁 Estrutura do projeto

```
TranscricaoAudio/
├─ transcricao_audio.py     # aplicativo (GUI + captura + transcrição)
├─ verificar_ambiente.py    # diagnóstico de libs/GPU/dispositivos
├─ requirements.txt         # dependências (CPU)
├─ requirements-gpu.txt     # dependências CUDA (opcional)
├─ TranscricaoAudio.spec    # configuração do PyInstaller
├─ README.md
└─ venv/                    # ambiente virtual (não versionar)
```

---

## 📜 Licença / uso

Uso pessoal. As dependências têm suas próprias licenças (MIT/BSD/etc.).
Os modelos Whisper são baixados do Hugging Face (Systran/faster-whisper-*).
