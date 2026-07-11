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

**Modo mais fácil (recomendado, usa a GPU):** dê **duplo-clique em
`Executar-GPU.bat`**.

Ou, pelo terminal com o ambiente ativado (`venv\Scripts\activate`):

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
3. **Modelo** — do mais leve/rápido ao mais preciso:
   - `tiny`/`base` = rápidos, menos precisos.
   - `small` = equilíbrio (padrão).
   - `medium` = mais preciso, mais pesado.
   - **`large-v3-turbo`** = **recomendado com GPU** — quase a qualidade do
     `large-v3` com muito mais velocidade.
   - `large-v3` = máxima qualidade (pesado; use com GPU).
4. **▶ Iniciar** — começa a transcrever. O texto aparece na caixa rolável.
5. **■ Parar** — encerra (processa os últimos trechos antes de parar).
6. **🧹 Limpar** — apaga o texto.
7. **💾 Salvar .txt** — grava a transcrição em um arquivo de texto (UTF-8).
8. **☀/🌙 (canto superior direito)** — alterna entre tema claro e escuro.
9. **⚙ Configurações** — motor (Local ou OpenAI), chave da API, e o campo de
   **vocabulário/contexto** (veja abaixo).

### 🎯 Como conseguir a MELHOR qualidade

A qualidade depende, em ordem de impacto:

1. **Use a GPU + um modelo grande.** O `.exe` roda só na CPU. Para usar sua
   **GPU NVIDIA**, execute pelo código-fonte — o jeito mais fácil é dar
   **duplo-clique em `Executar-GPU.bat`**. Aí escolha o modelo
   **`large-v3-turbo`** (ou `large-v3`). Isso sozinho já muda tudo.
2. **Preencha o Vocabulário/contexto** (⚙ Configurações): liste nomes, siglas e
   jargões do seu conteúdo (ex.: `Pridvê, MoCap, back-end, deploy, Lucas`). O
   modelo passa a acertar essas palavras.
3. **Corte em pausas (automático).** O app agora acumula o áudio e só transcreve
   ao detectar uma **pausa** (ou a cada ~15 s), mantendo o **contexto** entre
   trechos — muito melhor que blocos fixos curtos.
4. **Prefira "Microfone" ou "Sistema" a "Ambos"** quando possível. Misturar as
   duas fontes com falas sobrepostas atrapalha o reconhecimento.

### ☁ Usar a OpenAI (opcional, mais preciso)

Se quiser a melhor precisão sem depender do seu PC, use o motor **OpenAI**
(atenção: aí o áudio é enviado para a nuvem — deixa de ser 100% offline).

**Onde colocar a chave:**
1. Pegue uma chave em <https://platform.openai.com/api-keys> (começa com `sk-`).
2. No app: **⚙ Configurações** → marque **OpenAI (nuvem)** → cole a chave no
   campo **"Chave da API OpenAI"** → escolha o modelo → **Salvar**.
   - Modelos: `gpt-4o-mini-transcribe` (barato e ótimo — padrão),
     `gpt-4o-transcribe` (melhor), `whisper-1` (clássico).
   - A chave fica salva em `%USERPROFILE%\.transcricao_audio.json` (texto puro).
3. **Alternativa (sem salvar no disco):** defina a variável de ambiente
   `OPENAI_API_KEY` antes de abrir o app. Ex. no PowerShell:
   ```powershell
   setx OPENAI_API_KEY "sk-suachave"
   ```
   (feche e reabra o terminal/app depois do `setx`).

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
- **Segmentação por pausa (não por blocos fixos):** o áudio é acumulado e só
  transcrito ao detectar um **silêncio** (pausa natural) ou ao atingir ~15 s.
  Assim os cortes caem em pausas, não no meio de palavras. Cada trecho é
  transcrito com `beam_size=5`, `best_of=5`, `condition_on_previous_text=True`
  e um `initial_prompt` (vocabulário do usuário + fim do texto anterior), o que
  melhora muito nomes/jargão e a continuidade.
- **Motores:** `local` (faster-whisper, offline) ou `openai` (API na nuvem —
  `gpt-4o-mini-transcribe`/`gpt-4o-transcribe`/`whisper-1`), selecionável em
  ⚙ Configurações.
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
├─ Executar-GPU.bat         # abre o app pelo código-fonte usando a GPU
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
