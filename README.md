# TCC_BEC
# 🎙️ Transcrição de Áudio com Whisper Cloud (OpenAI)

Este projeto captura áudio pelo microfone, envia para a **API da OpenAI** utilizando os modelos de *speech-to-text* (`gpt-4o-transcribe` ou `gpt-4o-mini-transcribe`) e salva a transcrição em um arquivo `.txt`.

---

## 🚀 Funcionalidades

- 🎤 Captura áudio do microfone (em 16 kHz).
- ☁️ Envia para a **nuvem** (OpenAI API).
- 📝 Exibe a transcrição no terminal.
- 💾 Salva automaticamente em um arquivo `transcricao.txt`.

---

## 📦 Instalação

Clone o repositório e instale as dependências:

```bash
pip install openai sounddevice soundfile numpy python-dotenv

Configuração da API

Crie um arquivo .env na raiz do projeto.

Adicione sua chave da OpenAI:
OPENAI_API_KEY=sk-sua_chave_aqui
Adicione sua chave da Gemini:
GEMINI_API_KEY = <chave_Gemini>
▶️ Uso

Execute o script principal:

python cloud_teste.py


Fluxo do programa:

O programa grava 8 segundos de áudio (ajustável no código).

O arquivo captura.wav é salvo localmente.

O áudio é enviado para a API da OpenAI.

O texto transcrito é exibido no terminal.

A transcrição também é salva em transcricao.txt.

Iniciando gravação por 8.0s...
Áudio salvo em: captura.wav
Transcrevendo na nuvem...

=== TRANSCRIÇÃO (idioma original) ===
Olá, este é um teste de transcrição com o modelo Whisper via API.

Transcrição salva em 'transcricao.txt'


📂 Estrutura do Projeto
.
├── cloud_teste.py        # Script principal
├── .env                  # Chave da API (não subir no GitHub!)
├── captura.wav           # Áudio gravado
├── transcricao.txt       # Saída da transcrição
└── requirements.txt      # Dependências do projeto

📜 Licença

Este projeto é de uso acadêmico/pessoal.
Sinta-se à vontade para modificar e adaptar conforme necessário.

Exemplo de saída no terminal:
