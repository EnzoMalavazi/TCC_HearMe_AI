# -*- coding: utf-8 -*-
# UI em CustomTkinter (tema escuro) no layout do print
# pip install customtkinter sounddevice soundfile openai google-generativeai httpx numpy

import os
import time
import queue
import threading
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

import customtkinter as ctk
from tkinter import messagebox  # pode continuar usando os messagebox nativos

from openai import OpenAI
import google.generativeai as genai

# --- utilidades/API externas ---
import uuid
import httpx
import json

# ========================
# CONFIGS / CHAVES
# ========================
SUPABASE_URL = "chave"   # <- url do projeto
SUPABASE_KEY = "chave
OPENAI_API_KEY = "chave"                # <- chave da OpenAI
GEMINI_API_KEY = "chave"                # <- chave do Google Gemini 

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Defina SUPABASE_URL e SUPABASE_KEY no topo do arquivo.")
if not OPENAI_API_KEY:
    raise RuntimeError("Defina OPENAI_API_KEY no topo do arquivo.")

HEADERS_SUPABASE = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

# -----------------------------
# Configurações de áudio
# -----------------------------
SAMPLERATE = 16000   # 16 kHz
CHANNELS = 1

# -----------------------------
# Clientes de API
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def transcrever_via_api(
    audio_path: str,
    modelo: str = "gpt-4o-transcribe",
    prompt: str = "Contexto: consulta médica, português do Brasil.",
):
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=modelo,
            file=f,
            prompt=prompt,
        )
    return resp.text if hasattr(resp, "text") else str(resp)


def salvar_json_no_supabase(payload: dict) -> tuple[int, str]:
    try:
        r = httpx.post(
            f"{SUPABASE_URL}/rest/v1/dLog",
            headers=HEADERS_SUPABASE,
            json={"log": payload},
            timeout=20,
        )
        return r.status_code, r.text
    except Exception as e:
        return 0, f"ERRO de rede ao enviar ao Supabase: {e}"


# ========================
#       APP (CTk)
# ========================
class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Transcrição Cloud — CTk")
        self.geometry("920x620")
        self.minsize(880, 560)

        # Estado de captura
        self.is_recording = False
        self.stream = None
        self.buffer = []
        self.tmp_wav = None
        self.last_transcript_text = ""

        # Tema inicial
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._build_layout()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_layout(self):
        # Grid raiz: 1 linha, 2 colunas (sidebar fixa + conteúdo)
        self.grid_columnconfigure(0, weight=0)  # sidebar
        self.grid_columnconfigure(1, weight=1)  # conteúdo
        self.grid_rowconfigure(0, weight=1)

        # === Sidebar (esquerda) ===
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_rowconfigure(6, weight=1)  # empurra o restante para cima

        titulo = ctk.CTkLabel(self.sidebar, text="Transcrição Cloud", font=ctk.CTkFont(size=18, weight="bold"))
        titulo.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="w")

        self.btn_toggle = ctk.CTkButton(self.sidebar, text="Iniciar", command=self.toggle_recording)
        self.btn_toggle.grid(row=1, column=0, padx=16, pady=(4, 0), sticky="we")

        self.btn_clear = ctk.CTkButton(self.sidebar, text="Limpar", command=self.clear_text)
        self.btn_clear.grid(row=2, column=0, padx=16, pady=(8, 0), sticky="we")

        self.btn_resumo = ctk.CTkButton(self.sidebar, text="Resumo (Gemini)", command=self._resumir_gemini)
        self.btn_resumo.grid(row=3, column=0, padx=16, pady=(8, 0), sticky="we")

        # Tema
        tema_lbl = ctk.CTkLabel(self.sidebar, text="Tema", font=ctk.CTkFont(size=12, weight="bold"))
        tema_lbl.grid(row=4, column=0, padx=16, pady=(18, 6), sticky="w")
        self.opt_tema = ctk.CTkOptionMenu(
            self.sidebar,
            values=["system", "dark", "light"],
            command=self._trocar_tema
        )
        self.opt_tema.set("system")
        self.opt_tema.grid(row=5, column=0, padx=16, pady=(0, 16), sticky="we")

        # === Área principal (direita) ===
        self.main = ctk.CTkFrame(self, corner_radius=8)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        # linhas para se expandirem
        self.main.grid_rowconfigure(3, weight=1)
        self.main.grid_rowconfigure(6, weight=1)

        self.lbl_status = ctk.CTkLabel(self.main, text="Status: parado")
        self.lbl_status.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="w")

        lbl_tr = ctk.CTkLabel(self.main, text="Transcrição:")
        lbl_tr.grid(row=1, column=0, padx=10, pady=(2, 2), sticky="w")

        self.txt = ctk.CTkTextbox(self.main, height=180, wrap="word")
        self.txt.grid(row=3, column=0, padx=10, pady=(0, 6), sticky="nsew")

        lbl_rs = ctk.CTkLabel(self.main, text="Resumo (Gemini):")
        lbl_rs.grid(row=4, column=0, padx=10, pady=(6, 2), sticky="w")

        self.txt_resumo = ctk.CTkTextbox(self.main, height=160, wrap="word")
        self.txt_resumo.grid(row=6, column=0, padx=10, pady=(0, 6), sticky="nsew")

        self.lbl_tempo = ctk.CTkLabel(self.main, text="Tempo da última transcrição: -")
        self.lbl_tempo.grid(row=7, column=0, padx=10, pady=(6, 4), sticky="w")

        # Progress bar (decorativa; setamos 0/indeterminado ao processar)
        self.progress = ctk.CTkProgressBar(self.main)
        self.progress.set(0)
        self.progress.grid(row=8, column=0, padx=10, pady=(0, 10), sticky="we")

    # -----------------------------
    # Tema
    # -----------------------------
    def _trocar_tema(self, value: str):
        ctk.set_appearance_mode(value)

    # -----------------------------
    # Captura — Start/Stop
    # -----------------------------
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.buffer.clear()
        self.tmp_wav = None

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Áudio", f"Erro ao iniciar captura: {e}")
            return

        self.is_recording = True
        self.btn_toggle.configure(text="Parar")
        self._ui_set_status("Status: gravando…")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

    def stop_recording(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.is_recording = False
        self.btn_toggle.configure(text="Iniciar")
        self._ui_set_status("Status: processando…")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

        threading.Thread(target=self._finalizar_e_transcrever, daemon=True).start()

    # -----------------------------
    # Callbacks e pipeline
    # -----------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        self.buffer.append(indata.copy())

    def _finalizar_e_transcrever(self):
        t0 = time.time()
        try:
            if not self.buffer:
                self._ui_set_status("Status: nenhum áudio capturado.")
                self._stop_progress()
                return

            audio = np.concatenate(self.buffer, axis=0)
            audio = np.squeeze(audio)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                self.tmp_wav = tmp.name

            audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            sf.write(self.tmp_wav, audio_i16, SAMPLERATE, subtype="PCM_16")

            # Transcreve
            texto = transcrever_via_api(self.tmp_wav)

            # Envia log ao Supabase
            duracao_audio_seg = float(len(audio) / SAMPLERATE)
            payload = {
                "sessao_id": str(uuid.uuid4()),
                "modelo_asr": "gpt-4o-transcribe",
                "lingua": "pt-BR",
                "duracao_seg": duracao_audio_seg,
                "status": "ok" if (texto and len(texto.strip()) > 0) else "vazio",
                "transcricao": texto or "",
                "metadados": {"samplerate": SAMPLERATE, "channels": CHANNELS},
                "criado_em": datetime.utcnow().isoformat() + "Z",
            }
            status_code, body = salvar_json_no_supabase(payload)

            # Atualiza UI
            self.last_transcript_text = texto or ""
            dt = time.time() - t0
            self._ui_update_transcricao(texto or "")

            if status_code == 201:
                self._ui_set_status("Status: pronto — transcrição enviada ao Supabase (201).")
            else:
                self._ui_set_status(f"Status: atenção — Supabase retornou {status_code}: {body}")

            self._ui_set_tempo(dt)

        except Exception as e:
            self._ui_update_transcricao(f"[ERRO] {e}")
            self._ui_set_status("Status: erro ao transcrever.")
        finally:
            try:
                if self.tmp_wav and os.path.exists(self.tmp_wav):
                    os.remove(self.tmp_wav)
            except Exception:
                pass
            self.tmp_wav = None
            self._stop_progress()

    # -----------------------------
    # Resumo com Gemini
    # -----------------------------
    def _resumir_gemini(self):
        try:
            texto = (self.last_transcript_text or "").strip()
            if not texto:
                messagebox.showwarning("Resumo", "Não há transcrição recente em memória para resumir.")
                return

            resumo_prompt = (
                "Resuma de forma objetiva e estruturada o conteúdo abaixo em português do Brasil.\n"
                "- Foque em: queixas principais, sinais/sintomas, medicamentos citados, condutas e próximos passos.\n"
                "- Use no máximo 8 tópicos curtos.\n"
                "- Não invente informações.\n\n"
                "Transcrição:\n"
                f"\"\"\"\n{texto}\n\"\"\"\n"
            )

            model_g = genai.GenerativeModel("gemini-2.5-flash")
            response = model_g.generate_content(resumo_prompt)
            resumo = response.text or "(Sem retorno do Gemini.)"

            self.after(0, lambda: (
                self.txt_resumo.delete("1.0", "end"),
                self.txt_resumo.insert("end", resumo)
            ))
            self._ui_set_status("Resumo gerado a partir da transcrição em memória.")

        except Exception as e:
            messagebox.showerror("Resumo (Gemini)", f"Erro ao gerar resumo: {e}")

    # -----------------------------
    # Helpers de UI
    # -----------------------------
    def _ui_update_transcricao(self, texto: str):
        self.after(0, lambda: (self.txt.delete("1.0", "end"), self.txt.insert("end", texto + "\n")))

    def _ui_set_status(self, texto: str):
        self.after(0, lambda: self.lbl_status.configure(text=texto))

    def _ui_set_tempo(self, seconds: float):
        self.after(0, lambda: self.lbl_tempo.configure(text=f"Tempo da última transcrição: {seconds:.2f}s"))

    def _stop_progress(self):
        self.after(0, lambda: (self.progress.stop(), self.progress.configure(mode="determinate"), self.progress.set(0)))

    def clear_text(self):
        self.txt.delete("1.0", "end")
        self.txt_resumo.delete("1.0", "end")
        self.lbl_tempo.configure(text="Tempo da última transcrição: -")


if __name__ == "__main__":
    app = App()
    app.mainloop()
