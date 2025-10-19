# -*- coding: utf-8 -*-
# UI em CustomTkinter (tema escuro) + tela de PRONTUÁRIO
# pip install customtkinter sounddevice soundfile openai google-generativeai httpx numpy

import os
import time
import queue
import threading
import tempfile
from datetime import datetime
import uuid
import json
import re
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import httpx

import customtkinter as ctk
from tkinter import messagebox, filedialog  # messagebox + salvar JSON

from openai import OpenAI
import google.generativeai as genai

# ========================
# CONFIGS / CHAVES
# ========================
SUPABASE_URL = "chave"   # <- url do projeto
SUPABASE_KEY = "chave"                              # <- anon (com policies) ou service_role

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
            f"{SUPABASE_URL.strip().rstrip('/')}/rest/v1/dLog",
            headers=HEADERS_SUPABASE,
            json={"log": payload},
            timeout=20,
        )
        return r.status_code, r.text
    except Exception as e:
        return 0, f"ERRO de rede ao enviar ao Supabase: {e}"

# ========================
# PARSERS robustos (Resumo Gemini + Transcrição)
# ========================
WHITES = re.compile(r"\s+") # talvez retirar em caso de bug
MD_BOLD = re.compile(r"\*\*(.*?)\*\*")

def _clean_text(t: str, keep_newlines: bool = False) -> str:
    """Limpa texto; se keep_newlines=True, preserva quebras de linha."""
    if not t:
        return ""
    # tira **negrito** e bullets comuns
    t = MD_BOLD.sub(r"\1", t).replace("•", "-").replace("–", "-").replace("—", "-").replace("\u200b", "")

    if keep_newlines:
        # normaliza finais de linha e tira espaços excedentes POR LINHA
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = "\n".join(line.strip() for line in t.split("\n"))
        # comprime espaços dentro de cada linha, mas deixa \n
        t = re.sub(r"[ \t\f\v]+", " ", t)
        # tira linhas vazias duplas
        t = "\n".join([l for l in t.split("\n") if l.strip() != ""])
        return t.strip()
    else:
        # comportamento anterior (colapsa tudo, inclusive \n)
        return re.sub(r"\s+", " ", t).strip()

def _norm(s: Optional[str]) -> str:
    # mantém o comportamento antigo (sem quebras)
    return _clean_text(s or "", keep_newlines=False)


def _take_after(label: str, line: str) -> Optional[str]:
    l = line.strip()
    if l.lower().startswith(label.lower()):
        out = l[len(label):].strip(" :.-–—")
        return out if out else None
    return None

def _find_first(patterns, text, flags=re.I):
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            g = next((g for g in m.groups() if g is not None), None)
            return g.strip() if g else None
    return None

def parse_from_gemini(summary_text: str) -> Dict[str, Any]:
    """
    Lê um resumo no formato de 7 linhas:
      Nome do Paciente: ...
      Sexo: ...
      Queixa Principal: ...
      Sintomas: ...
      Medicamentos Citados: ...
      Conduta: ...
      Próximos Passos: ...
    """
    out = {"nome": "não descrito", "sexo": "", "sintomas": ["não descrito"]}
    if not summary_text:
        return out

    # <<< preserva as quebras de linha >>>
    txt = _clean_text(summary_text, keep_newlines=True)

    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    labels = {
        "nome do paciente": "nome",
        "sexo": "sexo",
        "queixa principal": "queixa",
        "sintomas": "sintomas",
        "medicamentos citados": "meds",
        "conduta": "conduta",
        "próximos passos": "proximos",
        "proximos passos": "proximos",
    }

    parsed: Dict[str, str] = {}
    for ln in lines:
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k in labels:
            parsed[labels[k]] = v

    nome = parsed.get("nome") or "não descrito"

    sx = (parsed.get("sexo") or "").lower()
    if sx.startswith("m"):
        sexo = "M"
    elif sx.startswith("f"):
        sexo = "F"
    else:
        sexo = ""

    sint_raw = parsed.get("sintomas") or "não descrito"
    if sint_raw.lower() == "não descrito":
        sintomas = ["não descrito"]
    else:
        sintomas = [s.strip(" .;,-") for s in re.split(r"[,;]", sint_raw) if s.strip()] or ["não descrito"]

    return {"nome": nome, "sexo": sexo, "sintomas": sintomas}


def parse_from_transcript(transcript: str) -> Dict[str, Any]:
    out = {"nome": None, "sexo": None, "sintomas": []}
    if not transcript:
        return out

    t = _clean_text(transcript)

    out["nome"] = _find_first([
        r"nome do paciente (?:é|eh)\s*([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÀ-ÖØ-öø-ÿ ']+)",
        r"paciente (?:se chama|chama\-se)\s*([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÀ-ÖØ-öø-ÿ ']+)",
        r"paciente\s*[:\-]\s*([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÀ-ÖØ-öø-ÿ ']+)"
    ], t)

    s = _find_first([r"\bsexo\s+(masculino|feminino)\b"], t)
    if s:
        out["sexo"] = "M" if s.lower().startswith("m") else "F"

    s = _find_first([
        r"\b(?:queixa(?: principal)?)\s*[:\-]\s*(.+?)(?:\.|$)",
        r"\bestá\s+com\s+(.+?)\b(?:\.\s|$)",
        r"\btem\s+(.+?)\b(?:\.\s|$)"
    ], t)
    if s:
        out["sintomas"] = [i.strip(" .;,-") for i in re.split(r"[,;]", s) if i.strip()][:6]

    return out

def merge_info(pref: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    nome = (pref.get("nome") or fallback.get("nome") or "não descrito")
    sexo = (pref.get("sexo") or fallback.get("sexo") or "")
    sintomas = (pref.get("sintomas") or fallback.get("sintomas") or ["não descrito"])
    return {"nome": nome, "sexo": sexo, "sintomas": sintomas}

# ========================
#       APP (CTk)
# ========================
class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Transcrição Cloud — CTk")
        self.geometry("1000x700")
        self.minsize(940, 640)

        # Estado de captura
        self.is_recording = False
        self.stream = None
        self.buffer = []
        self.tmp_wav = None
        self.last_transcript_text = ""

        # refs do prontuário (para autopreencher)
        self.prontuario_refs = None

        # Tema inicial
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._build_layout()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_layout(self):
        # Grid raiz
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === Sidebar ===
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(self.sidebar, text="Transcrição Cloud",
                     font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=16, pady=(16, 10), sticky="w")

        self.btn_toggle = ctk.CTkButton(self.sidebar, text="Iniciar", command=self.toggle_recording)
        self.btn_toggle.grid(row=1, column=0, padx=16, pady=(4, 0), sticky="we")

        self.btn_clear = ctk.CTkButton(self.sidebar, text="Limpar", command=self.clear_text)
        self.btn_clear.grid(row=2, column=0, padx=16, pady=(8, 0), sticky="we")

        self.btn_resumo = ctk.CTkButton(self.sidebar, text="Resumo (Gemini)", command=self._resumir_gemini)
        self.btn_resumo.grid(row=3, column=0, padx=16, pady=(8, 0), sticky="we")

        self.btn_prontuario = ctk.CTkButton(self.sidebar, text="🩺 Prontuário médico", command=self.abrir_prontuario)
        self.btn_prontuario.grid(row=4, column=0, padx=16, pady=(12, 0), sticky="we")

        ctk.CTkLabel(self.sidebar, text="Tema",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(row=6, column=0, padx=16, pady=(18, 6), sticky="w")
        self.opt_tema = ctk.CTkOptionMenu(self.sidebar, values=["system", "dark", "light"], command=self._trocar_tema)
        self.opt_tema.set("system")
        self.opt_tema.grid(row=7, column=0, padx=16, pady=(0, 16), sticky="we")

        # === Área principal ===
        self.main = ctk.CTkFrame(self, corner_radius=8)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(3, weight=1)
        self.main.grid_rowconfigure(6, weight=1)

        self.lbl_status = ctk.CTkLabel(self.main, text="Status: parado")
        self.lbl_status.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="w")

        ctk.CTkLabel(self.main, text="Transcrição:").grid(row=1, column=0, padx=10, pady=(2, 2), sticky="w")
        self.txt = ctk.CTkTextbox(self.main, height=180, wrap="word")
        self.txt.grid(row=3, column=0, padx=10, pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(self.main, text="Resumo (Gemini):").grid(row=4, column=0, padx=10, pady=(6, 2), sticky="w")
        self.txt_resumo = ctk.CTkTextbox(self.main, height=160, wrap="word")
        self.txt_resumo.grid(row=6, column=0, padx=10, pady=(0, 6), sticky="nsew")

        self.lbl_tempo = ctk.CTkLabel(self.main, text="Tempo da última transcrição: -")
        self.lbl_tempo.grid(row=7, column=0, padx=10, pady=(6, 4), sticky="w")

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
                self.stream.stop(); self.stream.close()
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
                "tipo": "transcricao",
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
        # -----------------------------
    # Resumo com Gemini (sem *, bullets ou **)
    # -----------------------------
        # -----------------------------
    # Resumo com Gemini (sem *, bullets ou **)
    # -----------------------------
       # -----------------------------
    # Resumo com Gemini (7 linhas, uma por campo)
    # -----------------------------
    def _resumir_gemini(self):
        try:
            texto = (self.last_transcript_text or "").strip()
            if not texto:
                messagebox.showwarning("Resumo", "Não há transcrição recente em memória para resumir.")
                return

            resumo_prompt = (
    "Resuma o conteúdo abaixo em português do Brasil e responda EXATAMENTE com 7 linhas, "
    "uma linha por campo, no formato 'Campo: valor', nesta ordem:\n"
    "Nome do Paciente: ...\n"
    "Sexo: ...\n"
    "Queixa Principal: ...\n"
    "Sintomas: ...\n"
    "Medicamentos Citados: ...\n"
    "Conduta: ...\n"
    "Próximos Passos: ...\n\n"
    "- Use 'não descrito' quando o valor não existir.\n"
    "- Não use bullets, asteriscos, negrito, markdown ou texto extra.\n"
    "- Separe múltiplos sintomas por vírgulas.\n\n"
    f"Transcrição:\n\"\"\"\n{texto}\n\"\"\"\n"
)


            model_g = genai.GenerativeModel("gemini-2.5-flash")
            response = model_g.generate_content(resumo_prompt)
            if getattr(response, "text", None):
                resumo = response.text
            else:
                try:
                    parts = response.candidates[0].content.parts
                    resumo = "".join(getattr(p, "text", "") for p in parts) or None
                except Exception:
                    resumo = None

            if not resumo:
                resumo = "(Sem retorno do Gemini.)"
            else:
                # normaliza quebras e tira espaços extras
                resumo = resumo.replace("\r\n", "\n").replace("\r", "\n")
                resumo = "\n".join(l.strip() for l in resumo.split("\n") if l.strip())

            self.after(0, lambda: (
                self.txt_resumo.delete("1.0", "end"),
                self.txt_resumo.insert("end", resumo)
            ))
            self._ui_set_status("Resumo gerado (7 linhas, formato fixo).")

        except Exception as e:
            messagebox.showerror("Resumo (Gemini)", f"Erro ao gerar resumo: {e}")



    # -----------------------------
    # PRONTUÁRIO (UI nova)
    # -----------------------------
    def abrir_prontuario(self):
        win = ctk.CTkToplevel(self)
        win.title("Prontuário do Paciente")
        win.geometry("900x720")
        win.minsize(860, 680)

        container = ctk.CTkScrollableFrame(win, corner_radius=8)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        title = ctk.CTkLabel(container, text="📄 Prontuário do Paciente", font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 10))

        # Helpers
        def L(r, c, txt, **kw):
            lab = ctk.CTkLabel(container, text=txt, **kw)
            lab.grid(row=r, column=c, sticky="w", padx=6, pady=4)
            return lab
        def E(r, c, w=280):
            ent = ctk.CTkEntry(container, width=w)
            ent.grid(row=r, column=c, sticky="we", padx=6, pady=4)
            return ent
        def T(r, c, h=80, w=600):
            tb = ctk.CTkTextbox(container, height=h, width=w, wrap="word")
            tb.grid(row=r, column=c, columnspan=3, sticky="we", padx=6, pady=4)
            return tb

        container.grid_columnconfigure(1, weight=1)
        container.grid_columnconfigure(3, weight=1)

        # Linha 1: Nº prontuário / Data abertura
        L(1,0,"Número do prontuário:"); ent_num = E(1,1,180)
        L(1,2,"Data de abertura:"); ent_data_ab = E(1,3,180); ent_data_ab.insert(0, datetime.now().strftime("%d/%m/%Y"))

        # Linha 2: Nome completo
        L(2,0,"Nome completo:"); ent_nome = E(2,1,520)

        # Linha 3: Data nasc / Sexo
        L(3,0,"Data de nascimento:"); ent_dn = E(3,1,180)
        L(3,2,"Sexo:")
        sexo_var = ctk.StringVar(value="")
        sex_frame = ctk.CTkFrame(container, fg_color="transparent")
        sex_frame.grid(row=3, column=3, sticky="w")
        ctk.CTkRadioButton(sex_frame, text="M", variable=sexo_var, value="M").pack(side="left", padx=4)
        ctk.CTkRadioButton(sex_frame, text="F", variable=sexo_var, value="F").pack(side="left", padx=4)

        # Linha 4: Endereço
        L(4,0,"Endereço:"); ent_end = E(4,1,520)

        # Linha 5: Telefones / Email
        L(5,0,"Telefone(s):"); ent_tel = E(5,1,240)
        L(5,2,"E-mail:"); ent_email = E(5,3,240)

        # Linha 6: Responsável/cuidador
        L(6,0,"Contato do responsável/cuidador (se aplicável):")
        txt_resp = T(6,1)

        # Linha 7: Médico(s)
        L(7,0,"Médico(s) do Paciente (se houver):"); ent_med = E(7,1,520)

        # Linha 8: Escolaridade / Ocupação
        L(8,0,"Escolaridade:"); ent_escol = E(8,1,240)
        L(8,2,"Ocupação:"); ent_ocup = E(8,3,240)

        # Linha 9: Limitações (checkboxes)
        L(9,0,"Limitação:")
        lim_frame = ctk.CTkFrame(container, fg_color="transparent")
        lim_frame.grid(row=9, column=1, columnspan=3, sticky="w", padx=6, pady=4)
        lim_vars = {
            "cognitiva": ctk.BooleanVar(value=False),
            "locomocao": ctk.BooleanVar(value=False),
            "visao": ctk.BooleanVar(value=False),
            "audicao": ctk.BooleanVar(value=False),
            "outras": ctk.BooleanVar(value=False),
        }
        ctk.CTkCheckBox(lim_frame, text="Cognitiva", variable=lim_vars["cognitiva"]).pack(side="left", padx=4)
        ctk.CTkCheckBox(lim_frame, text="Locomoção", variable=lim_vars["locomocao"]).pack(side="left", padx=4)
        ctk.CTkCheckBox(lim_frame, text="Visão", variable=lim_vars["visao"]).pack(side="left", padx=4)
        ctk.CTkCheckBox(lim_frame, text="Audição", variable=lim_vars["audicao"]).pack(side="left", padx=4)
        ctk.CTkCheckBox(lim_frame, text="Outras", variable=lim_vars["outras"]).pack(side="left", padx=4)

        # Linha 10: Alergia
        L(10,0,"Alergia:")
        txt_alerg = T(10,1)

        # --- guardar refs para autopreencher ---
        self.prontuario_refs = {
            "ent_num": ent_num,
            "ent_data_ab": ent_data_ab,
            "ent_nome": ent_nome,
            "ent_dn": ent_dn,
            "sexo_var": sexo_var,
            "ent_end": ent_end,
            "ent_tel": ent_tel,
            "ent_email": ent_email,
            "txt_resp": txt_resp,
            "ent_med": ent_med,
            "ent_escol": ent_escol,
            "ent_ocup": ent_ocup,
            "lim_vars": lim_vars,
            "txt_alerg": txt_alerg,
        }

        # Botões
        btns = ctk.CTkFrame(container, fg_color="transparent")
        btns.grid(row=11, column=0, columnspan=4, sticky="w", padx=6, pady=(8,0))

        # novo: preencher a partir do Resumo/Transcrição com parsers robustos
        ctk.CTkButton(btns, text="✨ Preencher do Resumo",
                      command=self.autopreencher_prontuario_do_resumo).pack(side="left", padx=(0,8))

        ctk.CTkButton(btns, text="💾 Salvar no Supabase", command=lambda: salvar()).pack(side="left")
        ctk.CTkButton(btns, text="⬇️ Exportar JSON", command=lambda: exportar()).pack(side="left", padx=8)

        # helpers internos
        def montar_payload():
            return {
                "tipo": "prontuario",
                "versao": 1,
                "cabecalho": {
                    "data_criacao_app": datetime.utcnow().isoformat() + "Z",
                },
                "dados_paciente": {
                    "numero_prontuario": ent_num.get().strip(),
                    "data_abertura": ent_data_ab.get().strip(),
                    "nome_completo": ent_nome.get().strip(),
                    "data_nascimento": ent_dn.get().strip(),
                    "sexo": sexo_var.get(),
                    "endereco": ent_end.get().strip(),
                    "telefones": ent_tel.get().strip(),
                    "email": ent_email.get().strip(),
                    "responsavel_contato": txt_resp.get("1.0", "end").strip(),
                    "medicos": ent_med.get().strip(),
                    "escolaridade": ent_escol.get().strip(),
                    "ocupacao": ent_ocup.get().strip(),
                    "limitacoes": {k: v.get() for k, v in lim_vars.items()},
                    "alergias": txt_alerg.get("1.0", "end").strip(),
                },
                "audit": {
                    "sessao_id": str(uuid.uuid4()),
                    "criado_em": datetime.utcnow().isoformat() + "Z",
                }
            }

        def validar_minimo(p):
            if not p["dados_paciente"]["nome_completo"]:
                raise ValueError("Nome completo é obrigatório.")
            if not p["dados_paciente"]["data_abertura"]:
                raise ValueError("Data de abertura é obrigatória.")

        def salvar():
            try:
                payload = montar_payload()
                validar_minimo(payload)
                code, body = salvar_json_no_supabase(payload)
                if code == 201:
                    messagebox.showinfo("Prontuário", "Prontuário salvo no Supabase (201).")
                else:
                    messagebox.showwarning("Prontuário", f"Supabase retornou {code}:\n{body}")
            except Exception as e:
                messagebox.showerror("Prontuário", f"Erro ao salvar: {e}")

        def exportar():
            try:
                payload = montar_payload()
                validar_minimo(payload)
                fname = filedialog.asksaveasfilename(
                    title="Salvar JSON do Prontuário",
                    defaultextension=".json",
                    filetypes=[("JSON", "*.json")],
                )
                if fname:
                    with open(fname, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    messagebox.showinfo("Exportar", f"Arquivo salvo:\n{fname}")
            except Exception as e:
                messagebox.showerror("Exportar", f"Erro ao exportar: {e}")

    # -----------------------------
    # Auto-preencher via parsers (Resumo + Transcrição)
    # -----------------------------
    def autopreencher_prontuario_do_resumo(self):
        if not self.prontuario_refs:
            messagebox.showwarning("Prontuário", "Abra o prontuário primeiro.")
            return

        # Resumo: manter \n para o parser
        resumo_txt = self.txt_resumo.get("1.0", "end")
        # Transcrição: pode normalizar sem \n
        transcr_txt = _norm(self.last_transcript_text)


        if not resumo_txt and not transcr_txt:
            messagebox.showwarning("Prontuário", "Não há resumo/transcrição para extrair informações.")
            return

        try:
            from_sum = parse_from_gemini(resumo_txt)
            from_trn = parse_from_transcript(transcr_txt)
            info = merge_info(from_sum, from_trn)

            r = self.prontuario_refs

            # Preenche campos com defaults "não descrito" onde preciso
            def fill_entry(ent, value):
                ent.delete(0, "end"); ent.insert(0, value)

            # Nome
            fill_entry(r["ent_nome"], info["nome"])

            # Sexo
            if info["sexo"] in ("M", "F"):
                r["sexo_var"].set(info["sexo"])
            else:
                r["sexo_var"].set("")

            # Coloca sintomas/queixa em Alergias se estiver vazio (ou você pode criar um campo "Queixas")
            sintomas_line = ", ".join(info["sintomas"])
            current = r["txt_alerg"].get("1.0", "end").strip()
            if not current:
                r["txt_alerg"].delete("1.0", "end")
                r["txt_alerg"].insert("end", f"Sintomas/Queixa: {sintomas_line}")

            # Campos restantes: se estiverem vazios, marcam "não descrito"
            for ent in (r["ent_dn"], r["ent_end"], r["ent_tel"], r["ent_email"],
                        r["ent_med"], r["ent_escol"], r["ent_ocup"]):
                if not ent.get().strip():
                    ent.insert(0, "não descrito")

            messagebox.showinfo("Prontuário", "Campos preenchidos a partir do Resumo/Transcrição.")
        except Exception as e:
            messagebox.showerror("Prontuário", f"Falha ao extrair dados: {e}")

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
