import streamlit as st
import torch
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

st.set_page_config(page_title="MindTrace", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

/* ══════════════════════════════════════════
   FORCE LIGHT MODE + BASE RESET
   ══════════════════════════════════════════ */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
section.main, section.main > div,
.block-container {
    background-color: #EEEDF8 !important;
    color: #16123A !important;
}
.stMarkdown p, .stMarkdown span, .stMarkdown div,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.element-container p, .element-container span {
    background: transparent !important;
    box-shadow: none !important;
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 720px;
}

/* ══════════════════════════════════════════
   HERO — dark card with gradient
   ══════════════════════════════════════════ */
.hero {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    padding: 2.6rem 2.4rem 2.4rem;
    margin-bottom: 2.6rem;
    background: #1A1545;
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 90% 90% at 105% -5%, #6C5FC7 0%, transparent 55%),
        radial-gradient(ellipse 70% 70% at -5% 105%, #8B6FD4 0%, transparent 50%);
    border-radius: 24px;
    pointer-events: none;
}
.hero-eyebrow {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #AFA9EC !important;
    -webkit-text-fill-color: #AFA9EC !important;
    margin-bottom: 18px;
}
.hero-eyebrow-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7B74C9;
    display: inline-block;
    flex-shrink: 0;
}
.hero-title {
    position: relative;
    font-family: 'Syne', sans-serif !important;
    font-size: 3.8rem;
    font-weight: 800;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    line-height: 1;
    letter-spacing: -2px;
    margin-bottom: 18px;
}
.hero-sub {
    position: relative;
    font-size: 13.5px;
    color: rgba(255,255,255,0.55) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.55) !important;
    line-height: 1.75;
    max-width: 440px;
    margin-bottom: 26px;
}
.hero-status {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    color: rgba(255,255,255,0.65) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.65) !important;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 999px;
    padding: 7px 16px;
}
.hero-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4ade80;
    display: inline-block;
    flex-shrink: 0;
}

/* ══════════════════════════════════════════
   SECTION LABEL
   ══════════════════════════════════════════ */
.sec-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7B74C9 !important;
    -webkit-text-fill-color: #7B74C9 !important;
    margin-bottom: 10px;
}

/* ══════════════════════════════════════════
   TEXTAREA
   ══════════════════════════════════════════ */
textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.7 !important;
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
    background: #fff !important;
    border: 1.5px solid #C8C4EF !important;
    border-radius: 14px !important;
    caret-color: #534AB7 !important;
}
textarea::placeholder {
    color: #B0ADCE !important;
    -webkit-text-fill-color: #B0ADCE !important;
}
textarea:focus {
    border-color: #534AB7 !important;
    box-shadow: 0 0 0 3px rgba(83,74,183,0.12) !important;
}
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] textarea,
.stTextArea textarea {
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
    background: #fff !important;
}

/* ══════════════════════════════════════════
   BUTTON
   ══════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 15px 20px;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    background: #534AB7 !important;
    border: none !important;
    border-radius: 14px !important;
    transition: background .15s, transform .1s !important;
    cursor: pointer;
}
div[data-testid="stButton"] > button:hover {
    background: #3C3489 !important;
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.985) !important;
}

/* ══════════════════════════════════════════
   DIVIDER
   ══════════════════════════════════════════ */
.custom-hr {
    border: none;
    border-top: 1px solid #C8C4EF;
    margin: 2rem 0;
}

/* ══════════════════════════════════════════
   RESULT HEADER
   ══════════════════════════════════════════ */
.result-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
}
.result-emoji {
    font-size: 2rem;
    line-height: 1;
    flex-shrink: 0;
}
.result-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 22px;
    font-weight: 700;
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
    line-height: 1.2;
}
.result-sub {
    font-size: 13px;
    color: #7B74C9 !important;
    -webkit-text-fill-color: #7B74C9 !important;
    margin-top: 3px;
}

/* ══════════════════════════════════════════
   RESULT CARDS — outline style
   ══════════════════════════════════════════ */
.cards-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
}
.rcard {
    border-radius: 18px;
    padding: 22px 20px;
    border: 1.5px solid #C8C4EF;
    background: #fff;
}
.rcard-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
    color: #9590C8 !important;
    -webkit-text-fill-color: #9590C8 !important;
}
.rcard-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.25;
    word-break: break-word;
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
}
.rcard-conf {
    font-size: 12.5px;
    margin-top: 8px;
    color: #534AB7 !important;
    -webkit-text-fill-color: #534AB7 !important;
}

/* distorsi card — subtle left accent */
.c-distort {
    border-color: #C8C4EF;
    border-left: 4px solid #534AB7;
}
.c-distort .rcard-value {
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
}

/* ok / tidak ada card */
.c-ok {
    border-color: #A8DEC9;
    border-left: 4px solid #22B07D;
}
.c-ok .rcard-label {
    color: #22B07D !important;
    -webkit-text-fill-color: #22B07D !important;
}
.c-ok .rcard-value {
    color: #0A3D2C !important;
    -webkit-text-fill-color: #0A3D2C !important;
}
.c-ok .rcard-conf {
    color: #22B07D !important;
    -webkit-text-fill-color: #22B07D !important;
}

/* jenis distorsi — ghost/muted style like reference */
.c-type {
    border-color: #C8C4EF;
    background: #F5F4FC;
}
.c-type .rcard-label {
    color: #9590C8 !important;
    -webkit-text-fill-color: #9590C8 !important;
}
.c-type .rcard-value {
    color: #534AB7 !important;
    -webkit-text-fill-color: #534AB7 !important;
}
.c-type .rcard-conf {
    color: #9590C8 !important;
    -webkit-text-fill-color: #9590C8 !important;
}

/* neutral / no type detected */
.c-neutral {
    border-color: #E0DFF5;
    background: #F9F8FE;
}
.c-neutral .rcard-label,
.c-neutral .rcard-conf {
    color: #C0BCDF !important;
    -webkit-text-fill-color: #C0BCDF !important;
}
.c-neutral .rcard-value {
    color: #C0BCDF !important;
    -webkit-text-fill-color: #C0BCDF !important;
}

/* ══════════════════════════════════════════
   INSIGHT PANEL — flat, border only
   ══════════════════════════════════════════ */
.insight {
    border-radius: 18px;
    padding: 24px;
    border: 1.5px solid #C8C4EF;
    background: #fff;
    margin-bottom: 14px;
}
.insight-top {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.insight-badge {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #534AB7 !important;
    -webkit-text-fill-color: #534AB7 !important;
    background: #EEEDFB;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid #C8C4EF;
}
.insight-name {
    font-family: 'Syne', sans-serif !important;
    font-size: 17px;
    font-weight: 700;
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
}
.insight-desc {
    font-size: 14px;
    color: #5C578F !important;
    -webkit-text-fill-color: #5C578F !important;
    line-height: 1.7;
}
.bar-wrap { margin-top: 20px; }
.bar-meta {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #9590C8 !important;
    -webkit-text-fill-color: #9590C8 !important;
    margin-bottom: 8px;
}
.bar-track {
    height: 5px;
    background: #E0DFF5;
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: #534AB7;
}

/* ══════════════════════════════════════════
   OK BANNER
   ══════════════════════════════════════════ */
.ok-banner {
    border-radius: 18px;
    padding: 22px 24px;
    border: 1.5px solid #A8DEC9;
    background: #fff;
    margin-bottom: 14px;
}
.ok-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 17px;
    font-weight: 700;
    color: #0A3D2C !important;
    -webkit-text-fill-color: #0A3D2C !important;
    margin-bottom: 5px;
}
.ok-desc {
    font-size: 13.5px;
    color: #22B07D !important;
    -webkit-text-fill-color: #22B07D !important;
    line-height: 1.6;
}

/* ══════════════════════════════════════════
   EXPANDER
   ══════════════════════════════════════════ */
div[data-testid="stExpander"] {
    border: 1.5px solid #C8C4EF !important;
    border-radius: 14px !important;
    overflow: hidden;
    background: #fff !important;
}
div[data-testid="stExpander"] summary {
    font-size: 13.5px !important;
    color: #534AB7 !important;
    -webkit-text-fill-color: #534AB7 !important;
    font-weight: 500 !important;
    padding: 14px 18px !important;
    background: #fff !important;
}
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] td,
div[data-testid="stExpander"] th {
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
    background: transparent !important;
}
div[data-testid="stExpander"] table {
    width: 100%;
    border-collapse: collapse;
    background: #fff !important;
}
div[data-testid="stExpander"] thead tr th {
    background: #F5F4FC !important;
    color: #534AB7 !important;
    -webkit-text-fill-color: #534AB7 !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 16px !important;
    border-bottom: 1.5px solid #E0DFF5 !important;
    text-align: left !important;
}
div[data-testid="stExpander"] tbody tr td {
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid #F0EEF9 !important;
    background: #fff !important;
    vertical-align: top;
    word-break: break-word;
}
div[data-testid="stExpander"] tbody tr:last-child td {
    border-bottom: none !important;
}
div[data-testid="stExpander"] tbody tr:nth-child(even) td {
    background: #FAFAFE !important;
}
/* Sample buttons inside expander */
div[data-testid="stExpander"] div[data-testid="stButton"] > button {
    background: #F5F4FC !important;
    color: #16123A !important;
    -webkit-text-fill-color: #16123A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-align: left !important;
    border: 1px solid #E0DFF5 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 4px !important;
}
div[data-testid="stExpander"] div[data-testid="stButton"] > button:hover {
    background: #EEEDFB !important;
    border-color: #C8C4EF !important;
}

/* ══════════════════════════════════════════
   FOOTER
   ══════════════════════════════════════════ */
.mt-footer {
    text-align: center;
    font-size: 11.5px;
    color: #9590C8 !important;
    -webkit-text-fill-color: #9590C8 !important;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #C8C4EF;
    letter-spacing: .04em;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
M1_PATH    = "./model_1_binary"
M2_PATH    = "./model_2_multiclass"
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESCRIPTIONS = {
    "All-or-nothing"           : "Berpikir dalam kategori hitam-putih, tanpa melihat nuansa di antaranya.",
    "Discounting the positives": "Mengabaikan atau meremehkan hal-hal positif yang nyata terjadi.",
    "Emotional Reasoning"      : "Menganggap perasaan negatif sebagai kebenaran faktual.",
    "Jumping to Conclusions"   : "Mengambil kesimpulan negatif tanpa bukti yang memadai.",
    "Labeling"                 : "Memberi label negatif secara menyeluruh pada diri sendiri atau orang lain.",
    "Mental filter"            : "Fokus berlebihan pada satu detail negatif, mengabaikan gambaran besar.",
    "Overgeneralization"       : "Menarik kesimpulan luas dari satu kejadian buruk.",
    "Personalization and Blame": "Menyalahkan diri sendiri atas hal-hal di luar kendali.",
    "Should statement"         : "Menetapkan standar kaku dengan kata 'harus' atau 'seharusnya'.",
}

SAMPLES = [
    "Aku merasa semua orang pasti membenciku.",
    "Kalau aku gagal sekali, berarti aku tidak akan pernah berhasil.",
    "Aku harus selalu sempurna dalam semua hal.",
    "Dia tidak membalas chatku, pasti dia marah padaku.",
    "Hari ini aku makan bersama keluarga dan merasa senang.",
]

# ── Helpers ────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource(show_spinner=False)
def load_models():
    le1    = joblib.load(f"{M1_PATH}/label_encoder.pkl")
    cfg1   = PeftConfig.from_pretrained(M1_PATH)
    base1  = AutoModelForSequenceClassification.from_pretrained(
        cfg1.base_model_name_or_path, num_labels=len(le1.classes_),
        ignore_mismatched_sizes=True)
    model1 = PeftModel.from_pretrained(base1, M1_PATH).eval().to(DEVICE)
    tok    = AutoTokenizer.from_pretrained(M1_PATH)

    le2    = joblib.load(f"{M2_PATH}/label_encoder.pkl")
    cfg2   = PeftConfig.from_pretrained(M2_PATH)
    base2  = AutoModelForSequenceClassification.from_pretrained(
        cfg2.base_model_name_or_path, num_labels=len(le2.classes_),
        ignore_mismatched_sizes=True)
    model2 = PeftModel.from_pretrained(base2, M2_PATH).eval().to(DEVICE)

    return tok, model1, le1, model2, le2

def predict(text, tok, model, le):
    inp = tok(clean_text(text), return_tensors="pt", truncation=True,
               padding=True, max_length=MAX_LENGTH)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inp).logits, dim=1).squeeze().cpu().numpy()
    pid = int(np.argmax(probs))
    return le.inverse_transform([pid])[0], float(probs[pid])

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow"><span class="hero-eyebrow-dot"></span>Cognitive Intelligence</div>
  <div class="hero-title">MindTrace</div>
  <div class="hero-sub">
    Deteksi dan klasifikasi cognitive distortion dari teks bahasa Indonesia
    menggunakan two-stage IndoBERT + LoRA.
  </div>
  <div class="hero-status">
    <span class="hero-dot"></span>Model siap digunakan
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────
with st.spinner("Memuat model..."):
    tok, model1, le1, model2, le2 = load_models()

# ── Input ──────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Teks analisis</div>', unsafe_allow_html=True)

if "txt" not in st.session_state:
    st.session_state.txt = ""

text_input = st.text_area(
    label="",
    value=st.session_state.txt,
    placeholder="Tulis atau tempelkan teks di sini...",
    height=130,
    label_visibility="collapsed",
    key="main_input",
)

with st.expander("Coba teks contoh"):
    for s in SAMPLES:
        if st.button(s, key=f"s_{s}", use_container_width=True):
            st.session_state.txt = s
            st.rerun()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
run = st.button("Analisis teks", use_container_width=True, type="primary")

# ── Result ─────────────────────────────────────────────────────
if run:
    text_input = st.session_state.get("main_input", text_input)
    if not text_input.strip():
        st.warning("Teks tidak boleh kosong.")
    else:
        with st.spinner("Menganalisis..."):
            lbin, cbin = predict(text_input, tok, model1, le1)
            lmul, cmul = (predict(text_input, tok, model2, le2)
                          if lbin == "Ya" else (None, None))

        st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

        result_sub = ("Ditemukan pola pikir terdistorsi."
                      if lbin == "Ya" else "Tidak ada distorsi yang terdeteksi.")
        st.markdown(f"""
<div class="result-header">
  <div class="result-emoji">🧠</div>
  <div>
    <div class="result-title">Hasil analisis</div>
    <div class="result-sub">{result_sub}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Cards
        cls1  = "c-distort" if lbin == "Ya" else "c-ok"
        val1  = "Ada distorsi" if lbin == "Ya" else "Tidak ada"
        conf1 = f"Confidence {cbin:.1%}"

        if lmul:
            cls2  = "c-type"
            val2  = lmul
            conf2 = f"Confidence {cmul:.1%}"
        else:
            cls2  = "c-neutral"
            val2  = "—"
            conf2 = "Tidak terdeteksi"

        st.markdown(f"""
<div class="cards-grid">
  <div class="rcard {cls1}">
    <div class="rcard-label">Deteksi</div>
    <div class="rcard-value">{val1}</div>
    <div class="rcard-conf">{conf1}</div>
  </div>
  <div class="rcard {cls2}">
    <div class="rcard-label">Jenis distorsi</div>
    <div class="rcard-value">{val2}</div>
    <div class="rcard-conf">{conf2}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        if lmul:
            desc = DESCRIPTIONS.get(lmul, "")
            pct  = int(cmul * 100)
            st.markdown(f"""
<div class="insight">
  <div class="insight-top">
    <span class="insight-badge">Distorsi</span>
    <span class="insight-name">{lmul}</span>
  </div>
  <div class="insight-desc">{desc}</div>
  <div class="bar-wrap">
    <div class="bar-meta">
      <span>Keyakinan model</span><span>{pct}%</span>
    </div>
    <div class="bar-track">
      <div class="bar-fill" style="width:{pct}%"></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        elif lbin == "Tidak":
            st.markdown("""
<div class="ok-banner">
  <div class="ok-title">✓ Tidak ada distorsi terdeteksi</div>
  <div class="ok-desc">Pola pikir dalam teks tampak sehat dan realistis. Pertahankan!</div>
</div>
""", unsafe_allow_html=True)

        with st.expander("Detail teknis"):
            st.markdown(f"""
| Field | Value |
|---|---|
| Teks asli | {text_input[:120]}{"…" if len(text_input)>120 else ""} |
| Setelah preprocessing | {clean_text(text_input)[:100]} |
| Model 1 output | {lbin} ({cbin:.4f}) |
| Model 2 output | {f"{lmul} ({cmul:.4f})" if lmul else "—"} |
| Device | {str(DEVICE).upper()} |
""")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="mt-footer">
  MindTrace &nbsp;·&nbsp; IndoLEM-IndoBERT + LoRA &nbsp;·&nbsp;
  Dataset: Cognitive Distortion Bahasa Indonesia
</div>
""", unsafe_allow_html=True)