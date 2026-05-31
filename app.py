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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 680px;
}

/* ── Hero ─────────────────────────────── */
.hero {
    position: relative;
    border-radius: 28px;
    overflow: hidden;
    padding: 2.8rem 2.2rem 2.2rem;
    margin-bottom: 2rem;
    background: #16123A;
}
.hero-bg {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 80% at 110% -10%, #7C6FE0 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at -10% 110%, #A78BFA 0%, transparent 55%);
    pointer-events: none;
    border-radius: 28px;
}
.eyebrow {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 10.5px;
    font-weight: 500;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #AFA9EC;
    margin-bottom: 14px;
}
.eyebrow-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #7F77DD;
    display: inline-block;
}
.hero-title {
    position: relative;
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.0;
    letter-spacing: -1.5px;
    margin-bottom: 10px;
}
.hero-title span { color: #AFA9EC; }
.hero-sub {
    position: relative;
    font-size: 13.5px;
    color: rgba(255,255,255,0.5);
    line-height: 1.7;
    max-width: 400px;
    margin-bottom: 0;
}
.status-pill {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 20px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 7px 16px;
    font-size: 12px;
    color: rgba(255,255,255,0.65);
}
.pulse {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    display: inline-block;
    position: relative;
}

/* ── Section label ─────────────────────── */
.sec-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}

/* ── Textarea override ─────────────────── */
textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.7 !important;
    color: #1a1a2e !important;
    background: #fff !important;
    border: 1.5px solid #DDD9F8 !important;
    border-radius: 16px !important;
}
textarea:focus {
    border-color: #7C6FE0 !important;
    box-shadow: 0 0 0 4px rgba(124,111,224,0.1) !important;
}

/* ── Streamlit button override ─────────── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 14px 20px;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
    color: #fff !important;
    background: #534AB7 !important;
    border: none !important;
    border-radius: 14px !important;
    transition: background .15s, transform .1s !important;
}
div[data-testid="stButton"] > button:hover {
    background: #3C3489 !important;
    border: none !important;
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.98) !important;
}

/* ── Result cards ──────────────────────── */
.result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}
.result-icon {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: #EEEDFE;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #534AB7;
    font-size: 20px;
    flex-shrink: 0;
}
.result-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 19px;
    font-weight: 700;
    color: #16123A;
}
.result-sub {
    font-size: 12.5px;
    color: #888;
    margin-top: 2px;
}
.cards-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 14px;
}
.rcard {
    border-radius: 20px;
    padding: 20px 18px;
    border: 1.5px solid transparent;
}
.rcard-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.rcard-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 19px;
    font-weight: 700;
    line-height: 1.2;
    word-break: break-word;
}
.rcard-conf {
    font-size: 12px;
    margin-top: 6px;
}

/* card variants */
.c-distort { background: #EEEDFE; border-color: #AFA9EC; }
.c-distort .rcard-label { color: #534AB7; }
.c-distort .rcard-value { color: #26215C; }
.c-distort .rcard-conf  { color: #534AB7; }

.c-ok { background: #E1F5EE; border-color: #9FE1CB; }
.c-ok .rcard-label { color: #0F6E56; }
.c-ok .rcard-value { color: #04342C; }
.c-ok .rcard-conf  { color: #0F6E56; }

.c-type { background: #16123A; border-color: #3C3489; }
.c-type .rcard-label { color: #AFA9EC; }
.c-type .rcard-value { color: #EEEDFE; }
.c-type .rcard-conf  { color: #7F77DD; }

.c-neutral { background: #f9f9f9; border-color: #eee; }
.c-neutral .rcard-label,
.c-neutral .rcard-conf  { color: #aaa; }
.c-neutral .rcard-value { color: #bbb; }

/* ── Insight panel ─────────────────────── */
.insight {
    border-radius: 20px;
    padding: 22px;
    border: 1.5px solid #CECBF6;
    background: #EEEDFE;
    margin-bottom: 12px;
}
.insight-top {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.insight-badge {
    background: #534AB7;
    color: #EEEDFE;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 999px;
}
.insight-name {
    font-family: 'Syne', sans-serif !important;
    font-size: 16px;
    font-weight: 700;
    color: #26215C;
}
.insight-desc {
    font-size: 13.5px;
    color: #3C3489;
    line-height: 1.65;
}
.bar-wrap { margin-top: 16px; }
.bar-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #534AB7;
    margin-bottom: 6px;
    font-weight: 500;
}
.bar-track {
    height: 6px;
    background: #CECBF6;
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: #534AB7;
}

/* ── OK banner ─────────────────────────── */
.ok-banner {
    border-radius: 20px;
    padding: 20px 22px;
    border: 1.5px solid #9FE1CB;
    background: #E1F5EE;
    margin-bottom: 12px;
}
.ok-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 16px;
    font-weight: 700;
    color: #04342C;
    margin-bottom: 4px;
}
.ok-desc {
    font-size: 13px;
    color: #0F6E56;
    line-height: 1.6;
}

/* ── Divider ───────────────────────────── */
.custom-hr {
    border: none;
    border-top: 1px solid #E8E5FA;
    margin: 1.75rem 0;
}

/* ── Footer ────────────────────────────── */
.mt-footer {
    text-align: center;
    font-size: 11px;
    color: #aaa;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #EAE8FA;
    letter-spacing: .04em;
}

/* ── Expander tweak ────────────────────── */
div[data-testid="stExpander"] {
    border: 1.5px solid #DDD9F8 !important;
    border-radius: 12px !important;
    overflow: hidden;
    margin-top: 10px;
    background: #fff;
}
div[data-testid="stExpander"] summary {
    font-size: 13px !important;
    color: #666 !important;
    padding: 12px 16px !important;
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
  <div class="hero-bg"></div>
  <div class="eyebrow"><span class="eyebrow-dot"></span>Cognitive Intelligence</div>
  <div class="hero-title">Mind<span>Trace</span></div>
  <div class="hero-sub">
    Deteksi dan klasifikasi cognitive distortion dari teks bahasa Indonesia
    menggunakan two-stage IndoBERT + LoRA.
  </div>
  <div class="status-pill"><span class="pulse"></span>Model siap digunakan</div>
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
    height=120,
    label_visibility="collapsed",
    key="main_input",
)

# Sample texts
with st.expander("Coba teks contoh"):
    for s in SAMPLES:
        if st.button(s, key=f"s_{s}", use_container_width=True):
            st.session_state.txt = s
            st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
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

        # Result header
        result_sub = ("Ditemukan pola pikir terdistorsi."
                      if lbin == "Ya" else "Tidak ada distorsi yang terdeteksi.")
        st.markdown(f"""
<div class="result-header">
  <div class="result-icon">🧠</div>
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

        # Insight / OK banner
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

        # Technical detail
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