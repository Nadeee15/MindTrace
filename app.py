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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 720px; }

/* ── Hero ───────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, #5b5ef4 0%, #8b5cf6 100%);
    border-radius: 14px;
    padding: 2.4rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-sub {
    color: rgba(255,255,255,0.82);
    font-size: 0.92rem;
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* ── Status ─────────────────────────────── */
.status-ok {
    background: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1.6rem;
}

/* ── Section label ──────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

/* ── Result cards ───────────────────────── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.2rem;
}
.rcard {
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid transparent;
}
.rcard-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.rcard-value {
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1.2;
}
.rcard-conf {
    font-size: 0.75rem;
    margin-top: 0.35rem;
    opacity: 0.7;
}

.rcard-red   { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
.rcard-green { background: #f0fdf4; border-color: #86efac; color: #166534; }
.rcard-blue  { background: #eff6ff; border-color: #bfdbfe; color: #1e40af; }
.rcard-gray  { background: #f9fafb; border-color: #e5e7eb; color: #4b5563; }

/* ── Distortion desc ────────────────────── */
.desc-wrap {
    border-left: 3px solid #8b5cf6;
    background: #faf5ff;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
    color: #3b0764;
    font-size: 0.875rem;
    line-height: 1.6;
}
.desc-name {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

/* ── Confidence bar ─────────────────────── */
.conf-wrap { margin-top: 1.1rem; }
.conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 5px;
}
.bar-bg {
    background: #e5e7eb;
    border-radius: 99px;
    height: 6px;
}
.bar-fg {
    background: linear-gradient(90deg, #5b5ef4, #8b5cf6);
    border-radius: 99px;
    height: 6px;
}

/* ── Ok banner ──────────────────────────── */
.ok-banner {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #166534;
    font-size: 0.875rem;
    margin-top: 1rem;
    line-height: 1.5;
}

/* ── Detail table ───────────────────────── */
.detail-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    color: #374151;
}
.detail-table td {
    padding: 0.45rem 0.7rem;
    border-bottom: 1px solid #f3f4f6;
    vertical-align: top;
}
.detail-table td:first-child {
    font-weight: 600;
    color: #6b7280;
    white-space: nowrap;
    width: 38%;
}

/* ── Footer ─────────────────────────────── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #f3f4f6;
}

/* ── Sample button tweak ────────────────── */
div[data-testid="stButton"] button {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource(show_spinner=False)
def load_models():
    le1     = joblib.load(f"{M1_PATH}/label_encoder.pkl")
    cfg1    = PeftConfig.from_pretrained(M1_PATH)
    base1   = AutoModelForSequenceClassification.from_pretrained(
        cfg1.base_model_name_or_path, num_labels=len(le1.classes_),
        ignore_mismatched_sizes=True)
    model1  = PeftModel.from_pretrained(base1, M1_PATH).eval().to(DEVICE)
    tok     = AutoTokenizer.from_pretrained(M1_PATH)

    le2     = joblib.load(f"{M2_PATH}/label_encoder.pkl")
    cfg2    = PeftConfig.from_pretrained(M2_PATH)
    base2   = AutoModelForSequenceClassification.from_pretrained(
        cfg2.base_model_name_or_path, num_labels=len(le2.classes_),
        ignore_mismatched_sizes=True)
    model2  = PeftModel.from_pretrained(base2, M2_PATH).eval().to(DEVICE)

    return tok, model1, le1, model2, le2

def predict(text, tok, model, le):
    inp = tok(clean_text(text), return_tensors="pt", truncation=True,
               padding=True, max_length=MAX_LENGTH)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inp).logits, dim=1).squeeze().cpu().numpy()
    pid = int(np.argmax(probs))
    return le.inverse_transform([pid])[0], float(probs[pid])

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">MindTrace</div>
  <div class="hero-sub">
    Deteksi dan klasifikasi cognitive distortion dari teks bahasa Indonesia<br>
    menggunakan two-stage IndoBERT + LoRA
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
with st.spinner("Memuat model..."):
    tok, model1, le1, model2, le2 = load_models()

st.markdown('<div class="status-ok">Model siap digunakan</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">Teks analisis</div>', unsafe_allow_html=True)

if "txt" not in st.session_state:
    st.session_state.txt = ""

text_input = st.text_area("", value=st.session_state.txt,
    placeholder="Ketik atau tempel teks di sini...",
    height=120, label_visibility="collapsed", key="main_input")

# Sample texts
with st.expander("Coba teks contoh"):
    for s in SAMPLES:
        if st.button(s, key=f"s_{s}", use_container_width=True):
            st.session_state.txt = s
            st.rerun()

st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
_, mid, _ = st.columns([2, 1, 2])
with mid:
    run = st.button("Analisis", use_container_width=True, type="primary")

# ── Result ────────────────────────────────────────────────────
if run:
    text_input = st.session_state.get("main_input", text_input)
    if not text_input.strip():
        st.warning("Teks tidak boleh kosong.")
    else:
        with st.spinner("Menganalisis..."):
            lbin, cbin = predict(text_input, tok, model1, le1)
            lmul, cmul = (predict(text_input, tok, model2, le2)
                          if lbin == "Ya" else (None, None))

        st.markdown("---")
        st.markdown('<div class="section-label">Hasil</div>', unsafe_allow_html=True)

        # Cards
        c1, c2 = st.columns(2)
        with c1:
            cls = "rcard-red" if lbin == "Ya" else "rcard-green"
            val = "Ada distorsi" if lbin == "Ya" else "Tidak ada distorsi"
            st.markdown(f"""
<div class="rcard {cls}">
  <div class="rcard-label">Deteksi</div>
  <div class="rcard-value">{val}</div>
  <div class="rcard-conf">Confidence {cbin:.1%}</div>
</div>""", unsafe_allow_html=True)

        with c2:
            if lmul:
                st.markdown(f"""
<div class="rcard rcard-blue">
  <div class="rcard-label">Jenis distorsi</div>
  <div class="rcard-value">{lmul}</div>
  <div class="rcard-conf">Confidence {cmul:.1%}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="rcard rcard-gray">
  <div class="rcard-label">Jenis distorsi</div>
  <div class="rcard-value">—</div>
  <div class="rcard-conf">Tidak terdeteksi</div>
</div>""", unsafe_allow_html=True)

        # Description
        if lmul:
            desc = DESCRIPTIONS.get(lmul, "")
            pct  = int(cmul * 100)
            st.markdown(f"""
<div class="desc-wrap">
  <div class="desc-name">{lmul}</div>
  {desc}
</div>
<div class="conf-wrap">
  <div class="conf-row"><span>Keyakinan model</span><span>{pct}%</span></div>
  <div class="bar-bg"><div class="bar-fg" style="width:{pct}%"></div></div>
</div>
""", unsafe_allow_html=True)

        elif lbin == "Tidak":
            st.markdown("""
<div class="ok-banner">
  Tidak terdeteksi adanya cognitive distortion.<br>
  <span style="opacity:0.75;font-size:0.82rem">
    Pola pikir dalam teks tampak sehat dan realistis.
  </span>
</div>
""", unsafe_allow_html=True)

        # Detail
        with st.expander("Detail teknis"):
            st.markdown(f"""
<table class="detail-table">
  <tr><td>Teks asli</td><td>{text_input[:120]}{"…" if len(text_input)>120 else ""}</td></tr>
  <tr><td>Setelah preprocessing</td><td>{clean_text(text_input)[:100]}</td></tr>
  <tr><td>Model 1 output</td><td>{lbin} &nbsp;({cbin:.4f})</td></tr>
  <tr><td>Model 2 output</td><td>{f"{lmul} ({cmul:.4f})" if lmul else "—"}</td></tr>
  <tr><td>Device</td><td>{str(DEVICE).upper()}</td></tr>
</table>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  MindTrace &nbsp;·&nbsp; IndoLEM-IndoBERT + LoRA &nbsp;·&nbsp;
  Dataset: Cognitive Distortion Bahasa Indonesia
</div>
""", unsafe_allow_html=True)