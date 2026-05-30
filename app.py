import streamlit as st
import torch
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MindTrace",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1rem;
    opacity: 0.88;
    margin-top: 0.5rem;
    margin-bottom: 0;
}

/* Card */
.card {
    background: white;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Result badge */
.badge-ya {
    display: inline-block;
    background: #fef3f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-tidak {
    display: inline-block;
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-jenis {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Metric card */
.metric-card {
    background: #f8f9ff;
    border: 1px solid #e0e4ff;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1f2937;
}
.metric-sub {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.2rem;
}

/* Desc box */
.desc-box {
    background: linear-gradient(135deg, #eff6ff, #f5f3ff);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.desc-box strong {
    color: #4338ca;
}

/* Progress bar custom */
.conf-bar-wrap {
    background: #e5e7eb;
    border-radius: 99px;
    height: 8px;
    margin-top: 0.5rem;
}
.conf-bar-fill {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 99px;
    height: 8px;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.78rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #f3f4f6;
}

/* Hide streamlit branding */
#MainMenu, footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────────────
M1_PATH    = "./model_1_binary"
M2_PATH    = "./model_2_multiclass"
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESCRIPTIONS = {
    "All-or-nothing"           : ("🔲", "Berpikir dalam kategori hitam-putih tanpa nuansa tengah."),
    "Discounting the positives": ("🚫", "Mengabaikan atau meremehkan hal-hal positif yang terjadi."),
    "Emotional Reasoning"      : ("💭", "Menganggap perasaan negatif sebagai kebenaran faktual."),
    "Jumping to Conclusions"   : ("⚡", "Mengambil kesimpulan negatif tanpa bukti yang memadai."),
    "Labeling"                 : ("🏷️", "Memberi label negatif secara menyeluruh pada diri atau orang lain."),
    "Mental filter"            : ("🔍", "Fokus berlebihan pada satu detail negatif, mengabaikan sisanya."),
    "Overgeneralization"       : ("🔄", "Menarik kesimpulan luas dari satu kejadian buruk."),
    "Personalization and Blame": ("👉", "Menyalahkan diri sendiri atas hal-hal di luar kendali."),
    "Should statement"         : ("📋", "Menetapkan standar kaku dengan kata 'harus' atau 'seharusnya'."),
}

# ── Helper ────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Load models ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    le1     = joblib.load(f"{M1_PATH}/label_encoder.pkl")
    config1 = PeftConfig.from_pretrained(M1_PATH)
    base1   = AutoModelForSequenceClassification.from_pretrained(
        config1.base_model_name_or_path,
        num_labels=len(le1.classes_),
        ignore_mismatched_sizes=True,
    )
    model1 = PeftModel.from_pretrained(base1, M1_PATH)
    model1.eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(M1_PATH)

    le2     = joblib.load(f"{M2_PATH}/label_encoder.pkl")
    config2 = PeftConfig.from_pretrained(M2_PATH)
    base2   = AutoModelForSequenceClassification.from_pretrained(
        config2.base_model_name_or_path,
        num_labels=len(le2.classes_),
        ignore_mismatched_sizes=True,
    )
    model2 = PeftModel.from_pretrained(base2, M2_PATH)
    model2.eval().to(DEVICE)

    return tokenizer, model1, le1, model2, le2

def predict(text, tokenizer, model, label_encoder):
    cleaned = clean_text(text)
    inputs  = tokenizer(cleaned, return_tensors="pt", truncation=True,
                        padding=True, max_length=MAX_LENGTH)
    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pid    = int(np.argmax(probs))
    return label_encoder.inverse_transform([pid])[0], float(probs[pid])

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 MindTrace</h1>
    <p>Deteksi pola pikir negatif dari teks bahasa Indonesia<br>
    menggunakan dua model <strong>IndoBERT + LoRA</strong></p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
with st.spinner("⏳ Memuat model AI, harap tunggu..."):
    tokenizer, model1, le1, model2, le2 = load_models()

st.markdown("""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
padding:0.6rem 1rem;margin-bottom:1.5rem;color:#15803d;font-size:0.88rem;">
✅ &nbsp; Model siap digunakan
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────
st.markdown("#### ✍️ Masukkan Teks")
text_input = st.text_area(
    label="",
    placeholder="Contoh: Aku merasa semua orang pasti membenciku...",
    height=130,
    label_visibility="collapsed"
)

# Contoh teks
with st.expander("💡 Coba teks contoh"):
    samples = [
        "Aku merasa semua orang pasti membenciku.",
        "Kalau aku gagal sekali, berarti aku memang tidak akan pernah berhasil.",
        "Aku harus selalu sempurna dalam semua hal.",
        "Dia tidak membalas chatku, pasti dia marah padaku.",
        "Hari ini aku makan bersama keluarga dan merasa senang.",
    ]
    for s in samples:
        if st.button(s, key=s, use_container_width=True):
            st.session_state["sample_text"] = s

if "sample_text" in st.session_state and not text_input.strip():
    text_input = st.session_state["sample_text"]

col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    analyze = st.button("🔍 Analisis", use_container_width=True, type="primary")

# ── Prediksi ──────────────────────────────────────────────────
if analyze:
    if not text_input.strip():
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        label_bin, conf_bin     = None, None
        label_multi, conf_multi = None, None

        with st.spinner("Menganalisis teks..."):
            label_bin, conf_bin = predict(text_input, tokenizer, model1, le1)
            if label_bin == "Ya":
                label_multi, conf_multi = predict(text_input, tokenizer, model2, le2)

        st.markdown("---")
        st.markdown("#### 📊 Hasil Analisis")

        # ── Metric cards ──────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            badge = f'<span class="badge-ya">⚠️ Ada Distorsi</span>' if label_bin == "Ya" \
                    else f'<span class="badge-tidak">✅ Normal</span>'
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Model 1 — Deteksi</div>
    <div style="margin: 0.4rem 0">{badge}</div>
    <div class="metric-sub">Confidence: {conf_bin:.1%}</div>
</div>
""", unsafe_allow_html=True)

        with col_b:
            if label_multi:
                badge2 = f'<span class="badge-jenis">🏷️ {label_multi}</span>'
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Model 2 — Jenis Distorsi</div>
    <div style="margin: 0.4rem 0">{badge2}</div>
    <div class="metric-sub">Confidence: {conf_multi:.1%}</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Model 2 — Jenis Distorsi</div>
    <div style="margin: 0.4rem 0"><span class="badge-tidak">— Tidak ada distorsi</span></div>
    <div class="metric-sub">Model 2 tidak dijalankan</div>
</div>
""", unsafe_allow_html=True)

        # ── Penjelasan distorsi ────────────────────────────────
        if label_multi:
            icon, desc = DESCRIPTIONS.get(label_multi, ("🔍", ""))
            st.markdown(f"""
<div class="desc-box">
    <strong>{icon} {label_multi}</strong><br>
    <span style="color:#374151;font-size:0.92rem">{desc}</span>
</div>
""", unsafe_allow_html=True)

            # Confidence bar
            pct = int(conf_multi * 100)
            st.markdown(f"""
<div style="margin-top:1rem">
    <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#6b7280;margin-bottom:4px">
        <span>Keyakinan model</span><span>{pct}%</span>
    </div>
    <div class="conf-bar-wrap">
        <div class="conf-bar-fill" style="width:{pct}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        elif label_bin == "Tidak":
            st.markdown("""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
padding:1rem 1.2rem;margin-top:1rem;color:#15803d;">
✅ <strong>Teks ini tidak terdeteksi mengandung cognitive distortion.</strong><br>
<span style="font-size:0.88rem;opacity:0.8">Pola pikir dalam teks tampak sehat dan realistis.</span>
</div>
""", unsafe_allow_html=True)

        # ── Detail teknis ──────────────────────────────────────
        with st.expander("🔬 Detail teknis"):
            st.markdown(f"""
| Field | Nilai |
|---|---|
| **Teks asli** | {text_input[:80]}{"..." if len(text_input)>80 else ""} |
| **Setelah preprocessing** | {clean_text(text_input)[:80]} |
| **Model 1** | {label_bin} ({conf_bin:.4f}) |
| **Model 2** | {label_multi if label_multi else "—"} {f"({conf_multi:.4f})" if conf_multi else ""} |
| **Device** | {str(DEVICE).upper()} |
""")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🧠 MindTrace &nbsp;·&nbsp; IndoLEM-IndoBERT + LoRA &nbsp;·&nbsp;
    Dataset: Cognitive Distortion Bahasa Indonesia (Universitas Udayana)
</div>
""", unsafe_allow_html=True)