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

# ── Konstanta ─────────────────────────────────────────────────
MODEL_NAME = "indolem/indobert-base-uncased"
M1_PATH    = "./model_1_binary"
M2_PATH    = "./model_2_multiclass"
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helper: clean text ────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Load models (cached) ──────────────────────────────────────
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Model 1 — binary ──────────────────────────────────────
    le1     = joblib.load(f"{M1_PATH}/label_encoder.pkl")
    config1 = PeftConfig.from_pretrained(M1_PATH)
    base1   = AutoModelForSequenceClassification.from_pretrained(
        config1.base_model_name_or_path,
        num_labels=len(le1.classes_),
        ignore_mismatched_sizes=True,
    )
    model1 = PeftModel.from_pretrained(base1, M1_PATH)
    model1.eval().to(DEVICE)

    # ── Model 2 — multi-class ─────────────────────────────────
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

# ── Fungsi prediksi ───────────────────────────────────────────
def predict(text: str, tokenizer, model, label_encoder):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pid    = int(np.argmax(probs))
    return label_encoder.inverse_transform([pid])[0], float(probs[pid])

# ── Penjelasan distorsi ───────────────────────────────────────
DESCRIPTIONS = {
    "All-or-nothing"           : "Berpikir dalam kategori hitam-putih tanpa nuansa.",
    "Discounting the positives": "Mengabaikan atau meremehkan hal-hal positif.",
    "Emotional Reasoning"      : "Menganggap perasaan negatif sebagai fakta.",
    "Jumping to Conclusions"   : "Mengambil kesimpulan tanpa bukti yang cukup.",
    "Labeling"                 : "Memberi label negatif pada diri sendiri atau orang lain.",
    "Mental filter"            : "Fokus berlebihan pada satu detail negatif.",
    "Overgeneralization"       : "Menarik kesimpulan umum dari satu kejadian.",
    "Personalization and Blame": "Menyalahkan diri sendiri atas hal di luar kendali.",
    "Should statement"         : "Menetapkan standar kaku dengan kata 'harus'.",
}

# ── UI ────────────────────────────────────────────────────────
st.title("🧠 MindTrace")
st.markdown(
    "Deteksi pola pikir negatif dari teks bahasa Indonesia "
    "menggunakan dua model IndoBERT + LoRA."
)
st.divider()

with st.spinner("Memuat model, harap tunggu..."):
    tokenizer, model1, le1, model2, le2 = load_models()
st.success("Model siap!", icon="✅")

st.divider()

text_input = st.text_area(
    "Masukkan teks di sini:",
    placeholder="Contoh: Aku merasa semua orang pasti membenciku.",
    height=120,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze = st.button("🔍 Analisis", use_container_width=True, type="primary")

# ── Prediksi ──────────────────────────────────────────────────
if analyze:
    if not text_input.strip():
        st.warning("Teks tidak boleh kosong!")
    else:
        with st.spinner("Menganalisis..."):
            label_bin, conf_bin = predict(text_input, tokenizer, model1, le1)

        # Jalankan model 2 di luar blok spinner supaya label_multi selalu terdefinisi
        label_multi, conf_multi = None, None
        if label_bin == "Ya":
            with st.spinner("Mengklasifikasi jenis distorsi..."):
                label_multi, conf_multi = predict(text_input, tokenizer, model2, le2)

        st.divider()
        st.subheader("📊 Hasil Analisis")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Model 1 — Deteksi", label_bin)
            st.caption(f"Confidence: {conf_bin:.1%}")

        with col_b:
            if label_multi is not None:
                st.metric("Model 2 — Jenis Distorsi", label_multi)
                st.caption(f"Confidence: {conf_multi:.1%}")
            else:
                st.metric("Model 2 — Jenis Distorsi", "-")
                st.caption("Tidak ada distorsi terdeteksi")

        if label_multi is not None:
            desc = DESCRIPTIONS.get(label_multi, "")
            if desc:
                st.info(f"**{label_multi}**: {desc}")
            st.progress(conf_multi, text=f"Keyakinan model: {conf_multi:.1%}")
        else:
            st.success("✅ Tidak terdeteksi adanya cognitive distortion pada teks ini.")

        with st.expander("🔬 Detail teknis"):
            st.markdown(f"""
| | Nilai |
|---|---|
| **Teks asli** | {text_input} |
| **Teks setelah preprocessing** | {clean_text(text_input)} |
| **Model 1 output** | {label_bin} ({conf_bin:.4f}) |
| **Model 2 output** | {label_multi if label_multi is not None else "-"} |
""")

st.divider()
st.caption("MindTrace — Deteksi Pola Pikir Negatif | IndoLEM-IndoBERT + LoRA")