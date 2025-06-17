# --- Импорты ---
import streamlit as st
import docx2txt
import pdfplumber
import tempfile
import pymorphy2
import re
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- Нормализация текста ---
morph = pymorphy2.MorphAnalyzer()
def normalize_text(text):
    text = re.sub(r"[^а-яА-Я0-9\\s]", " ", str(text).lower())
    return " ".join([morph.parse(word)[0].normal_form for word in text.split()])

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./finetuned_rut5_model", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_rut5_model")
    return tokenizer, model

tokenizer, model = load_model()

# --- Суммаризация текста ---
def summarize(text, level="среднее"):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    if level == "мягкое":
        min_len, max_len, penalty = 100, 384, 0.8
    elif level == "жёсткое":
        min_len, max_len, penalty = 20, 128, 2.2
    else:
        min_len, max_len, penalty = 50, 256, 1.2
    summary_ids = model.generate(
        input_ids,
        max_length=max_len,
        min_length=min_len,
        length_penalty=penalty,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Сохранение DOCX ---
def save_to_docx(text):
    doc = Document()
    doc.add_paragraph(text)
    path = "summary.docx"
    doc.save(path)
    return path

# --- Извлечение текста из файлов ---
def extract_text_from_docx(file): return docx2txt.process(file)
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Разбиение текста на блоки ---
def split_text(text, max_tokens=480):
    sentences = text.split('. ')
    current, chunks = "", []
    for s in sentences:
        if len(tokenizer.encode(current + s)) < max_tokens:
            current += s + '. '
        else:
            chunks.append(current.strip())
            current = s + '. '
    if current:
        chunks.append(current.strip())
    return chunks

# --- Настройки страницы ---
st.set_page_config(page_title="NLP Сокращение документов", page_icon="📄", layout="wide")

# --- Цветовая тема ---
st.markdown("""
<style>
body, .stApp {
    background-color: #F9FAFC;
    color: #1B1F3B;
}
h1, h2, h3, h4, h5, h6 {
    color: #0A4D8C;
}
.stButton>button {
    background-color: #0A4D8C;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    border: none;
    font-weight: bold;
}
.stSelectbox>div>div {
    background-color: #EAF2FB;
    color: #0A4D8C;
}
.stTextArea textarea {
    background-color: #FFFFFF;
    color: #1B1F3B;
}
.stInfo, .stWarning {
    background-color: #EAF2FB;
    color: #0A4D8C;
}
</style>
""", unsafe_allow_html=True)

# --- Заголовок ---
st.markdown("""
<h1 style='text-align: center; color: #0A4D8C;'>📄 Система автоматического сокращения текста</h1>
<p style='text-align: center; font-size: 16px; color: #1B1F3B;'>Загрузите технический документ или вставьте текст вручную — модель сократит его без потери смысла.</p>
""", unsafe_allow_html=True)

# --- Режим ввода ---
st.markdown("### 🔽 Выберите способ ввода текста")
input_mode = st.radio("Источник текста:", ["Загрузить файл", "Ввести вручную"])

text = ""

# --- Ввод вручную ---
if input_mode == "Ввести вручную":
    text = st.text_area("✍️ Вставьте сюда текст для сокращения", height=300)

# --- Загрузка файла ---
else:
    uploaded_file = st.file_uploader("📁 Загрузите файл (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name
        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_type == "docx":
            text = extract_text_from_docx(file_path)
        elif file_type == "pdf":
            text = extract_text_from_pdf(file_path)
        else:
            st.error("❌ Неподдерживаемый формат файла.")
            text = ""

# --- Обработка текста ---
if text:
    st.markdown("### ✂️ Уровень сокращения текста")
    reduction_level = st.selectbox("Выберите степень сокращения:", ["мягкое", "среднее", "жёсткое"])

    st.subheader("📑 Исходный текст")
    st.text_area("Исходный текст:", text, height=300)

    if st.button("🚀 Сократить"):
        with st.spinner("Модель обрабатывает текст..."):
            chunks = split_text(text)
            result_chunks = []
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                summary = summarize(chunk, level=reduction_level)
                result_chunks.append(summary)
                progress_bar.progress((i + 1) / len(chunks))
            result = "\n\n".join(result_chunks)

        original_word_count = len(text.split())
        summary_word_count = len(result.split())
        reduction_percent = round(100 * (original_word_count - summary_word_count) / original_word_count)

        st.subheader("✅ Сокращённый текст")
        st.success(result)

        st.markdown(f"""
        ### 📊 Статистика сокращения:
        - Было слов: **{original_word_count}**
        - Стало слов: **{summary_word_count}**
        - 🔻 Сокращение: **{reduction_percent}%**
        """)

        # --- Скачивание результата ---
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("💾 Скачать .txt", result, file_name="summary.txt", mime="text/plain")
        with col2:
            docx_path = save_to_docx(result)
            with open(docx_path, "rb") as file:
                st.download_button("📄 Скачать .docx", file.read(), file_name="summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

else:
    st.warning("⬆ Вставьте текст вручную или загрузите документ для начала обработки.")
