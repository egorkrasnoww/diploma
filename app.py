import streamlit as st
import docx2txt
import pdfplumber
import tempfile
import pymorphy2
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


morph = pymorphy2.MorphAnalyzer()


def normalize_text(text):
    text = re.sub(r"[^а-яА-Я0-9\\s]", " ", str(text).lower())
    return " ".join([morph.parse(word)[0].normal_form for word in text.split()])


# Загрузка модели ruT5
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./finetuned_rut5_model", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_rut5_model")
    return tokenizer, model




tokenizer, model = load_model()


def summarize(text, level="среднее"):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    if level == "мягкое":
        min_len, max_len, penalty = 100, 384, 0.8
    elif level == "жёсткое":
        min_len, max_len, penalty = 20, 128, 2.2
    else:  # среднее
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






def extract_text_from_docx(file):
    return docx2txt.process(file)


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

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



# Интерфейс
st.set_page_config(page_title="NLP Сокращение документов", page_icon="📄", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #2C3E50;'>📄 Система автоматического сокращения текста</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center; font-size: 16px; color: gray;'>Загрузите технический документ, и модель сократит его без потери смысла</p>",
    unsafe_allow_html=True,
)


col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📁 Загрузите файл (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])

with col2:
    st.info("Поддерживаются отчёты, акты, служебные записки. Максимальная длина — 512 токенов.")


st.markdown("### ✂️ Уровень сокращения текста")
reduction_level = st.selectbox(
    "Выберите степень сокращения:",
    ["мягкое", "среднее", "жёсткое"]
)

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

    if text:
        st.subheader("📑 Исходный текст")
        st.text_area("Текст из документа:", text, height=300)

        if st.button("🚀 Сократить"):
            with st.spinner("Модель обрабатывает текст..."):
                result = summarize(text, level=reduction_level)


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

else:
    st.warning("⬆ Загрузите документ, чтобы начать обработку.")
