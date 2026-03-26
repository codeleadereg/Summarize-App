import streamlit as st
from transformers import pipeline
import torch

# إعدادات الصفحة
st.set_page_config(page_title="AI Summarizer Pro", page_icon="📝")

# تحميل الموديل الأصلي (سيأخذ وقت في أول مرة فقط عند الـ Deploy)
@st.cache_resource
def load_summarizer():
    # استخدام موديل distilbart لأنه أخف وأسرع في الـ Deploy المجاني
    model_name = "sshleifer/distilbart-cnn-12-6"
    return pipeline(
        "summarization", 
        model=model_name, 
        framework="pt", 
        device=-1 # أجبار التشغيل على CPU
    )

st.title("📝 AI Text Summarizer")
st.write("Original Version - Local Model Hosting")

# التأكد من تحميل الموديل
with st.spinner("Loading AI Model into memory... Please wait."):
    summarizer = load_summarizer()

text = st.text_area("Input Text", height=250, placeholder="Paste your text here...")

if st.button("Summarize Now"):
    if not text.strip():
        st.warning("Please enter some text.")
    elif len(text.split()) < 10:
        st.error("Text is too short. Please enter at least 10 words.")
    else:
        with st.spinner("Summarizing..."):
            try:
                summary = summarizer(
                    text, 
                    max_length=100, 
                    min_length=30, 
                    do_sample=False
                )
                st.subheader("Summary Result:")
                st.success(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"Error: {e}")
