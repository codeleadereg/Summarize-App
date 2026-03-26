import streamlit as st
import os

# إعدادات الصفحة
st.set_page_config(page_title="AI Summarizer", page_icon="📝")

# بنستورد المكتبات جوه الـ function عشان نوفر رامات في البداية
@st.cache_resource
def get_summarizer():
    from transformers import pipeline
    import torch
    
    model_name = "sshleifer/distilbart-cnn-12-6"
    return pipeline(
        "summarization", 
        model=model_name, 
        framework="pt", 
        device=-1
    )

st.title("📝 Professional AI Summarizer")
st.info("Original Model Version (Local Hosting)")

# تحميل الموديل
try:
    summarizer = get_summarizer()
    
    text = st.text_area("Input Text", height=250, placeholder="Paste your text here...")

    if st.button("Summarize Now"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Processing..."):
                result = summarizer(text, max_length=100, min_length=30, do_sample=False)
                st.subheader("Summary Result:")
                st.success(result[0]['summary_text'])
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل المكتبات: {e}")
    st.info("تأكد من تحديث ملف requirements.txt لنسخة الـ CPU")
