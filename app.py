import streamlit as st
from transformers import pipeline

# تحميل الموديل (هيتحمل أول مرة بس)
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        framework="pt"  # السطر ده أساسي هنا
    )

summarizer = load_model()

# واجهة التطبيق
st.set_page_config(page_title="AI Summarizer", page_icon="📝")

st.title("📝 Text Summarizer")
st.write("Type your English text and get a summary")

text = st.text_area(" Input Text", height=200)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(
                text,
                max_length=80,
                min_length=30,
                do_sample=False
            )
            st.subheader(" Summary")
            st.write(summary[0]['summary_text'])
