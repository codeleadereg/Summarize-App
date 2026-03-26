import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Quick AI Summarizer", page_icon="📝")

@st.cache_resource
def load_model():
    # موديل T5-Small حجمه صغير جداً (حوالي 240 ميجا بس) 
    # ومثالي للسيرفرات المجانية
    return pipeline("summarization", model="t5-small")

st.title("📝 AI Text Summarizer")
st.write("Fast & Stable Version")

try:
    summarizer = load_model()
    
    text = st.text_area("Input Text", height=200)

    if st.button("Summarize"):
        if not text.strip():
            st.warning("Please enter text")
        else:
            with st.spinner("Summarizing..."):
                # T5 بيحتاج كلمة "summarize: " قبل النص عشان يفهم المهمة
                result = summarizer("summarize: " + text, max_length=150, min_length=30)
                st.subheader("Summary:")
                st.success(result[0]['summary_text'])
except Exception as e:
    st.error(f"Error: {e}")
