import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="Quick AI Summarizer", page_icon="📝")

@st.cache_resource
def load_summarizer():
    # استخدام أصغر موديل موجود لضمان عدم حدوث Error
    model_name = "t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # بناء الـ pipeline يدوياً
    return pipeline(
        "summarization", 
        model=model, 
        tokenizer=tokenizer,
        framework="pt"
    )

st.title("📝 AI Text Summarizer")
st.write("Fast & Stable Version")

try:
    summarizer = load_summarizer()
    
    text = st.text_area("Input Text (English)", height=200)

    if st.button("Summarize"):
        if not text.strip():
            st.warning("Please enter text")
        else:
            with st.spinner("AI is thinking..."):
                # T5 بيحتاج الكلمة دي في البداية
                result = summarizer("summarize: " + text, max_length=150, min_length=30)
                st.subheader("Summary:")
                st.success(result[0]['summary_text'])
except Exception as e:
    # لو لسه الـ KeyError موجود، هنعرض رسالة مساعدة
    st.error(f"Error: {e}")
    if "summarization" in str(e):
        st.info("السيرفر لسه بيحمل المكتبات الأساسية، جربي تعملي Refresh للصفحة بعد دقيقة.")
