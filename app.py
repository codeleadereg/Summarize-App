import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# إعدادات الصفحة
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝")

# تحميل الموديل والـ Tokenizer بشكل يدوي لضمان التوافق
@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    
    # تحميل الأدوات يدوياً
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # إنشاء الـ pipeline مع تحديد الـ framework والـ device
    # device=-1 معناه استخدام الـ CPU وهو المتاح في النسخ المجانية
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=-1
    )
    return summarizer

# استدعاء الموديل
try:
    summarizer = load_summarizer()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# واجهة المستخدم (UI)
st.title("📝 AI Text Summarizer")
st.markdown("### أدوات تلخيص النصوص بالذكاء الاصطناعي")
st.write("أدخل النص الإنجليزي بالأسفل للحصول على ملخص سريع وذكي.")

# مكان إدخال النص
input_text = st.text_area("Input Text (English)", height=250, placeholder="Paste your long text here...")

# أزرار التحكم
col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("Summarize")

if submit_button:
    if not input_text.strip():
        st.warning("Please enter some text first!")
    elif len(input_text.split()) < 30:
        st.info("The text is too short to summarize. Please enter at least 30 words.")
    else:
        with st.spinner("Wait a moment... AI is reading and summarizing..."):
            try:
                # عملية التلخيص
                result = summarizer(
                    input_text,
                    max_length=130,
                    min_length=30,
                    do_sample=False
                )
                
                # عرض النتيجة
                st.markdown("---")
                st.subheader("✅ Summary Result:")
                st.success(result[0]['summary_text'])
                
                # إضافة زر لنسخ النص (اختياري)
                st.code(result[0]['summary_text'], language="text")
                
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")

# تذييل الصفحة
st.markdown("---")
st.caption("Powered by Hugging Face Transformers & Streamlit")
