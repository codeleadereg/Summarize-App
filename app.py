import streamlit as st
import requests

# إعدادات الصفحة
st.set_page_config(page_title="AI Summarizer", page_icon="📝")

# ده الـ API بتاع Hugging Face (بيشتغل من سيرفراتهم مباشرة)
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# ملاحظة: سيبنا الـ Headers فاضية حالياً للنسخة العامة، لو طلب Token هقولك تجيبيه ازاي
headers = {}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

st.title("📝 Quick AI Summarizer")
st.write("Link-ready version (Fast & Light)")

text = st.text_area("Input Text", height=200, placeholder="Enter your text here...")

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        with st.spinner("AI is processing..."):
            output = query({"inputs": text})
            
            # التأكد من وصول النتيجة
            if isinstance(output, list) and len(output) > 0:
                st.subheader("Summary:")
                st.success(output[0]['summary_text'])
            else:
                st.error("The model is loading, please try again in 30 seconds.")
