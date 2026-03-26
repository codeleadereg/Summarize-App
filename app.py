import streamlit as st
import requests
import time

# إعدادات الصفحة
st.set_page_config(page_title="AI Summarizer", page_icon="📝")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {}

def query(payload):
    # محاولة الاتصال بالموديل 3 مرات لو لسه بيحمل
    for i in range(3):
        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()
        
        # لو الموديل لسه بيحمل، استنى 10 ثواني وجرب تاني
        if isinstance(output, dict) and "estimated_time" in output:
            wait_time = output.get("estimated_time", 10)
            st.info(f"الذكاء الاصطناعي بيجهز نفسه.. ثواني وهتظهر النتيجة (فاضل {int(wait_time)} ثانية)")
            time.sleep(10)
            continue
        return output
    return output

st.title("📝 Quick AI Summarizer")
st.write("Link-ready version (Fast & Light)")

text = st.text_area("Input Text", height=200, placeholder="Enter your text here...")

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        with st.spinner("AI is processing..."):
            output = query({"inputs": text})
            
            if isinstance(output, list) and len(output) > 0:
                st.subheader("Summary:")
                st.success(output[0]['summary_text'])
            else:
                st.error("السيرفر مشغول حالياً، جربي تضغطي على الزرار مرة تانية كمان لحظات.")
