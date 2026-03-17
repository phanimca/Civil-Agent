
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
OPENAI_TOKEN = st.secrets.get("OPENAI_TOKEN", "")

client = OpenAI(api_key=OPENAI_TOKEN, base_url="https://models.github.ai/inference")
MODEL_NAME = "openai/gpt-4o-mini"

@st.cache_resource
def load_model():
    try:
        if HF_TOKEN:
            model_path = hf_hub_download(
                repo_id="cazzz307/yolov8-crack-detection",
                filename="best.pt",
                token=HF_TOKEN
            )
            return YOLO(model_path)
    except Exception:
        pass

    if os.path.exists("models/best.pt"):
        return YOLO("models/best.pt")

    return YOLO("yolov8n.pt")

model = load_model()

def calculate_severity(box, conf, img_shape):
    x1, y1, x2, y2 = box.xyxy[0]
    area = (x2 - x1) * (y2 - y1)
    total = img_shape[0] * img_shape[1]
    ratio = area / total

    if ratio > 0.15 or conf > 0.8:
        return "🔴 High"
    elif ratio > 0.05:
        return "🟡 Medium"
    else:
        return "🟢 Low"

def generate_ai_report(detections):
    if not detections:
        detections = ["Minor surface anomaly"]

    prompt = f"""You are a civil engineering inspection expert.
Detected issues: {detections}
Generate report: summary, severity, causes, repair, prevention."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI report failed: {e}"

def style_report(text):
    return f"""<div style="background:#0f172a;padding:20px;border-radius:10px;color:white">
    <h2 style="color:#38bdf8;">AI Inspection Report</h2>
    <pre>{text}</pre></div>"""

def create_pdf(report, image):
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(file.name)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("AI Inspection Report", styles["Title"]))
    content.append(Spacer(1, 12))

    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    image.save(img_path)

    content.append(RLImage(img_path, width=400, height=250))
    content.append(Spacer(1, 12))

    for line in report.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 8))

    doc.build(content)
    return file.name

st.set_page_config(layout="wide")
st.title("AI Infrastructure Inspection")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns([1,2])

        with col1:
            st.image(image)

        with col2:
            results = model(img_np)[0]
            annotated = results.plot()
            st.image(annotated)

            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                severity = calculate_severity(box, conf, img_np.shape)
                text = f"Crack | {conf:.2f} | {severity}"
                detections.append(text)
                st.write(text)

            report = generate_ai_report(detections)
            st.markdown(style_report(report), unsafe_allow_html=True)

            pdf = create_pdf(report, image)
            with open(pdf, "rb") as f:
                st.download_button("Download PDF", f)

else:
    st.info("Upload images")
