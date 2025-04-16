import streamlit as st
import face_recognition
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import time

# Page setup
st.set_page_config(page_title="KYC Face Verification", layout="centered", page_icon="🧠")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #574fd6;
    }
    .stImage img {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 KYC Face Verification System")
st.markdown("Upload your **KYC document** (e.g., PAN, Aadhar) and verify using your webcam face.")

# Upload KYC Document
with st.sidebar:
    st.header("📄 Upload KYC Document")
    uploaded_doc = st.file_uploader("Upload PAN/Aadhar (with face)", type=["jpg", "jpeg", "png"])
    start_button = st.button("Start Verification")

# UI placeholders
status = st.empty()
confidence_display = st.empty()
image_display = st.empty()

if uploaded_doc:
    # Load image and process
    doc_image = Image.open(uploaded_doc).convert("RGB")
    doc_image = doc_image.resize((800, 600))  # Resize to help detection

    # Optional: enhance contrast
    enhancer = ImageEnhance.Contrast(doc_image)
    doc_image = enhancer.enhance(1.5)

    # Display image
    st.sidebar.image(doc_image, caption="🪪 KYC Document", use_column_width=True)

    # Convert to numpy for face_recognition
    doc_image_np = np.array(doc_image)

    # Use CNN model for better accuracy
    doc_face_locations = face_recognition.face_locations(doc_image_np, model='cnn')

    if not doc_face_locations:
        st.sidebar.error("❌ No face found in KYC document. Try a clearer or higher-resolution image.")
    else:
        # Encode reference face
        doc_face_encoding = face_recognition.face_encodings(doc_image_np, doc_face_locations)[0]

        if start_button:
            cap = cv2.VideoCapture(0)
            status.success("✅ Webcam started. Please look into the camera.")
            time.sleep(1)

            verified = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    status.error("❌ Failed to access webcam.")
                    break

                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    distance = face_recognition.face_distance([doc_face_encoding], face_encoding)[0]
                    match = distance < 0.5

                    top, right, bottom, left = face_location
                    color = (0, 255, 0) if match else (255, 0, 0)
                    label = "✅ MATCH" if match else "❌ NOT MATCHED"

                    # Draw box & label
                    cv2.rectangle(rgb_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(rgb_frame, f"{label} ({distance:.2f})", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Update UI
                    status.markdown(f"### {'🟢 FACE VERIFIED' if match else '🔴 FACE NOT MATCHED'}")
                    confidence_display.progress(min(1.0, 1 - distance))  # Show match confidence

                    if match:
                        verified = True
                        break

                image_display.image(rgb_frame, channels="RGB")

                # Exit if match found
                if verified:
                    st.success("🎉 KYC Face Verification Successful!")
                    break

            cap.release()
