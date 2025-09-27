import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Meowster Detective", layout="wide")
st.title("🐱 Meowster Detective — Cat Recognizer")

# --- Load model ---
model = load_model("cat_recognizer.h5")
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

# --- Cat detector ---
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")

# --- Sidebar for input ---
st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose input method:", ["Upload Image", "Use Webcam"])

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def predict_cat(face):
    preds = model.predict(face, verbose=0)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]
    top3_idx = preds.argsort()[-3:][::-1]
    return class_idx, confidence, top3_idx, preds

def draw_boxes(frame, cats, preds):
    for i, (x, y, w, h) in enumerate(cats):
        class_idx, confidence, top3_idx, all_preds = preds[i]
        # bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # label
        label = f"{class_names[class_idx]} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # confidence bar
        bar_x1, bar_y1 = x, y + h + 5
        bar_x2, bar_y2 = x + w, bar_y1 + 10
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), 1)
        fill_width = int(w * confidence)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_width, bar_y2), (0, 255, 0), -1)
    return frame

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a cat image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)[:, :, ::-1].copy()  # PIL RGB -> BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        preds_list = []
        for (x, y, w, h) in cats:
            face = frame[y:y+h, x:x+w]
            face_pre = preprocess_face(face)
            preds_list.append(predict_cat(face_pre))
        if len(cats) > 0:
            frame = draw_boxes(frame, cats, preds_list)
        else:
            st.warning("No cat faces detected!")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Processed Image", use_column_width=True)

elif option == "Use Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
    else:
        st.info("Press 'q' to stop the webcam.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            preds_list = []
            for (x, y, w, h) in cats:
                face = frame[y:y+h, x:x+w]
                face_pre = preprocess_face(face)
                preds_list.append(predict_cat(face_pre))
            if len(cats) > 0:
                frame = draw_boxes(frame, cats, preds_list)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
