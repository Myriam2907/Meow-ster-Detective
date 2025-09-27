import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# --- Load trained model ---
model = load_model("cat_recognizer.h5")

# --- Load class names ---
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

# --- Load OpenCV cat face detector ---
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")

# --- Parameters ---
CONF_THRESHOLD = 0.6         # if confidence < this, show suspicious cat warning
SMOOTHING_FRAMES = 5         # how many frames to smooth over
recent_labels = deque(maxlen=SMOOTHING_FRAMES)

# --- Open webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in cats:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop face and preprocess
        cat_face = frame[y:y + h, x:x + w]
        img = cv2.resize(cat_face, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict which cat
        preds = model.predict(img, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]

        if confidence < CONF_THRESHOLD:
            # 🚨 Unknown cat!
            label = "🚨 DANGER! Suspicious cat detected!"
        else:
            # Known cat
            label = f"{class_names[class_idx]} ({confidence*100:.1f}%)"

        # Add to recent labels
        recent_labels.append(label)

        # Pick the most common label from the last N frames
        final_label = max(set(recent_labels), key=recent_labels.count)

        # Choose color based on label
        color = (0, 0, 255) if "DANGER" in final_label else (255, 0, 0)

        # Display final label
        cv2.putText(frame, final_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # If it’s a known cat → draw confidence bar + top 3 preds
        if "DANGER" not in final_label:
            # --- Draw confidence bar ---
            bar_x1, bar_y1 = x, y + h + 5
            bar_x2, bar_y2 = x + w, bar_y1 + 10
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2),
                          (50, 50, 50), 1)  # background
            fill_width = int(w * confidence)
            cv2.rectangle(frame, (bar_x1, bar_y1),
                          (bar_x1 + fill_width, bar_y2), (0, 255, 0), -1)

            # --- Optional: show top 3 predictions ---
            top3_idx = preds.argsort()[-3:][::-1]
            for i, idx in enumerate(top3_idx):
                txt = f"{class_names[idx]}: {preds[idx]*100:.1f}%"
                cv2.putText(frame, txt, (x, y + h + 30 + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Cat Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
