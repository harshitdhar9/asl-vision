import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2 as cv

model = tf.keras.models.load_model('saved_models(h5)/image_processor.h5')

class_labels = [chr(i) for i in range(65, 91)] + ['blank']


if 'accumulated_text' not in st.session_state:
    st.session_state.accumulated_text = ""
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0

st.title("ASL Vision - Real-time Webcam Classifier")
st.write("Show your ASL gesture and press capture.")

img_data = st.camera_input("Capture Gesture")

if img_data is not None:

    image = Image.open(img_data).convert("RGB")
    st.image(image, caption='Captured Frame', use_column_width=True)

    img = np.array(image)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    st.write(f"Image dimensions: **{w} x {h}** (width x height)")
    max_dim = 256
    h, w = img.shape[:2]
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    img = cv.resize(img, (new_w, new_h))

    height, width = img.shape[:2]
    rect_width = int(width * 0.6)
    rect_height = int(height * 0.8)
    center_x, center_y = width // 2, height // 2
    x = max(center_x - rect_width // 2, 0)
    y = max(center_y - rect_height // 2, 0)
    rect = (x, y, min(rect_width, width - x), min(rect_height, height - y))

    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = img * mask2[:, :, np.newaxis]

    processed_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    st.image(processed_rgb, caption='Processed Image (Post-GrabCut)', use_column_width=True)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    final_img = cv.resize(gray, (256, 256))
    final_img = final_img.astype('float32') / 255.0
    final_img = final_img.reshape(1, 256, 256, 1)  

    prediction = model.predict(final_img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### Prediction: `{predicted_class.upper()}`")
    st.write(f"Confidence: {confidence:.2f}")

    if predicted_class.lower() != 'blank':
        st.session_state.accumulated_text += predicted_class.upper()

st.subheader("Generated Text")
st.text_area("Accumulated Text", value=st.session_state.accumulated_text, height=300, disabled=True)

if st.button("Reset Text"):
    st.session_state.accumulated_text = ""
