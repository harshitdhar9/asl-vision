import streamlit as st
import numpy as np
from PIL import Image
import tensorflow

model = tensorflow.keras.models.load_model('asl_vision_model.h5')

class_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'space','nothing','delete',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

st.title("ASL Vision - Sign Language Classifier")
st.write("Upload an image of a hand sign to identify the ASL character.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")  
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0 
    img_array = img_array.reshape(1, 64, 64, 1) 

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### Prediction: `{predicted_class.upper()}`")
    st.write(f"Confidence: {confidence:.2f}")
