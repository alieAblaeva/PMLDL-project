import streamlit as st
import requests
from PIL import Image
import io

st.title("Predict age by photo")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):

        image = Image.open(uploaded_file)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        response = requests.post("http://api:8000/predict", files={"file": img_bytes}
)

        st.write("Prediction result:", response.content)
