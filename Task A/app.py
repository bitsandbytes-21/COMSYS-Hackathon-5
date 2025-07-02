import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image


st.set_page_config(page_title="Gender Classifier", layout="centered")


st.title("👤 Gender Classification from Image")
st.markdown("Upload one or more face images to predict gender")


@st.cache_resource
def load_gender_model():
    return load_model("gender_classification.h5")

model = load_gender_model()


class_labels = ['female', 'male']  


uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))

    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx]:
            
            img = Image.open(uploaded_file).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

    
            pred = model.predict(img_array)
            pred_class = np.argmax(pred, axis=1)[0]
            pred_label = class_labels[pred_class]
            probs = dict(zip(class_labels, pred[0].round(4)))

    
            st.image(img, caption=f"Predicted: {pred_label}", use_container_width=True)
            st.write("Probabilities:", probs)
