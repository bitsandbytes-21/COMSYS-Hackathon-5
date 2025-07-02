import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input


class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

@st.cache_resource
def load_verification_model():
    return load_model("face_verifier.h5", custom_objects={"L1Dist": L1Dist})

model = load_verification_model()


def preprocess_image(img):
    img = img.resize((160, 160))
    img = np.array(img)
    if img.shape[-1] == 4:  
        img = img[..., :3]
    img = preprocess_input(img)
    return img

st.title(" Face Verification System")
st.write("Upload two face images to check if they belong to the same person.")

col1, col2 = st.columns(2)
with col1:
    img1 = st.file_uploader("Upload Anchor Image", type=["jpg", "png"], key="img1")
with col2:
    img2 = st.file_uploader("Upload Test Image", type=["jpg", "png"], key="img2")

if img1 and img2:
    image1 = Image.open(img1).convert("RGB")
    image2 = Image.open(img2).convert("RGB")

    
    col1.image(image1, caption="Anchor Image", use_container_width=True)
    col2.image(image2, caption="Test Image", use_container_width=True)


    img1_array = preprocess_image(image1)
    img2_array = preprocess_image(image2)

    pred = model.predict([np.expand_dims(img1_array, axis=0), np.expand_dims(img2_array, axis=0)])[0][0]

    st.markdown("---")
    st.subheader("Result")
    st.metric("Similarity Score", f"{pred:.4f}")
    if pred > 0.8:
        st.success("Faces Match!")
    elif pred > 0.5:
        st.warning("Uncertain Match. Use caution.")
    else:
        st.error(" Faces Do Not Match!")

