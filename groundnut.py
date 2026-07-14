import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

st.set_page_config(page_title="Groundnut Tikka Disease Recognition",page_icon="🥜",layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("GN_tikka_mobilenetv3_final.keras")
model=load_model()
CLASS_NAMES=["Groundnut Tikka Diseased","Groundnut Healthy"]

def model_prediction(test_image):
    img=image.load_img(test_image,target_size=(224,224))
    arr=image.img_to_array(img)
    arr=np.expand_dims(arr,0)
    arr=preprocess_input(arr)
    pred=model.predict(arr,verbose=0)
    return int(np.argmax(pred)), float(np.max(pred)*100), pred[0]

st.sidebar.title("Dashboard")
page=st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Management Strategies"])

if page=="Home":
    st.title("🥜 Groundnut Tikka Disease Recognition System")
    st.write("Upload a groundnut leaf image and predict whether it is healthy or affected by Tikka disease.")
    try: st.image("coverimage.jpg",use_container_width=True)
    except: pass
    c1,c2,c3=st.columns(3)
    c1.metric("Validation Accuracy","99.71%")
    c2.metric("Model","MobileNetV3Small")
    c3.metric("Input","224×224")
elif page=="About":
    st.title("About")
    st.write("Training images: 3135  •  Validation images: 349  •  Classes: 2")
    st.write("Developed at UAS Bangalore.")
elif page=="Disease Recognition":
    st.title("Disease Recognition")
    uploaded=st.file_uploader("Upload image",type=["jpg","jpeg","png"])
    if uploaded:
        st.image(uploaded,use_container_width=True)
        if st.button("Predict"):
            idx,conf,probs=model_prediction(uploaded)
            st.success(f"Prediction: {CLASS_NAMES[idx]}")
            st.metric("Confidence",f"{conf:.2f}%")
            st.progress(conf/100)
            st.subheader("Class Probabilities")
            st.write(f"Tikka Diseased: {probs[0]*100:.2f}%")
            st.write(f"Healthy: {probs[1]*100:.2f}%")
            st.bar_chart({"Probability":[probs[0],probs[1]]})
            if idx==0:
                st.error("Management: Remove infected leaves, maintain sanitation, rotate crops and follow local fungicide recommendations.")
            else:
                st.success("Leaf appears healthy. Continue routine monitoring.")
else:
    st.title("Management Strategies")
    st.markdown("- Crop rotation\n- Field sanitation\n- Resistant varieties\n- Follow local fungicide recommendations")
st.markdown("---")
st.caption("Groundnut Tikka Disease Recognition System | MobileNetV3Small")
