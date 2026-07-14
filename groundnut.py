import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("GN_tikka_mobilenetv3_final.keras")

model = load_model()

### Tensorflow Model Prediction
def model_prediction(test_image):

    img = image.load_img(
        test_image,
        target_size=(224,224)
    )

    input_arr = image.img_to_array(img)

    input_arr = np.expand_dims(input_arr, axis=0)

    # Same preprocessing used during training
    input_arr = preprocess_input(input_arr)

    prediction = model.predict(
        input_arr,
        verbose=0
    )

    result_index = np.argmax(prediction)

    confidence = float(np.max(prediction)*100)

    probabilities = prediction[0]

    return result_index, confidence, probabilities

## Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Management Strategies"])

### Home Page
if(app_mode=="Home"):
    st.header("🥜 GROUNDNUT TIKKA DISEASE RECOGNITION SYSTEM")
    image_path = "coverimage.jpg"
    st.image(image_path, use_column_width=True)

    st.markdown("""
    Welcome to the **Groundnut Tikka Disease Recognition System**!

    This system is designed to detect **Tikka disease in groundnut** using deep learning techniques.
    ---
    ### 🔍 How It Works
    1. Upload a groundnut leaf image
    2. Model analyzes the image 
    3. Get instant prediction
    ---
    ### 🌟 Supported Classes
    - **Tikka Diseased Leaf**
    - **Healthy Leaf**
    ---
    ### 🚀 Get Started
    Go to **Disease Recognition** and upload an image.
    ---
    ✨ *AI-based solution for early disease detection in groundnut.*

""")

### About Page 
elif(app_mode=="About"): 
    st.header("About")
    image_path_1 = "pipeline.png"
    st.image(image_path_1, use_column_width=True)
    st.markdown(""" 
    ## 📊 Dataset Information 
    This dataset is created using **image augmentation techniques** to improve model performance. 
    
    ### 📁 Dataset Content 
    - Training Set ~ 3200
    - Validation Set ~500
    
    --- 
    
    ## 🥜 About the Project 
    The **Groundnut Tikka Disease Recognition System** is developed to detect **Tikka disease in groundnut leaves** using deep learning. 
    
    --- 
    
    ## 👨‍🔬 Development Team
    
    **Maruthi Prasad B P**  
    Department of Genetics and Plant Breeding  
    University of Agricultural Sciences, Bangalore  
    
    **Harish J**  
    Department of Plant Pathology  
    University of Agricultural Sciences, Bangalore  
    
    **[Developer Name 3]**  
    Department: *(Add here)*  
    University: *(Add here)*  
    
    ---  
    
    ## 🏫 Acknowledgement 
    This work is supported by the **University of Agricultural Sciences, Bangalore**,  
    providing a strong foundation for research in agriculture and AI.
    
    ---  
    🌱 *Empowering agriculture with AI.*
""")
    
### Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("🥜 Disease Recognition")

    st.markdown("Upload a groundnut leaf image to detect Tikka disease.")

    test_image = st.file_uploader("📤 Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:

    # Automatically display uploaded image
    st.image(
        test_image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        st.write("### 🧠 Prediction Result")

        result_index, confidence, probabilities = model_prediction(test_image)

        class_names = [
            "Tikka Diseased Leaf",
            "Healthy Leaf"
        ]

        st.success(f"🌱 Model Prediction: **{class_names[result_index]}**")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        st.write(
            f"🟤 Tikka Diseased Leaf : {probabilities[0]*100:.2f}%"
        )

        st.write(
            f"🟢 Healthy Leaf : {probabilities[1]*100:.2f}%"
        )

        st.progress(confidence/100)

        # Disease-specific recommendation
        if result_index == 0:

            st.error("⚠️ Tikka Disease Detected")

            st.markdown("""
### Recommended Management

- Use resistant varieties.
- Remove infected leaves.
- Maintain proper plant spacing.
- Apply recommended fungicides (e.g., Mancozeb or Chlorothalonil) following local agricultural recommendations and label instructions.
- Follow crop rotation and field sanitation.
""")

        else:

            st.success("✅ Healthy Leaf")

            st.markdown("""
### Recommendation

- Continue regular crop monitoring.
- Use certified seeds.
- Maintain balanced fertilization.
- Follow good agronomic practices.
""")

else:

    st.warning("⚠️ Please upload an image to proceed.")
    ---
    
    
    ⚠️ *Consult agricultural experts before applying chemicals.*

""")

