import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

### Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('tikka_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

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
    image = "logo.jpg"  
    st.image(image, width= 400)
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
        if st.button("👁️ Show Image"):
            st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            st.write("### 🧠 Prediction Result")

            result_index = model_prediction(test_image)

            class_names = [
                "Tikka Diseased Leaf",
                "Healthy Leaf"
            ]

            st.success(f"🌱 Model Prediction: **{class_names[result_index]}**")

    else:
        st.warning("⚠️ Please upload an image to proceed.")

### Management Strategies Page
elif(app_mode=="Management Strategies"):
    st.header("🥜 Tikka Disease Management")

    st.markdown("""
    Effective management of **Tikka disease in groundnut** is essential for maintaining crop health and yield.
    
    ---
    ## 🟤 Tikka Diseased Leaf
    - Use resistant varieties  
    - Apply fungicides like **Mancozeb / Chlorothalonil**  
    - Maintain proper spacing  
    - Remove infected leaves  
    
    ---
    
    ## 🌱 Healthy Leaf
    - Use certified seeds  
    - Balanced fertilization  
    - Regular monitoring  
    
    ---
    
    ## 🌟 General Recommendations
    - Crop rotation  
    - Field sanitation  
    - Timely fungicide application  
    
    ---
    
    
    ⚠️ *Consult agricultural experts before applying chemicals.*

""")

