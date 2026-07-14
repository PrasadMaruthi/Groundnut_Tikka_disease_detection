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

### Tensorflow Model Prediction
def model_prediction(test_image):
    img=image.load_img(test_image,target_size=(224,224))
    arr=image.img_to_array(img)
    arr=np.expand_dims(arr,0)
    arr=preprocess_input(arr)
    pred=model.predict(arr,verbose=0)
    return int(np.argmax(pred)), float(np.max(pred)*100), pred[0]

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Dashboard")

pages = [
    "Home",
    "About",
    "Disease Recognition",
    "Management Strategies"
]

# Initialize page
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Home"

app_mode = st.sidebar.selectbox(
    "Select Page",
    pages,
    index=pages.index(st.session_state.app_mode)
)

# Keep session state updated
st.session_state.app_mode = app_mode
# -----------------------------------------------------------
# Get Started Button
# -----------------------------------------------------------

st.markdown("### 🚀 Ready to detect Groundnut Tikka Disease?")

if st.button(
    "🔍 Go to Disease Recognition",
    use_container_width=True
):
    st.session_state.app_mode = "Disease Recognition"
    st.rerun()
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

    test_image = st.file_uploader(
        "📤 Choose an Image:",
        type=["jpg", "jpeg", "png"]
    )

    if test_image is not None:

        # Display Uploaded Image
        st.image(
            test_image,
            caption="Uploaded Groundnut Leaf",
            use_container_width=True
        )

        if st.button("🔍 Predict"):

            with st.spinner("Analyzing image..."):

                result_index, confidence, probabilities = model_prediction(test_image)

            class_names = [
                "Tikka Diseased Leaf",
                "Healthy Leaf"
            ]

            st.write("## 🧠 Prediction Result")

            # Prediction
            st.success(
                f"🌱 Model Prediction: **{class_names[result_index]}**"
            )

            # Confidence
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            # Confidence Progress Bar
            st.progress(confidence/100)

            st.markdown("---")

            # Class Probabilities
            st.subheader("📊 Class Probabilities")

            st.write(
                f"🟤 Tikka Diseased Leaf : **{probabilities[0]*100:.2f}%**"
            )

            st.write(
                f"🟢 Healthy Leaf : **{probabilities[1]*100:.2f}%**"
            )

            # Probability Chart
            st.bar_chart({
                "Probability": {
                    "Tikka Diseased": probabilities[0],
                    "Healthy": probabilities[1]
                }
            })

            st.markdown("---")

            # Recommendation
            if result_index == 0:

                st.error("⚠️ Tikka Disease Detected")

                st.markdown("""
### 🩺 Recommended Management

- ✅ Remove infected leaves immediately.
- ✅ Maintain proper field sanitation.
- ✅ Avoid overcrowding by maintaining recommended spacing.
- ✅ Follow crop rotation.
- ✅ Use resistant varieties whenever available.
- ✅ Apply recommended fungicides (e.g., Mancozeb or Chlorothalonil) according to local agricultural recommendations and product labels.
- ✅ Regularly monitor disease progression.
                """)

            else:

                st.success("✅ Healthy Leaf Detected")

                st.markdown("""
### 🌱 Recommendation

- Continue regular crop monitoring.
- Use certified seeds.
- Maintain balanced fertilization.
- Practice field sanitation.
- Irrigate and manage the crop according to recommended agronomic practices.
                """)

            st.markdown("---")

            st.info("""
**Model Information**

- Backbone: **MobileNetV3Small**
- Input Size: **224 × 224**
- Validation Accuracy: **99.71%**
- Classes: **2**
            """)

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

