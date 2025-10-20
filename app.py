#           (Use VS code if possible, because i made it in VScode.)
#  Note: Right click the folder (S.Lavan_AI_Handwritten_Digit_Recognition) and click openwith vscode. opening the single app.py file wont run the program
#    How to run:     
#    Step 1.  Open terminal in this folder.
#    Step 2.  Run: pip install -r requirements.txt  
#    Step 3.  Run: streamlit run app.py
#    Step 4.  The app opens in your browser.
# Please folloow these steps to run the program correctly. 
# Thankyou, 
# S.Lavan Chary


import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# ------------------------------
# Original Code: Title and Model
# ------------------------------
st.set_page_config(page_title="AI Handwritten Digit Recognition", page_icon="✍️", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0E86D4;'>AI Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:gray;'>Developer: S. Lavan Chary</h3>", unsafe_allow_html=True)

# ------------------------------
# UX Enhancement: Background Gradient
# ------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e0f7fa, #ffffff);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Model with Spinner
# ------------------------------
@st.cache_resource
def load_model():
    with st.spinner('⚡ Loading AI model, please wait...'):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        model.fit(x_train / 255.0, y_train, epochs=2, verbose=0)
        time.sleep(1)  # gives smooth UX
    st.success("✅ Model loaded successfully!")
    return model

model = load_model()

# ------------------------------
# Sidebar: Upload & Draw Digit
# ------------------------------
st.sidebar.title("🖊️ Draw or Upload a Digit")
st.sidebar.info("Draw a number (0–9) on paper or upload an image. The AI will predict instantly!")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# ------------------------------
# Techy Feature: Refresh Model
# ------------------------------
if st.sidebar.button("🔄 Refresh Model"):
    with st.spinner("Reloading model, please wait..."):
        model = load_model()
    st.sidebar.success("Model reloaded successfully!")

# ------------------------------
# Techy Feature: Random Example
# ------------------------------
if st.sidebar.button("🎲 Try Random Digit"):
    mnist = keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    idx = np.random.randint(0, len(x_test))
    image_array = x_test[idx].reshape(1, 28, 28)
    with st.spinner("Predicting random digit..."):
        predicted_digit = np.argmax(model.predict(image_array))
        time.sleep(0.5)
    st.image(x_test[idx], caption=f"Random Digit: {y_test[idx]}", width=150)
    st.success(f"### 🧠 Predicted Digit: **{predicted_digit}**")
    # Plot probabilities
    probs = model.predict(image_array)[0]
    fig, ax = plt.subplots()
    ax.bar(range(10), probs, color='skyblue')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)

# ------------------------------
# Original File Upload Prediction with UX
# ------------------------------
if uploaded_file:
    with st.spinner("🔍 Processing your image, please wait..."):
        image = tf.keras.utils.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
        image_array = tf.keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = 255 - image_array  # invert color for better prediction
        image_array = image_array / 255.0
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        time.sleep(0.5)  # smooth transition

    st.image(image, caption="Your Uploaded Digit", width=150)
    st.success(f"### 🧠 Predicted Digit: **{predicted_digit}**")

    # Techy Feature: Show probability distribution
    st.markdown("#### Prediction Confidence")
    prob_df = {str(i): f"{prediction[0][i]*100:.2f}%" for i in range(10)}
    st.json(prob_df)

    # Plot probabilities
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0], color='orange')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)
else:
    st.info("👈 Upload a digit image to get prediction.")
