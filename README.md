AI Handwritten Digit Recognition 🖊️

Developer: S. Lavan Chary

This project is a web-based AI application built using Streamlit and TensorFlow/Keras to recognize handwritten digits (0–9) from images or drawings. Users can either draw a digit or upload an image, and the AI will predict the digit instantly along with a confidence score.

Features:
Draw or Upload: Users can draw digits on paper or upload images in png, jpg, or jpeg formats.
Instant Prediction: The model predicts the digit immediately.
Confidence Visualization: Displays a probability bar chart and JSON format confidence for all digits.
Random Example: Users can try a random MNIST digit to see predictions.
Refresh Model: Reload the AI model anytime from the sidebar.
Enhanced UX: Smooth animations, background gradient, and friendly user experience.

Requirements:
Install all dependencies using:
pip install -r requirements.txt

Minimum requirements include:
Python 3.8+
Streamlit
TensorFlow
NumPy
Matplotlib
Pillow

How to Run:
Open VS Code in the project folder S.Lavan_AI_Handwritten_Digit_Recognition.
Open the terminal in VS Code.

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Your browser will open with the AI Handwritten Digit Recognition interface.

Usage:
upload files. Or
Click buttons to try random digits or refresh the model.
View prediction and confidence for all digits.

Notes:
The AI model is trained on the MNIST dataset.
Images are automatically resized to 28x28 grayscale and normalized for prediction.
Model loading and predictions include spinner animations for better user experience.

Screenshots:
Optional: Add screenshots of the app interface here for better clarity.

Developer Contact:
S. Lavan Chary
Email: lavans110@gmail.com / LinkedIn: www.linkedin.com/in/lavan-srungari-493266297 / GitHub: https://github.com/lavan2003

Enjoy predicting digits with AI!
