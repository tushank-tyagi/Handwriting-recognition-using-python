import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained OCR model
model = load_model('handwritten_digit_ocr_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Resize the image to 28x28 (same as the input size of the CNN model)
    resized = cv2.resize(thresh, (100, 100), interpolation=cv2.INTER_AREA)
    # Reshape the image to match the input shape of the model
    reshaped = resized.reshape((1, 100, 100, 1))
    # Normalize the pixel values
    normalized = reshaped.astype('float32') / 255.0
    return normalized

# Load the input image
input_image = cv2.imread('test.jpg')

# Preprocess the input image
processed_image = preprocess_image(input_image)

# Use the model to predict the digit
prediction = model.predict(processed_image)

# Get the predicted digit
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
