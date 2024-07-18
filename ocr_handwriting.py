import cv2
import numpy as np
from keras.models import load_model
import argparse

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    image = np.expand_dims(image, axis=-1)
    image = image / 255.0
    return image

def recognize_handwriting(model, image_path):
    image = load_and_preprocess_image(image_path)
    prediction = model.predict(np.array([image]))
    recognized_text = ''.join([chr(np.argmax(char)) for char in prediction[0]])
    return recognized_text

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to pre-trained handwriting OCR model")
    ap.add_argument("-i", "--image", required=True, help="Path to input handwritten image")
    args = vars(ap.parse_args())

    print("[INFO] Loading handwriting OCR model...")
    model = load_model(args["model"])

    recognized_text = recognize_handwriting(model, args["image"])
    print(f"Recognized text: {recognized_text}")
