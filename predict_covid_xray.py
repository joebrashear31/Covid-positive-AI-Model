import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys
import os

# CONFIG
model_path = "covid_xray_model.h5"
img_size = (128, 128)

# Load model
model = load_model(model_path)
print("Model loaded.")

# Load and preprocess image
def load_and_predict(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "COVID Positive" if prediction >= 0.5 else "Normal"

    # Display image and prediction
    plt.imshow(img)
    plt.title(f"Prediction: {label} ({prediction:.2f})")
    plt.axis("off")
    plt.show()

# Run prediction from CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_covid_xray.py path_to_image.jpg")
    else:
        load_and_predict(sys.argv[1])
