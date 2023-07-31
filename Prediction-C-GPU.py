import cupy as np
import cv2
import pickle
from google.colab import files
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os

def preprocess_image(image):
     # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the mean value of the preprocessed image
    mean_value = np.mean(image_gray)

    # Determine the thresholding type based on the mean value
    if mean_value < 128:  # If background is black (white text)
        _, image_thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
    else:  # If background is white (black text)
        _, image_thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Normalize the pixel values
    image_norm = image_thresh / 255.0

    # Reshape the image to match the input shape of the model
    image_reshaped = image_norm.reshape(1, 1, 28, 28).astype(np.float32)

    # Show the preprocessed image
    plt.imshow(image_reshaped.squeeze(), cmap='gray')
    plt.title("Preprocessed Image")
    plt.show()

    return image_reshaped


# Function to predict the digit from the input image
def predict_digit(image):
    # Load the trained model
    with open('model.pkl', 'rb') as f:
        network = pickle.load(f)

    # Preprocess the image
    image = preprocess_image(image)

    # Perform the prediction using the trained model
    prediction = network.predict(np.array(image))

    # Get the predicted digit
    digit = np.argmax(prediction)

    return digit.get()


uploaded = files.upload()

if len(uploaded) == 0:
    print("No image file uploaded.")
    exit(1)

image_path = next(iter(uploaded))

image = cv2.imread(image_path)

if image is None:
    print("Failed to load the image.")
    exit(1)

predicted_digit = predict_digit(image)
print("Predicted Digit:", predicted_digit)
os.remove(image_path)