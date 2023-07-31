# import cupy as cp
import cupy as np
import cv2
import pickle
from google.colab import files


# Function to preprocess the input image
# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to 28x28 pixels and convert it to grayscale
    image_gray = cv2.cvtColor(cv2.resize(image, (28, 28)), cv2.COLOR_BGR2GRAY)

    # Threshold the image to make the background white
    _, image_thresh = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)

     # Check if the background is white (255) or black (0)
    if np.mean(image_thresh) > 128:
        # Reverse the colors (black to white, white to black)
        image_thresh_reversed = cv2.bitwise_not(image_thresh)
    else:
        # Keep the colors as is (black background, white text)
        image_thresh_reversed = image_thresh

    # Normalize the pixel values
    image_norm = image_thresh_reversed / 255.0

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


# Upload the image file
uploaded = files.upload()

# Check if an image file was uploaded
if len(uploaded) == 0:
    print("No image file uploaded.")
    exit(1)

# Get the uploaded image file
image_path = next(iter(uploaded))

# Load the input image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Failed to load the image.")
    exit(1)

# Perform the prediction
predicted_digit = predict_digit(image)

# Print the predicted digit
print("Predicted Digit:", predicted_digit)
