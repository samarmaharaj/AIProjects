import numpy as np
import tensorflow as tf
import cv2

# Load the trained model in the new format
model = tf.keras.models.load_model('mnist_model.keras')

# Function to preprocess the image for prediction
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    reshaped = resized.reshape(1, 28, 28, 1) / 255.0
    return reshaped

# Set up video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest (ROI) where the number is expected to appear
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)

    # Preprocess the ROI
    processed_img = preprocess_image(roi)

    # Predict the digit
    prediction = model.predict(processed_img)
    digit = np.argmax(prediction)

    # Display the resulting frame
    cv2.putText(frame, f'Prediction: {digit}', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
