# import cv2
# import pickle
# import numpy as np
# import os

# class HandGestureDetector:
#     def __init__(self, model_path):
#         # Load the pre-trained model
#         with open(model_path, 'rb') as file:
#             self.model = pickle.load(file)

#         # Define the label names
#         self.label_names = {
#             1: "Pataka",
#             2: "Tripataka",
#             3: "Ardhapataka",
#             4: "Kartari Mukha",
#             5: "Mayura",
#             6: "Ardhachandra",
#             7: "Mushti",
#             8: "Shikhara",
#             9: "Kapittha",
#             10: "Katamukha",
#             11: "Sarpasirsha",
#             12: "Mrigashirsha",
#             13: "Simhamukha",
#             14: "Kangula",
#             15: "Alapadma",
#             16: "Chatura",
#             17: "Bhramara",
#             18: "Hamsasya",
#             19: "Hamsapaksha",
#             20: "Samdamsha",
#             21: "Mukula",
#             22: "Tamrachuda",
#             23: "Trishula",
#             24: "Pasham"
#         }

#     def preprocess_image(self, img):
#         # Convert image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Resize the image to 128x94 pixels
#         resized = cv2.resize(gray, (128, 94))

#         # Flatten the resized image
#         flattened = resized.flatten()

#         # Reshape the image to have 9408 features
#         img_features = flattened.reshape(1, -1)

#         # Normalize the pixel values to the range [0, 1]
#         img_features = img_features / 255.0

#         print("Image Features Shape:", img_features.shape)

#         return img_features

#     def detect_gesture(self, img_features):
#         # Make predictions on the image using the pre-trained model
#         predictions = self.model.predict(img_features)

#         # Get the predicted class
#         predicted_class = predictions[0]

#         # Get the class name based on the predicted class
#         class_name = self.label_names[predicted_class]

#         return class_name


# if __name__ == '__main__':
#     # Create Hand Gesture Detector Object
#     model_path = "hand_gesture_model.pkl"  # Path to the pre-trained model
#     detector = HandGestureDetector(model_path)

#     # Directory containing the images
#     image_dir = "testimage"

#     # Iterate over the images in the directory
#     for filename in os.listdir(image_dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             # Read image from file
#             image_path = os.path.join(image_dir, filename)
#             img = cv2.imread(image_path)

#             # Preprocess the image
#             img_features = detector.preprocess_image(img)

#             # Detect gesture and get the predicted class
#             predicted_class = detector.detect_gesture(img_features)

#             # Print the predicted class
#             print("Predicted Class:", predicted_class)

#             # Display image
#             cv2.imshow("Image", img)

#             # Wait for key press
#             cv2.waitKey(0)

#     # Close windows
#     cv2.destroyAllWindows()
import cv2
import os
import pandas as pd
import pickle

# Define the label names
label_names = {
    1: "Pataka",
    2: "Tripataka",
    3: "Ardhapataka",
    4: "Kartari Mukha",
    5: "Mayura",
    6: "Ardhachandra",
    7: "Mushti",
    8: "Shikhara",
    9: "Kapittha",
    10: "Katamukha",
    11: "Sarpasirsha",
    12: "Mrigashirsha",
    13: "Simhamukha",
    14: "Kangula",
    15: "Alapadma",
    16: "Chatura",
    17: "Bhramara",
    18: "Hamsasya",
    19: "Hamsapaksha",
    20: "Samdamsha",
    21: "Mukula",
    22: "Tamrachuda",
    23: "Trishula",
    24: "Pasham"
}

def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def extract_data(csv_file):
    ids = csv_file['id'].tolist()
    return ids

def load_images(image_folder, image_ids):
    images = []
    valid_indices = []  # Store the indices of successfully loaded images
    for i, image_id in enumerate(image_ids):
        image_path = os.path.join(image_folder, str(image_id) + '.jpg')
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}: Invalid image file")
                continue
            images.append(image)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
    return images, valid_indices

def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Resize the image to desired dimensions
        resized_image = cv2.resize(image, (128, 94))

        # Flatten the image dimensions
        flattened_image = resized_image.flatten()
        preprocessed_images.append(flattened_image)
    return preprocessed_images

def test_model(test_ids):
    # Load test images
    test_image_folder = 'images'
    test_images, valid_indices = load_images(test_image_folder, test_ids)
    test_ids = [test_ids[i] for i in valid_indices]  # Update test_ids accordingly

    # Preprocess test images
    preprocessed_test_images = preprocess_images(test_images)

    # Load the trained model
    model_path = 'hand_gesture_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Make predictions on test images
    predictions = []
    for image_features in preprocessed_test_images:
        # Reshape the image features
        image_features = image_features.reshape(1, -1)

        # Make prediction using the trained model
        predicted_class = model.predict(image_features)
        class_name = label_names[predicted_class[0]]
        predictions.append(class_name)

    return predictions

# Read test.csv
test_file_path = 'test_169.csv'
test_data = read_csv_file(test_file_path)
if test_data is not None:
    test_ids = extract_data(test_data)
    print("Test data:")
    print("IDs:", test_ids)

    # Test the model on the test data
    predictions = test_model(test_ids)
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print(f"ID: {test_ids[i]}, Prediction: {prediction}")
else:
    print("Error reading test.csv file.")
