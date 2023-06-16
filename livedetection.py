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

# Function to draw a rectangle and label on an image
def draw_label(image, label, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)  # Green color for label
    thickness = 2

    # Get the size of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Calculate the position to place the label text
    x, y = position
    y -= 10  # Adjust the y-coordinate to display the label above the rectangle

    # Draw the rectangle
    cv2.rectangle(image, (x, y), (x + label_width, y - label_height - 10), color, cv2.FILLED)

    # Draw the label text
    cv2.putText(image, label, (x, y - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

# Function to perform hand gesture detection using webcam feed
def perform_hand_gesture_detection():
    # Load the trained model
    model_path = 'hand_gesture_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Hand Gesture Detection', frame)

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (128, 94))
        flattened_frame = resized_frame.flatten()

        # Reshape the frame features
        frame_features = flattened_frame.reshape(1, -1)

        # Make prediction using the trained model
        predicted_class = model.predict(frame_features)
        class_name = label_names[predicted_class[0]]

        # Draw rectangle and label on the frame
        label_position = (10, 30)
        draw_label(frame, class_name, label_position)

        # Show the frame with rectangle and label
        cv2.imshow('Hand Gesture Detection', frame)

        # Print the label name in the terminal
        print("Label: ", class_name)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Perform hand gesture detection using live webcam feed
perform_hand_gesture_detection()
