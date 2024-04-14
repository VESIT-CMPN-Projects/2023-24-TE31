import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model



# Define a function to load and preprocess video frames
def load_video_frames(folder_path, target_size=(64, 64)):
    frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
            cap.release()
            print(f"{filename} completed")
    return frames

# Define a function to preprocess the data and extract labels
def preprocess_data_and_labels(data_folder):
    X = []
    y = []
    label_to_int = {}
    labels = os.listdir(data_folder)
    for idx, label in enumerate(labels):
        label_folder = os.path.join(data_folder, label)
        frames = load_video_frames(label_folder)
        X.extend(frames)
        y.extend([label] * len(frames))
        label_to_int[label] = idx
    return np.array(X), np.array(y), label_to_int

# Specify your data folder (e.g., 'videoSorted') and preprocess the data
data_folder = 'D:\eb'
X, y, label_to_int = preprocess_data_and_labels(data_folder)
labels = os.listdir(data_folder)
# Convert string labels to integers using the label-to-integer mapping
y = np.array([label_to_int[label] for label in y])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data further (normalize and one-hot encode labels)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
num_classes = len(label_to_int)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))


# Evaluate the model
score = model.evaluate(X_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('your_model.h5')  # You saved the model with this line

# Generate predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate the classification report
report = classification_report(np.argmax(y_test, axis=1), y_pred_classes, target_names=labels)

# Print the classification report
print("\nClassification Report:\n", report)









