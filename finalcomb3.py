import os
import cv2
import numpy as np
from keras.models import load_model
import moviepy.editor as mp
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import easyocr

# Function to load and preprocess video frames
def load_video_frames(video_path, target_size=(64, 64)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Function to extract text from video frames using EasyOCR
def extract_text_from_video_easyocr(video_path, language='en'):
    # Load the video clip
    clip = VideoFileClip(video_path)

    # Initialize EasyOCR reader
    reader = easyocr.Reader([language])

    # Extract text from each frame
    extracted_text = ""
    for frame in clip.iter_frames(fps=1):  # Change fps value as needed
        # Perform OCR on the frame
        results = reader.readtext(frame)

        # Extract text from results and append to the result
        for detection in results:
            text = detection[1]
            extracted_text += text + "\n"

    return extracted_text

# Function to extract speech from video using SpeechRecognition
def extract_speech_from_video(video_path):
    # Define your categories (sets of words)
    categories = {
        "satellite": {"satellite", "container", "PSLV C51",
                      "solar panel", "antenna", "GSAT", "payload", "spacecraft", "propulsion"
                      , "kilogram", "bus", "phases", "transponders", "performance", "cylinder"
                      , "deployed", "deployable", "INSAT", "orbit", "space", "communication", "GSL", "GSLV",
                      "telemetry", "propulsion system", "payload deployment", "communication link",
                      "Earth observation", "solar array", "attitude control", "ground station",
                      "remote sensing", "orbital maneuver", "space weather"},
        "Indoor Lab": {"sensors", "analyzed", "instruments", "computers", "ground", "application", "data",
                       "elements",
                       "ISV", "observatory", "Kelvin", "temperature", "cool", "cold", "thermal", "cryogenic",
                       "pumps", "semiconductor", "climate", "vaccum", "chambers", "evaporators", "machine",
                       "material", "system", "laboratory equipment", "scientific research", "experiment setup",
                       "environmental control", "data analysis", "specimen handling", "laboratory safety",
                       "research methodology", "data visualization", "scientific literature", "lab notebook",
                       "lab assistant"},
        "technology": {"computer", "internet", "software", "hardware", "programming", "coding", "algorithm",
                       "database", "network", "cybersecurity", "artificial intelligence", "machine learning",
                       "cloud computing", "virtual reality", "augmented reality", "nanotechnology"},
        "Indoor generic": {"payload", "view", "launch", "milestone", "presentation", "orbital velocity",
                           "engineers", "chemical", "mining",
                           "resources", "moon", "radiation", "UV rays", "collaboration", "application",
                           "presentation slides", "project management", "team collaboration", "research findings",
                           "academic publication", "technical report", "innovation", "brainstorming",
                           "problem-solving", "creativity", "project timeline", "progress review"},
        "Graphics": {"outer space", "astronaut", "youngsters", "Indians", "gravity", "biomedicine", "journey",
                     "psycological", "solar system", "object", "information",
                     "temperature", "satellite", "communication", "weather", "electromagnetic", "",
                     "visual effects", "computer-generated imagery (CGI)", "animation", "3D modeling",
                     "rendering", "special effects", "storyboard", "concept art", "motion capture",
                     "texture mapping", "character design", "rendering engine"},
        "launch": {"flight", "launch", "rocket", "time"}
    }

    # Load the video clip
    clip = mp.VideoFileClip(video_path)

    # Extract audio from the video
    audio = clip.audio

    # Save the audio as a temporary WAV file
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)

    # Use SpeechRecognition to transcribe the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    # Perform speech-to-text conversion
    try:
        text = recognizer.recognize_google(audio_data)
        print("Extracted speech from the video:", text)

        # Check if any word in the text belongs to any category
        for category, words in categories.items():
            if any(word in text.lower() for word in words):
                print("Classified as:", category)
                break  # If a match is found, stop checking for other categories
        else:
            print("No words match any category")

    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    # Clean up temporary audio file
    os.remove(audio_path)

    return text  # Return the extracted speech text

# Function to classify videos based on extracted text
def classify_video_by_text(extracted_text):
    # Define keywords or patterns for classification
    graphics_keywords = ["outer space", "astronaut", "youngsters", "Indians", "gravity", "biomedicine", "journey",
                         "psycological", "solar system", "object", "information",
                         "temperature", "satellite", "communication", "weather", "electromagnetic",
                         "telemetry", "propulsion system", "payload deployment", "communication link",
                         "Earth observation", "solar array", "attitude control", "ground station",
                         "remote sensing", "orbital maneuver", "space weather"]
    launch_keywords = ["flight", "launch", "rocket", "time"]
    satellite_keywords = ["satellite", "container", "PSLV C51",
                          "solar panel", "antenna", "GSAT", "payload", "spacecraft", "propulsion"
                          , "kilogram", "bus", "phases", "transponders", "performance", "cylinder"
                          , "deployed", "deployable", "INSAT", "orbit", "space", "communication", "GSL", "GSLV"]
    indoorlab_keywords = ["sensors", "analyzed", "instruments", "computers", "ground", "application", "data",
                          "elements",
                          "ISV", "observatory", "Kelvin", "temperature", "cool", "cold", "thermal", "cryogenic",
                          "pumps", "semiconductor", "climate", "vaccum", "chambers", "evaporators", "machine",
                          "material", "system", "laboratory equipment", "scientific research", "experiment setup",
                          "environmental control", "data analysis", "specimen handling", "laboratory safety",
                          "research methodology", "data visualization", "scientific literature", "lab notebook",
                          "lab assistant"]
    indoorgeneric_keywords = ["payload", "view", "launch", "milestone", "presentation", "orbital velocity",
                              "engineers", "chemical", "mining",
                              "resources", "moon", "radiation", "UV rays", "collaboration", "application",
                              "presentation slides", "project management", "team collaboration", "research findings",
                              "academic publication", "technical report", "innovation", "brainstorming",
                              "problem-solving", "creativity", "project timeline", "progress review"]
    graphics_keywords = ["visual effects", "computer-generated imagery (CGI)", "animation", "3D modeling",
                         "rendering", "special effects", "storyboard", "concept art", "motion capture",
                         "texture mapping", "character design", "rendering engine"]

    # Initialize classification
    classification = []

    # Check for keywords and classify accordingly
    if any(keyword in extracted_text.lower() for keyword in satellite_keywords):
        classification.append('satellite')

    if any(keyword in extracted_text.lower() for keyword in launch_keywords):
        classification.append('launch')

    if any(keyword in extracted_text.lower() for keyword in indoorlab_keywords):
        classification.append('indoor lab')
    if any(keyword in extracted_text.lower() for keyword in indoorgeneric_keywords):
        classification.append('indoor generic')
    if any(keyword in extracted_text.lower() for keyword in graphics_keywords):
        classification.append('graphics')

    return classification

# Specify the path to your video file
video_path = r"D:\eb\IndoorLab\Video005-Scene-056.mp4"

# Call the function to extract text from the video using EasyOCR
result_text = extract_text_from_video_easyocr(video_path)

# Call the function to extract speech from the video using SpeechRecognition
result_speech = extract_speech_from_video(video_path)

# Call the function to classify the video based on extracted text
video_classification_by_text = classify_video_by_text(result_text)
print("Video classification based on text:", video_classification_by_text)

# Print the extracted text from video frames
print("Extracted text from video frames:")
print(result_text)

# Load the saved model for video categorization
model_path = r'C:\Users\ANMOL GYANMOTE\Desktop\mini project codes\your_model.h5'
model = load_model(model_path)

# Preprocess the new video frames
new_frames = load_video_frames(video_path)

# Corrected line
new_frames = new_frames.astype('float32') / 255.0

# Make predictions on the new frames
predictions = model.predict(new_frames)

# Post-process the predictions
predicted_classes = np.argmax(predictions, axis=1)
class_probabilities = np.max(predictions, axis=1)

label_to_int = {
    'Crowd': 0,
    'DisplayScreen': 1,
    'Graphics': 2,
    'IndoorGeneric': 3,
    'IndoorLab': 4,
    'Launch': 5,
    'OutdoorGeneric': 6,
    'OutdoorLaunchpad': 7,
    'PersonCloseUp': 8,
    'Satellite': 9,
    # Add more class labels as needed
}

# Map integer labels back to original class labels
int_to_label = {idx: label for label, idx in label_to_int.items()}
predicted_labels = [int_to_label[idx] for idx in predicted_classes]

# Print the predicted labels and probabilities along with extracted text and speech
for label, prob in zip(predicted_labels, class_probabilities):
    
    print("Confidence:", prob)
    print("Extracted Text:", result_text)
    print("Extracted Speech:", result_speech)
    print("Predicted Label according to Text,Speech and CNN:", label)
