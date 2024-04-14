# Smart Sorting: Deep Learning-Powered Metadata Creation For Documentary Video Catalog
Automate video classification, metadata extraction, and object detection using deep learning and computer vision techniques. This project streamlines data management by categorizing videos, extracting text, transcribing audio, and detecting objects, reducing manual effort across industries. Explore various deep learning approaches for optimal results.

# Steps
* First a 2D-CNN model was trained using around 1500 videos that are classified into 9 different categories such as Launch, Satellite,Graphics,etc..
* An accuracy of 95% was obtained for the model. This classified the video based on the frames extracted through it.
* Next involved the extraction of metadata through the speech and the text in the video. Which involved the techniques of OCR and Google Speech Recognition.
* Specific words were targeted and a dictionary of the words that are related to each category was created.
* After extraction of text and speech words from the extraction were targeted and then the category was predicted.
* Further the metadata also involved the detection of objects in the video.
* For this a YOLOv8 model is trained for 9 custom objects.
* The accuracy for the model was around 99%.
  
# Getting Start
* Paste the path of the video file in the code.
* Run the code.
* The output will display the extracted text and speech if any and then classify the video into respective categories.

# Contributors
* [Pranav Rane (Leader)](https://github.com/pranavrane26)
* [Amisha Chandwani (Member)](https://github.com/AmishaChandwani)
* [Anmol Gyanmote (Member)](https://github.com/anmolcodinglegend)
