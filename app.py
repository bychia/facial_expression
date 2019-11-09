from io import BytesIO
from io import BufferedReader
from flask import Flask, render_template, request, jsonify

from keras.models import load_model
from werkzeug import secure_filename
from PIL import Image
# from matplotlib import pyplot as plt
import cv2
import numpy as np
import face_recognition
import keras
import copy
import os

app = Flask(__name__)

# haarcascade front face model
haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

# expression hdf5 model
model = load_model("data/model_v6_23.hdf5")

# expression dictionary
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

image_dir = os.path.join(app.root_path, 'images')

def converted_to_gray(image):
    # Converting to grayscale
    return(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

def convert_to_RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# resizing the image
def array_from_image(image):
    face_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return(np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1]))


def predict_expression(image, model=model):
    predicted_class = np.argmax(model.predict(image))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    return predicted_label


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/facial-expression', methods=['POST'])
def facial_expression():
    result = {"Expression": ""}

    if request.method == 'POST':
        image = request.files['image']
        image.save(os.path.join(image_dir, "img_captured.jpg"))

        original_image = cv2.imread(os.path.join(image_dir, "img_captured.jpg"))
        processed_image = copy.copy(original_image)
        processed_image = converted_to_gray(processed_image)
        faces_rects = haar_cascade_face.detectMultiScale(processed_image, scaleFactor=1.2, minNeighbors=5);

        if (len(faces_rects) > 0):
            # only supporting just 1 face for simplicity
            (x, y, w, h) = faces_rects[0]
            face_image = processed_image[y:y + h, x:x + w]
            (width, height) = (48, 48)
            face_image = cv2.resize(face_image, (width, height))
            cv2.imwrite(os.path.join(image_dir, 'resized_img_captured.jpg'), face_image)
            face_image = cv2.imread(os.path.join(image_dir, 'resized_img_captured.jpg'))
            expression = predict_expression(array_from_image(face_image))
            result = {"Expression": expression}
        return jsonify(result), 201

    return jsonify({'error': 'Not found'}), 404

if __name__ == "__main__":
    #app.run(host="0.0.0.0")
    app.run(host="0.0.0.0", port=8081)
