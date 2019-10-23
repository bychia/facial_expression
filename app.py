import numpy as np
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from flask import request, jsonify

#app = Flask(__name__)
#FETCH_TOP_N_FEATURES = 10

def detect_face(image):
    test_image = cv2.imread(image)
    # Converting to grayscale
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # Displaying the grayscale image
    plt.imshow(test_image_gray, cmap='gray')


#@app.route('/upload-photo', methods=['POST'])
def upload_photo():
    # if request.method == 'POST':
    #     file = request.files['image']
        detect_face("images/20191022_181144_994.jpg")
    #     return jsonify(out_dict), 201
    # return jsonify({'error': 'Not found'}), 404

upload_photo()

# if __name__ == "__main__":
#     #app.run(host="0.0.0.0")
#     app.run(host="0.0.0.0", port=8081)
