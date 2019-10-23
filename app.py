import cv2

app = Flask(__name__)
FETCH_TOP_N_FEATURES = 10

def detect_face(image):
    in_file = model.stream.read()
    bytes_in_file = BytesIO(in_file)
    buffered_reader_in_file = BufferedReader(bytes_in_file)
    result = {"Incompatible model": "True"}
    try:
        #out_dict = dill.load(buffered_reader_in_file)
        out_dict = joblib.load(buffered_reader_in_file)
        model = out_dict["model"]
        model_type = out_dict["model_type"]
        model_obj = model["model_object"]
        feature_list = dict(zip(model["features"][:FETCH_TOP_N_FEATURES], (model_obj.feature_importances_.tolist())[:FETCH_TOP_N_FEATURES]))
        result = {"_model_type": model_type,
                 "threshold_to_hitrate": out_dict['threshold_to_hitrate'],
                  "_top10_feature_list": sorted(feature_list.items(), key=lambda k: k[1], reverse=True)}
    except:
        pass
    return result


@app.route('/upload-photo', methods=['POST'])
def upload_photo():
    if request.method == 'POST':
        file = request.files['image']
        out_dict = detect_face(file)
        print(out_dict)
        return jsonify(out_dict), 201
    return jsonify({'error': 'Not found'}), 404


if __name__ == "__main__":
    #app.run(host="0.0.0.0")
    app.run(host="0.0.0.0", port=8081)
