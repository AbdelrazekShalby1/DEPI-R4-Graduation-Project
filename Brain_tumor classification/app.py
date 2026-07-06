from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "brain_tumor_mobilenet.keras"
)

model = load_model(MODEL_PATH)

def predict_tumor(img_path):
    img = image.load_img(
        img_path,
        target_size=(128, 128)
    )
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "Tumor Detected"
        confidence = pred * 100
    else:
        label = "No Tumor"
        confidence = (1 - pred) * 100

    return label, confidence

@app.route('/')
def home():
    return render_template('index.html') 


@app.route('/brain-classification')
def brain():
    return render_template(
        'brain_classification.html'
    )


@app.route('/brain-predict', methods=['POST'])
def brain_predict():
    file = request.files['file']
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )
    file.save(filepath)

    result, conf = predict_tumor(filepath)
    image_path = 'uploads/' + file.filename

    return render_template(
        'brain_classification.html',
        prediction=result,
        confidence=round(conf, 2),
        image=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)