import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import librosa
import librosa.display
import numpy as np
import cv2
from time import time
from matplotlib import pylab
UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'wav', 'wave'}
model = load_model('./model.hdf5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def identifyImage(audio_path):
    if audio_path.find('Q1_0_Q2_0') != -1:
        diagnoses = np.array([0, 0])
    elif audio_path.find('Q1_0_Q2_1') != -1:
        diagnoses = np.array([0, 1])
    elif audio_path.find('Q1_1_Q2_0') != -1:
        diagnoses = np.array([1, 0])
    elif audio_path.find('Q1_1_Q2_1') != -1:
        diagnoses = np.array([1, 1])
    else:
        print('Name Error!')
        return 0

    audio, sr = librosa.load(audio_path)
    # For MFCCS
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=120)
    mfccsscaled = np.mean(mfccs.T, axis=0)

    # Mel Spectogram
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    melspec  = librosa.feature.melspectrogram(y=audio,sr=sr)
    s_db     = librosa.power_to_db(melspec, ref=np.max)
    librosa.display.specshow(s_db)

    savepath = os.path.join('./',str(time())+'.png')
    pylab.savefig(savepath, bbox_inches=None, pad_inches=0)
    pylab.close()

    features = np.array(mfccsscaled)
    imgpaths = (savepath)

    features = features.reshape(1, 120)
    diagnoses = diagnoses.reshape(1, 2)

    img = cv2.imread(imgpaths)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = np.array(img / 255)
    img = img.reshape(1, 64, 64, 3)
    os.remove(audio_path)
    os.remove(imgpaths)
    predic = model.predict([features, img, diagnoses])
    return predic


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print('Post Request!')
        # check if the post request has the file part
        if 'file' not in request.files:
            print('Empty Request!')
            return "someting went wrong 1"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('file not found!')
            return jsonify({
                "status":"Failure",
                "prediction":"File Not Found!",
                "confidence":"File Not Found!",
                })
        if file and allowed_file(file.filename):
            path = os.path.join(os.getcwd()+'\\static\\'+file.filename)
            file.save(path)
            prediction = identifyImage(path)
            if prediction > 0.5:
                labels = 'COVID Positive'
            elif prediction == 0:
                labels = 'Wrong Naming format'
            else:
                labels = 'COVID Negative'
            print('Results', labels, str(prediction))
            return jsonify({
                "status":"success",
                "prediction":labels,
                "confidence":str(prediction),
                })
    return jsonify({
                "status":"Failuer",
                "prediction":"Missing file request!",
                "confidence":"Missing file request!",
                })



@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
