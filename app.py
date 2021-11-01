import requests
import pickle
import numpy as np
import sys
import os
import re
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from werkzeug.utils import secure_filename
app = Flask(__name__)


model_pneumonia = load_model('pneumonia_disease.h5')

@app.route('/',methods=['GET'])
@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html')

def pneumonia_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x=image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x, axis=0)
    preds = model_pneumonia.predict(x)
    return preds


@app.route('/pneumoniadisease', methods=['GET', 'POST'])
def pneumoniadisease():
    if request.method=="GET":
        return render_template('pneumoniadisease.html', title='Pneumonia Disease')
    else:
        f=request.files["file"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',  secure_filename(f.filename))
        f.save(file_path)
        prediction = pneumonia_predict(file_path)
        pred=np.argmax(prediction, axis=1)
        if pred[0]==1:
            return render_template('pneumonia_prediction.html', prediction_text="Oops! This Chest X-Ray shows an area of lung inflammation indicating the presence of Pneumonia.", file_name = f.filename, title='Pneumonia Disease')
        else:
            return render_template('pneumonia_prediction.html', prediction_text="Great! You don't have Pneumonia.", file_name= f.filename, title='Pneumonia Disease')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__=='__main__':
	app.run(debug=True)
