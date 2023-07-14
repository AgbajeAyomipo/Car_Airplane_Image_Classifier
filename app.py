import gradio as gr
import os
os.environ['KERAS-BACKEND'] = 'tensorflow'

import keras_core as keras
import numpy as np
from keras.models import load_model
import cv2
import tensorflow as tf
import tensorflow.image




def image_predict(img_):
    model = load_model('models/final_model.h5')

    img = img_ / 255.0
    img = tf.image.central_crop(img, central_fraction = .85).numpy()
    img = cv2.resize(img, dsize = [224, 224])
    img = np.expand_dims(img, axis = 0)

    pred = model.predict(img, verbose = 1)
    pred = np.argmax(pred, axis = 1)

    if pred == 0:
        answer = "The inputted image is an Airplane"
    elif pred == 1:
        answer = "The inputted image is a Car"

    return answer


# image_ = gr.Image(label = 'Input Image to be predicted')
# output = gr.Textbox(label = 'Prediction')

# demo = gr.Interface(fn = image_predict, inputs = [image_], outputs = output)


with gr.Blocks() as demo:
    image_ = gr.Image(label = 'Input Image to be predicted')
    output = gr.Textbox(label = 'Prediction')
    btn = gr.Button('Predict')
    btn.click(fn = image_predict, inputs = [image_], outputs = output)

demo.launch(share = False)