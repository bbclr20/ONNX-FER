"""An example of using the model of FER+ Emotion Recognition.
https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
"""
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image


emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def preprocess(image_path):
    input_shape = (1, 1, 64, 64)
    img = Image.open(image_path)
    img = img.resize((64, 64), Image.ANTIALIAS)
    img_data = np.array(img)
    img_data = np.resize(img_data, input_shape)
    img_data = img_data.astype(np.float32)
    return img_data

def softmax(x):
    e_x = np.exp(x - np.max(x) + 1e-7)
    return e_x / e_x.sum(axis=1) # only difference

ort_session = ort.InferenceSession('emotion_ferplus/model.onnx')
img_data = preprocess("image/image0002170.jpg")
outputs = ort_session.run(None, {'Input3': img_data})

# print(outputs[0])
# print(softmax(outputs[0]))

print(emotions[np.argmax(outputs[0])])
