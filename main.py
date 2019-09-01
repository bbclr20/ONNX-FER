import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort


image_path = "image/image0002636.jpg"
model_path = "emotion_ferplus/model.onnx"
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def preprocess(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image =  cv2.resize(image, (64,64))
    image = image.astype(np.float32)
    image = image.reshape((1, 1, 64, 64))
    return image


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

# Read image and run the model
img_data = preprocess(image_path)
plt.imshow(img_data[0][0])
ort_session = ort.InferenceSession(model_path)
outputs = ort_session.run(None, {'Input3': img_data})

# # Result
# print(outputs)
print(softmax(outputs[0]))
print(emotions[np.argmax(outputs[0])])
plt.show()
