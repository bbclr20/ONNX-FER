"""Compare the inference result with the answer.
https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
"""
import numpy as np
import onnx
import os
import glob
from onnx import backend
import onnxruntime as ort
from onnx import numpy_helper
import matplotlib.pyplot as plt


model = onnx.load('emotion_ferplus/model.onnx')
test_data_dir = 'emotion_ferplus/test_data_set_2'

# Load inputs
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))
print(inputs)


# Load reference outputs
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# Run the model on the backend
ort_session = ort.InferenceSession('emotion_ferplus/model.onnx')
outputs = ort_session.run(None, {'Input3': inputs[0]})


# Compare the results with reference outputs.
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o, decimal=5)


# Visualize the prediction
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

result = softmax(outputs[0])
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
print(emotions[np.argmax(outputs[0])])

plt.imshow(inputs[0][0][0].astype(int))
plt.show()
