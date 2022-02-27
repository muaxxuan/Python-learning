# Tutorial for libaries used in Machine Learning

# 1 - NumPy
# used for multi-dimensional array and matrix processing
# with the help of a large collection of
# high-level mathematical functions.

# Python program using NumPy for some basic mathematical operations
import numpy as np
# create 2 arrays of rank 3
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
# create 2 arrays of rank 1
v = np.array([9, 10])
w = np.array([11, 12])
# inner product of vectors
print(np.dot(v, w), "\n")
# matrix and vector product
print(np.dot(x, v), "\n")
# matrix and matrix product
print(np.dot(x, y), "\n")

# 2 - SciPy
# contains different modules for optimization, linear algebra,
# integration and statistics

# Python script using SciPy for image manipulation
from scipy.misc import imread, imsave, imresize
# read a JPEG image into a numpy array
img = imread('/Users/xuanle/Pictures/catexample.jpg.webp') # path of the image
print(img.dtype, img.shape)
# tinting the image
img_tint = img*[1, 0.45, 0.3]
# saving the tinted image
imsave('/Users/xuanle/Pictures/catexample_tinted.jpg.webp', img_tint)
# resizing the tinted image to be 300x300 pixels
img_tint_resize = imresize(img_tint, (300, 300))
# saving the resized tinted image
imsave('/Users/xuanle/Pictures/catexample_tinted_resized.jpg.webp', img_tint_resize)

# 3 - Scikit-learn
# library for classical ML algorithms, can also be used
# for data-mining and data-analysis

# Python script using scikit-learn for Decision Tree Classifier
# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# load the iris dataset
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier
model.fit(dataset.data, dataset.targer)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# 4 - Theano
# used to define, evaluate and optimize mathematical expression
# involving multi-dimensional arrays in an efficient manner

# Python program using Theano for computing a Logistic Function
import theano
import theano.tensor as T
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
logistic([[0, 1],[-1, -2]])

# 5 - TensorFLow
# can train and run deep neural networks that can be used to
# develop server AI applications. Widely used in Deep Learning

# Python program using TensorFlow for multiplying two arrays
import tensorflow as tf
# initialize 2 constants
x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])
# multiply
result = tf.multiply(x1, x2)
# initialize the session
sess = tf.Session()
# print the result
print(sess.run(result))
# close the session
sess.close()

# 6 - Keras
# used to build and design a neural network

# 7 - PyTorch
# tools and libaries that support computer vision, natural language
# processing (NLP) and many more ML programs. Performs computations
# on Tensors with GPU acceleration and also helps in creating compu-
# tational graph

# Python program using Pytorch for defining tensors fit a 2-layer
# network to randon data and calculating the loss
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0) Uncomment this to run on GPU

# N is bath sizel D_in is input dimensions; H is hidden dimensions;
# D_out is output dimensions
N, D_in, H, D_out = 64, 1000, 100, 10
# create random input and output data
x = torch.random(N, D_in, device = device, dtype = dtype)
y = torch.random(N, D_out, device = device, dtype = dtype)
# Randomly initialize weights
w1 = torch.random(D_in, H, device = device, dtype = dtype)
w2 = torch.random(H, D_out, device = device, dtype = dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# 8 - Pandas
# used for data analysis, extraction and preparation

# Python program using Pandas for arranging a given set of data into a table
import pandas as pd

data = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
        "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
        "area": [8.516, 17.10, 3.286, 9.597, 1.221],
        "population": [200.4, 143.5, 1252, 1357, 52.98]}

data_table = pd.DataFrame(data)
print(data_table)

# 9 - Matplotlib
# used for data visualization

# Python program using Matplotlib for forming a linear plot
import matplotlib.pyplot as plt
import numpy as np
# prepare the data
x = np.linspace(0, 10, 100)
# plot the data
plt.plot(x, x, label = 'Linear plot')
# add legend
plt.legend()
# show plot
plt.show()


