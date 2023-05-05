import tensorflow as tf
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from reframeData import cleanData
import random
from comfort3 import getUserInfo
dftrain = read_csv("Training.csv")

cleanData(dftrain)

y_train = dftrain.pop("prognosis")
# load the dataset
# split into input and output columns
# ensure all data are floating point values
# encode strings to integer
# print(y_train.unique())
list_of_diseases = list(y_train.unique())
number_of_classes = len(list(y_train.unique()))
y_train_encoder = LabelEncoder()
y_train = y_train_encoder.fit_transform(y_train)

X_train, X_test, y_train, y_test = train_test_split(
    dftrain, y_train, test_size=0.33)
print(X_test)
# split into train and test datasets
# print(y_train_encoder.classes_)
# print(y_train_encoder.inverse_transform([0]))

# determine the number of input features
number_of_features = len(list(dftrain.keys()))

model = tf.keras.models.load_model('saved_model/malaria_model')

row=getUserInfo()
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# Check its architecture
#model.summary()
count = 0
for prediction in yhat[0]:
    print(
        f" showing a {prediction*100} % chance of {y_train_encoder.classes_[count]}")
    count += 1

print(y_train_encoder.classes_[argmax(yhat)])
print(y_train_encoder.inverse_transform([argmax(yhat)]))
