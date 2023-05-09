from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated,AllowAny
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from rest_framework.pagination import PageNumberPagination
import tensorflow as tf
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from .cleanData import clean_csv_file
import random
from .modelTrainer import getUserInfo
from .models import EmailList
from django.core.mail import send_mail

#we need to fine tune the model.
#It is not very accurate
#Maybe adjust the dataset to make it more accurate

def getdir():
   # "C:\Users\personal\Desktop\programming stuff\Final-Year\a.i"
    model_path=(os.path.join(os.getcwd(),"saved_model/malaria_model"))
    return model_path



@api_view(["POST"])
@permission_classes([AllowAny])
def predictor (request):
    dftrain = read_csv("Training.csv")

    #clean_csv_file(dftrain)

    y_train = dftrain.pop("prognosis")
    # load the dataset
    # split into input and output columns
    # ensure all data are floating point values
    # encode strings to integer
    # print(y_train.unique())
    
    y_train_encoder = LabelEncoder()
    y_train = y_train_encoder.fit_transform(y_train)

    X_train, X_test, y_train, y_test = train_test_split(
        dftrain, y_train, test_size=0.33)
    #print(request.data.keys())
    #print(request.data.values())

    # split into train and test datasets
    # print(y_train_encoder.classes_)
    # print(y_train_encoder.inverse_transform([0]))

    # determine the number of input features
    #number_of_features = len(list(dftrain.keys()))
    print(os.getcwd())
    model = tf.keras.models.load_model(getdir())

    row=getUserInfo(request.data)
    yhat = model.predict([row])
    print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
    # Check its architecture
    #model.summary()
    probability_of_malaria=None
    print(yhat[0])
    
    count = 0
    for prediction in yhat[0]:
        if y_train_encoder.classes_[count]=="Malaria":
            print("bitch")
            probability_of_malaria=prediction*100
        print(f" showing a {prediction*100} % chance of {y_train_encoder.classes_[count]}")
        count += 1

    print(y_train_encoder.classes_[argmax(yhat)])
    print(y_train_encoder.inverse_transform([argmax(yhat)]))
    print(probability_of_malaria)
    

    return JsonResponse({"message": probability_of_malaria})

@api_view(["POST"])
@permission_classes([AllowAny])
def submitEmail (request):
    try:
        email=EmailList.objects.create(email=request.data['email'])
        email.save()
        return JsonResponse({"message": "Email submitted successfully"})
    except Exception as e:
        print("btic3h")
        return JsonResponse({"message": "Something went wrong 🫠"})
