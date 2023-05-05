import tensorflow as tf
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from cleanData import cleanData
import random
from modelTrainer import getUserInfo

