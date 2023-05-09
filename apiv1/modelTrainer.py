# mlp for multiclass classification
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential, optimizers, metrics, losses, callbacks,utils
from tensorflow.keras.layers import Dense, Dropout
#from .cleanData import clean_csv_file
import random
import os
import time
from matplotlib import pyplot
# ddd

module_dir=os.path.dirname(__file__)

file_path=os.path.join(module_dir,"newTraining.csv")

dftrain = read_csv(file_path)


#clean_csv_file(dftrain)

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
    dftrain, y_train, test_size=0.005)
# print(i)
# split into train and test datasets
# print(y_train_encoder.classes_)
# print(y_train_encoder.inverse_transform([0]))

# determine the number of input features
number_of_features = len(list(dftrain.keys()))
    

def getUserInfo(report):
    print(report)
    headers = []
    userReply = []
    count=0
    for key in dftrain.keys():
        if count == 0:
            pass
        
        else:
            headers.append(key)
        count+=1

    for header in headers:
        if header.lower() in report.keys():

            if report[header] == "true":
                userReply.append(1)
            else:
                userReply.append(0)
        else:
            userReply.append(0)
    count = 0
    for i in headers:
        print(f"{headers[count]} --> {userReply[count]}")
        count += 1
    return userReply

# define model

'''

def create_model():
    model = Sequential()
    random_first_layer= random.randint(5, 30)
    random_second_layer = random.randint(5, 15)
    random_third_layer = random.randint(1, 20)
    model.add(Dense(random_first_layer,
                    activation='relu', kernel_initializer='he_normal',
                    input_shape=(number_of_features,)))
    model.add(Dropout(0.5))
    model.add(Dense(random_second_layer,activation='relu', kernel_initializer='he_normal' ))
    model.add(Dense(number_of_classes, activation='softmax'))
    # compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                 
                  metrics=[metrics.SparseCategoricalAccuracy()])
    return model



loss, acc = (1, 0)

if __name__=="__main__":
    while loss > 0.0004:
        random_epoch = random.randint(5, 180)
        random_batch_size = random.randint(1, 100)
        random_patience=random.randint(2,10)
        model = create_model()
        print(f"This training session has {random_batch_size} batch size and {random_epoch} number of epoch ")
        es=callbacks.EarlyStopping(monitor="val_loss",patience=random_patience)
        
        history=model.fit(X_train, y_train, epochs=random_epoch,callbacks=[es],
                  validation_split=0.3,
                batch_size=random_batch_size, verbose=1,)
        # evaluate the model
        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        acc=acc*100
    
        print(model.summary())

        
        print('Test Accuracy: %.3f,' % acc)
        print(f'Test Loss: {loss}')
        time.sleep(2.5)        


    
    print("DONE CALIBRATING ERI")
    # make a prediction
    model.save("a.i_model/malaria_model")
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    print(history.history)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.show()
    

    #
    row = getUserInfo()
    # yhat = model.predict([[0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]])
    yhat = model.predict([row])
    print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

    count = 0
    for prediction in yhat[0]:
        print(
            f" showing a {prediction*100} % chance of {y_train_encoder.classes_[count]}")
        count += 1

    print(y_train_encoder.classes_[argmax(yhat)])
    print(y_train_encoder.inverse_transform([argmax(yhat)]))

    '''
