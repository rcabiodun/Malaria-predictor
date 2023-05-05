import pandas as pd
import tensorflow as tf
from reframeData import cleanData
import csv
dftrain=pd.read_csv("Training.csv")


cleanData(dftrain)

y_train=dftrain.pop("prognosis")
feature_columns=[]#this is what we are giving to the linear model
y_train_vocabulary=y_train.unique()
y_train_vocabulary_list=[]
for key in dftrain.keys():
    feature_columns.append(tf.feature_column.numeric_column(key,dtype=tf.float32))


for disease in y_train_vocabulary:
    y_train_vocabulary_list.append(disease)

def getUserInfo():
    headers=[]    
    userReply=[]
    for key in dftrain.keys():
        headers.append(key)
    
    for header in headers:
        if  header.lower() in ["chills","vomiting","high_fever","loss_of_appetite","fatigue","sweating","headache","nausea","muscle_pain"]:
            reply=input(f"Do you get or feel like {header}, y/n: ")
            if reply =="y":
                userReply.append(1)
            else:
                userReply.append(0)

        else:
            userReply.append(0)
    count=0
    for i in headers:
        print(f"{headers[count]} --> {userReply[count]}")
        count+=1
    filename = "patient_details.csv"
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        headers.append("prognosis")
        # writing the fields 
        csvwriter.writerow(headers) 
        userReply.append("AIDS")   
        # writing the data rows 
        csvwriter.writerows([userReply])
#print(tf.feature_column.categorical_column_with_vocabulary_list("prognosis",y_train_vocabulary))
getUserInfo()
dfEval=pd.read_csv("Testing.csv")
y_eval=dfEval.pop("prognosis")
patient_df=pd.read_csv("patient_details.csv")
patient_y_df=patient_df.pop("prognosis")

def make_input_fn(data_df,label_df,num_of_epochs=30,shuffle=True,batch_size=52):
    def input_function():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds=ds.shuffle(2000)
        ds=ds.batch(batch_size).repeat(num_of_epochs)
        return ds
    return input_function
train_input_fn=make_input_fn(dftrain,y_train)
eval_input_fn=make_input_fn(dfEval,y_eval,num_of_epochs=1,shuffle=False)

patient_input_fn=make_input_fn(patient_df,None,num_of_epochs=1,shuffle=False)#data gotten from user
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns,n_classes=len(y_train_vocabulary_list),label_vocabulary=y_train_vocabulary_list)

linear_est.train(train_input_fn)
result=linear_est.evaluate(eval_input_fn)
print(f"{result}")
diagnosis_result=list(linear_est.predict(patient_input_fn))[0]
print(diagnosis_result['classes'])
count=0

for probability in diagnosis_result['probabilities']:
    print(f"{probability * 100}% chance of {diagnosis_result['all_classes'][count]}")
    count+=1
#print(tf.feature_column.categorical_column_with_vocabulary_list("prognosis",y_train_vocabulary))
