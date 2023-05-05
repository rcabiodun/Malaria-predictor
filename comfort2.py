import pandas as pd
import tensorflow as tf
import csv

df=pd.read_csv("seattle-weather.csv")
x_train=df[:1200]
x_testing=df[1200:]
y_train=x_train.pop("weather")
y_testing=x_testing.pop("weather")
y_train_vocabulary=[]

for weather_type in  y_train.unique():
    y_train_vocabulary.append(weather_type)

CATEGORICAL_COLUMNS=["date"]
NUMERICAL_COLUMNS=["precipitation","temp_max","temp_min","wind"]
feature_columns=[]

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary=x_train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))

for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

def make_input_fn(data_df,label_df,num_of_epochs=23,shuffle=True,batch_size=35):
    def input_function():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_of_epochs)
        return ds
    return input_function
train_input_fn=make_input_fn(x_train,y_train)
eval_input_fn=make_input_fn(x_testing,y_testing,num_of_epochs=1,shuffle=False)
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns,n_classes=len(y_train_vocabulary),label_vocabulary=y_train_vocabulary)
linear_est.train(train_input_fn)
result=linear_est.evaluate(eval_input_fn)
print(result)
diagnosis_result=list(linear_est.predict(eval_input_fn))[180]

print(diagnosis_result)
print(x_testing.loc[1380],y_testing[1380])
#print(tf.feature_column.categorical_column_with_vocabulary_list("prognosis",y_train_vocabulary))
