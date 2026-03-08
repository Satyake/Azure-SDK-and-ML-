import argparse
import pandas as pd 
import numpy as np 
#import mltable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import mlflow
#from azure.ai.ml import command 
#from azure.ai.ml import MLClient
#from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from sklearn.preprocessing import LabelEncoder
import pickle
import mlflow
import os 
parser=argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
parser.add_argument('--training_data', type=str, dest='train')
parser.add_argument('--solver', type=str, dest='solver')
args=parser.parse_args()

path=os.path.join(args.train, 'diabetes.csv')
df=pd.read_csv(path)
df.dropna(inplace=True)

y = df[['Outcome']]

x = df[['Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age']]
LE=LabelEncoder()
y=LE.fit_transform(y)
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3)

model=LogisticRegression(solver=args.solver,C=1/args.reg_rate)
model.fit(x_train,y_train)
os.makedirs("outputs",exist_ok=True)
with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)
    #mlflow.sklearn.log_model(model, "model")
#mlflow.log_artifact("model.pkl")
y_hat=model.predict(x_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_hat, y_test)
mlflow.log_metric("Accuracy",acc)
print(f"Accuracy is {acc}")
#mlflow.log_metric("accuracy", acc)




