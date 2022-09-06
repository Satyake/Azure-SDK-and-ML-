import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from azureml.core import Run
from azureml.core import Workspace
from azureml.core import Experiment
from interpret.ext.blackbox import TabularExplainer
import argparse
from argparse import ArgumentParser
AP=ArgumentParser()
AP.add_argument('--input_dataset',type=str)
args=AP.parse_args()
run=Run.get_context()
ws=run.experiment.workspace
input_ds=run.input_datasets['raw_data'].to_pandas_dataframe()
input_ds=pd.get_dummies(input_ds)
x=input_ds.iloc[:,:-1]
y=input_ds.iloc[:,-1]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
rfc=RandomForestClassifier()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
features=list(x.columns)
classes=['NotGreater','Greater']
tab_explainer = TabularExplainer(rfc, x_train,features=features, classes=classes)
explanation=tab_explainer.explain_global(x_train)
from azureml.interpret import ExplanationClient
#using explanation clients
explain_client=ExplanationClient.from_run(run)
#uploading the explanations
explain_client.upload_model_explanation(explanation,comment='Explainations for Random Forest_Tabular')
run.complete()