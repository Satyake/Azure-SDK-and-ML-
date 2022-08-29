from azureml.core import Workspace, Experiment, Run,Datastore,Dataset
import pandas as pd
import os
new_run=Run.get_context()
ws=new_run.experiment.workspace
#import argument parser
from argparse import ArgumentParser as AP
AP=AP()
AP.add_argument('--n_neighbors',type=int)
AP.add_argument('--input-data',type=str)
args=AP.parse_args()
nn=args.n_neighbors
credit_df=new_run.input_datasets['raw_data'].to_pandas_dataframe()
x=credit_df.iloc[:,:-1].values
y=credit_df.iloc[:,-1].values
#splits and imports 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
KNC=KNeighborsClassifier(n_neighbors=nn)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
KNC.fit(x_train,y_train)
y_pred=KNC.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
#logging of metrics 
cm_dict=   {
       "schema_type": "confusion_matrix",
       "schema_version": "1.0.0",
       "data": {
           "class_labels": ["Default", "Non Default"],
           "matrix": cm.tolist()
       }
   }
new_run.log('Accuracy',accuracy_score(y_test,y_pred))
new_run.log_confusion_matrix('Confusion_Matrix',cm_dict)
new_run.complete()