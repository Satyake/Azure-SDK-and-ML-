from azureml.core import Workspace, Experiment, Run,Datastore,Dataset
import pandas as pd
import os
new_run=Run.get_context()
ws=new_run.experiment.workspace
#import argument parser
from argparse import ArgumentParser as AP
parser=AP()
parser.add_argument('--datafolder',type=str)
args=parser.parse_args()
path=os.path.join(args.datafolder,'credit_df_.csv')

credit_df=pd.read_csv(path)
x=credit_df.iloc[:,:-1].values
y=credit_df.iloc[:,-1].values
#splits and imports 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
KNC=KNeighborsClassifier(n_neighbors=5)
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
new_run.wait_for_completion()