import joblib
from azureml.core.model import Model
import pandas as pd 
import json
def init():
    global ref_cols, predictor
    model_path=Model.get_model_path('adultincome')
    ref_cols,predictor=joblib.load(model_path)
#run function for predictions
def run(raw_data):
    data_dict=json.loads(raw_data)['data']
    data=pd.DataFrame.from_dict(data_dict)
    data_enc=pd.get_dummies(data)
    deploy_cols=data_enc.columns
    missing_cols=ref_cols.difference(deploy_cols)
    for cols in missing_cols:
        data_enc[cols]=0
    data_enc=data_enc[ref_cols]
    predictions=predictor.predict(data_enc)
    classes=['Less than 50k','Greater than 50k']
    predicted_classes=[]
    for prediction in predictions:
        predicted_classes.append(classes[prediction])
    return json.dumps(predicted_classes)
            

