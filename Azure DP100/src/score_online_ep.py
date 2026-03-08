import json 
import numpy as np 
import pickle 
import os 
import pandas as pd
def init():
 global model

 model_path=os.path.join(os.getenv("AZUREML_MODEL_DIR"),'model.pkl')
 with open(model_path,"rb") as f:
    model=pickle.load(f)

def run(raw_data):
  #results=[]
  #for file_path in mini_batch:
  data=json.loads(raw_data)
  #data=np.array(data)
  #df=pd.read_csv(file_path)
  df=pd.DataFrame(data)
  X = df[['Pregnancies', 'Glucose', 'BloodPressure',
                'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']]
  preds=model.predict(X)
  #df['Prediction']=preds
  #results.append(df)
  return preds.tolist()#pd.concat(results)
