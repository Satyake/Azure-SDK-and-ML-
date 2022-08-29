from azureml.core import Workspace, Experiment, Run,Datastore,Dataset,Run
import pandas as pd
#load _workspace
new_run=Run.get_context()
ws = new_run.experiment.workspace

credit_df=new_run.input_datasets['raw_data'].to_pandas_dataframe()

credit_df.isnull().sum()
#replace missing value with median
#missing value treatments
credit_df['BILL_AMT1'].fillna(credit_df['BILL_AMT1'].median(),inplace=True)
credit_df['BILL_AMT2'].fillna(credit_df['BILL_AMT2'].median(),inplace=True)
credit_df['BILL_AMT3'].fillna(credit_df['BILL_AMT3'].median(),inplace=True)
credit_df['BILL_AMT4'].fillna(credit_df['BILL_AMT4'].median(),inplace=True)
credit_df['BILL_AMT5'].fillna(credit_df['BILL_AMT5'].median(),inplace=True)
credit_df['BILL_AMT6'].fillna(credit_df['BILL_AMT6'].median(),inplace=True)
credit_df['PAY_AMT1'].fillna(credit_df['PAY_AMT1'].median(),inplace=True)
credit_df['PAY_AMT2'].fillna(credit_df['PAY_AMT2'].median(),inplace=True)
credit_df['PAY_AMT3'].fillna(credit_df['PAY_AMT3'].median(),inplace=True)
credit_df['PAY_AMT4'].fillna(credit_df['PAY_AMT4'].median(),inplace=True)
credit_df['PAY_AMT5'].fillna(credit_df['PAY_AMT5'].median(),inplace=True)
credit_df['PAY_AMT6'].fillna(credit_df['PAY_AMT6'].median(),inplace=True)
credit_df.isnull().sum()

from argparse import ArgumentParser as AP
parser=AP()
parser.add_argument('--datafolder',type=str)
args=parser.parse_args()

import os 
os.makedirs(args.datafolder, exist_ok=True)
path=os.path.join(args.datafolder,'credit_df_.csv')
credit_df.to_csv(path)
new_run.complete()
