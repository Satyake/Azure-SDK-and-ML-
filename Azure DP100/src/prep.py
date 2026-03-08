#create data prep
import argparse
import pandas as pd 
import os 

parser=argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--output_data", type=str)
args=parser.parse_args()

df=pd.read_csv(os.path.join(args.input_data,"diabetes.csv"))
df=df.dropna()

os.makedirs(args.output_data, exist_ok=True)
df.to_csv(os.path.join(args.output_data,'cleaned.csv'), index=False)
print("Data Prep Complete")
