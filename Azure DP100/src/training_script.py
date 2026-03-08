import argparse
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--training_data", type=str)
parser.add_argument("--model_output", type=str)

args = parser.parse_args()

df = pd.read_csv(os.path.join(args.training_data, "cleaned.csv"))

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

os.makedirs(args.model_output, exist_ok=True)

with open(os.path.join(args.model_output, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("Training completed")
