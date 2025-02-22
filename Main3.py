#Imports
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

#Read data
df=pd.read_csv("data.csv")
df = df.loc[:,~df.columns.str.contains('^Unnamed')]
data=df.drop(["class"],axis=1)
clas=df["class"]

trainx,textx,trainy,texty=train_test_split(data,clas,test_size=0.2)


#Create Model
model=LogisticRegression()
model.fit(trainx,trainy)

with open("LogisticModel.joblib", "wb") as f:
    dump(model, f, protocol=5)