import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tr
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as pr
from sklearn.metrics import recall_score as rec


data = pd.read_csv("transfusion.csv" ,delimiter =',')

#variable selection, splitting into features and class
features = data.drop(["Donated"], axis = 1)
clas = data["Donated"]

#splitting into training and testing
f_train, f_test, c_train, c_test = tr(features, clas, test_size=0.2, random_state=1)
print("Features Training\n",f_train)
print("Features Testing\n",f_test)
print("Class Training\n",c_train)
print("Class Testing\n",c_test)

naive = GaussianNB()
naive.fit(f_train, c_train)
hasil = naive.predict(f_test)
print("hasil",hasil)

print("f1",f1(c_test,hasil,average='macro'))

print("acc",acc(c_test,hasil))

print("precision",pr(c_test,hasil,average='macro'))

print("recall",rec(c_test,hasil,average='macro'))






