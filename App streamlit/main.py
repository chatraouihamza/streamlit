import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,classification_report
from sklearn.preprocessing import StandardScaler


def get_cleandata():
    data=pd.read_csv("data/data.csv")
    data=data.drop(['Unnamed: 32','id'],axis=1)
    # print(data.isna().sum())
    data['diagnosis']=data['diagnosis'].map({ 'M':1, 'B':0 })  
    return data

def createmodel(data):
    X=data.drop(['diagnosis'],axis=1)
    Y=data['diagnosis']
    # we have to do the standarisation of our data
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    # split the data
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    # train data
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,Y_train)
    # predict the data
    Y_pred=model.predict(X_test)
    Y_pred2=model.predict(X_train)
    # print the accuracy
    print("Accuracy of the train model is : ",accuracy_score(Y_train,Y_pred2))
    print("Accuracy of the test model is : ",accuracy_score(Y_test,Y_pred))
    # classification report
    print("classification report:\t",classification_report(Y_test,Y_pred))
    return model,scaler

def main():

    data=get_cleandata()
    
    model,scaler= createmodel(data)
    # Ensure the 'model' directory exists
    os.makedirs('model', exist_ok=True)
    
    # Save the trained model to a file
    joblib.dump(model, 'model/random_forest_model.joblib')
    
    # Save the scaler to a file
    joblib.dump(scaler, 'model/scaler.joblib')
    
    # Load the model from the file
    loaded_model = joblib.load('model/random_forest_model.joblib')
    
    # Load the scaler from the file
    loaded_scaler = joblib.load('model/scaler.joblib')

if __name__=='__main__':
   
   main()    