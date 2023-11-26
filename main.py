"""
Main file to define and host API related methods
"""

# Imports
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
import os
from ml.data import process_data
from ml.model import inference
import pandas as pd
import numpy as np

file_dir = os.path.dirname(__file__)
modelpath = './model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Decalre class using the first row of census.csv as sample
class InputData(BaseModel):
    age: int = Field(None, example=50)
    workclass: str = Field(None, example='Private')
    fnlgt: int = Field(None, example=234721)
    education: str = Field(None, example='Doctorate')
    education_num: int = Field(None, example=16)
    marital_status: str = Field(None, example='Separated')
    occupation: str = Field(None, example='Exec-managerial')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='Black')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=0)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=50)
    native_country: str = Field(None, example='United-States')

# Initialize API object
app = FastAPI(title='Inference API ML Model',
description='API interface to run get predictions from sample data', 
version='0.0.1')

# load model artifacts on startup of the application to reduce latency
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    # if saved model exits, load the model from disk
    if os.path.isfile(os.path.join(file_dir, modelpath,filename[0])):
        model = pickle.load(open(os.path.join(file_dir, modelpath,filename[0]), "rb"))
        encoder = pickle.load(open(os.path.join(file_dir, modelpath,filename[1]), "rb"))
        lb = pickle.load(open(os.path.join(file_dir, modelpath,filename[2]), "rb"))

# Greetings
@app.get("/")
async def greetings():
    return "Welcome!"

# This allows sending of data (our InferenceSample) via POST to the API.
@app.post("/inference/")
async def ingest_data(inference: InputData):
    data = {  'age': inference.age,
                'workclass': inference.workclass, 
                'fnlgt': inference.fnlgt,
                'education': inference.education,
                'education-num': inference.education_num,
                'marital-status': inference.marital_status,
                'occupation': inference.occupation,
                'relationship': inference.relationship,
                'race': inference.race,
                'sex': inference.sex,
                'capital-gain': inference.capital_gain,
                'capital-loss': inference.capital_loss,
                'hours-per-week': inference.hours_per_week,
                'native-country': inference.native_country,
                }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
    cat_features = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                    ]

    # if saved model exits, load the model from disk
    if os.path.isfile(os.path.join(file_dir, modelpath,filename[0])):
        model = pickle.load(open(os.path.join(file_dir, modelpath,filename[0]), "rb"))
        encoder = pickle.load(open(os.path.join(file_dir, modelpath,filename[1]), "rb"))
        lb = pickle.load(open(os.path.join(file_dir, modelpath,filename[2]), "rb"))
        
    X,_,_,_ = process_data(sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )

    # get model prediction which is a one-dim array like [1]                            
    prediction = model.predict(X)

    # convert prediction to label and add to data output
    if prediction[0]>0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K', 
    data['prediction'] = prediction

    return data


if __name__ == '__main__':
    pass