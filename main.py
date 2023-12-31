"""
Main file to define and host API related methods
"""

# Imports
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import inference

file_dir = os.path.dirname(__file__)
modelpath = './model'
filename = ['rfc_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Decalre class using the first row of census.csv as sample


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                'age': 50,
                'workclass': "Private",
                'fnlgt': 234721,
                'education': "Doctorate",
                'education_num': 16,
                'marital_status': "Separated",
                'occupation': "Exec-managerial",
                'relationship': "Not-in-family",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 50,
                'native_country': "United-States"
            }
        }


# Initialize API object
app = FastAPI(
    title='Inference API ML Model',
    description='API interface to run get predictions from sample data',
    version='0.0.1')

# load model artifacts on startup of the application to reduce latency


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    # if saved model exits, load the model from disk
    if os.path.isfile(os.path.join(file_dir, modelpath, filename[0])):
        model = pickle.load(
            open(
                os.path.join(
                    file_dir,
                    modelpath,
                    filename[0]),
                "rb"))
        encoder = pickle.load(
            open(
                os.path.join(
                    file_dir,
                    modelpath,
                    filename[1]),
                "rb"))
        lb = pickle.load(
            open(
                os.path.join(
                    file_dir,
                    modelpath,
                    filename[2]),
                "rb"))

# Greetings


@app.get("/")
async def greetings():
    """
    Function to Greet user when he visits the website or hits default GET API
    """
    return "Welcome!"

# This allows sending of data (our InferenceSample) via POST to the API.


@app.post("/inference/")
async def ingest_data(inference: InputData):
    """
    Function to call model's predict method and serve the output
    """
    data = {'age': inference.age,
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
    print(f"*******{os.path.join(file_dir, modelpath,filename[0])}*******")
    if os.path.isfile(os.path.join(file_dir, modelpath, filename[0])):
        model = pickle.load(
            open(
                os.path.join(
                    file_dir,
                    modelpath,
                    filename[0]),
                "rb"))
        encoder = pickle.load(
            open(
                os.path.join(
                    file_dir,
                    modelpath,
                    filename[1]),
                "rb"))
        lb = pickle.load(
            open(
                os.path.join(
                    file_dir,
                    modelpath,
                    filename[2]),
                "rb"))

        X, _, _, _ = process_data(sample,
                                  categorical_features=cat_features,
                                  training=False,
                                  encoder=encoder,
                                  lb=lb
                                  )

        # get model prediction which is a one-dim array like [1]
        prediction = model.predict(X)

        # convert prediction to label and add to data output
        if prediction[0] > 0.5:
            prediction = '>50K'
        else:
            prediction = '<=50K'
        data['prediction'] = prediction

    return data


if __name__ == '__main__':
    pass
