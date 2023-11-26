from fastapi.testclient import TestClient
import os
import sys
import json

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome!"


def test_predict_positive():
    data = {"age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 15024,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    response = client.post("/inference", data=json.dumps(data))
    # assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_predict_negative():
    data = {"age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    response = client.post("/inference", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json()["prediction"] == ['<=50K']


def test_predict_invalid():
    data = {}
    response = client.post("/inference", json=json.dumps(data))
    assert response.status_code == 422