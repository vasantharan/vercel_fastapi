from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
from sklearn.datasets import load_iris
from mangum import Mangum

target_names = load_iris().target_names

app = FastAPI()
rf_model = load('./Random_Forest.joblib')
knn_model = load('./KNN.joblib')

class input(BaseModel):
    features: list

@app.get("/")
async def home():
    return {"message": "This is a home page"}

@app.get("/about")
async def test():
    return {"hello world": "hello world"}

@app.post("/predict/rf")
async def predict_sklearn(data: input):
    x_input = np.array(data.features).reshape(1, -1)
    prediction = rf_model.predict(x_input)
    return {"prediction": target_names[prediction[0]]}

@app.post("/predict/knn")
async def predict_knn(data: input):
    x_input = np.array(data.features).reshape(1, -1)
    prediction = knn_model.predict(x_input)
    return {"prediction": target_names[prediction][0]}

handler = Mangum(app) 