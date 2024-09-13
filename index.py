import numpy as np
from joblib import load
from pydantic import BaseModel
from sklearn.datasets import load_iris

from fastapi import FastAPI

target_names = load_iris().target_names

app = FastAPI()
rf_model = load('./Random_Forest.joblib')
knn_model = load('./KNN.joblib')

class input(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "This is a home page"}

@app.get("/about")
def test():
    return {"hello world": "hello world"}

@app.post("/predict/rf")
def predict_sklearn(data: input):
    x_input = np.array(data.features).reshape(1, -1)
    prediction = rf_model.predict(x_input)
    return {"prediction": target_names[prediction[0]]}

@app.post("/predict/knn")
def predict_knn(data: input):
    x_input = np.array(data.features).reshape(1, -1)
    prediction = knn_model.predict(x_input)
    return {"prediction": target_names[prediction][0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)