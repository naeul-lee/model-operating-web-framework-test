import uvicorn
import pandas as pd
from fastapi import FastAPI, Response
import pickle

from .schemas import Features, FraudLabel

model = pickle.load(open("./artifacts/sample_model_feature_50_20220712.pkl", "rb"))
app = FastAPI()


@app.get("/")
async def root():
    return "RMS inference server run by fastAPI"


@app.post("/predict", response_model=FraudLabel)
async def predict(response: Response, payload: Features):
    def predict_score(model, X, threshold):
        prob = model.predict_proba(X)
        prob = prob[:, 1]
        prob[prob >= threshold] = 1
        prob[prob < threshold] = 0
        return prob

    payload_dict = payload.dict()
    payload_df = pd.DataFrame([payload_dict])
    prediction = predict_score(model, payload_df, 0.3)[0]
    response.headers["X-model-score"] = str(prediction)
    return FraudLabel(label=prediction)


@app.get("/ping")
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5002, reload=True)
