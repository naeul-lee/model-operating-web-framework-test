import pickle
from pydantic import BaseModel
import numpy
import onnxruntime as rt
from fastapi import FastAPI

app = FastAPI()

with open('./app/features.pickle', 'rb') as f:
    feature = pickle.load(f)
    print("features:", feature)

session = rt.InferenceSession("./app/xgboost_model.onnx")
first_input_name = session.get_inputs()[0].name
first_output_name = session.get_outputs()[0].name


# Creating objects i.e feature names
class Data(BaseModel):
    online_daily_payment_tpv_max_90days: float
    online_daily_payment_tpv_max_30days: float
    online_daily_payment_trx_max_365days: float
    online_daily_payment_trx_mean_30days: float
    online_manwon_payment_unit_trx_365days: float
    online_daily_payment_tpv_mean_365days: float
    brand_tpv_weight_365days: float
    online_daily_payment_trx_mean_365days: float
    consumption_trx_ratio_365days: float
    online_payment_method_card_trx_weight_90days: float
    online_payment_method_card_trx_weight_7days: float
    brand_trx_ratio_365days: float
    brand_tpv_ratio_365days: float
    online_manwon_payment_total_tpv_30days: float
    consumption_trx_365days: float
    online_payment_cycle_cycle_median_365days: float
    brand_trx_weight_365days: float
    brand_tpv_180days: float
    online_daily_payment_trx_max_30days: float
    consumption_age_tpv_ratio_365days: float
    consumption_trx_ratio_30days: float
    consumption_trx_180days: float
    online_payment_method_card_trx_weight_30days: float
    consumption_age_trx_ratio_365days: float
    consumption_trx_weight_180days: float
    consumption_trx_ratio_180days: float
    online_daily_payment_trx_max_180days: float
    consumption_age_gender_trx_ratio_365days: float
    consumption_tpv_ratio_365days: float
    brand_tpv_365days: float
    online_daily_payment_trx_max_7days: float
    online_daily_payment_tpv_max_365days: float
    online_daily_payment_trx_mean_180days: float
    online_payment_method_card_trx_weight_180days: float
    consumption_tpv_365days: float
    consumption_age_gender_trx_ratio_180days: float
    consumption_trx_ratio_90days: float
    online_payment_cycle_cycle_mean_90days: float
    online_payment_cycle_cycle_max_365days: float
    online_daily_payment_tpv_max_180days: float
    password_cnt_password_fin_7days: float
    online_payment_cycle_cycle_mean_365days: float
    consumption_age_trx_ratio_90days: float
    consumption_trx_90days: float
    online_daily_payment_trx_max_90days: float
    brand_trx_30days: float
    consumption_trx_weight_30days: float
    brand_trx_ratio_30days: float
    online_manwon_payment_total_tpv_max_365days: float
    brand_trx_ratio_90days: float


@app.post("/predict")
def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = [data_dict[feature] for feature in feature]

        # dict to array

        to_predict = numpy.array(to_predict).reshape(1, -1)
        print("array:", to_predict)
        pred_onx = session.run([], {first_input_name: to_predict.astype(numpy.float32)})[0]
        return {"prediction": float(pred_onx[0])}
    except Exception as e:
        return {"prediction": "error", "message": str(e)}
