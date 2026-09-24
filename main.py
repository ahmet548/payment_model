from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import json

app = FastAPI(title='E-Ticaret Sahtekarlik Tespiti', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Geliştirme ortamı için tümüne izin veriyoruz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('fraud_model.pkl')
encoder = joblib.load('encoder.pkl')

with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)

class TransactionData(BaseModel):
    account_age_days: int
    total_transactions_user: int
    avg_amount_user: float
    amount: float
    promo_used: int
    avs_match: int
    cvv_result: int
    three_ds_flag: int
    shipping_distance_km: float
    transaction_time: str  
    country: str
    bin_country: str
    channel: str
    merchant_category: str

@app.post('/predict')
def preict_fraud(data: TransactionData):

      df = pd.DataFrame([data.dict()])

      df['transaction_time'] = pd.to_datetime(df['transaction_time'])
      df['process_hour'] = df['transaction_time'].dt.hour
      df['is_country_match'] = (df['country'] == df['bin_country']).astype(int)
      df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_user'] + 0.001)

      xd = encoder.transform(df[['country','bin_country','channel','merchant_category']]).toarray()
      column_names = encoder.get_feature_names_out()

      df[column_names] = xd

      df.drop(columns=['country','bin_country','channel','merchant_category','transaction_time'])

      df = df.reindex(columns=model_columns, fill_value=0)

      fraud_probability = model.predict_proba(df)[0][1]
        
      threshold = 0.40
      is_fraud = bool(fraud_probability >= threshold)
        
      return {
          "fraud_probability": round(fraud_probability, 4),
          "is_fraud": is_fraud,
          "action": "REJECT" if is_fraud else "APPROVE"
      }