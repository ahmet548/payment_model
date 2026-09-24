import joblib
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('transactions.csv')

df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df['process_hour'] = df['transaction_time'].dt.hour

df['is_country_match'] = (df['country'] == df['bin_country']).astype(int)

df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_user'] + 0.001)

ohe = OneHotEncoder()
xd = ohe.fit_transform(df[['country','bin_country','channel','merchant_category']]).toarray()
columns_name = ohe.get_feature_names_out()
df[columns_name] = xd

df = df.drop(columns=['transaction_id', 'user_id','transaction_time','country','bin_country','channel','merchant_category'])

y = df['is_fraud']
x = df.drop('is_fraud', axis=1)

vc = df['is_fraud'].value_counts()
plt.pie(vc.values, autopct='%1.1f%%')
plt.show()

x_train,x_test,y_train,y_test=train_test_split(
    x,y,random_state=42,train_size=0.88,stratify=y
)

class_weight_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

lgbm = LGBMClassifier(
    scale_pos_weight=class_weight_ratio * 0.5, 
    n_estimators=1000,          
    learning_rate=0.01,         
    max_depth=6,                
    num_leaves=31,             
    colsample_bytree=0.8,      
    subsample=0.8,            
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model = lgbm.fit(x_train,y_train)
model.score(x_test,y_test)

y_prob = model.predict_proba(x_test)[:, 1]

threshold_value = 0.85
y_pred = pd.DataFrame()
y_pred['pred'] = (y_prob >= threshold_value).astype(int)
y_pred['y'] = y_test.values

print(df.info())

print(y_pred)

print(classification_report(y_test,y_pred['pred'].values))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test,y_pred['pred'].values), annot=True,fmt='d',cmap='Blues')
plt.title('confusion_matrix')
plt.ylabel('real')
plt.xlabel('pred')
plt.show()

joblib.dump(model, 'fraud_model.pkl')
joblib.dump(ohe, 'encoder.pkl')

import json
with open('model_columns.json', 'w') as f:
    json.dump(list(x_train.columns), f)