from pathlib import Path
import pickle
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

app = FastAPI()

model =LinearRegression()
data = pd.read_csv('Experience-Salary.csv')
print("THE DATA SO FAR : ",data.shape)

plt.scatter(data['exp(in months)'],data['salary(in thousands)'])
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')

X=data[['exp(in months)']].values
Y = data['salary(in thousands)'].values
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=42)
model.fit(X_train,Y_train)

rate_of_increase = model.coef_
starting_point =model.intercept_
print("Rate of Increase per Salary: ",rate_of_increase,"\nStarting value with no experience :",starting_point)

#now downloading the model
with open ("./model/model.pkl", 'wb') as file:
  pickle.dump(model, file)
  
class RequestMessage(BaseModel):
  experience_months: float

class ResponseMessage(BaseModel):
  ans: float

@app.get("/")
def serve_frontend():
  return FileResponse(Path(__file__).with_name("index.html"))

@app.post('/predict', response_model=ResponseMessage)
def predict_salary(request: RequestMessage):
  num = np.array([[float(request.experience_months)]], dtype=float)
  ans = model.predict(num)
  return {"ans": float(ans[0])}
