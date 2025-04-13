import streamlit as st 
import numpy as np 
import tensorflow 
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from keras import backend


backend.clear_session()
model=load_model('model.keras')

with open('label_encoder.pkl','rb') as file:
    Lab_encode=pickle.load(file)
    
with open('OneHotEncoding.pkl','rb') as file:
    OHE=pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
st.title('Customer Churn Prediction')
geography=st.selectbox('Geography',OHE.categories_[0])
gender=st.selectbox('Gender',Lab_encode.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('credit_score')
estimated_salary=st.number_input('Estimated_salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of product',1,4)
has_cr_card=st.selectbox('Has Credict card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame({'CreditScore':[credit_score],
                         'Gender' :Lab_encode.transform([gender])[0],
                         'Age' :[age],
                         'Tenure' :[tenure],
                         'Balance':[balance],
                         'NumOfProducts':[num_of_products],
                         'HasCrCard':[has_cr_card],
                         'IsActiveMember':[is_active_member],
                         'EstimatedSalary':[estimated_salary]})


geo_encode=OHE.transform([[geography]]).toarray()
geo_encode_df=pd.DataFrame(geo_encode,columns=OHE.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encode_df],axis=1)

scaled_data=scaler.transform(input_data)


prediction=model.predict(scaled_data)
pred_prob=prediction[0][0]
st.write(f'Pred Prob : {pred_prob}')
if pred_prob>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')
