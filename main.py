import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut
#use other api providers through 1 simple interface
client = OpenAI(
  base_url = 'https://api.groq.com/openai/v1',
  api_key = os.getenv('GROQ_KEY')
)
def load_model(filename):
  with open(filename,'rb') as file:
    return pickle.load(file)


dt_model = load_model('dt_model.pkl')
knn_model = load_model('knn_model.pkl')
nb_model = load_model('nb_model.pkl')
rf_model = load_model('rf_model.pkl')
svm_model = load_model('svm_model.pkl')
xgboost_model = load_model('xgb_model.pkl')
voting_classifier = load_model('voting_classifier.pkl')
xgb_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgb_feature_engineered_model = load_model('xgboostfeatureEngineered.pkl')

#why return input_dict?
def prepare_input(credit_score,location,gender,age,tenure,balance,num_products,has_credit_card,is_active_member,estimated_salary):
  input_dict = {
    'CreditScore':credit_score,
    'Age': age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_products,
    'HasCrCard':int(has_credit_card),
    'IsActiveMember': int(is_active_member),
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if location=='France' else 0,
    'Geography_Germany': 1 if location=='Germany' else 0,
    'Geography_Spain': 1 if location=='Spain' else 0,
    'Gender_Male': 1 if gender=='Male' else 0,
    'Gender_Female': 1 if gender=='Female' else 0,
  }
  input_df = pd.DataFrame([input_dict])
  return input_df,input_dict

def make_predictions(input_df,input_dict):
  probabilities = {
    #what is predict_proba, sk_learn?
    #predict_proba gives us predictive class and predictive probability
    #predict gives the prediction (exited or not exited ) [1,0] while predict_proba gives probability that is is a certain classification
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'Random Forest': rf_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1]
  }
  avg_probability = np.mean(list(probabilities.values()))
  #difference with markdown and st.write() ? markdown lets you do **bold**  # for headers. It lets u use the markdown language
  st.markdown('# Model probabilities')
  col1,col2 = st.columns(2)
  with col1:
    st.markdown(f'### Average Probability:{round(avg_probability * 100,1)}%')

  with col2:
    st.markdown(f'### Probability by model')
    plt_bar_char = ut.bar_chart(probabilities['Random Forest'],probabilities['XGBoost'],probabilities['K-Nearest Neighbors'])
#   for model,prob in probabilities.items():
#     st.write(f'{model} :{prob}')
#   st.write(f'Average Probability: {avg_probability}')
#   return avg_probability
# st.title("Customer Churn Prediction")

  return avg_probability
def explain_prediction(probability,input_dict,surname):
  #set option is used before the .describe() this is because in output pandas may set a limit to the number of columns, typically to 20. by saying .display.max_columns, and then None we say no limits to the output of columns. 
  #  .describe() then gives us summary statistics like mean, median, and standard deviation among others.

  prompt = f""" You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a {round(probability*100,1)} % probability of churning, based on the information provided below.
  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting churn:
  Feature              | Importance
  ------------------------------------
  NumOfProducts        | 0.323888
  IsActiveMember       | 0.164146
  Age                  | 0.109550
  Geography_Germany    | 0.091373
  Balance              | 0.052786
  Geography_France     | 0.046463
  Gender_Female        | 0.045283
  Geography_Spain      | 0.036855
  CreditScore          | 0.035005
  EstimatedSalary      | 0.032655
  HasCrCard            | 0.031940
  Tenure               | 0.030054
  Gender_Male          | 0.000000

  {pd.set_option('display.max_columns',None)}

  Here are summary statistics for non-churned customers:
  {df[df['Exited']==0].describe()}

  -If the customer has over 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  - If the customer has less than a 40% risk of churning generate a 3 sentence explanation of why they might not be at risk of churning.
  - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

  Don't mention the probability of churning, or the machine learning model, or say anything technical that you would not tell the layman.
  This is meant for system admins who are laymans in data science not for the customer.
  
  """
  

  raw_response = client.chat.completions.create(
    model = 'llama-3.2-3b-preview',
    messages = [{
      'role':'user',
      'content':prompt
    }]
)
  return raw_response.choices[0].message.content

def generate_email(probability,input_dict,explanation,surname):
  prompt = f""" You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.
  You noticed a customer named {surname} has a {round(probability*100),1}% probability of churning.
  Here is the customer's information:
  {input_dict}
  Here is some explanation as to why the customer might be at risk of churning"
  {explanation}

  Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

  Make sure to list out a set of incentives to stay based on their information in buullet format. Don't ever mention the probability, machine learning model or anything technical at all whatsoever because this is meant for a layman. This is specifically for the customer, now make their personalized email.
  
  """
  raw_response = client.chat.completions.create(
    model='llama-3.1-8b-instant',
    messages = [{
      'role':'user',
      'content':prompt
    }]
  )
  
  return raw_response.choices[0].message.content

st.title('Customer Predictive modeling for churning')
df = pd.read_csv('churn.csv')
customers = [f"{row['CustomerId']}-{row['Surname']}" for _,row in df.iterrows()]

selected_customer_option = st.selectbox('Select a customer',customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split('-')[0])
  

  #without iloc it is still a pandas series with indexing. Get the customer without indexing by doing iloc[0]
  selected_customer = df.loc[df['CustomerId']==selected_customer_id].iloc[0]
  col1,col2= st.columns(2)
  
  selected_surname = selected_customer_option.split('-')[1]

  with col1:
    #will get default value from dataframe but user can adjust according to min and max rules
    credit_score = st.number_input(
      'Credit Score',
      min_value=300,
      max_value =850,
      value = int(selected_customer['CreditScore'])
    )
    #to establish default for select box an index must be put in place.
    #first get geography value of customer, this will give u its index and value from df
    #use iloc to get just the value. From there using that value find what index that
    #corresponds to according to the selectbox. Remember index from dataset is irrelevant
    #but we need to get index from select box to get the default value.
    location = st.selectbox(
      'location',['Spain','France','Germany'],
      index = ['Spain','France','Germany'].index(
        selected_customer['Geography']
      )
    )
    gender = st.radio('Gender',['Male','Female'],
                  index = 0 if selected_customer['Gender']=='Male' else 1
                     )

    age = st.number_input(
      'Age',
      min_value=18,
      max_value=100,
      value = int(selected_customer['Age'])
    )
    tenure = st.number_input(
      'Tenure (years)',
      min_value = 0,
      max_value = 50,
      value = int(selected_customer['Tenure'])
    )

    
    
    
  with col2:
    balance = st.number_input('Balance',
                             min_value = 0.0,
                             value = float(selected_customer['Balance']))
    num_products = st.number_input('Number of Products',
                              min_value = 1,
                              max_value = 10,
                              value = int(selected_customer['NumOfProducts'])
                             )
    has_credit_card = st.checkbox('Has Credt Card',
                                 value = bool(selected_customer['HasCrCard']))

    is_active_member = st.checkbox('Is Active Member',
                                  value = bool(selected_customer['IsActiveMember']))

    estimated_salary = st.number_input('Estimated Salary', min_value = 0.0, value = float(selected_customer['EstimatedSalary']))


input_df, input_dict = prepare_input(credit_score,location,gender,age,tenure,balance,num_products,has_credit_card,is_active_member,estimated_salary)

avg_probability = make_predictions(input_df,input_dict)
explanation = explain_prediction(avg_probability,input_dict,selected_customer['Surname'])

st.markdown("---")
st.subheader('Explanation of Prediction')

st.markdown(explanation)
personalized_email = generate_email(avg_probability,input_dict,explanation,selected_customer['Surname'])
st.markdown('---')
st.subheader('Personalized Email')
st.markdown(personalized_email)





    
  

  
