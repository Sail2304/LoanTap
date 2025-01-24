import streamlit as st 
import os
import pandas as pd
from src.LoanTapPred.pipeline.prediction_pipeline import predict, InputData


# Streamlit Inputs
st.title("Loan Application Form")

loan_amnt = st.number_input("Loan Amount", min_value=0.0, step=1000.0)
term = st.selectbox("Term", ['36 months', '60 months'])
int_rate = st.number_input("Interest Rate", min_value=0.0, step=0.1)
grade = st.selectbox("Grade", ['B', 'A', 'C', 'E', 'D', 'F', 'G'])
sub_grad = st.slider("Sub-Grade", 1, 5)
sub_grad = grade+str(sub_grad)
emp_length = st.selectbox("Employment Length", ['10+ years', '4 years', '< 1 year', '6 years', '9 years', '2 years', '3 years', '8 years', '7 years', '5 years', '1 year'])
home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'ANY', 'NONE'])
annual_inc = st.number_input("Annual Income", min_value=0.0, step=1000.0)
verification_status = st.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
purpose = st.selectbox("Purpose", ['vacation', 'debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'major_purchase', 'other', 'medical', 'wedding', 'car', 'moving', 'house', 'educational', 'renewable_energy'])
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, step=0.1)
open_acc = st.number_input("Open Credit Accounts", min_value=0, step=1)
pub_rec = st.selectbox("Public Records", [0, 1])
revol_bal = st.number_input("Revolving Balance", min_value=0.0, step=1000.0)
revol_util = st.number_input("Revolving Utilization", min_value=0.0, step=0.1)
total_acc = st.number_input("Total Accounts", min_value=0, step=1)
initial_list_status = st.selectbox("Initial List Status", ['w', 'f'])
application_type = st.selectbox("Application Type", ['INDIVIDUAL', 'JOINT', 'DIRECT_PAY'])
mort_acc = st.number_input("Mortgage Accounts", min_value=0, step=1)
pub_rec_bankruptcies = st.selectbox("Public Record Bankruptcies", [0, 1])
zip_code = st.selectbox("Zip Code", ['22690', '05113', '00813', '11650', '30723', '70466', '29597', '48052', '86630', '93700'])

# Create inputdata instance when user presses the submit button
if st.button("Submit"):
    loan_app = InputData(
        loan_amnt, term, int_rate, grade, sub_grad, emp_length, home_ownership, 
        annual_inc, verification_status, purpose, dti, open_acc, pub_rec, revol_bal, 
        revol_util, total_acc, initial_list_status, application_type, mort_acc, 
        pub_rec_bankruptcies, zip_code
    )
    data = loan_app.preprocess()
    res, prob = predict(data)

    # Display created Loan Application info
    if res:
        st.write(f"There might be risk in giving credit to this person with probability {prob}")
    else:
        st.write(f"There is no risk in giving credit to this person with probability {prob}")

