import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pathlib import Path
import joblib

def missing_value_imputation(df):
    ### Impute missing values in mor_acc
    df['mort_acc'] = df['mort_acc'].fillna(df.groupby('total_acc')['mort_acc'].transform('median'))
    ## drop missing values from remaining columns
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def feature_engineering(df):
    df['zip_code'] = df.address.apply(lambda x: x[-5:])
    df['city_code']=df.address.apply(lambda x: x[-8:-6])
    df=df.drop(columns=['address', 'issue_d','earliest_cr_line', 'emp_title', 'title', 'installment'])
    df['term'] = df['term'].str.strip()
    return df

def nominal_feature_encoding(df, path:Path):
    ohe=OneHotEncoder(drop=['36 months','RENT','Not Verified','vacation','w','INDIVIDUAL','22690'])
    ohe_encoded_features = ohe.fit_transform(df[['term','home_ownership','verification_status','purpose','initial_list_status','application_type','zip_code']]).toarray()
    ohe_data=pd.DataFrame(ohe_encoded_features, columns=ohe.get_feature_names_out())
    df = pd.concat([df, ohe_data], axis=1)
    df = df.drop(columns=['term','home_ownership','verification_status','purpose','initial_list_status','application_type','zip_code', 'city_code'])
    
    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x=='Charged Off' else 0)
    joblib.dump(ohe, path)

    return df

def ordinal_feature_encoding(df, ordinal_features:list, paths:list):
    for feature, path in zip(ordinal_features, paths):
        le = LabelEncoder()
        label_encodings=le.fit_transform(df[feature].values)
        le_df = pd.DataFrame({f'le_{feature}':label_encodings.reshape(-1,)})
        df = pd.concat([df,le_df], axis=1)
        df = df.drop(columns=[feature])
        joblib.dump(le, path)

    return df

def drop_collinear_features(df):
    df = df.drop(columns=['int_rate', 'le_sub_grade','purpose_debt_consolidation', 'total_acc'])
    return df

def scale_data(df, scaler_path):
    sc=StandardScaler()
    X_scaled=sc.fit_transform(df)
    joblib.dump(sc,scaler_path)
    return X_scaled

    




    

# Index(['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
#        'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
#        'verification_status', 'issue_d', 'loan_status', 'purpose', 'title',
#        'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
#        'revol_util', 'total_acc', 'initial_list_status', 'application_type',
#        'mort_acc', 'pub_rec_bankruptcies', 'address'],
#       dtype='object')


# 'address', 'issue_d','earliest_cr_line', 'emp_title', 'title', 'installment'
# 'term','home_ownership','verification_status','purpose','initial_list_status','application_type','zip_code', 'city_code'
#  int_rate', 'le_sub_grade','purpose_debt_consolidation', 'total_acc


# loan_amnt Int,
# annual_inc Int,
# dti,
# open_acc,
# pub_rec,
# revol_bal,
# revol_util,
# mort_acc,
# pub_rec_bankruptcies,
# term_60 months,
# home_ownership_ANY,home_ownership_MORTGAGE,home_ownership_NONE,home_ownership_OTHER,home_ownership_OWN,
# verification_status_Source Verified,verification_status_Verified,purpose_car,purpose_credit_card,
# purpose_educational,purpose_home_improvement,purpose_house,purpose_major_purchase,purpose_medical,purpose_moving,purpose_other,purpose_renewable_energy,purpose_small_business,purpose_wedding,
# initial_list_status_f,
# application_type_DIRECT_PAY,application_type_JOINT,
# zip_code_00813,zip_code_05113,zip_code_11650,zip_code_29597,zip_code_30723,zip_code_48052,zip_code_70466,zip_code_86630,zip_code_93700,
# le_grade,
# le_emp_length