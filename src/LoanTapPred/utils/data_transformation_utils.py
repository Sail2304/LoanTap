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
    df = df.reset_index()
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

    




    

    

    