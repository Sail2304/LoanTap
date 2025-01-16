import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

def change_data_types(data):   
    data['MMM-YY'] = pd.to_datetime(data['MMM-YY'], format="%m/%d/%y")
    data['Dateofjoining'] = pd.to_datetime(data['Dateofjoining'], format="%d/%m/%y")
    data['LastWorkingDate'] = pd.to_datetime(data['LastWorkingDate'], format="%d/%m/%y")

    return data

def missing_value_imputation(data):

    data['Total Business Value'].replace(to_replace=0, value = np.nan, inplace=True)
    # fill missing age and gender 
    data['Age'] = data.groupby('Driver_ID')['Age'].transform(lambda x: x.fillna(x.min()))
    data['Gender'] = data.groupby('Driver_ID')['Gender'].transform(lambda x: x.fillna(x.min()))

    ######## let's use KNNImputer to fill missing values in Total Business Value
    ## separate numeric columns
    df_num = data.loc[:,(data.dtypes=='int64')|(data.dtypes=='float64')]
    df_num.drop(columns='Sr',axis=1,inplace=True)

    ##scaling data as KNN is distance based algorithm
    Scaler=MinMaxScaler()
    df_num_scaled = Scaler.fit_transform(df_num)
    imputer = KNNImputer(n_neighbors=5)
    df_num_knn = pd.DataFrame(imputer.fit_transform(df_num_scaled),columns = df_num.columns)
    df_num_1 = pd.DataFrame(Scaler.inverse_transform(df_num_knn),columns=df_num.columns)
    df_non_num=data.loc[:,(data.dtypes!='int64')&(data.dtypes!='float64')]
    df = pd.concat([df_num_1,df_non_num], axis=1)

    return df


def group_transform_data(df):
    function_dict = {'Age':'max', 'Gender':'first','City':'first',
                 'Education_Level':'last', 'Income':'last', 
                 'Joining Designation':'last','Grade':'last', 
                 'Dateofjoining':'last','LastWorkingDate':'last',
                 'Total Business Value':'sum','Quarterly Rating':'last'}
    new_train=df.groupby(['Driver_ID','MMM-YY']).aggregate(function_dict)
    df=new_train.sort_index( ascending=[True,True])
    df = df.reset_index()
    df1=pd.DataFrame()
    df1['Driver_ID']=df['Driver_ID'].unique()
    df1['Age'] = list(df.groupby('Driver_ID',axis=0).max('MMM-YY')['Age'])
    df1['Gender'] = list(df.groupby('Driver_ID').agg({'Gender':'last'})['Gender'])
    df1['City'] = list(df.groupby('Driver_ID').agg({'City':'last'})['City'])
    df1['Education'] = list(df.groupby('Driver_ID').agg({'Education_Level':'last'})['Education_Level'])
    df1['Income'] = list(df.groupby('Driver_ID').agg({'Income':'last'})['Income'])
    df1['Joining_Designation'] = list(df.groupby('Driver_ID').agg({'Joining Designation':'last'})['Joining Designation'])
    df1['Grade'] = list(df.groupby('Driver_ID').agg({'Grade':'last'})['Grade'])
    df1['Total_Business_Value'] = list(df.groupby('Driver_ID',axis=0).sum('Total Business Value')['Total Business Value'])
    df1['Last_Quarterly_Rating'] = list(df.groupby('Driver_ID').agg({'Quarterly Rating':'last'})['Quarterly Rating'])
    df1['First_Quarterly_Rating'] = list(df.groupby('Driver_ID').agg({'Quarterly Rating':'first'})['Quarterly Rating'])
    df1['First_Income'] = list(df.groupby('Driver_ID').agg({'Income':'first'})['Income'])
    df1['Quarterly_Rating_Increased'] = df1.apply(lambda x: 1 if x['Last_Quarterly_Rating']>x['First_Quarterly_Rating'] else 0, axis=1)
    df1['Salary_Increased'] = df1.apply(lambda x: 1 if x['Income']>x['First_Income'] else 0, axis=1)
    df1['Last_Working_Date'] = df.groupby('Driver_ID').agg({'LastWorkingDate':'last'})['LastWorkingDate']
    #creating target variable
    df1['Target'] = np.zeros(shape=(len(df1)))
    df1.loc[~(df1['Last_Working_Date'].isna()), 'Target'] = 1.0
    df1.drop(columns=['Driver_ID','First_Quarterly_Rating', 'First_Income', 'Last_Working_Date'], inplace=True)
    return df1

    

    