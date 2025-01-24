import joblib
from pathlib import Path
import pandas as pd
from src.LoanTapPred.utils.data_transformation_utils import drop_collinear_features
import numpy as np

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


# loan_amnt Int,  done 
# annual_inc Int, done
# dti float,       done      
# open_acc Int,        done
# pub_rec Int,      done
# revol_bal Int,    done
# revol_util Int,       done    
# mort_acc Int,     done
# pub_rec_bankruptcies Int, Done
# term_60 months, categries ['36 months' '60 months']
# home_ownership_ANY,home_ownership_MORTGAGE,home_ownership_NONE,home_ownership_OTHER,home_ownership_OWN, categories ['RENT' 'MORTGAGE' 'OWN' 'OTHER' 'ANY' 'NONE']
# verification_status_Source Verified,verification_status_Verified ['Not Verified' 'Source Verified' 'Verified']
# purpose_car,purpose_credit_card,purpose_educational,purpose_home_improvement,purpose_house,purpose_major_purchase,purpose_medical,purpose_moving,purpose_other,purpose_renewable_energy,purpose_small_business,purpose_wedding,
#['vacation' 'debt_consolidation' 'credit_card' 'home_improvement'
#  'small_business' 'major_purchase' 'other' 'medical' 'wedding' 'car'
# 'moving' 'house' 'educational' 'renewable_energy']
# initial_list_status_f, ['w' 'f']
# application_type_DIRECT_PAY,application_type_JOINT, ['INDIVIDUAL' 'JOINT' 'DIRECT_PAY']
# zip_code_00813,zip_code_05113,zip_code_11650,zip_code_29597,zip_code_30723,zip_code_48052,zip_code_70466,zip_code_86630,zip_code_93700, 
#['22690' '05113' '00813' '11650' '30723' '70466' '29597' '48052' '86630''93700']
# le_grade ['B' 'A' 'C' 'E' 'D' 'F' 'G']
# le_emp_length ['10+ years' '4 years' '< 1 year' '6 years' '9 years' '2 years' '3 years'
#  '8 years' '7 years' '5 years' '1 year']


# term--> ['36 months' '60 months']--> 2
# grade--> ['B' 'A' 'C' 'E' 'D' 'F' 'G']--> 7
# sub_grade--> ['B4' 'B5' 'B3' 'A2' 'C5' 'C3' 'A1' 'B2' 'C1' 'A5' 'E4' 'A4' 'A3' 'D1'
#  'C2' 'B1' 'D3' 'D5' 'D2' 'E1' 'E2' 'E5' 'F4' 'E3' 'D4' 'G1' 'F5' 'G2'
#  'C4' 'F1' 'F3' 'G5' 'G4' 'F2' 'G3']--> 35
# emp_length--> ['10+ years' '4 years' '< 1 year' '6 years' '9 years' '2 years' '3 years'
#  '8 years' '7 years' '5 years' '1 year']--> 11
# home_ownership--> ['RENT' 'MORTGAGE' 'OWN' 'OTHER' 'ANY' 'NONE']--> 6
# verification_status--> ['Not Verified' 'Source Verified' 'Verified']--> 3
# loan_status--> ['Fully Paid' 'Charged Off']--> 2
# purpose--> ['vacation' 'debt_consolidation' 'credit_card' 'home_improvement'
#  'small_business' 'major_purchase' 'other' 'medical' 'wedding' 'car'
#  'moving' 'house' 'educational' 'renewable_energy']--> 14
# initial_list_status--> ['w' 'f']--> 2
# application_type--> ['INDIVIDUAL' 'JOINT' 'DIRECT_PAY']--> 3
# zip_code--> ['22690' '05113' '00813' '11650' '30723' '70466' '29597' '48052' '86630'
#  '93700']--> 10

class InputData:
    def __init__(self, loan_amnt, term, int_rate, grade, sub_grad,
                emp_length, home_ownership, annual_inc,
                verification_status, purpose,
                dti, open_acc, pub_rec, revol_bal,
                revol_util, total_acc, initial_list_status, application_type,
                mort_acc, pub_rec_bankruptcies, zip_code):
        
        self.loan_amnt=loan_amnt   
        self.term=term
        self.int_rate=int_rate
        self.grade=grade
        self.sub_grad=sub_grad
        self.emp_length=emp_length
        self.home_ownership=home_ownership
        self.annual_inc=annual_inc
        self.verification_status=verification_status
        self.purpose=purpose
        self.dti=dti
        self.open_acc=open_acc 
        self.pub_rec=pub_rec
        self.revol_bal=revol_bal
        self.revol_util=revol_util
        self.total_acc=total_acc
        self.initial_list_status=initial_list_status 
        self.application_type=application_type
        self.mort_acc=mort_acc
        self.pub_rec_bankruptcies=pub_rec_bankruptcies
        self.zip_code=zip_code



    def preprocess(self):
        num = pd.DataFrame({'loan_amnt':self.loan_amnt,'annual_inc':self.annual_inc,'dti':self.dti,'open_acc':self.open_acc,
                            'pub_rec':self.pub_rec,'revol_bal':self.revol_bal,'revol_util':self.revol_util,'mort_acc':self.mort_acc,
                            'pub_rec_bankruptcies':self.pub_rec_bankruptcies,'total_acc':self.total_acc, 'int_rate':self.int_rate}, index=[0])
        ohe=joblib.load(Path("artifacts/data_transformation/ohencoder.joblib"))
        le_grade=joblib.load(Path('artifacts/data_transformation/le_grade.joblib'))
        le_subgrade=joblib.load(Path('artifacts/data_transformation/le_subgrade.joblib'))
        le_emp_length=joblib.load(Path('artifacts/data_transformation/le_emp_length.joblib'))

        ohe_features = pd.DataFrame(ohe.transform([[self.term,self.home_ownership,self.verification_status,self.purpose,
                                                    self.initial_list_status,self.application_type,self.zip_code]]).toarray(), columns=ohe.get_feature_names_out())
        le_df = pd.DataFrame({'le_grade':le_grade.transform([self.grade]),
                              'le_sub_grade':le_subgrade.transform([self.sub_grad]),
                              'le_emp_length':le_emp_length.transform([self.emp_length])})
        df = pd.concat([num,ohe_features,le_df], axis=1)
        df = drop_collinear_features(df)
        sc = joblib.load(Path('artifacts/model_trainer/scaler.joblib'))
        scaled_inputs = sc.transform(df)
        return scaled_inputs
        
def predict(data):
    model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
    pred = model.predict(data)[0]
    pred_prob = round(np.max(model.predict_proba(data)[0]),2)
    # res = model.predict(data.loc[0, :].to_numpy().reshape(1,-1))
    return pred, pred_prob


if __name__=="__main__":
    Input = InputData(10000.0,'36 months', 11.44, 'B', 'B4', '10+ years', 'RENT', 117000.0, 
                      'Not Verified', 'vacation', 26.24, 16.0, 0.0, 36369.0, 41.8, 25.0, 
                      'w','INDIVIDUAL',0.0,0.0, '22690')
    data = Input.preprocess()
    pred, prob = predict(data)
    print(pred, prob)


#     loan_amnt,term,int_rate,installment,grade,sub_grade,emp_title,emp_length,home_ownership,annual_inc,verification_status,issue_d,loan_status,purpose,title,dti,earliest_cr_line,open_acc,pub_rec,revol_bal,revol_util,total_acc,initial_list_status,application_type,mort_acc,pub_rec_bankruptcies,address
# 10000.0, 36 months,11.44,329.48,B,B4,Marketing,10+ years,RENT,117000.0,Not Verified,Jan-2015,Fully Paid,vacation,Vacation,26.24,Jun-1990,16.0,0.0,36369.0,41.8,25.0,w,INDIVIDUAL,0.0,0.0,"0174 Michelle Gateway
# Mendozaberg, OK 22690"
