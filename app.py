from flask import Flask, request
import os
import pandas as pd
from src.OLAChurnPred.pipeline.prediction_pipeline import churnpredict


app = Flask(__name__)


@app.route("/train", methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"

#defining the endpoint which will make the prediction
@app.route("/predict", methods=['POST'])
def prediction():
    """ Returns loan application status using ML model
    """
    # Age,Gender,Education,Income,Joining_Designation,Grade,Total_Business_Value,Last_Quarterly_Rating,Quarterly_Rating_Increased,
    # Salary_Increased,Target,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C2,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C3,C4,C5,C6,C7,C8,C9

    churn_req = request.get_json()
    churn_req = dict(churn_req)
    data = pd.DataFrame(churn_req, index=[0])
    res = churnpredict(data)
    print(res)

    return {"response": int(res)}

# if __name__=="__main__":
#     # app.run(host="0.0.0.0",port=8080)        
#     app.run(host='0.0.0.0', port=8080)        

