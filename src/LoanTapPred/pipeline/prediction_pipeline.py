import joblib
from pathlib import Path
import pandas as pd


def churnpredict(data):
    ohe = joblib.load(Path('artifacts/data_transformation/ohencoder.joblib'))
    model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    enc = ohe.transform(data['City'].values.reshape(1,-1))
    enc_df = pd.DataFrame(enc.toarray(), columns=ohe.categories_[0][1:])
    data = pd.concat([data,enc_df], axis=1)
    data = data.drop(columns=['City'])
    res = model.predict(data.loc[0, :].to_numpy().reshape(1,-1))
    return res
