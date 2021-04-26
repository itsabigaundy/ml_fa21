import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder


def split_data(data : pd.DataFrame, train_percent : float) -> tuple:
    if train_percent == 0:
        return data, None
    num_train = int(data.shape[0] * train_percent)
    return data.iloc[:num_train,:], data.iloc[num_train:,:]

def get_aussie_data(train_percent=0):
    df = pd.read_csv('datasets/aussieRain/weatherAUS.csv')
    df = df[df['Location'] == 'Albury'].drop(['Location'], axis=1)

    dates = OrdinalEncoder().fit_transform(df['Date'].to_numpy().reshape((-1, 1))).astype(int)
    df['Date'] = dates

    return split_data(df, train_percent)

def get_sarcasm_data(train_percent=0):
    return pd.read_json('datasets/sarcasmHeadlines/Sarcasm_Headlines_Dataset.json', lines=True)

def get_income_data():
    return pd.read_csv('datasets/incomeEvaluation/income_evaluation.csv')

def get_churn_data():
    df = pd.read_csv('datasets/churnModeling/Churn_Modeling.csv').drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    df.loc[df['Gender'] == 'Female', 'Gender'] = 0
    df.loc[df['Gender'] == 'Male', 'Gender'] = 1

    df = pd.concat([df, pd.get_dummies(df, prefix='loc', columns=['Geography',])], axis=1).drop(['Geography'], axis=1)

    return df.drop('Exited', axis=1), df['Exited']

def get_stroke_data():
    df = pd.read_csv('datasets/strokePrediction/strokeData.csv').drop(['id'], axis=1)

    df = df[np.isfinite(df['bmi'])]

    df = df[df['gender'] != 'Other']
    df.loc[df['gender'] == 'Female', 'gender'] = 0
    df.loc[df['gender'] == 'Male', 'gender'] = 1

    df.loc[df['ever_married'] == 'No', 'ever_married'] = 0
    df.loc[df['ever_married'] == 'Yes', 'ever_married'] = 1

    df.loc[df['work_type'] == 'children', 'work_type'] = 0
    df.loc[df['work_type'] == 'Never_worked', 'work_type'] = 0
    df.loc[df['work_type'] == 'Govt_job', 'work_type'] = 1
    df.loc[df['work_type'] == 'Private', 'work_type'] = 1
    df.loc[df['work_type'] == 'Self-employed', 'work_type'] = 1

    df.loc[df['Residence_type'] == 'Rural', 'Residence_type'] = 0
    df.loc[df['Residence_type'] == 'Urban', 'Residence_type'] = 1

    df.loc[df['smoking_status'] == 'never smoked', 'smoking_status'] = 0
    df.loc[df['smoking_status'] == 'Unknown', 'smoking_status'] = 1
    df.loc[df['smoking_status'] == 'formerly smoked', 'smoking_status'] = 2
    df.loc[df['smoking_status'] == 'smokes', 'smoking_status'] = 3

    return df.drop('stroke', axis=1), df['stroke']

def get_cross_validation_data(dataset):
    n = dataset.shape[0]
    num_train = int(n * 9 / 10)

    idxs = np.arange(n)
    np.random.shuffle(idxs)

    return dataset.iloc[idxs[:num_train],:], dataset.iloc[idxs[num_train:],:]