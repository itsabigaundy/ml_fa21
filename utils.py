from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder


def getAussieData(train_percent : float = 0) -> pd.DataFrame:
    df = pd.read_csv('datasets/aussieRain/weatherAUS.csv')
    df = df[df['Location'] == 'Albury'].drop(['Location'], axis=1)

    dates = OrdinalEncoder().fit_transform(df['Date'].to_numpy().reshape((-1, 1))).astype(int)
    df['Date'] = dates

    return df


def getSarcasmData(train_percent : float = 0) -> pd.DataFrame:
    return pd.read_json('datasets/sarcasmHeadlines/Sarcasm_Headlines_Dataset.json', lines=True)


def getIncomeData() -> pd.DataFrame:
    return pd.read_csv('datasets/incomeEvaluation/income_evaluation.csv')


def getChurnData() -> pd.DataFrame:
    df = pd.read_csv('datasets/churnModeling/Churn_Modeling.csv').drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    df.loc[df['Gender'] == 'Female', 'Gender'] = 0
    df.loc[df['Gender'] == 'Male', 'Gender'] = 1

    return pd.concat([df, pd.get_dummies(df, prefix='loc', columns=['Geography',])], axis=1).drop(['Geography'], axis=1)


def getStrokeData() -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def getTitanicData() -> Tuple[pd.DataFrame, pd.DataFrame]:
    def read_csv(whichSet : str) -> pd.DataFrame:
        # TODO: Expand Drop List
        drop_list = ['PassengerId', 'Name']
        return pd.read_csv('datasets/titanic/' + whichSet + '.csv').drop(drop_list, axis=1)
    return read_csv('train'), read_csv('test')


def getZooData() -> pd.DataFrame:
    ret = pd.concat([pd.read_csv('datasets/zoo/zoo3.csv'), pd.read_csv('datasets/zoo/zoo2.csv')])
    return ret.drop( [ 'animal_name', ], axis=1 )



def getCreditRiskData() -> pd.DataFrame:
    ret = pd.read_csv('datasets/creditRisk/customer_data.csv').drop( [ 'id', ], axis=1 )
    return ret[pd.notna(ret['fea_2'])]