from typing import Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def formatDateCol(df: pd.DataFrame, date_col: str, fmt: str) -> pd.DataFrame:
    df[date_col] = [datetime.strptime(x, fmt) for x in df[date_col]]
    df = df.sort_values(by=[date_col], ignore_index=True)
    df[date_col] = [x - df[date_col][0] for x in df[date_col]]
    df[date_col] = [x.total_seconds() for x in df[date_col]]
    df[date_col] /= np.gcd.reduce(df[date_col].astype(int))
    df = df.drop_duplicates(subset=[date_col], ignore_index=True)

    return df


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


def getSzegedData() -> pd.DataFrame:
    ret = pd.read_csv('datasets/szeged/weatherHistory.csv')
    ret = ret.drop( [ 'Loud Cover', 'Daily Summary', 'Summary', ], axis=1 )

    # convert dates to numeric format and normalize
    ret['Formatted Date'] = [x[:19] + x[23:] for x in ret['Formatted Date']]
    ret = formatDateCol(ret, 'Formatted Date', "%Y-%m-%d %H:%M:%S %z")

    # convert precip type 
    ret['Precip Type'] = ret['Precip Type'].replace(np.nan, 'clear')
    enc = OrdinalEncoder(categories=[[ 'clear', 'rain', 'snow', ]])
    ret['Precip Type'] = enc.fit_transform(ret['Precip Type'].values.reshape(-1, 1)).flatten()

    return ret


def getEpicuriousData() -> pd.DataFrame:
    df = pd.read_csv('datasets/epicurious/epi_r.csv')[[
        'rating',
        'calories',
        'protein',
        'fat',
        'sodium',
        'healthy',
    ]]

    df = df.dropna()

    return df