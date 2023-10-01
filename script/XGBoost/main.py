import xgboost as xgb
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold

import copy
import warnings
warnings.simplefilter('ignore')

import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, MultiLineString

from tqdm import tqdm
tqdm.pandas()

OUTPUT_DIR = '../output/'
DIR = '../output/'

kf = StratifiedKFold(n_splits=3,shuffle=False)
def stratifiedKFold_on_val(X,y):
    """_summary_

    Args:
        X (_type_): X[0]: User2, X[1]: User3
        y (_type_): y[0]: User2, y[1]: User3

    Returns:
        _type_: user2,user3ごとに分けたindexを返す
    """
    val_2 = []
    test_2 = []
    val_3 = []
    test_3 = []
    for i, (val, test) in enumerate(kf.split(X[0], y[0])):
        val_2.append(val)
        test_2.append(test)
    for i, (val, test) in enumerate(kf.split(X[1], y[1])):
        val_3.append(val)
        test_3.append(test)
    # indexを返す
    return val_2,test_2,val_3,test_3

def exec_XGBoost():
    train_features_bus_stop = pd.read_csv(OUTPUT_DIR + 'train/Hand/location_feature_every_5s.csv').iloc[:, 1:]
    train_features_bus_stop = train_features_bus_stop['distance_bus_stops']

    train_features = pd.read_csv(DIR + 'training_great_feature.csv').iloc[:, 1:]
    train_features = pd.concat([train_features, train_features_bus_stop], axis=1)

    val_features_bus_stop = pd.read_csv(OUTPUT_DIR + 'validate/Hand/location_feature_every_5s.csv').iloc[:, 1:]
    val_features_bus_stop = val_features_bus_stop['distance_bus_stops']

    val_features = pd.read_csv(DIR + 'validate_great_feature.csv').iloc[:, 1:]
    val_features = pd.concat([val_features, val_features_bus_stop], axis=1)

    test_features_bus_stop = pd.read_csv(OUTPUT_DIR + 'test/location_feature_every_5s.csv').iloc[:, 1:]
    test_features_bus_stop = test_features_bus_stop['distance_bus_stops']

    test_features = pd.read_csv(DIR + 'test_great_feature.csv').iloc[:, 1:]
    test_features = pd.concat([test_features, test_features_bus_stop], axis=1)

    train_label = np.load(OUTPUT_DIR + 'train/Label_every_5s.npy')
    val_label = np.load(OUTPUT_DIR + 'validate/Label_every_5s.npy')

    thresh = 1.5e+8
    change_indexes = np.where(np.abs(np.diff(val_label[:, 0, 0])) > thresh)[0] + 1

    user2 = val_features.iloc[:change_indexes[2]]
    user2_label = val_label[:change_indexes[2], 0, 1]

    user3 = val_features.iloc[change_indexes[2]:]
    user3_label = val_label[change_indexes[2]:, 0, 1]


    val_2, test_2, val_3, test_3 = stratifiedKFold_on_val([user2, user3], [user2_label, user3_label])
    result = np.zeros((len(val_label),8))
    models = []
    for i in range(len(val_2)):
        val = np.hstack([val_2[i],val_3[i]+change_indexes[2]])
        test = np.hstack([test_2[i],test_3[i]+change_indexes[2]])    
        
        model = xgb.XGBClassifier(max_depth=21, min_child_weight=10, learning_rate=0.0102, gamma=0.005,
                                    subsample=0.8, colsample_bytree=0.5, n_estimators=1000,
                                    n_jobs=-1, tree_method='gpu_hist', gpu_id=0)

                    # 'eval_metric': 'merror',  # early_stopping_roundsの評価指標
        model.fit(train_features, train_label[:, 0, 1], early_stopping_rounds=65, eval_set=[(train_features, train_label[:, 0, 1]), (val_features.iloc[val], val_label[val, 0, 1])],
                eval_metric='merror', verbose=False)
        models.append(copy.copy(model))
        pred = model.predict_proba(val_features.iloc[test])
        result[test] = pred

    predict = np.argmax(result, axis=1)

    test_predict = model.predict_proba(test_features)
    # 平均する
    predict = models[0].predict_proba(test_features)
    for model in models[1:]:
        predict += model.predict_proba(test_features)
    predict /= 3

    test_acc = np.load(OUTPUT_DIR + 'test/Acc_every_5s.npy')[:, 0, 0]
    test_predict = np.concatenate([np.reshape(test_acc, [-1, 1]), 
                                predict], axis=1)
    result = np.concatenate([
        np.reshape(val_label[:, 0, 0], [-1, 1]),
        result
    ], axis=1)

    np.save(OUTPUT_DIR + 'val_predict_xgboost_0626_001', result)
    np.save(OUTPUT_DIR + 'test_predict_xgboost_0626_001', test_predict)

if __name__ == '__main__':
    exec_XGBoost()