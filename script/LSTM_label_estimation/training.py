import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import  Dense, Dropout, Layer,LSTM,Input, Concatenate,Conv1D,MaxPooling1D,Input,Multiply
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint,TensorBoard
from tensorflow.keras.utils import Sequence
from tensorflow_addons.optimizers import AdamW
from sklearn.model_selection import train_test_split
import math
import numpy as np
import pandas as pd

import os
import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,f1_score,classification_report

from utils import config,plot_confusion_matrix,split_val
from MSRLSTM import MSRLSTM,cosine_schedule,DataGenerator,get_model

OUTPUT_DIR = '../../output/model/MSRLSTM/'
NORM_PARAM = '../../output/data/every_500/Hand_norm_params.npy'
POS = ['Bag','Hand','Hips','Torso']
TRAINING_PATH = '../../data/every_500/train/'
VAL_PATH = '../../data/every_500/validate/'
TEST_PATH = '../../data/every_500/test/'
DATA_FILENAME = 'glolacc_gyr_mag_normalized'
LABEL_FILENAME = 'Label_nan_dropped'


def load_features(type,pos='Hand'):
    LOC_FILENAME = "location_feature_every_5s.csv"
    USING_FEATURE_NAMES =[
        "Accuracy",
        "Altitude",
        "speed",
        "distance_bus_routes",
        "distance_railways",
        "distance_subways",
        "distance_car_roads",
        "distance_railways_station",
        "distance_subways_station",
        "distance_bus_stops"
        ]
    if type != 'test':
        location = pd.read_csv(os.path.join('../../data/every_500/',type,pos,LOC_FILENAME))
    else:
        location = pd.read_csv(os.path.join('../../data/every_500/',type,LOC_FILENAME))
    location = location[USING_FEATURE_NAMES]
    print(len(np.where(location.isnull().any(axis=1))[0]))
    location = location[USING_FEATURE_NAMES].clip(0,location[USING_FEATURE_NAMES].quantile(0.99),axis=1)
    location = location.fillna(method='bfill',limit=50)
    location = location.fillna(method='ffill',limit=50)
    
    print(len(np.where(location.isnull().any(axis=1))[0]))
    location_na = (~location.isnull().any(axis=1))
    
    location = location.fillna(location.mean())
    location = np.c_[location_na.T,location.values]
    print(location.shape)
    return location

def load_data():
    TRAINING_PATH = '../../data/every_500/train/'
    VAL_PATH = '../../data/every_500/validate/'
    TEST_PATH = '../../data/every_500/test/'
    DATA_FILENAME = 'glolacc_gyr_mag_normalized'
    LABEL_FILENAME = 'Label_nan_dropped'
    X_train = np.load(os.path.join(TRAINING_PATH,"Hand",DATA_FILENAME+".npy"))
    y_train = np.load(os.path.join(TRAINING_PATH,"Hand",LABEL_FILENAME+".npy"))[:,1]-1
    X_val_raw = [
    np.load(os.path.join(VAL_PATH,"Hand",DATA_FILENAME+"_user2.npy")),
    np.load(os.path.join(VAL_PATH,"Hand",DATA_FILENAME+"_user3.npy")),
    ]
    y_val_raw = [
        np.load(os.path.join(VAL_PATH,"Hand",LABEL_FILENAME+"_user2.npy"))[:,1]-1,
        np.load(os.path.join(VAL_PATH,"Hand",LABEL_FILENAME+"_user3.npy"))[:,1]-1,
    ]
    l_train = load_features('train',use_location=True)
    l_train = np.delete(l_train,120950,0)
    l_val_raw = load_features('validate',use_location=True)
    l_val_raw = [l_val_raw[:14843],l_val_raw[14843:]]
    l_test_raw  =load_features('test',use_location=True)
    l_test_raw =[l_test_raw[:11204],l_test_raw[11204:]]
    time_val_raw = np.load(os.path.join(VAL_PATH,"Hand","Label.npy"))[:,0,0]
    time_val_raw = [time_val_raw[:14843],time_val_raw[14843:]]
    X_test_raw = [
        np.load(os.path.join(TEST_PATH,DATA_FILENAME+"_user2.npy")),
        np.load(os.path.join(TEST_PATH,DATA_FILENAME+"_user3.npy")),
    ]
    time_test_raw = [
        np.load(os.path.join(TEST_PATH,LABEL_FILENAME+"_user2.npy")),
        np.load(os.path.join(TEST_PATH,LABEL_FILENAME+"_user3.npy")),
    ]
    return [X_train,l_train,y_train],[X_val_raw,l_val_raw,y_val_raw,time_val_raw],[X_test_raw,l_test_raw,time_test_raw]
  


def train(train_data,config=config):    
    if os.path.isdir(os.path.join(OUTPUT_DIR,config.model_dirname)):
        raise ValueError(os.path.join(OUTPUT_DIR,config.model_dirname)+" already exists.")

    X_train,l_train,y_train = train_data
    
    checkpoint_path = os.path.join(OUTPUT_DIR,config.model_dirname,"cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = get_model()
    model.fit(
    DataGenerator(X_train,y_train,l_train,batch_size=config.batch_size),
    epochs=config.epochs,
    steps_per_epoch=len(X_train) // config.batch_size,
    callbacks=[
        LearningRateScheduler(CosineDecayRestarts( config.learning_rate,5,)),
        ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1),
        TensorBoard(log_dir='../../output/logs/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),batch_size=config.batch_size,histogram_freq = 1,write_grads=True)
    ],
    verbose=1
    )