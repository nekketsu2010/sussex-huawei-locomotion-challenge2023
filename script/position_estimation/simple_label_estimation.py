import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import confusion_matrix,f1_score, classification_report
import matplotlib.pyplot as plt
import pickle
import yaml
from const_pos import POS,LABEL,RAWNPY_DIR_500,F_SIMPLE_LABEL_DIR,RELIABLE_INDEXES_DIR

def load_data(input_dir,label_dir):
    train_arr = []
    val_arr = []
    for pos in POS:
        train_arr.append(np.load(os.path.join(input_dir,'train',pos+'.npy'))[:,1:])
        val_arr.append(np.load(os.path.join(input_dir,'validate',pos+'.npy'))[:,1:])
    X_train = np.concatenate(train_arr)
    X_val = np.concatenate(val_arr)
    X_test = np.load(os.path.join(input_dir,'test','test.npy'))[:,1:]
    # label
    train_label = np.load(os.path.join(label_dir,'train','Label_every_5s.npy'))[:,0,1]-1
    val_label = np.load(os.path.join(label_dir,'validate','Label_every_5s.npy'))[:,0,1]-1
    y_train = np.r_[train_label,train_label,train_label,train_label]
    y_val = np.r_[val_label,val_label,val_label,val_label]
    return X_train,y_train,X_val,y_val,X_test
    
def train(X_train,y_train):
    model = xgb.XGBClassifier(max_depth=18, min_child_weight=7, learning_rate=0.1, gamma=0.005, sub_sample=0.9, colsample_bytree=0.8, 
                          n_jobs=-1, tree_method='gpu_hist', gpu_id=0)
    model.fit(X_train,y_train,verbose=False)
    return model

def predict(model,X_val,X_test):
    pred_val = model.predict(X_val)
    pred_test_proba = model.predict_proba(X_test)
    return pred_val,pred_test_proba

def save_reliable_indexes(pred_test_proba,output_dir):
    test_reliable_samples = []
    for i in range(8):
        arr = np.where(pred_test_proba[:,i] > 0.75)
        print(LABEL[i],len(arr[0]))
        test_reliable_samples.append(arr)
    for i in range(8):
        os.makedirs(os.path.join(output_dir),exist_ok=True)
        np.save(os.path.join(output_dir,LABEL[i]+'.npy'),test_reliable_samples[i])
   
        
def calc_stats(pred_test_proba):
    label_argmax = []
    for i in range(len(LABEL)):
        label_argmax.append(np.where(np.argmax(pred_test_proba,axis=1)==i)[0])

    label_mean = np.zeros(8)
    label_se = np.zeros(8)
    for i in range(len(LABEL)):
        label_mean[i]=np.mean(pred_test_proba[label_argmax[i],i])
        label_se[i] = np.std(pred_test_proba[label_argmax[i],i],ddof=1)/np.sqrt(label_argmax[i].shape)
    print(label_mean,label_se)
    return label_mean,label_se

def main(config):
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    label_dir = os.path.join(config['data_dir'],RAWNPY_DIR_500)
    features_dir = os.path.join(config['data_dir'],F_SIMPLE_LABEL_DIR)
    output_dir =  os.path.join(config['data_dir'],RELIABLE_INDEXES_DIR)
    X_train,y_train,X_val,y_val,X_test = load_data(features_dir,label_dir)
    model = train(X_train,y_train)
    pred_val,pred_test_proba = predict(model,X_val,X_test)
    save_reliable_indexes = save_reliable_indexes(pred_test_proba,output_dir)

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)