import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import confusion_matrix,f1_score, classification_report
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split,KFold
import yaml
from const_pos import RAWNPY_DIR_500,F_SIMPLE_LABEL_DIR,RELIABLE_INDEXES_DIR,POS,LABEL


def load_data(features_dir,label_dir,reliable_indexes_dir,label=[2,3,5]):
    load_val_f = [np.load(os.path.join(features_dir,'validate',pos+'.npy'))[:,1:]for pos in POS]

    val_pos_label = [np.ones(load_val_f[i].shape[0])* i for i in range(len(load_val_f))]

    load_val_l = np.load(os.path.join(label_dir,'validate','Label_every_5s.npy'))[:,0,1]
    load_val_l = [load_val_l,load_val_l,load_val_l,load_val_l]

    val_f = np.zeros((load_val_f[0].shape[0]*4,load_val_f[0].shape[1]))
    val_l = np.zeros((val_pos_label[0].shape[0]*4))
    val_state_l = np.zeros((load_val_l[0].shape[0]*4,))

    for i in range(len(load_val_f[0])):
        for j in range(4):
            val_f[i*4+j] = load_val_f[j][i]
            val_l[i*4+j] = val_pos_label[j][i]
            val_state_l[i*4+j] = load_val_l[j][i]
    print(val_f.shape,val_l.shape,val_l[:5])
    load_test_f = np.load(os.path.join(features_dir,'test','test.npy'))[:,1:]
    test_indexes = [np.load(os.path.join(reliable_indexes_dir,LABEL[l-1]+'.npy')).squeeze() for l in label] 
    val_indexes = [np.where(val_state_l == l)[0] for l in label]

    X_val = []
    y_val = []
    X_test = []
    for i in range(len(label)):
        X_val.append(val_f[val_indexes[i],:])    
        y_val.append(val_l[val_indexes[i]])   
        X_test.append(load_test_f[test_indexes[i]])
    return X_val,y_val,X_test

def cross_val_predict(X_val,y_val):
    model = xgb.XGBClassifier(max_depth=18, min_child_weight=7, learning_rate=0.1, gamma=0.005, sub_sample=0.9, colsample_bytree=0.8, 
                          n_jobs=-1, tree_method='gpu_hist', gpu_id=0,random_state=42)

    pred_val = cross_val_predict(model,X_val,y_val,cv=KFold(n_splits=4))
    return pred_val

def predict(X_val,y_val,X_test):
    model = xgb.XGBClassifier(max_depth=18, min_child_weight=7, learning_rate=0.1, gamma=0.005, sub_sample=0.9, colsample_bytree=0.8, 
                        n_jobs=-1, tree_method='gpu_hist', gpu_id=0,random_state=42)
    X_train,X_val,y_train,y_val = train_test_split(X_val,y_val,test_size=0.25,random_state=42,shuffle=False)
    model.fit(X_train,y_train)
    pred_test_proba = model.predict_proba(X_test)
    return pred_test_proba

def save_reliable_indexes(pred_test_proba,output_dir):
    test_reliable_samples = []
    for i in range(8):
        arr = np.where(pred_test_proba[:,i] > 0.75)
        print(LABEL[i],len(arr[0]))
        test_reliable_samples.append(arr)
    for i in range(8):
        os.makedirs(os.path.join(output_dir),exist_ok=True)
        np.save(os.path.join(output_dir,LABEL[i]+'.npy'),test_reliable_samples[i])
   
        
def calc_stats(proba_arr):
    test_stats = []
    for i,pred_test_proba in enumerate(proba_arr):
        n = len(pred_test_proba)
        label_argmax = []
        for i in range(len(POS)):
            label_argmax.append(np.where(np.argmax(pred_test_proba,axis=1)==i)[0])
        n_label = [i.shape[0] for i in label_argmax]
        label_mean = np.zeros(4)
        label_se = np.zeros(4)
        for i in range(len(POS)):
            label_mean[i]=np.mean(pred_test_proba[label_argmax[i],i])
            if n_label[i]:
                label_se[i] = np.std(pred_test_proba[label_argmax[i],i],ddof=1)/np.sqrt(n_label[i])
            else:    
                label_se[i] = None
        test_stats.append([n_label,label_mean,label_se,])

    for i in test_stats:
        print('n:',i[0],'\tmean:',i[1],'\tse:',i[2])
    return test_stats

def print_result(pred_proba,label,p=0.75):
    p = 0.99
    print(f"number of labels which probability is < {p}")
    for i in range(len(label)):
        print('----'+LABEL[label[i]-1]+'----')
        for j,e in enumerate(POS):
            print("{}:{}".format(e, np.where(pred_proba[i][:, j]>=p)[0].shape[0]))

def main(config):
    label_dir = os.path.join(config['data_dir'],RAWNPY_DIR_500)
    features_dir = os.path.join(config['data_dir'],F_SIMPLE_LABEL_DIR)
    reliable_indexes_dir = os.path.join(config['data_dir'],RELIABLE_INDEXES_DIR)
    used_labels = config['position_estimation']['used_labels']
    X_val,y_val,X_test = load_data(features_dir,label_dir,reliable_indexes_dir,used_labels)
    pred_val = []
    pred_test_proba = []
    for i in range(len(used_labels)):
        pred_val.append(cross_val_predict(X_val,y_val))
        pred_test_proba.append(predict(X_val,y_val,X_test))
    calc_stats(pred_test_proba)
    print_result(pred_test_proba,used_labels)   


if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main()