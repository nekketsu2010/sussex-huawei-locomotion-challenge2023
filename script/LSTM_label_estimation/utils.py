import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import yaml

class config:
  epochs=155
  num_layers=3
  embed_layer_size=128
  fc_layer_size=256
  dropout=0.2
  optimizer='adamw'
  amsgrad=False
  label_smoothing=0.1
  learning_rate=1e-3
  weight_decay=5e-5
  warmup_steps=10
  batch_size=512
  global_clipnorm=3.0
  data_dim = 9
  location_dim = 11
  earlystopping_patience=30
  model_dirname= '0624'
  model_dirname_finetune= model_dirname+'_finetune8'
  lr_finetuning = 5e-5
  epoch_finetuning = 70


def get_config():
    PATH = '../config.yaml'
    with open(PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_confusion_matrix(test_y,pred_y,normalize=True, fontsize=16, vmin=0, vmax=1, axis=1):
    class_names = ['Still', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
    cm = (confusion_matrix(test_y,pred_y))
    if normalize:
        cm_rate = cm.astype('float') / cm.sum(axis=axis, keepdims=True)
    
    if len(class_names) <= 3:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(cm_rate, interpolation='nearest', cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label\n',
           xlabel='\nPredicted label')
    ax.set_ylabel('True label\n', fontsize=fontsize)
    ax.set_xlabel('\nPredicted label', fontsize=fontsize)
    ax.set_xticklabels(class_names, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm_rate[i, j] * 100, fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_rate[i, j] > 0.5 else 'black', fontsize=fontsize)
            ax.text(j, i+0.3, '( ' + str(cm[i, j]) + ' )', ha="center",
                    va="center",
                    color="white" if cm_rate[i, j] > 0.5 else 'black', fontsize=fontsize//2)
    fig.tight_layout()
    
    print(f1_score(test_y, pred_y, average='macro'))
    print(classification_report(test_y, pred_y))
    return fig

def split_val2(X,y,user_split=False,p=0.33):
    vx2,tx2,vy2,ty2 = train_test_split(X[0],y[0],test_size=p,random_state=1,stratify=y[0])
    vx3,tx3,vy3,ty3 = train_test_split(X[1],y[1],test_size=p,random_state=1,stratify=y[1])
    vx = np.r_[vx2,vx3]
    vy = np.r_[vy2,vy3]
    tx = np.r_[tx2,tx3]

    ty = np.r_[ty2,ty3]
    if user_split:
        return ([vx2,vy2,tx2,ty2],[vx3,vy3,tx3,ty3])
    return vx,vy,tx,ty

def stratifiedKFold_on_val(X,y):
    kf = StratifiedKFold(n_splits=3,shuffle=False)
    """_summary_

    Args:
        X (_type_): X[0]: User2, X[1]: User3
        y (_type_): y[0]: User2, y[1]: User3

    Returns:
        _type_: Returns an array containing K-fold indexes for each user2 and user3
    """
    val_2 = []
    test_2 = []
    val_3 = []
    test_3 = []
    for (val, test) in tqdm(kf.split(X[0][:,0], y[0])):
        val_2.append(val)
        test_2.append(test)
    for (val, test) in tqdm(kf.split(X[1][:,0], y[1])):
        val_3.append(val)
        test_3.append(test)
    return [val_2,val_3],[test_2,test_3]

def reformat(index,data,type):
    if type == 'validate':
        arr = np.load('../../data/every_500/validate/Hand/label_dummy.npy')
    elif type== 'test':
        arr = np.load('../../data/every_500/test/label_dummy.npy')
    print(arr.shape)
    bias = 14843 if type == 'validate' else 11204
    for i in range(len(index)): # 2
        for j in range(len(index[i])):#3
            print(i,j,max(index[i][j]))
            if i == 0:
                arr[index[i][j]] = data[j][i]
            else:
                arr[index[i][j]+bias] = data[j][i]
    return arr