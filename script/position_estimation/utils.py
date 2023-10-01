import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score

def plot_confusion_matrix(test_y,pred_y,class_names,normalize=True, fontsize=16, vmin=0, vmax=1, axis=1,title=None):
    cm = (confusion_matrix(test_y,pred_y))
    if normalize:
        cm_rate = cm.astype('float') / cm.sum(axis=axis, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
 

    im = ax.imshow(cm_rate, interpolation='nearest', cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label\n',
           xlabel='\nPredicted label')
    if title != None:
        ax.set_title(title)
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
    return ax

def plot_confusion_matrix_multi(test_y,pred_y,class_names,normalize=False, fontsize=16,figsize=None, vmin=0, vmax=1, axis=1,titles=None,suptitle=None):
    n = len(test_y)
    if figsize is None:
        figsize = (n*2+1.5,n*2-1.5)
    fig  = plt.figure(figsize =figsize)
    fig.supylabel('True label\n', fontsize=fontsize)
    fig.supxlabel('Predicted label', fontsize=fontsize)
    if suptitle != None:
        fig.suptitle(suptitle,fontsize=fontsize)
    for x in range(n):
        ax = fig.add_subplot(1,n,x+1)
        ax.grid(False)
        cm = (confusion_matrix(test_y[x],pred_y[x]))
        if normalize:
            cm_rate = cm.astype('float') / cm.sum(axis=axis, keepdims=True)
            
        if titles != None:
            ax.set_title(titles[x],fontsize=fontsize)
    

        im = ax.imshow(cm_rate, interpolation='nearest', cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
        if x == 0:
            ax.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                )
    
            ax.set_yticklabels(class_names, fontsize=fontsize-2)
        else:
            ax.set(xticks=np.arange(cm.shape[1]),
                yticks=[],
                xticklabels=class_names,
                )    
        ax.set_xticklabels(class_names, fontsize=fontsize-2)
            
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
                        color="white" if cm_rate[i, j] > 0.5 else 'black', fontsize=fontsize-5.5)
        print(f1_score(test_y[x], pred_y[x], average='macro'))
    fig.tight_layout()
    plt.show()
    return

def file_exist(path):
    if os.path.isfile(path):
        print(path+" already exists.")
        return True
    else:
        return False
    
# def savenpy(path, obj):
# save np.array to path, creating dirctories
def savenpy(path,ary):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    np.save(path, ary)  

# convert ['train','validate','test'] to [0,1,2]
def type2idx(type):
    TYPE = ['train','validate','test']
    for i, elem in enumerate(TYPE):
        if elem == type:
            return i
    return None