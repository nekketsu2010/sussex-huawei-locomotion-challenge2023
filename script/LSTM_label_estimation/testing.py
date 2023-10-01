import pathlib
import glob
import os
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from MSRLSTM import get_model,load_weights
from utils import config,plot_confusion_matrix

OUTPUT_DIR = '../../output/model/MSRLSTM/'

def predict_all_epochs(data,user,n_fold,config=config,return_best_loss=True):
    model = get_model()
    files = glob.glob(os.path.join(OUTPUT_DIR,config.model_dirname,'*.ckpt.index'))
    n_epochs = len(files)
    scce = SparseCategoricalCrossentropy()

    val_acc = np.zeros(n_epochs)
    val_loss = np.ones(n_epochs)*3
    for i,f in enumerate(tqdm(files[60:76])):
        file_ckpt = os.path.splitext(f)[0]
        model.load_weights(file_ckpt)
        pred = model.predict(data[:2],verbose=0)
        val_acc[i] = accuracy_score(data[2],np.argmax(pred,axis=1))
        val_loss[i] = scce(data[2],pred)
    print(f"max val_acc:epoch {np.argmax(val_acc)+1}, {np.max(val_acc)}")
    print(f"min val_loss:epoch {np.argmin(val_loss)+1}, {np.min(val_loss)}")
    fig,axes = plt.subplots(2,1)
    axes[0].plot(np.arange(n_epochs),val_acc)
    axes[1].plot(np.arange(n_epochs),val_loss)
    fig.show()
    fig.savefig(os.path.join(OUTPUT_DIR,config.model_dirname,f"learning_curve_user{user}_fold{n_fold}.png"))
    if return_best_loss:
        return np.argmin(val_loss)+1+60
    else:
        return np.argmax(val_acc)+1+60

def predict(data,user,config=config,model=get_model(),finetuning=False,earlystopping=True,get_score=False,epoch=None):
    if not finetuning:
        dir =  os.path.join(OUTPUT_DIR,config.model_dirname)
        dir2 = os.path.join(OUTPUT_DIR,config.model_dirname,config.model_dirname)
    else:
        dir =  os.path.join(OUTPUT_DIR,config.model_dirname_finetune+'_'+str(user))
        dir2 =  os.path.join(OUTPUT_DIR,config.model_dirname_finetune+'_'+str(user),config.model_dirname_finetune+'_'+str(user))
    load_weights(model,dir,config=config,epoch=epoch,earlystopping=earlystopping)
    pred = model.predict(data[:2])
    if get_score:
        fig = plot_confusion_matrix(np.argmax(pred,axis=1),data[2])
        fig.savefig(os.path.join(dir2,'cm.png'))
    return np.concatenate([data[-1].reshape(-1,1),pred],axis=1)