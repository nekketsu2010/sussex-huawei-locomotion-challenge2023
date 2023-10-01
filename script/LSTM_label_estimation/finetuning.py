from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

import numpy as np
import os

from sklearn.metrics import confusion_matrix,f1_score,classification_report

from utils import config,plot_confusion_matrix
from MSRLSTM import cosine_schedule,DataGenerator,get_model,load_weights

OUTPUT_DIR = '../../output/model/MSRLSTM/'

def freeze_model(model):
    freeze_l = []
    train_l = []
    for i,l in enumerate(model.layers):
        n =l.name
        if n.startswith("residual") or n.startswith("conv1d"):
            freeze_l.append(i)
        elif n.startswith("lstm") or n.startswith("dense"):
            train_l.append(i)
    model.trainable = True
    for i in freeze_l:
        model.layers[i].trainable = False
    for i in train_l:
        model.layers[i].trainable = True
    
    return model

def finetuning(val_data,val_test_data,test_data,user,n_fold,config=config,epoch=None):
    model = get_model()
    if epoch is not None:
        load_weights(model,os.path.join(OUTPUT_DIR,config.model_dirname),epoch=epoch)
    else:
        epoch = load_weights(model,os.path.join(OUTPUT_DIR,config.model_dirname))
    print(epoch)
    model = freeze_model(model)

    X_val,l_val,y_val,t_val = val_data
    X_val_test,l_val_test,y_val_test,t_val_test = val_test_data
    X_test,l_test,t_test = test_data
    print(X_val_test.shape,y_val_test.shape,l_val_test.shape)
    print(np.unique(y_val,return_counts=True))
    print(np.unique(y_val_test,return_counts=True))
    checkpoint_path = os.path.join(OUTPUT_DIR,config.model_dirname_finetune+'_'+str(user)+'_'+str(n_fold),"cp-{epoch:04d}.ckpt")

    model.fit(
    DataGenerator(X_val,y_val,l_val,batch_size=config.batch_size),
    validation_data = DataGenerator(X_val_test,y_val_test,l_val_test,batch_size=config.batch_size),
    epochs=config.epoch_finetuning,
    steps_per_epoch=len(X_val) // config.batch_size,
    validation_steps=len(X_val_test) // config.batch_size,
    callbacks=[
        LearningRateScheduler(cosine_schedule(base_lr=config.lr_finetuning, total_steps=config.epoch_finetuning, warmup_steps=config.warmup_steps)),
        ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,monitor='val_loss',save_best_only=True,verbose=0),
        # EarlyStopping(monitor="val_accuracy", mode='max', min_delta=0.001, patience=config.earlystopping_patience),
        # TensorBoard(log_dir='../../output/logs/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),batch_size=config.batch_size,histogram_freq = 1,write_grads=True)
    ],
    verbose=1
    )
    out = model.predict([X_val_test,l_val_test])
    pred = np.argmax(out, axis=-1)
    fig = plot_confusion_matrix(y_val_test,pred)
    pred_test = model.predict([X_test,l_test])
    model_dir=os.path.join(OUTPUT_DIR,config.model_dirname_finetune+'_'+str(user),config.model_dirname_finetune+'_'+str(user)+'_'+str(n_fold))
    os.makedirs(os.path.join(model_dir),exist_ok=True)
    np.save(os.path.join(model_dir,'pred_val_'+str(user)+'.npy'),out)
    np.save(os.path.join(model_dir,'true_val_'+str(user)+'.npy'),y_val_test)
    fig.savefig(os.path.join(model_dir,'cm_'+str(user)+'.png'))
    return  np.concatenate([t_val_test.reshape(-1,1),out],axis=1),np.concatenate([t_test.reshape(-1,1),pred_test],axis=1)