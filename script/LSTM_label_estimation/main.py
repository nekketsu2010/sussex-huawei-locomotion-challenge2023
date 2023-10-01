import numpy as np
import os

from utils import stratifiedKFold_on_val,reformat,config
from training import train,load_data
from testing import predict,predict_all_epochs
from finetuning import finetuning

OUTPUT_DIR = '../../output/model/MSRLSTM/'

def main(train_data,val_data_raw,test_data,do_train = False):
    # train: X,location,y
    # val: X,location,y,timestamp
    # test: X,location,timestamp
    pred_val_prefinetune = [[],[],[]]
    pred_val = [[],[],[]]
    pred_test = [[],[],[]]
    val_i,val_test_i = stratifiedKFold_on_val(val_data_raw[0],val_data_raw[2])
    print(f"fold number : {len(val_i)}")
    for i in range(len(val_i[0])):
        print(f"fold {i} val: user2:{len(val_i[0][i])}, user3:{len(val_i[1][i])}")
        print(f"fold {i} test: user2:{len(val_test_i[0][i])}, user3:{len(val_test_i[1][i])}")
    if do_train:
        train(train_data,None)
    else:
        del train_data
    for i in range(len(val_i[0])):
        for j,user in enumerate([2,3]):   
            val_data = [data[j][val_i[j][i]] for data in val_data_raw]
            val_test_data = [data[j][val_test_i[j][i]] for data in val_data_raw]
            test_data_ = [data[j] for data in test_data]  
            best_epoch = predict_all_epochs(val_test_data[:-1],user=user,n_fold=i+1,return_best_loss=True)
            pred = predict(val_test_data,user=user,get_score=False,epoch=best_epoch)
            pred_val_prefinetune[i].append(pred)
            val_pred,test_pred = finetuning(val_data,val_test_data,test_data_,user=user,n_fold=i+1,epoch=best_epoch)
            pred_val[i].append(val_pred)
            pred_test[i].append(test_pred)
    pred_val_prefinetune_formatted = reformat(val_test_i,pred_val_prefinetune,type='validate')
    pred_val_formatted = reformat(val_test_i,pred_val,type='validate')
    pred_test_formatted = [np.concatenate(arr,axis=0) for arr in pred_test]
    dir =  os.path.join(OUTPUT_DIR,config.model_dirname) 
    dir2 =  os.path.join(OUTPUT_DIR,config.model_dirname_finetune) 
    os.makedirs(dir,exist_ok=True)
    os.makedirs(dir2,exist_ok=True)
    np.save(os.path.join(dir,'pred_val.npy'),pred_val_prefinetune_formatted)
    np.save(os.path.join(dir2,'pred_val.npy'),pred_val_formatted)
    np.save(os.path.join(dir2,'pred_test_1.npy'),pred_test_formatted[0])
    np.save(os.path.join(dir2,'pred_test_2.npy'),pred_test_formatted[1])
    np.save(os.path.join(dir2,'pred_test_3.npy'),pred_test_formatted[2])


if __name__ == '__main__':
    train_data,val_data,test_data=load_data()
    main(train_data,val_data,test_data)