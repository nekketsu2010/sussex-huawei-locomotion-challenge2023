import numpy as np
import os 
from utils import savenpy,file_exist
import const_prep as const
# require magnitude .py files 
def extract_features(input_path,output_path):
    """
    extract features for position estimation.
    needs to call for each position.
    Args:
        input_path (str): path of input magnitude.npy file 
        output_path (str): path of output .npy file
    """
    data = np.load(input_path)
    timestamp = data[:,0,0]
    data = data[:,:,1:]
    data_ac = data-np.expand_dims(data.mean(axis=1),axis=1)
    data_fft = np.fft.fft(data_ac,n=data.shape[1],axis=1)[:,:data.shape[1]//2,:]
    data_fft = np.abs(data_fft)
    indexes = np.argpartition(-data_fft,1,axis=1)[:,:2,:]
    print("indexes.shape",indexes.shape)

    max_indexes = indexes[:,0,:]
    print("data_fft.shape",data_fft.shape,np.expand_dims(indexes[:,0,:], axis=1).shape)
    max_values = np.take_along_axis(data_fft, np.expand_dims(indexes[:,0,:], axis=1), axis=1)
    second_values = np.take_along_axis(data_fft, np.expand_dims(indexes[:,1,:], axis=1), axis=1)
    ratio = (max_values/second_values).squeeze()

    means = data.mean(axis=1)[:,:2]
    stds = data.std(axis=1) 
    features = np.hstack([timestamp.reshape((-1,1)),max_indexes[:,0].reshape((-1,1)),ratio[:,0].reshape((-1,1)),means,stds])
    savenpy(output_path,features)


def extract_magnitude(type,input_dir,output_dir):
    """
    extract magnitude of each sensors and positions.
    input and output data shape is (-1, 500, 4).
    Args:
        type (str): type of data to extract. [train, validate, test]
        input_dir (dir): input directory of 5s sensor .npy file. 
        output_dir (_type_): output directory of magnitude .npy file
    """
    if type=='test':
        extract_magnitude_test(input_dir,output_dir)
        return
    for pos in const.POS:
        if file_exist(os.path.join(output_dir,type,pos,const.OUTPUT_FILENAME)):
            continue
        acc = np.load(os.path.join(input_dir,type,pos,const.SENSOR_IN[0]))
        gyr = np.load(os.path.join(input_dir,type,pos,const.SENSOR_IN[1]))
        mag = np.load(os.path.join(input_dir,type,pos,const.SENSOR_IN[2])) 
        timestamp = acc[:,:,0]
        acc = np.linalg.norm(acc[:,:,1:],axis = 2)
        gyr = np.linalg.norm(gyr[:,:,1:],axis = 2)
        mag = np.linalg.norm(mag[:,:,1:],axis = 2)
        result = np.stack([timestamp,acc,gyr,mag],axis=2)
        print(pos,mag.shape,result.shape)
        savenpy(os.path.join(output_dir,type,pos,const.OUTPUT_FILENAME),result)

def extract_magnitude_test(input_dir,output_dir):
    if file_exist(os.path.join(output_dir,'test',const.OUTPUT_FILENAME)):
        return
    acc = np.load(os.path.join(input_dir,'test',const.SENSOR_IN[0]))
    gyr = np.load(os.path.join(input_dir,'test',const.SENSOR_IN[1]))
    mag = np.load(os.path.join(input_dir,'test',const.SENSOR_IN[2])) 
    timestamp = acc[:,:,0]
    acc = np.linalg.norm(acc[:,:,1:],axis = 2)
    gyr = np.linalg.norm(gyr[:,:,1:],axis = 2)
    mag = np.linalg.norm(mag[:,:,1:],axis = 2)
    result = np.stack([timestamp,acc,gyr,mag],axis=2)
    print(mag.shape,result.shape)
    savenpy(os.path.join(output_dir,'test',const.OUTPUT_FILENAME),result)
