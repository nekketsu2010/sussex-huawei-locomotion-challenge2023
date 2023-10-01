import numpy as np
import const_prep as const
import os
from tqdm import tqdm

def txt2npy(input_directory, position, filename, output_directory):
    if os.path.isfile(output_directory + position + '/' + filename + '.npy'):
        print(output_directory + position + '/' + filename + '.npy' + 'is already exists.')
        return
    
    if filename == 'GPS':
        x = np.genfromtxt(input_directory + position + '/' + filename + '.txt', dtype='str', delimiter=',')
    else:
        x = np.genfromtxt(input_directory + position + '/' + filename + '.txt')
    np.save(output_directory + position + '/' + filename, x)

def exec_txt2npy(config):
    paths = [
            config["data_preparation"]["train_txt_dir"],
            config["data_preparation"]["val_txt_dir"],
            config["data_preparation"]["test_txt_dir"]
            ]
    output_directories = [
        os.path.join(config['data_dir'],const.RAWNPY_DIR,'train/'),
        os.path.join(config['data_dir'],const.RAWNPY_DIR,'validate/'),
        os.path.join(config['data_dir'],const.RAWNPY_DIR,'test/'),
    ]

    for i in range(len(output_directories)):
        # When testing, there is no need to consider the phone location.
        if('test' in output_directories[i]):
            os.makedirs(output_directories[i],exist_ok=True)
            for filename in const.RAW_FILENAMES:
                txt2npy(paths[i][:-1], '', filename, output_directories[i])
            txt2npy(paths[i], '', 'Label_idx', output_directories[i])
        else:
            for position in tqdm(const.POS):
                os.makedirs(output_directories[i] + position, exist_ok=True)
                for filename in const.RAW_FILENAMES:
                    txt2npy(paths[i], position, filename, output_directories[i])
                txt2npy(paths[i], position, 'Label', output_directories[i])