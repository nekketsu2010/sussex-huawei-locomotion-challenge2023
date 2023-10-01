import numpy as np

def extract_change_indexes(label, type):
    change_indexes = np.where(np.abs(np.diff(label[:, 0])) > 10)[0]
    np.save('../output/' + type + '/ChangeIndexes', change_indexes)

def exec_extract_change_indexes(config):
    training_label = np.load(config['data_dir'] + '/npy/train/Bag/Label.npy')
    extract_change_indexes(training_label, 'train')

    validation_label = np.load(config['data_dir'] + '/npy/validate/Bag/Label.npy')
    extract_change_indexes(validation_label, 'validate')

    test_label = np.load(config['data_dir'] + '/npy/test/Label_idx.npy')
    extract_change_indexes(test_label, 'test')